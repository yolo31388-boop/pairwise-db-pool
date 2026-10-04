"""
游戏内数据库连接池与分库分表系统 - 核心模块

包含五个子系统：
- ConnectionPool   连接池：最大连接数限制 + 获取超时
- Sharding         分库分表：哈希取模规则 + 查询路由
- ReadWriteSplit   读写分离：读写规则 + 主从延迟窗口内读主库
- LoadBalancer     负载均衡：平滑加权轮询 + 健康检查
- TransactionManager 事务管理：两阶段提交 + 失败补偿回滚
"""
import hashlib
import itertools
import threading
import time
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple


class PoolExhaustedError(TimeoutError):
    """连接池已达最大连接数，且在超时时间内未获取到连接。"""


class NoHealthyReplicaError(RuntimeError):
    """没有健康从库可用。"""


class TransactionError(RuntimeError):
    """事务执行失败。"""


@dataclass
class Config:
    pass


@dataclass
class Replica:
    """从库节点：weight 用于负载均衡权重，healthy 由健康检查维护，lag 为主从延迟（秒）。"""

    id: str
    weight: int = 1
    healthy: bool = True
    lag: float = 0.0


# ---------------------------------------------------------------- 连接池


class _Connection:
    _ids = itertools.count(1)

    def __init__(self) -> None:
        self.id = next(self._ids)
        self.created_at = time.monotonic()

    def __repr__(self) -> str:
        return f"<Connection #{self.id}>"


class ConnectionPool:
    """带最大连接数和获取超时的连接池，连接可被复用而不是每次查询新建。"""

    def __init__(self, max_connections: int = 10, timeout: float = 1.0) -> None:
        if max_connections <= 0:
            raise ValueError("max_connections 必须为正数")
        if timeout < 0:
            raise ValueError("timeout 不能为负数")
        self.max_connections = max_connections
        self.timeout = timeout
        self._idle: List[_Connection] = []
        self._in_use = 0
        self._cond = threading.Condition()

    @property
    def size(self) -> int:
        """当前已创建的连接总数（空闲 + 使用中）。"""
        return len(self._idle) + self._in_use

    @property
    def in_use(self) -> int:
        return self._in_use

    def acquire(self, timeout: Optional[float] = None) -> _Connection:
        """获取一个连接；池满时最多等待 timeout 秒，超时抛出 PoolExhaustedError。"""
        wait = self.timeout if timeout is None else timeout
        deadline = time.monotonic() + wait
        with self._cond:
            while True:
                if self._idle:
                    self._in_use += 1
                    return self._idle.pop()
                if self.size < self.max_connections:
                    self._in_use += 1
                    return _Connection()
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    raise PoolExhaustedError(
                        f"获取连接超时：连接池已达最大连接数 {self.max_connections}"
                    )
                self._cond.wait(remaining)

    def release(self, conn: _Connection) -> None:
        """归还连接，唤醒等待中的获取请求。"""
        with self._cond:
            if self._in_use <= 0:
                raise RuntimeError("释放的连接不属于本连接池")
            self._in_use -= 1
            self._idle.append(conn)
            self._cond.notify()


# ---------------------------------------------------------------- 分库分表


class Sharding:
    """分库分表：按 key 的哈希取模确定分片与分表，查询按同一规则路由。"""

    def __init__(self, num_shards: int = 4, tables_per_shard: int = 4,
                 table_prefix: str = "t") -> None:
        if num_shards <= 0 or tables_per_shard <= 0:
            raise ValueError("分片数和分表数必须为正数")
        self.num_shards = num_shards
        self.tables_per_shard = tables_per_shard
        self.table_prefix = table_prefix
        # shard_id -> {table_name -> {key: value}}
        self._data: Dict[int, Dict[str, Dict[Any, Any]]] = {
            shard: {} for shard in range(num_shards)
        }

    @staticmethod
    def _hash(key: Any) -> int:
        return int(hashlib.md5(str(key).encode("utf-8")).hexdigest(), 16)

    def shard_for(self, key: Any) -> int:
        """分库规则：hash(key) % num_shards。"""
        return self._hash(key) % self.num_shards

    def table_for(self, key: Any) -> str:
        """分表规则：hash(key) % tables_per_shard。"""
        return f"{self.table_prefix}_{self._hash(key) % self.tables_per_shard}"

    def route(self, key: Any) -> Tuple[int, str]:
        """路由：返回 key 应读写的 (分片, 分表)。"""
        return self.shard_for(key), self.table_for(key)

    def write(self, key: Any, value: Any) -> Tuple[int, str]:
        shard, table = self.route(key)
        self._data[shard].setdefault(table, {})[key] = value
        return shard, table

    def read(self, key: Any) -> Any:
        shard, table = self.route(key)
        return self._data[shard].get(table, {}).get(key)

    def shard_sizes(self) -> Dict[int, int]:
        """每个分片上的数据量，用于验证数据确实分布到多个库。"""
        return {
            shard: sum(len(rows) for rows in tables.values())
            for shard, tables in self._data.items()
        }


# ---------------------------------------------------------------- 负载均衡


class LoadBalancer:
    """从库负载均衡：平滑加权轮询 + 健康检查，不健康的从库不再接收请求。"""

    def __init__(self, replicas: List[Replica]) -> None:
        self.replicas = list(replicas)
        self._current: Dict[str, int] = {r.id: 0 for r in self.replicas}

    def healthy_replicas(self) -> List[Replica]:
        return [r for r in self.replicas if r.healthy]

    def health_check(self, replica_id: str, ok: bool) -> None:
        """健康检查结果上报：ok=False 的从库会被摘出负载均衡。"""
        for replica in self.replicas:
            if replica.id == replica_id:
                replica.healthy = bool(ok)
                return
        raise KeyError(f"未知从库: {replica_id}")

    def select_replica(self, candidates: Optional[List[Replica]] = None) -> Replica:
        """平滑加权轮询：一个完整周期内各从库被选次数严格等于其权重。"""
        pool = [r for r in (candidates if candidates is not None else self.replicas)
                if r.healthy]
        if not pool:
            raise NoHealthyReplicaError("没有健康从库可用")
        total = sum(max(r.weight, 0) for r in pool)
        if total <= 0:
            total = len(pool)
            weights = {r.id: 1 for r in pool}
        else:
            weights = {r.id: max(r.weight, 0) for r in pool}
        best = None
        for replica in pool:
            self._current[replica.id] += weights[replica.id]
            if best is None or self._current[replica.id] > self._current[best.id]:
                best = replica
        self._current[best.id] -= total
        return best


# ---------------------------------------------------------------- 读写分离


_READ_PREFIXES = ("SELECT", "SHOW", "DESCRIBE", "DESC", "EXPLAIN")


class ReadWriteSplit:
    """读写分离：写请求走主库，读请求走健康且低延迟的从库；
    写入后的主从延迟窗口内，读请求回主库以保证读到最新数据。"""

    def __init__(self, balancer: LoadBalancer, max_lag: float = 0.5) -> None:
        self.balancer = balancer
        self.max_lag = max_lag
        self._last_write_at: Optional[float] = None

    @staticmethod
    def is_read(sql: str) -> bool:
        """分离规则：按 SQL 首关键字判断读/写。"""
        parts = sql.lstrip().split(None, 1)
        return bool(parts) and parts[0].upper() in _READ_PREFIXES

    def execute(self, sql: str) -> Tuple[str, str]:
        """返回 (目标类型, 节点id)，目标类型为 "master" 或 "replica"。"""
        if not self.is_read(sql):
            self._last_write_at = time.monotonic()
            return ("master", "master")
        # 主从延迟窗口内的读请求走主库，避免读不到刚写入的数据
        if (self._last_write_at is not None
                and time.monotonic() - self._last_write_at < self.max_lag):
            return ("master", "master")
        # 跳过延迟超过阈值的从库，防止读到过期数据
        candidates = [r for r in self.balancer.healthy_replicas()
                      if r.lag <= self.max_lag]
        if not candidates:
            return ("master", "master")
        replica = self.balancer.select_replica(candidates)
        return ("replica", replica.id)


# ---------------------------------------------------------------- 事务管理


_MISSING = object()


class Transaction:
    """跨分片事务：commit 时先两阶段提交，提交阶段失败则对已提交分片做补偿回滚。"""

    def __init__(self, manager: "TransactionManager", txn_id: int) -> None:
        self.manager = manager
        self.txn_id = txn_id
        self._ops: List[Tuple[Any, Any]] = []
        self.state = "active"  # active -> committed | aborted

    def write(self, key: Any, value: Any) -> None:
        if self.state != "active":
            raise TransactionError("事务已结束，不能再写入")
        self._ops.append((key, value))

    def rollback(self) -> None:
        """主动回滚：提交前丢弃所有缓冲的写操作。"""
        if self.state != "active":
            raise TransactionError("事务已结束")
        self._ops.clear()
        self.state = "aborted"

    def commit(self) -> None:
        if self.state != "active":
            raise TransactionError("事务已结束")
        sharding = self.manager.sharding
        # 记录每项写入的旧值，作为补偿回滚的依据
        undo = []
        for key, _value in self._ops:
            shard, table = sharding.route(key)
            old = sharding._data[shard].get(table, {}).get(key, _MISSING)
            undo.append((shard, table, key, old))

        # 阶段一：prepare，任一分片失败则整个事务中止，此时未写入任何数据
        for shard, _table, _key, _old in undo:
            if shard in self.manager.fail_prepare_on:
                self.state = "aborted"
                raise TransactionError(f"分片 {shard} prepare 失败，事务已回滚")

        # 阶段二：commit，失败时补偿已提交的分片
        committed = []
        try:
            for (key, value), (shard, table, _key, old) in zip(self._ops, undo):
                if shard in self.manager.fail_commit_on:
                    raise TransactionError(f"分片 {shard} commit 失败")
                sharding._data[shard].setdefault(table, {})[key] = value
                committed.append((shard, table, key, old))
        except TransactionError:
            for shard, table, key, old in reversed(committed):
                if old is _MISSING:
                    sharding._data[shard][table].pop(key, None)
                else:
                    sharding._data[shard][table][key] = old
            self.state = "aborted"
            raise
        self.state = "committed"


class TransactionManager:
    """事务管理器：基于分片路由的两阶段提交 + 补偿。
    fail_prepare_on / fail_commit_on 用于注入分片故障（测试与演练用）。"""

    def __init__(self, sharding: Sharding) -> None:
        self.sharding = sharding
        self.fail_prepare_on: set = set()
        self.fail_commit_on: set = set()
        self._ids = itertools.count(1)

    def begin(self) -> Transaction:
        return Transaction(self, next(self._ids))


# ---------------------------------------------------------------- 门面


_DEFAULT_CONFIG = {
    "max_connections": 10,
    "connection_timeout": 1.0,
    "num_shards": 4,
    "tables_per_shard": 4,
    "table_prefix": "t",
    "replicas": [{"id": "replica-1", "weight": 1}],
    "max_replication_lag": 0.5,
}


class DBPool:
    """游戏内数据库连接池与分库分表系统门面。"""

    def __init__(self, config: Optional[Dict] = None):
        merged = dict(_DEFAULT_CONFIG)
        merged.update(config or {})
        self.config = merged
        self._state: Dict = {}
        self._history: List = []
        self._build()

    def _build(self) -> None:
        cfg = self.config
        self._pool = ConnectionPool(cfg["max_connections"], cfg["connection_timeout"])
        self._sharding = Sharding(
            cfg["num_shards"], cfg["tables_per_shard"], cfg["table_prefix"]
        )
        replicas = [
            r if isinstance(r, Replica) else Replica(**r) for r in cfg["replicas"]
        ]
        self._balancer = LoadBalancer(replicas)
        self._rw_split = ReadWriteSplit(self._balancer, cfg["max_replication_lag"])
        self._txn_manager = TransactionManager(self._sharding)

    def connection_pool(self) -> ConnectionPool:
        return self._pool

    def sharding(self) -> Sharding:
        return self._sharding

    def read_write_split(self) -> ReadWriteSplit:
        return self._rw_split

    def load_balance(self) -> LoadBalancer:
        return self._balancer

    def transaction_manager(self) -> TransactionManager:
        return self._txn_manager

    def update(self, dt: float):
        pass

    def reset(self):
        self._state = {}
        self._history = []
        self._build()
