"""
游戏内数据库连接池与分库分表系统 - 核心模块

子系统：
- ConnectionPool      连接池（最大连接数 + 获取超时 + 连接复用）
- ShardingRouter      分库分表（确定性哈希规则 + 路由）
- ReadWriteSplitter   读写分离（SQL 分类规则 + 主从延迟窗口内读主库）
- LoadBalancer        负载均衡（平滑加权轮询 + 健康检查）
- TransactionManager  事务管理（两阶段提交 + 失败补偿）
"""
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple
import zlib


class PoolTimeoutError(Exception):
    """在超时时间内无法从连接池获取连接"""


class TransactionError(Exception):
    """分布式事务提交失败"""


@dataclass
class Config:
    max_connections: int = 4
    pool_timeout: float = 1.0
    num_shards: int = 2
    tables_per_shard: int = 2
    replication_delay: float = 0.5


# ---------------------------------------------------------------------------
# 模拟数据库节点
# ---------------------------------------------------------------------------
class Database:
    """模拟数据库节点（主库或从库），数据保存在内存表中。"""

    def __init__(self, name: str, role: str, weight: int = 1, lag: float = 0.0):
        self.name = name
        self.role = role  # "master" / "slave"
        self.weight = weight
        self.lag = lag
        self.healthy = True
        self.fail_prepare = False
        self.fail_commit = False
        self.tables: Dict[str, Dict] = {}
        self.binlog: List[Tuple] = []  # 主库复制日志: (ts, op, table, key, value)
        self.read_count = 0
        self.write_count = 0
        # 负载均衡内部状态
        self._current_weight = 0

    # -- 数据操作 ---------------------------------------------------------
    def write_row(self, table: str, key, value, now: float, log: bool = True):
        self.tables.setdefault(table, {})[key] = value
        self.write_count += 1
        if log and self.role == "master":
            self.binlog.append((now, "set", table, key, value))

    def delete_row(self, table: str, key, now: float, log: bool = True):
        self.tables.get(table, {}).pop(key, None)
        self.write_count += 1
        if log and self.role == "master":
            self.binlog.append((now, "delete", table, key, None))

    def read_row(self, table: str, key):
        self.read_count += 1
        return self.tables.get(table, {}).get(key)

    def has_row(self, table: str, key) -> bool:
        return key in self.tables.get(table, {})


class Shard:
    """一个分片：1 主 + N 从。"""

    def __init__(self, index: int, master: Database, replicas: List[Database]):
        self.index = index
        self.master = master
        self.replicas = replicas
        self._applied = {id(r): 0 for r in replicas}  # 每个从库已应用的 binlog 位置


# ---------------------------------------------------------------------------
# 连接池
# ---------------------------------------------------------------------------
class PooledConnection:
    def __init__(self, pool: "ConnectionPool", cid: int):
        self.pool = pool
        self.id = cid
        self.query_count = 0

    def execute(self, sql: str, key=None, value=None):
        self.query_count += 1
        return self.pool.system.read_write_split().execute(sql, key=key, value=value)

    def close(self):
        self.pool.release(self)


class ConnectionPool:
    """连接池：复用连接、限制最大连接数、获取超时。"""

    def __init__(self, system: "DBPool", max_connections: int, timeout: float):
        self.system = system
        self.max_connections = max_connections
        self.timeout = timeout
        self._idle: List[PooledConnection] = []
        self._active: List[PooledConnection] = []
        self._created = 0

    @property
    def created_count(self) -> int:
        return self._created

    @property
    def active_count(self) -> int:
        return len(self._active)

    @property
    def idle_count(self) -> int:
        return len(self._idle)

    def acquire(self, timeout: Optional[float] = None) -> PooledConnection:
        # 1) 优先复用空闲连接
        if self._idle:
            conn = self._idle.pop()
            self._active.append(conn)
            return conn
        # 2) 未达上限则新建
        if self._created < self.max_connections:
            self._created += 1
            conn = PooledConnection(self, self._created)
            self._active.append(conn)
            return conn
        # 3) 池已满：等待至超时后报错，绝不无限等待
        wait = self.timeout if timeout is None else timeout
        self.system._advance(wait)
        raise PoolTimeoutError(
            "连接池已满（max=%d），等待 %.2fs 后超时" % (self.max_connections, wait)
        )

    def release(self, conn: PooledConnection):
        if conn in self._active:
            self._active.remove(conn)
            self._idle.append(conn)


# ---------------------------------------------------------------------------
# 分库分表
# ---------------------------------------------------------------------------
class ShardingRouter:
    """分库分表路由：基于 crc32 的确定性哈希规则。"""

    def __init__(self, num_shards: int, tables_per_shard: int):
        if num_shards < 1 or tables_per_shard < 1:
            raise ValueError("分片数和每库表数必须 >= 1")
        self.num_shards = num_shards
        self.tables_per_shard = tables_per_shard

    @staticmethod
    def _hash(key) -> int:
        return zlib.crc32(str(key).encode("utf-8"))

    def shard_index(self, key) -> int:
        return self._hash(("shard", key)) % self.num_shards

    def table_name(self, key) -> str:
        return "table_%d" % (self._hash(("table", key)) % self.tables_per_shard)

    def route(self, key) -> Tuple[int, str]:
        """返回 (分库下标, 表名)，同一 key 永远路由到同一位置。"""
        return self.shard_index(key), self.table_name(key)


# ---------------------------------------------------------------------------
# 负载均衡
# ---------------------------------------------------------------------------
class LoadBalancer:
    """平滑加权轮询 + 健康检查。"""

    def __init__(self, replicas: List[Database]):
        self.replicas = replicas
        self.last_health_check: Optional[float] = None

    def health_check(self, now: float = 0.0) -> Dict[str, bool]:
        """探测所有从库，返回 {节点名: 是否健康}。不健康的节点不再接收请求。"""
        self.last_health_check = now
        return {r.name: r.healthy for r in self.replicas}

    def healthy_replicas(self) -> List[Database]:
        return [r for r in self.replicas if r.healthy]

    def pick(self) -> Optional[Database]:
        """平滑加权轮询：权重高的节点获得更多请求；无健康节点返回 None。"""
        candidates = self.healthy_replicas()
        if not candidates:
            return None
        total = sum(r.weight for r in candidates)
        best = None
        for r in candidates:
            r._current_weight += r.weight
            if best is None or r._current_weight > best._current_weight:
                best = r
        best._current_weight -= total
        return best


# ---------------------------------------------------------------------------
# 读写分离
# ---------------------------------------------------------------------------
class ReadWriteSplitter:
    """读写分离：写走主库，读走从库；主从延迟窗口内读主库保证读到最新数据。"""

    READ_KEYWORDS = {"SELECT", "SHOW", "DESCRIBE", "EXPLAIN"}

    def __init__(self, system: "DBPool"):
        self.system = system

    @classmethod
    def classify(cls, sql: str) -> str:
        keyword = sql.strip().split(None, 1)[0].upper() if sql.strip() else ""
        return "read" if keyword in cls.READ_KEYWORDS else "write"

    def execute(self, sql: str, key=None, value=None):
        if self.classify(sql) == "read":
            return self.read(key)
        return self.write(key, value)

    def write(self, key, value):
        shard_idx, table = self.system.sharding().route(key)
        shard = self.system._shards[shard_idx]
        shard.master.write_row(table, key, value, self.system.now)
        self.system._last_write[(shard_idx, table, key)] = self.system.now
        return value

    def read(self, key):
        shard_idx, table = self.system.sharding().route(key)
        shard = self.system._shards[shard_idx]
        balancer = self.system.load_balance()[shard_idx]
        replica = balancer.pick()
        if replica is None:
            # 没有健康从库：回退主库，保证可用性
            return shard.master.read_row(table, key)
        # 主从延迟窗口内的写，读从库可能拿到旧数据，改走主库
        last_write = self.system._last_write.get((shard_idx, table, key))
        if last_write is not None and self.system.now - last_write < replica.lag:
            return shard.master.read_row(table, key)
        return replica.read_row(table, key)


# ---------------------------------------------------------------------------
# 事务管理（两阶段提交 + 补偿）
# ---------------------------------------------------------------------------
class Transaction:
    def __init__(self, manager: "TransactionManager"):
        self.manager = manager
        self._ops: List[dict] = []
        self._finished = False

    def insert(self, key, value, compensate=None):
        """登记一条写操作；compensate 为可选的自定义补偿动作。"""
        if self._finished:
            raise TransactionError("事务已结束")
        system = self.manager.system
        shard_idx, table = system.sharding().route(key)
        self._ops.append({
            "shard_idx": shard_idx,
            "table": table,
            "key": key,
            "value": value,
            "compensate": compensate,
        })

    def commit(self):
        """两阶段提交：prepare 全部成功才 commit；commit 失败则补偿已提交分片。"""
        if self._finished:
            raise TransactionError("事务已结束")
        self._finished = True
        system = self.manager.system
        if not self._ops:
            return

        # ---- 阶段一：prepare ----
        participants = sorted({op["shard_idx"] for op in self._ops})
        for idx in participants:
            master = system._shards[idx].master
            if not master.healthy or master.fail_prepare:
                # 任一参与者 prepare 失败：全局中止，不产生任何部分写入
                raise TransactionError("prepare 失败：分片 %d 不可用，事务已回滚" % idx)

        # ---- 阶段二：commit ----
        committed: List[dict] = []
        try:
            for op in self._ops:
                master = system._shards[op["shard_idx"]].master
                if master.fail_commit:
                    raise TransactionError("commit 失败：分片 %d" % op["shard_idx"])
                master.write_row(op["table"], op["key"], op["value"], system.now)
                committed.append(op)
        except TransactionError:
            # ---- 补偿：撤销已提交的分片，恢复原子性 ----
            for op in reversed(committed):
                master = system._shards[op["shard_idx"]].master
                if op["compensate"] is not None:
                    op["compensate"]()
                else:
                    master.delete_row(op["table"], op["key"], system.now)
            raise

    def rollback(self):
        self._finished = True
        self._ops = []


class TransactionManager:
    def __init__(self, system: "DBPool"):
        self.system = system

    def begin(self) -> Transaction:
        return Transaction(self)


# ---------------------------------------------------------------------------
# 系统门面
# ---------------------------------------------------------------------------
DEFAULT_REPLICAS = (
    {"name": "slave-1", "weight": 1},
    {"name": "slave-2", "weight": 3},
)


class DBPool:
    def __init__(self, config: Optional[Dict] = None):
        self.config = dict(config or {})
        self._build()

    # -- 构建/重置 ---------------------------------------------------------
    def _build(self):
        cfg = self.config
        self._now = 0.0
        self._last_write: Dict[Tuple, float] = {}
        self._history = []

        max_conn = int(cfg.get("max_connections", Config.max_connections))
        timeout = float(cfg.get("pool_timeout", Config.pool_timeout))
        num_shards = int(cfg.get("num_shards", Config.num_shards))
        tables_per_shard = int(cfg.get("tables_per_shard", Config.tables_per_shard))
        delay = float(cfg.get("replication_delay", Config.replication_delay))
        replica_cfgs = cfg.get("replicas") or [dict(r) for r in DEFAULT_REPLICAS]

        self._shards: List[Shard] = []
        for i in range(num_shards):
            master = Database("master-%d" % i, role="master")
            replicas = [
                Database(
                    "%s-s%d" % (rc.get("name", "slave"), i),
                    role="slave",
                    weight=int(rc.get("weight", 1)),
                    lag=float(rc.get("lag", delay)),
                )
                for rc in replica_cfgs
            ]
            self._shards.append(Shard(i, master, replicas))

        self._pool = ConnectionPool(self, max_conn, timeout)
        self._router = ShardingRouter(num_shards, tables_per_shard)
        self._balancers = [LoadBalancer(s.replicas) for s in self._shards]
        self._splitter = ReadWriteSplitter(self)
        self._tx_manager = TransactionManager(self)

    def reset(self):
        self._build()

    # -- 虚拟时钟：推进时间并应用主从复制 -----------------------------------
    @property
    def now(self) -> float:
        return self._now

    def _advance(self, dt: float):
        if dt < 0:
            raise ValueError("dt 必须 >= 0")
        self._now += dt
        for shard in self._shards:
            for replica in shard.replicas:
                applied = shard._applied[id(replica)]
                log = shard.master.binlog
                while applied < len(log) and log[applied][0] + replica.lag <= self._now:
                    _, op, table, key, value = log[applied]
                    if op == "set":
                        replica.tables.setdefault(table, {})[key] = value
                    else:
                        replica.tables.get(table, {}).pop(key, None)
                    applied += 1
                shard._applied[id(replica)] = applied

    def update(self, dt: float):
        self._advance(dt)

    # -- 子系统入口 ---------------------------------------------------------
    def connection_pool(self) -> ConnectionPool:
        return self._pool

    def sharding(self) -> ShardingRouter:
        return self._router

    def read_write_split(self) -> ReadWriteSplitter:
        return self._splitter

    def load_balance(self) -> List[LoadBalancer]:
        return self._balancers

    def transaction_manager(self) -> TransactionManager:
        return self._tx_manager

    # -- 便捷接口：从连接池取连接执行 SQL ------------------------------------
    def execute(self, sql: str, key=None, value=None):
        conn = self._pool.acquire()
        try:
            return conn.execute(sql, key=key, value=value)
        finally:
            self._pool.release(conn)
