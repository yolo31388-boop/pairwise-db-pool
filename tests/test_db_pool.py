"""
游戏内数据库连接池与分库分表系统 - 验收测试
运行方式：python -m pytest tests/test_db_pool.py -q
共 16 个测试用例
"""
import time

import pytest
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from db.pool import (
    DBPool,
    NoHealthyReplicaError,
    PoolExhaustedError,
    TransactionError,
)


def make_system(**overrides):
    config = {
        "max_connections": 3,
        "connection_timeout": 0.05,
        "num_shards": 4,
        "tables_per_shard": 4,
        "replicas": [
            {"id": "r1", "weight": 3},
            {"id": "r2", "weight": 1},
            {"id": "r3", "weight": 2},
        ],
        "max_replication_lag": 0.5,
    }
    config.update(overrides)
    return DBPool(config)


def keys_on_different_shards(sharding, count=2):
    """找出落在不同分片上的 key，用于跨分片事务测试。"""
    found = {}
    for i in range(1000):
        key = "player_%d" % i
        shard = sharding.shard_for(key)
        if shard not in found:
            found[shard] = key
        if len(found) >= count:
            return list(found.values())
    raise AssertionError("分片数不足，无法构造跨分片数据")


class TestDBPool:
    def setup_method(self):
        self.system = make_system()

    # ------------------------------------------------------------ 连接池

    def test_case_01(self):
        """连接池复用连接：归还后再次获取拿到同一个连接，而不是每次新建。"""
        pool = self.system.connection_pool()
        conn = pool.acquire()
        pool.release(conn)
        again = pool.acquire()
        assert again is conn
        assert pool.size == 1

    def test_case_02(self):
        """连接池有最大连接数：已创建连接数不会超过上限。"""
        pool = self.system.connection_pool()
        conns = [pool.acquire() for _ in range(3)]
        assert pool.size == 3
        assert pool.size <= pool.max_connections
        with pytest.raises(PoolExhaustedError):
            pool.acquire()
        for conn in conns:
            pool.release(conn)

    def test_case_03(self):
        """连接池有获取超时：拿不到连接时不会永远等待。"""
        pool = self.system.connection_pool()
        for _ in range(3):
            pool.acquire()
        start = time.monotonic()
        with pytest.raises(PoolExhaustedError):
            pool.acquire()
        elapsed = time.monotonic() - start
        assert 0.04 <= elapsed < 1.0

    def test_case_04(self):
        """释放连接后池恢复可用，且连接总数不无限增长。"""
        pool = self.system.connection_pool()
        conns = [pool.acquire() for _ in range(3)]
        pool.release(conns[0])
        reused = pool.acquire()
        assert reused is conns[0]
        assert pool.size == 3

    # ------------------------------------------------------------ 分库分表

    def test_case_05(self):
        """分库规则生效：数据按 key 分布到多个分片，而不是全在一个库。"""
        sharding = self.system.sharding()
        for i in range(200):
            sharding.write("player_%d" % i, {"level": i})
        sizes = sharding.shard_sizes()
        used = [shard for shard, n in sizes.items() if n > 0]
        assert len(used) > 1
        assert sum(sizes.values()) == 200

    def test_case_06(self):
        """分表有明确规则：表名由 hash(key) % tables_per_shard 决定且结果确定。"""
        sharding = self.system.sharding()
        table = sharding.table_for("player_1")
        assert table == sharding.table_for("player_1")
        assert table.startswith("t_")
        index = int(table.split("_")[1])
        assert 0 <= index < sharding.tables_per_shard

    def test_case_07(self):
        """分表有路由：写入后按路由规则能准确读回，且只落在路由的分片上。"""
        sharding = self.system.sharding()
        shard, table = sharding.write("player_42", {"level": 99})
        assert (shard, table) == sharding.route("player_42")
        assert sharding.read("player_42") == {"level": 99}
        for other_shard, tables in sharding._data.items():
            if other_shard != shard:
                for rows in tables.values():
                    assert "player_42" not in rows

    # ------------------------------------------------------------ 读写分离

    def test_case_08(self):
        """分离有规则：SELECT 走从库，INSERT/UPDATE/DELETE 走主库。"""
        rw = self.system.read_write_split()
        target, _ = rw.execute("SELECT * FROM player")
        assert target == "replica"
        time.sleep(0.6)  # 越过主从延迟窗口
        for sql in ("INSERT INTO player VALUES (1)",
                    "UPDATE player SET level = 2",
                    "DELETE FROM player WHERE id = 1"):
            target, node = rw.execute(sql)
            assert (target, node) == ("master", "master")
            time.sleep(0.6)

    def test_case_09(self):
        """分离考虑延迟：写入后主从延迟窗口内的读请求走主库，能读到最新数据。"""
        rw = self.system.read_write_split()
        rw.execute("UPDATE player SET level = 2 WHERE id = 1")
        target, node = rw.execute("SELECT level FROM player WHERE id = 1")
        assert (target, node) == ("master", "master")

    def test_case_10(self):
        """延迟超阈值的从库被跳过：全部从库延迟过高时读回退到主库。"""
        rw = self.system.read_write_split()
        for replica in self.system.load_balance().replicas:
            replica.lag = 10.0
        target, node = rw.execute("SELECT * FROM player")
        assert (target, node) == ("master", "master")

    # ------------------------------------------------------------ 负载均衡

    def test_case_11(self):
        """均衡有健康检查：挂掉的从库不再接收请求，恢复后重新加入。"""
        balancer = self.system.load_balance()
        balancer.health_check("r2", False)
        for _ in range(12):
            assert balancer.select_replica().id != "r2"
        balancer.health_check("r2", True)
        assert "r2" in {balancer.select_replica().id for _ in range(12)}

    def test_case_12(self):
        """均衡有权重：一个完整周期内各从库被选次数与权重成正比。"""
        balancer = self.system.load_balance()
        counts = {}
        for _ in range(6):  # 权重 3+1+2 = 6，一个完整周期
            replica = balancer.select_replica()
            counts[replica.id] = counts.get(replica.id, 0) + 1
        assert counts == {"r1": 3, "r2": 1, "r3": 2}

    def test_case_13(self):
        """全部从库不健康时负载均衡明确报错，而不是把请求发给挂掉的从库。"""
        balancer = self.system.load_balance()
        for replica_id in ("r1", "r2", "r3"):
            balancer.health_check(replica_id, False)
        with pytest.raises(NoHealthyReplicaError):
            balancer.select_replica()

    # ------------------------------------------------------------ 事务管理

    def test_case_14(self):
        """两阶段提交成功：跨分片的写操作全部生效，事务原子提交。"""
        sharding = self.system.sharding()
        key1, key2 = keys_on_different_shards(sharding)
        txn = self.system.transaction_manager().begin()
        txn.write(key1, {"gold": 100})
        txn.write(key2, {"gold": 200})
        txn.commit()
        assert txn.state == "committed"
        assert sharding.read(key1) == {"gold": 100}
        assert sharding.read(key2) == {"gold": 200}

    def test_case_15(self):
        """prepare 阶段失败：整个事务回滚，任何分片都不会部分提交。"""
        sharding = self.system.sharding()
        manager = self.system.transaction_manager()
        key1, key2 = keys_on_different_shards(sharding)
        manager.fail_prepare_on.add(sharding.shard_for(key2))
        txn = manager.begin()
        txn.write(key1, {"gold": 100})
        txn.write(key2, {"gold": 200})
        with pytest.raises(TransactionError):
            txn.commit()
        assert txn.state == "aborted"
        assert sharding.read(key1) is None
        assert sharding.read(key2) is None

    def test_case_16(self):
        """commit 阶段失败：已提交的分片被补偿回滚，不会出现部分成功。"""
        sharding = self.system.sharding()
        manager = self.system.transaction_manager()
        key1, key2 = keys_on_different_shards(sharding)
        # 第一笔写入成功提交后，第二笔所在分片 commit 失败
        manager.fail_commit_on.add(sharding.shard_for(key2))
        txn = manager.begin()
        txn.write(key1, {"gold": 100})
        txn.write(key2, {"gold": 200})
        with pytest.raises(TransactionError):
            txn.commit()
        assert txn.state == "aborted"
        assert sharding.read(key1) is None  # 已提交但被补偿回滚
        assert sharding.read(key2) is None


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
