"""
游戏内数据库连接池与分库分表系统 - 验收测试
运行方式：python -m pytest tests/test_db_pool.py -q
共 16 个测试用例
"""
import pytest
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from db.pool import DBPool, PoolTimeoutError, TransactionError


def key_for_shard(system, shard_idx, prefix="k"):
    """找到一个路由到指定分片的 key。"""
    i = 0
    while True:
        key = "%s-%d" % (prefix, i)
        if system.sharding().shard_index(key) == shard_idx:
            return key
        i += 1


class TestDBPool:
    def setup_method(self):
        self.system = DBPool()

    # ---------------- 连接池 ----------------
    def test_case_01(self):
        """连接池复用连接，不是每次查询都新建连接"""
        pool = self.system.connection_pool()
        for _ in range(5):
            conn = pool.acquire()
            conn.execute("SELECT * FROM t", key="user-1")
            pool.release(conn)
        assert pool.created_count == 1

    def test_case_02(self):
        """连接池有最大连接数限制，连接数不会无限增长"""
        pool = self.system.connection_pool()
        conns = [pool.acquire() for _ in range(4)]
        assert pool.created_count == 4
        assert pool.created_count <= pool.max_connections
        with pytest.raises(PoolTimeoutError):
            pool.acquire()
        for c in conns:
            pool.release(c)

    def test_case_03(self):
        """池满后获取连接会在超时后报错，而不是永远等待"""
        pool = self.system.connection_pool()
        for _ in range(4):
            pool.acquire()
        before = self.system.now
        with pytest.raises(PoolTimeoutError):
            pool.acquire()
        assert self.system.now - before == pytest.approx(pool.timeout)

    def test_case_04(self):
        """释放的连接归还池中，可被再次获取"""
        pool = self.system.connection_pool()
        conn = pool.acquire()
        pool.release(conn)
        assert pool.idle_count == 1
        assert pool.active_count == 0
        again = pool.acquire()
        assert again.id == conn.id
        assert pool.active_count == 1

    # ---------------- 分库分表 ----------------
    def test_case_05(self):
        """数据按规则分布到多个库，而不是全在一个库"""
        splitter = self.system.read_write_split()
        for i in range(50):
            splitter.write("user-%d" % i, {"id": i})
        used = [
            idx for idx, shard in enumerate(self.system._shards)
            if any(shard.master.tables.get(t) for t in shard.master.tables)
        ]
        assert len(used) > 1

    def test_case_06(self):
        """分片规则确定：同一 key 永远路由到同一库同一表"""
        router = self.system.sharding()
        key = "player-42"
        first = router.route(key)
        for _ in range(10):
            assert router.route(key) == first
        shard_idx, table = first
        assert 0 <= shard_idx < router.num_shards
        assert table.startswith("table_")

    def test_case_07(self):
        """分表有规则：数据分布到一个库内的多张表"""
        splitter = self.system.read_write_split()
        for i in range(50):
            splitter.write("item-%d" % i, i)
        tables = set()
        for shard in self.system._shards:
            tables.update(shard.master.tables.keys())
        assert len(tables) > 1

    def test_case_08(self):
        """查询有路由：写入后能按 key 路由到正确的库表查到数据"""
        splitter = self.system.read_write_split()
        splitter.write("hero-7", {"level": 30})
        shard_idx, table = self.system.sharding().route("hero-7")
        # 数据确实落在路由指向的库和表
        assert self.system._shards[shard_idx].master.has_row(table, "hero-7")
        # 读路径经路由能找到数据
        assert splitter.read("hero-7") == {"level": 30}

    # ---------------- 读写分离 ----------------
    def test_case_09(self):
        """写操作走主库，不走从库"""
        self.system.execute("INSERT INTO t VALUES (1)", key="w-1", value=1)
        for shard in self.system._shards:
            for replica in shard.replicas:
                assert replica.write_count == 0
        assert sum(s.master.write_count for s in self.system._shards) == 1

    def test_case_10(self):
        """读操作走从库，不走主库"""
        splitter = self.system.read_write_split()
        splitter.write("r-1", "v")
        self.system.update(10.0)  # 等待主从复制完成、越过延迟窗口
        assert splitter.read("r-1") == "v"
        slave_reads = sum(
            r.read_count for s in self.system._shards for r in s.replicas
        )
        assert slave_reads == 1
        assert sum(s.master.read_count for s in self.system._shards) == 0

    def test_case_11(self):
        """主从延迟窗口内读走主库拿到最新数据，复制完成后读走从库"""
        splitter = self.system.read_write_split()
        splitter.write("d-1", "new")
        # 延迟窗口内：从库尚未同步，读必须走主库才能拿到最新值
        assert splitter.read("d-1") == "new"
        assert sum(s.master.read_count for s in self.system._shards) == 1
        # 推进时间超过复制延迟后：从库已同步，读走从库
        self.system.update(10.0)
        assert splitter.read("d-1") == "new"
        slave_reads = sum(
            r.read_count for s in self.system._shards for r in s.replicas
        )
        assert slave_reads == 1

    # ---------------- 负载均衡 ----------------
    def test_case_12(self):
        """负载均衡按权重分配请求（默认权重 1:3）"""
        balancer = self.system.load_balance()[0]
        counts = {}
        for _ in range(8):
            replica = balancer.pick()
            counts[replica.name] = counts.get(replica.name, 0) + 1
        names = sorted(counts)
        assert counts[names[0]] == 2   # weight=1
        assert counts[names[1]] == 6   # weight=3

    def test_case_13(self):
        """健康检查剔除挂掉的从库；全部从库挂掉时回退主库"""
        balancer = self.system.load_balance()[0]
        down = balancer.replicas[1]
        down.healthy = False
        balancer.health_check(self.system.now)
        for _ in range(6):
            assert balancer.pick() is not down
        # 全部从库挂掉：读请求回退主库，不丢请求
        for shard in self.system._shards:
            for replica in shard.replicas:
                replica.healthy = False
            self.system.load_balance()[shard.index].health_check(self.system.now)
        splitter = self.system.read_write_split()
        splitter.write("h-1", "v")
        self.system.update(10.0)
        assert splitter.read("h-1") == "v"
        assert sum(s.master.read_count for s in self.system._shards) >= 1

    # ---------------- 事务管理 ----------------
    def test_case_14(self):
        """两阶段提交：跨库事务全部提交成功"""
        tx = self.system.transaction_manager().begin()
        k0 = key_for_shard(self.system, 0, "tx-a")
        k1 = key_for_shard(self.system, 1, "tx-b")
        tx.insert(k0, "A")
        tx.insert(k1, "B")
        tx.commit()
        for key, val in ((k0, "A"), (k1, "B")):
            idx, table = self.system.sharding().route(key)
            assert self.system._shards[idx].master.read_row(table, key) == val

    def test_case_15(self):
        """prepare 阶段任一参与者失败：全局回滚，无部分提交"""
        self.system._shards[1].master.healthy = False
        tx = self.system.transaction_manager().begin()
        k0 = key_for_shard(self.system, 0, "rb-a")
        k1 = key_for_shard(self.system, 1, "rb-b")
        tx.insert(k0, "A")
        tx.insert(k1, "B")
        with pytest.raises(TransactionError):
            tx.commit()
        idx0, table0 = self.system.sharding().route(k0)
        assert not self.system._shards[idx0].master.has_row(table0, k0)

    def test_case_16(self):
        """commit 阶段失败：已提交的分片被补偿撤销，恢复原子性"""
        self.system._shards[1].master.fail_commit = True
        tx = self.system.transaction_manager().begin()
        k0 = key_for_shard(self.system, 0, "cp-a")
        k1 = key_for_shard(self.system, 1, "cp-b")
        tx.insert(k0, "A")
        tx.insert(k1, "B")
        with pytest.raises(TransactionError):
            tx.commit()
        # 分片 0 已提交的数据被补偿删除，分片 1 没有残留
        idx0, table0 = self.system.sharding().route(k0)
        idx1, table1 = self.system.sharding().route(k1)
        assert not self.system._shards[idx0].master.has_row(table0, k0)
        assert not self.system._shards[idx1].master.has_row(table1, k1)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
