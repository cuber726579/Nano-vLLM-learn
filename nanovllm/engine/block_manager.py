from collections import deque
import xxhash
import numpy as np

from nanovllm.engine.sequence import Sequence


class Block:
    """
    表示 KV cache 中的一个物理 block, 最多有 config.num_kvcache_blocks 个 block 可用.
    每个block有 config.kvcache_block_size 个slots,
    每个slot可以存储一个 token id 所对应的 hf_config.num_hidden_layers 个层中的 K/V 张量数据.

    一个 Sequence 的 block_table 记录的是逻辑 block 号到物理 Block 号的映射.
    类 Block 自身不保存真正的 K/V 张量数据, K/V 张量存放在各 Attention 层的
    k_cache/v_cache 中; 本类保存的是调度和 prefix cache 需要的元信息.
    """

    def __init__(self, block_id):
        # 物理 block 编号, 也是访问 Attention.k_cache/v_cache 第一维时使用的索引(model_runner.kv_cache 第三个维度的索引)
        self.block_id = block_id 

        # 引用计数, 表示归属于多少个序列, 多个序列命中相同 prefix cache block 时会共享同一个物理 block
        self.ref_count = 0

        # 当前 block 对应 token_ids 的哈希值; -1 表示还没有形成可复用的完整缓存块
        # 即当前 block 中的 token 数还没有达到 config.kvcache_block_size
        self.hash = -1

        # 当前 block 覆盖的 token id, 用于哈希命中后再做一次精确校验,避免哈希碰撞或误命中
        self.token_ids = []

    def update(self, hash: int, token_ids: list[int]):
        """记录当前 block 的 prefix-cache 哈希和对应 token id"""
        # 通常当一个block中存放token数达到config.kvcache_block_size时, 才会更新hash和token_ids
        self.hash = hash
        self.token_ids = token_ids

    def reset(self):
        """将空闲 block 重新分配给一个序列时, 重置它的运行时元信息"""
        self.ref_count = 1
        self.hash = -1
        self.token_ids = []


class BlockManager:
    """
    KV cache 物理 block 的分配器

    BlockManager 只管理 KV cache 在 block(config.num_kvcache_blocks) 维度上的地址分配, 不直接保存
    真实 K/V 张量.真实 K/V 数据存放在每一层 Attention 的 k_cache/v_cache 中,
    形状通常类似 [num_blocks, block_size, num_kv_heads, head_dim].

    负责:
      1. 为 Sequence 分配 block, 写入 block_table 建立逻辑 block -> 物理 block 的映射;
      2. 维护 free/used block 集合, 在请求结束或被抢占时释放 block;
      3. 通过 hash_to_block_id 支持 prefix cache, 让相同完整 prefix block
         可以被多个 Sequence 共享;
      4. decode 过程中在跨 block 时追加新物理 block, 并在 block 填满后
         计算 hash, 使其后续可被 prefix cache 复用.
    """

    def __init__(self, num_blocks: int, block_size: int):
        # 每个物理 block 能容纳多少个 token 的 K/V
        self.block_size = block_size

        # 所有物理 block 的元信息, block_id 与列表下标一一对应
        self.blocks: list[Block] = [Block(i) for i in range(num_blocks)]

        # prefix cache 索引: block 的 hash -> 物理 block id
        self.hash_to_block_id: dict[int, int] = dict()

        # 双端队列当前空闲的物理 block ids
        self.free_block_ids: deque[int] = deque(range(num_blocks))

        # 当前已经被分配出去的物理 block ids
        self.used_block_ids: set[int] = set()

    @classmethod
    def compute_hash(cls, token_ids: list[int], prefix: int = -1):
        """
        计算一个完整 block 的 prefix-cache hash

        prefix 是 token_ids 所属 sequence 的前一个 block 的 hash. 把 prefix 也纳入 hash 后, 
        第 i 个 block的 hash 不只取决于本 block 的 token_ids, 还取决于它前面的完整前缀,
        因而可以区分“本 block token 相同但前文不同”的情况.
        """
        h = xxhash.xxh64()
        if prefix != -1:
            h.update(prefix.to_bytes(8, "little")) # 用 prefix 更新 hash
        h.update(np.array(token_ids).tobytes()) # 用当前块的 token_ids 更新 hash
        return h.intdigest()

    def _allocate_block(self, block_id: int) -> Block:
        """把一个空闲物理 block 标记为已使用, 并重置其运行时元信息"""
        block = self.blocks[block_id]
        assert block.ref_count == 0
        block.reset()
        self.free_block_ids.remove(block_id)
        self.used_block_ids.add(block_id)
        return block

    def _deallocate_block(self, block_id: int) -> Block:
        """把 ref_count 已经降到 0 的物理 block 放回空闲池"""
        # 注意不会清空 block 的 hash 和 token_ids, 
        # 仍然可以被其他序列命中 : cache_miss and not in self.used_block_ids
        assert self.blocks[block_id].ref_count == 0
        self.used_block_ids.remove(block_id)
        self.free_block_ids.append(block_id)

    def can_allocate(self, seq: Sequence) -> bool:
        """Prefill 阶段判断当前空闲 block 数是否足够一次性放下整个 seq 的 block_table"""
        return len(self.free_block_ids) >= seq.num_blocks

    def allocate(self, seq: Sequence):
        """
        Prefill阶段为没有 block_table 的 Sequence 一次行分配所有所需的物理 block.

        分配时会从第 0 个逻辑 block 开始尝试 prefix caching:
        - prefix caching 的本质是复用已经计算好的 KV 向量, 为了能复用, 必须严格保证上下文一致;
        - 上下文的 Token 不一样则不能复用, 即使两个 block 的 token_ids 相同, 但是 KV 向量不同;
        - 如果当前完整 block 的 hash 命中且 token_ids 完全一致, 复用已有物理 block;
        - 一旦遇到 cache miss, 后续 block 都重新分配, 不再继续尝试命中;
        - 未满 block 的 hash 为 -1, 不会进入 prefix cache.
        """
        assert not seq.block_table
        h = -1
        cache_miss = False
        for i in range(seq.num_blocks):
            # 取出 seq 的第 i 个逻辑 block 对应的 token ids
            token_ids = seq.block(i)

            # 只有完整/满的 block 才计算 hash; 最后一个未满 block 不能作为 prefix cache key
            h = self.compute_hash(token_ids, h) if len(token_ids) == self.block_size else -1
            block_id = self.hash_to_block_id.get(h, -1)

            # hash 命中后还要比较 token_ids, 防止哈希碰撞或 stale entry 造成误用
            if block_id == -1 or self.blocks[block_id].token_ids != token_ids:
                cache_miss = True
            if cache_miss:
                # 第一次 miss 之后, 当前和后续 block 都分配新的物理 block
                block_id = self.free_block_ids[0]
                block = self._allocate_block(block_id)
            else:
                # 命中 prefix cache: 这一个完整 block 不需要重新 prefill
                seq.num_cached_tokens += self.block_size
                if block_id in self.used_block_ids:
                    # 物理 block 正被其他 seq 使用, 增加引用计数实现共享
                    block = self.blocks[block_id]
                    block.ref_count += 1
                else:
                    # 物理 block 在空闲池中但 hash 仍可用, 重新标记为 used
                    block = self._allocate_block(block_id)

            if h != -1:
                # 完整 block 才记录 hash, 供后续请求做 prefix cache 查询
                block.update(h, token_ids)
                self.hash_to_block_id[h] = block_id
            # 记录当前 seq 的逻辑 block i 映射到哪个物理 block.
            seq.block_table.append(block_id)

    def can_append(self, seq: Sequence) -> bool:
        """decode阶段前检查是否有足够空间追加 1 个 token"""
        return len(self.free_block_ids) >= (len(seq) % self.block_size == 1)

    def may_append(self, seq: Sequence):
        """
        decode 阶段必要时分配新块和维护 block 填满情况下的 prefix-cache hash

        调用时 seq.num_tokens 已经包含即将计算/写入 KV cache 的最后一个 token (上一次 decode 的结果):
        - 如果刚进入新 block, 为 seq.block_table 追加一个新的物理 block;
        - 如果最后一个 block 正好被填满, 计算并记录它的 prefix-cache hash;
        - 如果 block 尚未填满, 保持 hash == -1.
        """
        block_table = seq.block_table
        last_block = self.blocks[block_table[-1]]
        if len(seq) % self.block_size == 1: 
            # 新 token 是 block 的首个 token, 需要分配新 block
            assert last_block.hash != -1
            block_id = self.free_block_ids[0]
            self._allocate_block(block_id)
            block_table.append(block_id)
        elif len(seq) % self.block_size == 0:
            # 新 token 是 block 最后一个 token, 可以计算 hash 并加入 prefix cache
            assert last_block.hash == -1
            token_ids = seq.block(seq.num_blocks - 1)
            prefix = self.blocks[block_table[-2]].hash if len(block_table) > 1 else -1
            h = self.compute_hash(token_ids, prefix)
            last_block.update(h, token_ids)
            self.hash_to_block_id[h] = last_block.block_id
        else:
            # 最后一个 block 还没填满, 不能作为可复用 prefix cache block
            assert last_block.hash == -1

    def deallocate(self, seq: Sequence):
        """释放一个 Sequence 占用或引用的所有物理 block"""
        for block_id in reversed(seq.block_table):
            block = self.blocks[block_id]
            block.ref_count -= 1
            if block.ref_count == 0:
                self._deallocate_block(block_id)
        seq.num_cached_tokens = 0
        seq.block_table.clear()

