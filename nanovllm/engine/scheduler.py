from collections import deque

from nanovllm.config import Config
from nanovllm.engine.sequence import Sequence, SequenceStatus
from nanovllm.engine.block_manager import BlockManager


class Scheduler:
    """
    Nano-vLLM 的请求调度器

    这个调度器采用 V0 风格的阶段式调度:
    1. 新请求先进入 waiting 队列, 等待 prompt prefill;
    2. prefill 完成后转入 running 队列, 后续参与 decode;
    3. decode 生成结束后释放 KV cache, 并从 running 中移除

    调度器同时负责和 BlockManager 协作, 为 prefill/decode 阶段分配, 追加或释放 KV cache block
    """

    def __init__(self, config: Config):
        """初始化调度器的调度上限、KV cache 管理器和请求队列"""
        # 单轮调度最多允许同时处理多少条序列
        self.max_num_seqs = config.max_num_seqs

        # prefill时 单轮 batch 最多允许处理多少个 token
        self.max_num_batched_tokens = config.max_num_batched_tokens

        # tokenizer 的 EOS token id, 用于判断序列是否生成结束
        self.eos = config.eos

        # 管理 paged KV cache 的物理 block 分配, 追加和释放
        self.block_manager = BlockManager(config.num_kvcache_blocks, config.kvcache_block_size)

        # 存放等待 prefill 的新请求和被抢占的请求的队列
        self.waiting: deque[Sequence] = deque()

        # 已完成 prefill, 正在 decode 的序列队列
        self.running: deque[Sequence] = deque()

    def is_finished(self):
        """当没有 waiting 和 running 序列时, 说明所有请求都已完成"""
        return not self.waiting and not self.running

    def add(self, seq: Sequence):
        """添加一条新请求序列, 先放入 waiting 队列等待 prefill"""
        self.waiting.append(seq)

    def schedule(self) -> tuple[list[Sequence], bool]:
        "v0 scheduling: 调度器优先做 prefill, 没有新 prefill 可做时, 再做 decode"
        # prefill
        scheduled_seqs = [] # 本次被调度的序列
        num_batched_tokens = 0 # 本次被调度的Token总数
        while self.waiting and len(scheduled_seqs) < self.max_num_seqs: # 优先处理Prefill请求
            seq = self.waiting[0] # 始终从 waiting 队首取请求,保持 FIFO 调度顺序

            # 对边界情况用 max(..., 1) 保证后续逻辑至少按 1 个 token 处理
            num_tokens = max(seq.num_tokens - seq.num_cached_tokens, 1) # 序列剩余待 prefill 的 token 数

            remaining = self.max_num_batched_tokens - num_batched_tokens # 当前 batch 还剩多少 token 预算

            # 没有 token 预算了, 或者这是该序列第一次进入 prefill 且 KV block 不够分配, 那么就停止继续往本轮 batch 里塞请求
            if remaining == 0 or (not seq.block_table and not self.block_manager.can_allocate(seq)):
                break

            # 如果剩余预算不足以完整容纳当前序列：
            # - 当本轮 batch 还是空的, 允许把这第一条超长序列切成一个 chunk 来做 prefill;
            # - 当本轮已经有别的序列时, 就不再切 chunk,直接停下, 避免调度过碎.
            if remaining < num_tokens and scheduled_seqs: # Chunked Prefill 简化: 每次调度只支持首个序列是超长序列
                break

            # 第一次调度这个序列时,先为序列分配完整 block_table.
            # 后续 chunked prefill 只是逐步把 token 写入这些已分配好的 block.
            if not seq.block_table:
                self.block_manager.allocate(seq)
                # allocate 可能命中 prefix cache 并更新 num_cached_tokens,
                # 因此需要重新计算本轮剩余待 prefill 的 token 数.
                num_tokens = max(seq.num_tokens - seq.num_cached_tokens, 1)
            # 本轮真正执行的 token 数,不能超过“剩余待 prefill token”与“batch 剩余预算”的较小值.
            seq.num_scheduled_tokens = min(num_tokens, remaining)

            # 如果这轮已经覆盖了这个序列剩余的全部 prefill token,
            # 说明序列完成了 prompt prefill,可以从 waiting 转入 running.
            if seq.num_scheduled_tokens == num_tokens:
                seq.status = SequenceStatus.RUNNING
                self.waiting.popleft()
                self.running.append(seq)
            scheduled_seqs.append(seq) # 记录到本轮执行列表里；如果是 chunked prefill,同一个序列可能会连续多轮出现在这里
            num_batched_tokens += seq.num_scheduled_tokens # 累加本轮已经占用的 token 预算
        if scheduled_seqs: # 只要成功组出了 prefill batch,这一轮就优先执行 prefill,不进入 decode 分支
            return scheduled_seqs, True

        # decode
        # 只有在本轮没有 prefill 请求可执行时,才进入 decode 调度
        while self.running and len(scheduled_seqs) < self.max_num_seqs:
            # 从 running 队首取一个已经完成 prefill、正在生成的序列
            seq = self.running.popleft()
            # decode 每次只追加 1 个 token
            # 如果当前 KV cache 空间不足以容纳这个新 token,就需要先腾空间
            while not self.block_manager.can_append(seq):
                # 尽量保住当前队首这个更早进入 decode 的序列,维持较好的公平性
                if self.running:
                    self.preempt(self.running.pop()) # 优先抢占 running 队列尾部的序列
                else:
                    # 如果 running 里只剩当前这个序列, 说明已经没有别的序列可抢占,
                    # 之前也没有调度到序列, 则会 assert scheduled_seqs 报错
                    self.preempt(seq)
                    break
            else:
                # 能追加时,decode 阶段本轮只调度 1 个 token
                seq.num_scheduled_tokens = 1
                # 为这个新 token 预留 / 更新 block 状态
                self.block_manager.may_append(seq)
                # 加入本轮 decode batch
                scheduled_seqs.append(seq)

        # 走到 decode 分支时,必须至少调度出一个序列；否则说明调度状态异常
        assert scheduled_seqs
        # 本轮被调度执行的序列仍然应该保持在 running 队列前部,
        # 以便下一轮继续按相近顺序参与 decode.
        self.running.extendleft(reversed(scheduled_seqs)) # reversd 抵消 extendleft 的反转效果
        return scheduled_seqs, False

    def preempt(self, seq: Sequence):
        """
        抢占一条正在运行的序列

        抢占会释放该序列当前占用的 KV cache block, 清空 seq.block_table,
        并把序列放回 waiting 队首. 后续如果还能被调度, 需要重新经过 prefill 阶段来恢复 KVCache,
        (不一定清空 block 中的内容, 所以可以通过 prefix caching 可以避免一部分重复计算).
        """
        seq.status = SequenceStatus.WAITING # 标记为等待状态, 表示序列不能继续 decode
        self.block_manager.deallocate(seq)
        self.waiting.appendleft(seq) # 放回队首, 让被抢占的序列优先重新尝试 prefill

    def postprocess(self, seqs: list[Sequence], token_ids: list[int], is_prefill: bool):
        """
        处理模型本轮执行后的结果

        Args:
            seqs: 本轮被 schedule() 选中的序列.
            token_ids: 模型为每条序列采样得到的下一个 token id
            is_prefill: 本轮是否是 prefill 阶段

        prefill 阶段:
        - 更新 num_cached_tokens, 表示本轮新算出的 prompt token 已经写入 KVCache;
        - 如果只是 chunked prefill 的中间片段, 或者是被抢占后重新 prefill
          尚未回到正常 decode 的情况, 则不追加新生成 token.

        decode 阶段:
        - 把采样出的 tokens 追加到各个序列;
        - 如果遇到 EOS 或达到 max_tokens, 标记为 FINISHED 并释放 KVCache. 
        """
        for seq, token_id in zip(seqs, token_ids):
            if is_prefill:
                # prefill 不一定一次算完整个 prompt: chunked prefill 会分多轮写入 KVCache.
                # num_cached_tokens 记录已经可复用/已写入 KVCache 的 token 数.
                seq.num_cached_tokens = min(seq.num_cached_tokens + seq.num_scheduled_tokens, seq.num_tokens)
                if seq.num_cached_tokens < seq.num_tokens or seq.num_completion_tokens > 0: 
                    # chunked prefill 或者是被抢占后重新 prefill 的情况
                    # 这类情况只是在恢复/ KVCache, 不消费模型输出作为新 token. 
                    seq.num_scheduled_tokens = 0
                    continue

            # prefill 完成后的首轮会采样出第一个 completion token;
            # decode 阶段每轮也会采样出一个新 token.
            seq.append_token(token_id)
            # 新 token 的 KV 也已经写入 cache, 因此 cached token 数同步加一
            seq.num_cached_tokens += 1
            # 清空本轮调度 token 数, 避免影响下一轮调度
            seq.num_scheduled_tokens = 0

            if (not seq.ignore_eos and token_id == self.eos) or seq.num_completion_tokens == seq.max_tokens:
                # 生成结束, 标记序列状态为 FINISHED, 并释放占用的 KV block
                seq.status = SequenceStatus.FINISHED
                self.block_manager.deallocate(seq)
                self.running.remove(seq)
