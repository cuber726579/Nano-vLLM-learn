from collections import deque

from nanovllm.config import Config
from nanovllm.engine.sequence import Sequence, SequenceStatus
from nanovllm.engine.block_manager import BlockManager


class Scheduler:

    def __init__(self, config: Config):
        self.max_num_seqs = config.max_num_seqs
        self.max_num_batched_tokens = config.max_num_batched_tokens
        self.eos = config.eos
        self.block_manager = BlockManager(config.num_kvcache_blocks, config.kvcache_block_size)
        self.waiting: deque[Sequence] = deque()
        self.running: deque[Sequence] = deque()

    def is_finished(self):
        return not self.waiting and not self.running

    def add(self, seq: Sequence):
        self.waiting.append(seq)

    def schedule(self) -> tuple[list[Sequence], bool]:
        "调度器优先做 prefill, 没有新 prefill 可做时, 再做 decode"
        # prefill
        scheduled_seqs = [] # 本次被调度的序列
        num_batched_tokens = 0 # 本次被调度的Token总数
        while self.waiting and len(scheduled_seqs) < self.max_num_seqs: # 优先处理Prefill请求
            seq = self.waiting[0] # 始终从 waiting 队首取请求，保持 FIFO 调度顺序
            # 还需要被 prefill 的 token 数。
            # 对普通 prefill 来说，这是“总 token - 已缓存 token”；
            # 对边界情况用 max(..., 1) 保证后续逻辑至少按 1 个 token 处理
            num_tokens = max(seq.num_tokens - seq.num_cached_tokens, 1)
            remaining = self.max_num_batched_tokens - num_batched_tokens # 当前 batch 还剩多少 token 预算
            # 没有 token 预算了，或者这是该序列第一次进入 prefill 且 KV block 不够分配，那么就停止继续往本轮 batch 里塞请求
            if remaining == 0 or (not seq.block_table and not self.block_manager.can_allocate(seq)):    # no budget
                break
            # 如果剩余预算不足以完整容纳当前序列：
            # - 当本轮 batch 还是空的，允许把这第一条序列切成一个 chunk 来做 prefill；
            # - 当本轮已经有别的序列时，就不再切 chunk，直接停下，避免调度过碎。
            if remaining < num_tokens and scheduled_seqs: # 简化: 每次调度只支持第一个超长序列的 Chunked Prefill
                break
            # 第一次调度这个序列时，先为它分配完整 block_table。
            # 后续 chunked prefill 只是逐步把 token 写入这些已分配好的 block。
            if not seq.block_table:
                self.block_manager.allocate(seq)
            # 本轮真正执行的 token 数，不能超过“剩余待 prefill token”与“batch 剩余预算”的较小值。
            seq.num_scheduled_tokens = min(num_tokens, remaining)
            # 如果这轮已经覆盖了这个序列剩余的全部 prefill token，
            # 说明它完成了 prompt prefill，可以从 waiting 转入 running。
            if seq.num_scheduled_tokens == num_tokens:
                seq.status = SequenceStatus.RUNNING
                self.waiting.popleft()
                self.running.append(seq)
            scheduled_seqs.append(seq) # 记录到本轮执行列表里；如果是 chunked prefill，同一个序列可能会连续多轮出现在这里
            num_batched_tokens += seq.num_scheduled_tokens # 累加本轮已经占用的 token 预算
        if scheduled_seqs: # 只要成功组出了 prefill batch，这一轮就优先执行 prefill，不进入 decode 分支
            return scheduled_seqs, True

        # decode
        # 只有在本轮没有 prefill 请求可执行时，才进入 decode 调度
        while self.running and len(scheduled_seqs) < self.max_num_seqs:
            # 从 running 队首取一个已经完成 prefill、正在生成的序列
            seq = self.running.popleft()
            # decode 每次只追加 1 个 token。
            # 如果当前 KV cache 空间不足以容纳这个新 token，就需要先腾空间。
            while not self.block_manager.can_append(seq):
                # 优先抢占 running 队列尾部的序列，把它的 KV block 释放出来，
                # 尽量保住当前队首这个更早进入 decode 的序列，维持较好的公平性。
                if self.running:
                    self.preempt(self.running.pop())
                else:
                    # 如果 running 里只剩当前这个序列，说明已经没有别的序列可抢占，
                    # 那就只能把它自己也抢占回 waiting，等后面重新 prefill / 恢复执行。
                    self.preempt(seq)
                    break
            else:
                # 能追加时，decode 阶段本轮只调度 1 个 token。
                seq.num_scheduled_tokens = 1
                # 为这个新 token 预留 / 更新 block 状态。
                self.block_manager.may_append(seq)
                # 加入本轮 decode batch。
                scheduled_seqs.append(seq)
        # 走到 decode 分支时，必须至少调度出一个序列；否则说明调度状态异常。
        assert scheduled_seqs
        # 本轮被调度执行的序列仍然应该保持在 running 队列前部，
        # 以便下一轮继续按相近顺序参与 decode。
        self.running.extendleft(reversed(scheduled_seqs))
        return scheduled_seqs, False

    def preempt(self, seq: Sequence):
        seq.status = SequenceStatus.WAITING
        self.block_manager.deallocate(seq)
        self.waiting.appendleft(seq)

    def postprocess(self, seqs: list[Sequence], token_ids: list[int], is_prefill: bool):
        for seq, token_id in zip(seqs, token_ids):
            if is_prefill:
                seq.num_cached_tokens = min(seq.num_cached_tokens + seq.num_scheduled_tokens, seq.num_tokens)
                if seq.num_cached_tokens < seq.num_tokens or seq.num_completion_tokens > 0:    # chunked prefill or re prefill after preemption
                    seq.num_scheduled_tokens = 0
                    continue
            seq.append_token(token_id)
            seq.num_cached_tokens += 1
            seq.num_scheduled_tokens = 0
            if (not seq.ignore_eos and token_id == self.eos) or seq.num_completion_tokens == seq.max_tokens:
                seq.status = SequenceStatus.FINISHED
                self.block_manager.deallocate(seq)
                self.running.remove(seq)
