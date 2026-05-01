from dataclasses import dataclass
import torch


@dataclass(slots=True)
class Context:
    """
    模型执行期的全局变量容器

    ModelRunner.run() 每轮执行前会根据 Scheduler 返回的 seqs 构造
    attention 和 KV cache 元数据, 并通过 set_context() 写入这里;
    Attention.forward()、ParallelLMHead.forward() 等模型层再通过
    get_context() 读取这些元数据. run() 结束后调用 reset_context()
    清空, Context 不跨 batch 保存状态.

    Prefill 阶段使用 cu_seqlens_q/cu_seqlens_k/max_seqlen_q/max_seqlen_k
    描述变长 batch 中一个序列的起始位置, 并用 slot_mapping 把本轮新 token 的 K/V 写入
    paged KV cache; 如果有 prefix cache, 还会使用 block_tables 访问已缓存前缀.

    例如本轮 prefill 有 2 条 seq:
      seq0: prompt 长度 4, 本轮需要计算 4 个 token
      seq1: prompt 长度 6, 但前 2 个 token 命中 prefix cache, 本轮只计算后 4 个 token
    那么 q 表示本轮实际要算的 token, 长度分别是 [4, 4],
    cu_seqlens_q = [0, 4, 8]; k 表示 attention 可见的完整上下文(seq_len),
    长度分别是 [4, 6], cu_seqlens_k = [0, 4, 10].

    Decode 阶段每条 seq 通常只计算一个 token, 主要使用 slot_mapping,
    context_lens 和 block_tables: 前者表示新 token 的 KV 写入位置,
    后两者用于 flash_attn_with_kvcache 读取已有上下文.
    """
    is_prefill: bool = False

    # Attention 相关元数据
    cu_seqlens_q: torch.Tensor | None = None # Schedule 时使用的各个Seq的Q Token累计长度
    cu_seqlens_k: torch.Tensor | None = None # Schedule 时使用的各个Seq的KV Token累计长度
    max_seqlen_q: int = 0 # 当前Batch中Query的最大长度
    max_seqlen_k: int = 0 # 当前Batch中Key&Value的最大长度

    # KV cache 相关元数据
    slot_mapping: torch.Tensor | None = None # 本轮调度得到的每个Token，应当写到KV cache的何处物理槽位
    context_lens: torch.Tensor | None = None # Decode时使用的每个Seq的上下文Token数量
    block_tables: torch.Tensor | None = None # 每个Seq对应的Block Table

_CONTEXT = Context()

def get_context():
    return _CONTEXT

def set_context(is_prefill, cu_seqlens_q=None, cu_seqlens_k=None, max_seqlen_q=0, max_seqlen_k=0, slot_mapping=None, context_lens=None, block_tables=None):
    global _CONTEXT
    _CONTEXT = Context(is_prefill, cu_seqlens_q, cu_seqlens_k, max_seqlen_q, max_seqlen_k, slot_mapping, context_lens, block_tables)

def reset_context():
    global _CONTEXT
    _CONTEXT = Context()
