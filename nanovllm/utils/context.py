from dataclasses import dataclass
import torch


@dataclass
class Context:
    is_prefill: bool = False
    cu_seqlens_q: torch.Tensor | None = None # Prefill时使用的各个Seq的Q Token累计长度
    cu_seqlens_k: torch.Tensor | None = None # Prefill时使用的各个Seq的KV Token累计长度
    max_seqlen_q: int = 0 # 当前Batch中Query的最大长度
    max_seqlen_k: int = 0 # 当前Batch中Key&Value的最大长度
    slot_mapping: torch.Tensor | None = None # 本轮算出来的每个Token，应当写到KV cache的哪个绝对槽位的映射表, 已经考虑了块维度
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
