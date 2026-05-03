import argparse
import os

import torch
from torch import nn
import triton
import triton.language as tl

from flash_attn import flash_attn_varlen_func, flash_attn_with_kvcache
from nanovllm.utils.context import get_context


@triton.jit
def store_kvcache_kernel(
    key_ptr,
    key_stride,
    value_ptr,  
    value_stride,
    k_cache_ptr,
    v_cache_ptr,
    slot_mapping_ptr,
    D: tl.constexpr,
    DEBUG: tl.constexpr,
):
    """
        张量在底层是一段线性内存, 可以按照一位地址的计算来读写
        参数:
        key_ptr: 指向当前层新生成 Key 的首地址。逻辑 shape 为
            [N, num_kv_heads, head_dim]，在 kernel 中按 [N, D] 视角访问，
            其中 D = num_kv_heads * head_dim。
        key_stride: key 在第 0 维上的 stride, 单位是元素个数。
            表示相邻两个 token 的 Key 向量起始地址间隔。
        value_ptr: 指向当前层新生成 Value 的首地址。逻辑 shape 为
            [N, num_kv_heads, head_dim]，在 kernel 中同样按 [N, D] 视角访问。
        value_stride: value 在第 0 维上的 stride, 单位是元素个数。
        k_cache_ptr: 指向当前层 Key cache 的首地址。逻辑 shape 为
            [num_blocks, block_size, num_kv_heads, head_dim]，在 kernel 中按
            [num_slots, D] 视角线性写入，其中 num_slots = num_blocks * block_size。
        v_cache_ptr: 指向当前层 Value cache 的首地址, shape 与 k_cache_ptr 相同。
        slot_mapping_ptr: 指向槽位映射表的首地址, shape 为 [N]。
            slot_mapping[idx] 表示第 idx 个 token 的 KV 要写入哪个线性物理槽位。
        D: 一个 token 在当前层上的 KV 向量长度, D = num_kv_heads * head_dim。
    """
    # 一个 Triton program 对应一个 token 的 K/V 写入任务
    idx = tl.program_id(0)
    # 读取第 idx 个 token 应写入的目标槽位
    slot = tl.load(slot_mapping_ptr + idx)
    # -1 表示这个位置无需写入，直接跳过
    if slot == -1: return
    if DEBUG:
        tl.device_print("store_kvcache idx/slot", idx, slot)
    # 计算 key/value 中第 idx 个 token 的整段向量地址范围
    key_offsets = idx * key_stride + tl.arange(0, D)
    value_offsets = idx * value_stride + tl.arange(0, D)
    # 将当前 token 的整段 K/V 从寄存器前的全局内存读出
    key = tl.load(key_ptr + key_offsets)
    value = tl.load(value_ptr + value_offsets)
    # 线性槽位 slot 展开后对应 cache 中一段长度为 D 的连续地址
    cache_offsets = slot * D + tl.arange(0, D)
    # 将当前 token 的整段 K/V 写入目标物理槽位
    tl.store(k_cache_ptr + cache_offsets, key)
    tl.store(v_cache_ptr + cache_offsets, value)


def store_kvcache(
    key: torch.Tensor,
    value: torch.Tensor,
    k_cache: torch.Tensor,
    v_cache: torch.Tensor,
    slot_mapping: torch.Tensor,
    debug: bool = False,
):
    """
    将当前 batch 新产生的 K/V 写入单层 paged KV cache

    参数:
        key: 当前层新计算出的 Key, shape 为 [N, num_heads, head_dim]
            其中 N 是本轮需要写入 cache 的 token 数
        value: 当前层新计算出的 Value, shape 为 [N, num_heads, head_dim]
        k_cache: 当前层的 Key cache, 逻辑 shape 为
            [num_blocks, block_size, num_heads, head_dim]
            Triton kernel 会把它按 [num_slots, D] 视角线性写入，
            其中 num_slots = num_blocks * block_size, D = num_heads * head_dim。
        v_cache: 当前层的 Value cache, shape 与 k_cache 相同
        slot_mapping: 长度为 N 的一维张量, shape 为 [N]
            slot_mapping[i] 表示第 i 个 token 的 KV 应写入哪个物理槽位。
    """
    # N 是本轮待写入 token 数, D 是单个 token 在该层上的扁平 KV 长度
    N, num_heads, head_dim = key.shape
    D = num_heads * head_dim
    # 保证最后一维连续，便于把 [num_heads, head_dim] 视作一段长度 D 的向量
    assert key.stride(-1) == 1 and value.stride(-1) == 1
    # 保证相邻两个 head 之间紧邻排布，没有额外空洞
    assert key.stride(1) == head_dim and value.stride(1) == head_dim
    # 保证 cache 中相邻两个 token 槽位正好间隔 D 个元素
    assert k_cache.stride(1) == D and v_cache.stride(1) == D
    # 每个待写入 token 都必须有一个对应槽位
    assert slot_mapping.numel() == N
    # 启动 N 个 Triton program, 分别负责 N 个 token 的写入, 并传入第 0 维 stride 作为逐 token 访存步长
    store_kvcache_kernel[(N,)](
        key,
        key.stride(0),
        value,
        value.stride(0),
        k_cache,
        v_cache,
        slot_mapping,
        D,
        DEBUG=debug,
    )


class Attention(nn.Module):

    def __init__(
        self,
        num_heads,
        head_dim,
        scale,
        num_kv_heads,
    ):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = head_dim
        self.scale = scale
        self.num_kv_heads = num_kv_heads
        self.k_cache = self.v_cache = torch.tensor([])
        # Model Runner在分配KV Cache时, 进行各个层的统一分配, 再将给每个层的KV Cache划分区域
        # KV Cache分配后, self.k_cache维度是[Block ID, Token Id, num_kv_heads, head_dim]

    def forward(self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor):
        context = get_context()
        k_cache, v_cache = self.k_cache, self.v_cache
        if k_cache.numel() and v_cache.numel(): # model_runner.warmup_model 时还没有分配 KVCache
            store_kvcache(k, v, k_cache, v_cache, context.slot_mapping)
        if context.is_prefill:
            if context.block_tables is not None: # prefix cache
                k, v = k_cache, v_cache
            o = flash_attn_varlen_func(q, k, v,
                                       max_seqlen_q=context.max_seqlen_q, cu_seqlens_q=context.cu_seqlens_q,
                                       max_seqlen_k=context.max_seqlen_k, cu_seqlens_k=context.cu_seqlens_k,
                                       softmax_scale=self.scale, causal=True, block_table=context.block_tables)
        else: # decode
            o = flash_attn_with_kvcache(q.unsqueeze(1), k_cache, v_cache,
                                        cache_seqlens=context.context_lens, block_table=context.block_tables, 
                                        softmax_scale=self.scale, causal=True)
        return o





def _run_store_kvcache_debug(interpret: bool, debug_kernel: bool, use_pdb: bool):
    if interpret:
        os.environ["TRITON_INTERPRET"] = "1"
    device = "cuda" if torch.cuda.is_available() else "cpu"
    if device != "cuda":
        raise RuntimeError("This debug entry expects CUDA tensors. Set up a CUDA device before running it.")

    def _build_store_kvcache_debug_case(device: str):
        N, num_heads, head_dim = 3, 2, 4
        num_blocks, block_size = 2, 4
        key = torch.arange(N * num_heads * head_dim, dtype=torch.float32, device=device).reshape(N, num_heads, head_dim)
        value = key + 1000
        k_cache = torch.full((num_blocks, block_size, num_heads, head_dim), -1.0, dtype=key.dtype, device=device)
        v_cache = torch.full((num_blocks, block_size, num_heads, head_dim), -1.0, dtype=value.dtype, device=device)
        slot_mapping = torch.tensor([0, 3, 5], dtype=torch.int32, device=device)
        return key, value, k_cache, v_cache, slot_mapping

    key, value, k_cache, v_cache, slot_mapping = _build_store_kvcache_debug_case(device)
    slot_indices = slot_mapping.to(dtype=torch.long)

    print(f"device={device}")
    print(f"TRITON_INTERPRET={os.environ.get('TRITON_INTERPRET', '0')}")
    print(f"key.shape={tuple(key.shape)}, k_cache.shape={tuple(k_cache.shape)}")
    print(f"slot_mapping={slot_mapping.tolist()}")

    if use_pdb:
        breakpoint()

    store_kvcache(key, value, k_cache, v_cache, slot_mapping, debug=debug_kernel)
    torch.cuda.synchronize()

    linear_k_cache = k_cache.view(-1, key.size(1), key.size(2))
    linear_v_cache = v_cache.view(-1, value.size(1), value.size(2))
    written_k = linear_k_cache[slot_indices]
    written_v = linear_v_cache[slot_indices]

    print("written_k:")
    print(written_k.cpu())
    print("written_v:")
    print(written_v.cpu())

    assert torch.equal(written_k, key)
    assert torch.equal(written_v, value)
    print("store_kvcache debug run passed.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Debug entry for nanovllm.layers.attention.store_kvcache")
    parser.add_argument("--interpret", action="store_true", help="Enable Triton interpreter mode before kernel launch.")
    parser.add_argument("--debug-kernel", action="store_true", help="Enable device-side Triton debug prints.")
    parser.add_argument("--pdb", action="store_true", help="Break into pdb before launching the Triton kernel.")
    args = parser.parse_args()
    _run_store_kvcache_debug(args.interpret, args.debug_kernel, args.pdb)
