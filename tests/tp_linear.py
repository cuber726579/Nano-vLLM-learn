import sys
from contextlib import contextmanager
from pathlib import Path

import torch
import torch.distributed as dist


from nanovllm.layers.linear import (
    ColumnParallelLinear,
    MergedColumnParallelLinear,
    QKVParallelLinear,
    ReplicatedLinear,
    RowParallelLinear,
)


@contextmanager
def _mock_tp_env(rank: int, tp_size: int, all_reduce=None):
    """
    在单进程证明函数里模拟当前 TP rank.

    这些证明函数的目标是验证本文件中的真实 Linear 类, 不启动真正的多进程
    process group. 因此这里只替换构造模块时依赖的 rank/world_size 查询.
    """
    original_get_rank = dist.get_rank
    original_get_world_size = dist.get_world_size
    original_all_reduce = dist.all_reduce
    dist.get_rank = lambda: rank
    dist.get_world_size = lambda: tp_size
    if all_reduce is not None:
        dist.all_reduce = all_reduce
    try:
        yield
    finally:
        dist.get_rank = original_get_rank
        dist.get_world_size = original_get_world_size
        dist.all_reduce = original_all_reduce


def _replicated_linear_output(
    x: torch.Tensor,
    weight: torch.Tensor,
    bias: torch.Tensor | None = None,
) -> torch.Tensor:
    """
    ReplicatedLinear 的参考输出.

    ReplicatedLinear.forward 本质上就是 F.linear(x, weight, bias).
    这里真实构造 ReplicatedLinear, 让参考结果也走本文件中的类实现.
    """
    with _mock_tp_env(rank=0, tp_size=1):
        layer = ReplicatedLinear(weight.size(1), weight.size(0), bias is not None)
        layer.weight.weight_loader(layer.weight, weight)
        if bias is not None:
            layer.bias.weight_loader(layer.bias, bias)
        return layer(x)


def prove_column_parallel_linear_equivalence() -> bool:
    """
    证明 ColumnParallelLinear 和 ReplicatedLinear 等价.

    Column Parallel 沿输出维切分 weight:
        weight = [weight_0; weight_1; ...]

    每个 TP rank 计算自己的局部输出, 最后等价于 all_gather 后沿最后一维拼接.
    """
    torch.manual_seed(0)
    batch_size, input_size, output_size, tp_size = 3, 8, 12, 3
    x = torch.randn(batch_size, input_size)
    weight = torch.randn(output_size, input_size)
    bias = torch.randn(output_size)

    y_replicated = _replicated_linear_output(x, weight, bias)

    local_outputs = []
    for rank in range(tp_size):
        with _mock_tp_env(rank, tp_size):
            layer = ColumnParallelLinear(input_size, output_size, bias=True)
            layer.weight.weight_loader(layer.weight, weight)
            layer.bias.weight_loader(layer.bias, bias)
            local_outputs.append(layer(x))

    # 模拟 ColumnParallelLinear 输出散落在不同 rank 后, 通过 all_gather 拼回完整输出.
    y_tp = torch.cat(local_outputs, dim=-1)
    assert torch.allclose(y_tp, y_replicated, atol=1e-6)
    return True


def prove_merged_column_parallel_linear_equivalence() -> bool:
    """
    证明 MergedColumnParallelLinear 和 ReplicatedLinear 等价.

    Merged Column Parallel 先把多个输出分支合并, 例如 [gate, up],
    再让每个分支分别按 TP 切分. 每个 rank 的局部输出布局是:
        [local_gate, local_up]

    因此要恢复 ReplicatedLinear 的完整输出顺序, 需要分别 gather gate/up,
    再拼成 [full_gate, full_up].
    """
    torch.manual_seed(0)
    batch_size, input_size, tp_size = 3, 6, 3
    output_sizes = [12, 12]
    output_size = sum(output_sizes)
    x = torch.randn(batch_size, input_size)
    weight = torch.randn(output_size, input_size)
    bias = torch.randn(output_size)

    y_replicated = _replicated_linear_output(x, weight, bias)

    local_outputs = []
    for rank in range(tp_size):
        with _mock_tp_env(rank, tp_size):
            layer = MergedColumnParallelLinear(input_size, output_sizes, bias=True)
            offset = 0
            for shard_id, shard_output_size in enumerate(output_sizes):
                shard_weight = weight.narrow(0, offset, shard_output_size)
                shard_bias = bias.narrow(0, offset, shard_output_size)
                layer.weight.weight_loader(layer.weight, shard_weight, shard_id)
                layer.bias.weight_loader(layer.bias, shard_bias, shard_id)
                offset += shard_output_size
            local_outputs.append(layer(x))

    # 每个 rank 的 local_output 是 [local_shard_0, local_shard_1, ...].
    local_sizes = [size // tp_size for size in output_sizes]
    outputs_by_shard = [[] for _ in output_sizes]
    for local_output in local_outputs:
        for shard_id, local_shard_output in enumerate(local_output.split(local_sizes, dim=-1)):
            outputs_by_shard[shard_id].append(local_shard_output)

    # 按原始分支顺序恢复: [full_shard_0, full_shard_1, ...].
    y_tp = torch.cat([torch.cat(outputs, dim=-1) for outputs in outputs_by_shard], dim=-1)
    assert torch.allclose(y_tp, y_replicated, atol=1e-6)
    return True


def prove_qkv_parallel_linear_equivalence() -> bool:
    """
    证明 QKVParallelLinear 和 ReplicatedLinear 等价.

    QKVParallelLinear 可以看成 Attention 专用的 MergedColumnParallelLinear:
        [Q, K, V] = x @ qkv_weight.T

    区别是 Q heads 和 KV heads 的数量可能不同, 所以 Q/K/V 的输出大小不同.
    每个 rank 的局部输出布局是 [local_q, local_k, local_v].
    """
    torch.manual_seed(0)
    batch_size, hidden_size, head_size, tp_size = 3, 8, 4, 3
    total_num_heads, total_num_kv_heads = 6, 3
    q_size = total_num_heads * head_size
    kv_size = total_num_kv_heads * head_size
    shard_sizes = [q_size, kv_size, kv_size]

    x = torch.randn(batch_size, hidden_size)
    q_weight = torch.randn(q_size, hidden_size)
    k_weight = torch.randn(kv_size, hidden_size)
    v_weight = torch.randn(kv_size, hidden_size)
    weight = torch.cat([q_weight, k_weight, v_weight], dim=0)

    y_replicated = _replicated_linear_output(x, weight)

    local_outputs = []
    for rank in range(tp_size):
        with _mock_tp_env(rank, tp_size):
            layer = QKVParallelLinear(
                hidden_size,
                head_size,
                total_num_heads,
                total_num_kv_heads,
                bias=False,
            )
            layer.weight.weight_loader(layer.weight, q_weight, "q")
            layer.weight.weight_loader(layer.weight, k_weight, "k")
            layer.weight.weight_loader(layer.weight, v_weight, "v")
            local_outputs.append(layer(x))

    # 每个 rank 的 local_output 是 [local_q, local_k, local_v],
    # 恢复完整输出时需要分别 gather Q/K/V, 再拼成 [full_q, full_k, full_v].
    local_sizes = [size // tp_size for size in shard_sizes]
    outputs_by_shard = [[] for _ in shard_sizes]
    for local_output in local_outputs:
        for shard_id, local_shard_output in enumerate(local_output.split(local_sizes, dim=-1)):
            outputs_by_shard[shard_id].append(local_shard_output)

    y_tp = torch.cat([torch.cat(outputs, dim=-1) for outputs in outputs_by_shard], dim=-1)
    assert torch.allclose(y_tp, y_replicated, atol=1e-6)
    return True


def prove_row_parallel_linear_equivalence() -> bool:
    """
    证明 RowParallelLinear 和 ReplicatedLinear 等价.

    Row Parallel 沿输入维切分:
        x = [x_0, x_1, ...]
        weight = [weight_0, weight_1, ...]

    矩阵乘法沿输入维是求和关系, 所以每个 rank 先算局部贡献,
    再 all_reduce(sum), 就能得到完整 ReplicatedLinear 输出.
    """
    torch.manual_seed(0)
    batch_size, input_size, output_size, tp_size = 3, 12, 5, 3
    x = torch.randn(batch_size, input_size)
    weight = torch.randn(output_size, input_size)
    bias = torch.randn(output_size)

    y_replicated = _replicated_linear_output(x, weight, bias)

    local_outputs = []
    for rank in range(tp_size):
        local_x = x.chunk(tp_size, dim=-1)[rank]
        with _mock_tp_env(rank, tp_size, all_reduce=lambda tensor: tensor):
            layer = RowParallelLinear(input_size, output_size, bias=True)
            layer.weight.weight_loader(layer.weight, weight)
            layer.bias.weight_loader(layer.bias, bias)
            local_outputs.append(layer(local_x))

    # 模拟 RowParallelLinear.forward 里的 dist.all_reduce(sum).
    y_tp = sum(local_outputs)
    assert torch.allclose(y_tp, y_replicated, atol=1e-6)
    return True


def prove_all_tp_linear_equivalence() -> bool:
    """一次性运行本文件中所有 TP Linear 的等价性证明函数."""
    prove_column_parallel_linear_equivalence()
    prove_merged_column_parallel_linear_equivalence()
    prove_qkv_parallel_linear_equivalence()
    prove_row_parallel_linear_equivalence()
    return True


if __name__ == "__main__":
    proof_fns = [
        prove_column_parallel_linear_equivalence,
        prove_merged_column_parallel_linear_equivalence,
        prove_qkv_parallel_linear_equivalence,
        prove_row_parallel_linear_equivalence,
    ]

    for proof_fn in proof_fns:
        proof_fn()
        print(f"{proof_fn.__name__}: passed")

    print("All TP Linear equivalence proofs passed.")
