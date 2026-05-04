import torch
from torch import nn
import torch.nn.functional as F
import torch.distributed as dist
from contextlib import contextmanager


def divide(numerator, denominator):
    assert numerator % denominator == 0
    return numerator // denominator


class LinearBase(nn.Module):
    """
    项目中所有线性层的基类

    这里统一创建 weight/bias 参数, 记录当前 TP rank 和 TP size,
    并把每个参数的 weight_loader 绑定到当前模块的加载函数.
    具体怎么切分权重、forward 后是否通信, 由子类实现.
    """

    def __init__(
        self,
        input_size: int,
        output_size: int,
        bias: bool = False,
        tp_dim: int | None = None,
    ):
        super().__init__()

        # tp_dim 指定权重加载时沿哪个维度切分:
        #   None: 不切分
        #   0: 沿输出维切分, 对应 Column Parallel
        #   1: 沿输入维切分, 对应 Row Parallel
        self.tp_dim = tp_dim # 指定按哪个维度切分权重
        self.tp_rank = dist.get_rank() # 当前进程在tp维度上的rank
        self.tp_size = dist.get_world_size() # tp维度的总进程数

        # PyTorch Linear 的权重布局是 [out_features, in_features].
        # 这里的 input_size/output_size 可能已经是子类切分后的局部尺寸.
        self.weight = nn.Parameter(torch.empty(output_size, input_size))

        # 给 Parameter 动态挂载 weight_loader.
        # load_model 读取 safetensors 后会优先调用这个函数, 让参数自己决定
        # 是完整加载、按 TP 切片加载, 还是写入合并参数中的某一段.
        self.weight.weight_loader = self.weight_loader
        if bias:
            self.bias = nn.Parameter(torch.empty(output_size))
            self.bias.weight_loader = self.weight_loader
        else:
            # 显式注册为 None, 让模块结构和 nn.Linear 一样拥有 bias 字段.
            self.register_parameter("bias", None)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """基类不定义计算逻辑, 由具体并行策略的子类实现"""
        raise NotImplementedError


class ReplicatedLinear(LinearBase):
    """
    不做 tensor parallel 切分的普通线性层 (项目中没有使用)

    每个 rank 都保存完整的 weight 和 bias, forward 也直接执行完整的
    F.linear(x, weight, bias). 这种层适合参数量较小, 或者不需要切分的投影.
    """

    def __init__(
        self,
        input_size: int,
        output_size: int,
        bias: bool = False,
    ):
        # tp_dim=None 表示权重加载时不沿任何维度切分.
        super().__init__(input_size, output_size, bias)

    def weight_loader(self, param: nn.Parameter, loaded_weight: torch.Tensor):
        """直接加载完整权重, 不做 TP shard 切片"""
        param.data.copy_(loaded_weight)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # 每个 rank 都有完整参数, 因此这里不需要 collective 通信
        return F.linear(x, self.weight, self.bias)


class ColumnParallelLinear(LinearBase):
    """
    按输出维度切分的并行线性层

    完整 Linear 的权重形状是 [output_size, input_size].
    Column Parallel 会沿着 output_size 维度切分权重, 也就是每个 rank
    只保存一部分输出通道对应的权重行: local_weight: [output_size / tp_size, input_size]

    forward 时每个 rank 都接收完整输入 x, 但只计算自己的局部输出.
    因为输出本来就是按最后一维分片的, 所以这里不需要 all_reduce.
    """

    def __init__(
        self,
        input_size: int,
        output_size: int,
        bias: bool = False,
    ):
        tp_size = dist.get_world_size()
        # 当前 rank 只持有 output_size / tp_size 个输出通道.
        # tp_dim=0 表示加载权重时沿 weight 的第 0 维切分
        # 第 0 维: output_dim/正常矩阵乘法中的列方向
        super().__init__(input_size, divide(output_size, tp_size), bias, 0)

    def weight_loader(self, param: nn.Parameter, loaded_weight: torch.Tensor):
        """从完整权重中切出当前 rank 负责的输出通道分片"""
        param_data = param.data
        # param_data 是本 rank 的局部参数, 其第 0 维大小就是单个 shard 的大小.
        shard_size = param_data.size(self.tp_dim) # <=> param_data.shape[self.tp_dim]
        start_idx = self.tp_rank * shard_size

        # 从完整权重 loaded_weight 中取出 [start_idx, start_idx + shard_size)
        # 这一段输出通道, 写入当前 rank 的局部参数.
        loaded_weight = loaded_weight.narrow(self.tp_dim, start_idx, shard_size) # slice : tensor.narrow(dim, start, length) <=> tensor[:, start:start+length, ...] if dim=1
        param_data.copy_(loaded_weight)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # 输入 x 不切分, 每个 rank 用自己的局部 weight 计算局部输出.
        # 输出 shape 的最后一维是 output_size / tp_size,
        # 输出的不同维度散落在不同 TP 上, 方便后续继续进行 Row Parallel,
        # 如果需要合并可以使用 all_gather.
        return F.linear(x, self.weight, self.bias)


class MergedColumnParallelLinear(ColumnParallelLinear):
    """
    多个 Column Parallel 线性层合并后的实现.

    典型用途是把 MLP 中的 gate_proj 和 up_proj 合并成一个大线性层:
        [gate, up] = x @ merged_weight.T

    合并后的完整输出大小是 sum(output_sizes), 然后仍然沿输出维做 TP 切分.
    因此每个 rank 的局部参数布局是:
        [local_shard_0, local_shard_1, ...]
    例如 gate/up 合并时, 本 rank 保存 [local_gate, local_up].
    """

    def __init__(
        self,
        input_size: int,
        output_sizes: list[int],
        bias: bool = False,
    ):
        # output_sizes 记录合并前每个线性层的完整输出大小.
        # 例如 gate_proj 和 up_proj 合并时是 [intermediate_size, intermediate_size].
        self.output_sizes = output_sizes

        # 父类 ColumnParallelLinear 会把 sum(output_sizes) 按 tp_size 切分,
        # 当前 rank 最终只保存 sum(output_sizes) / tp_size 个输出通道.
        super().__init__(input_size, sum(output_sizes), bias)

    def weight_loader(self, param: nn.Parameter, loaded_weight: torch.Tensor, loaded_shard_id: int):
        """
        将某个合并前线性层的权重加载到切分后的参数 param 的对应局部区域

        loaded_shard_id 表示当前 loaded_weight 属于第几个原始线性层.
        param 需要保存不同 loaded_shard_id Linear 中属于自己 TP 的权重.
        例如 gate/up 合并时:
            loaded_shard_id=0 表示 gate_proj
            loaded_shard_id=1 表示 up_proj
        """
        param_data = param.data

        # 在当前 rank 的切分参数中, 找到该 shard 应该写入的局部偏移和大小.
        # 因为 param_data 已经是 TP 切分后的局部参数, 所以 offset/size 也要除以 tp_size.
        shard_offset = sum(self.output_sizes[:loaded_shard_id]) // self.tp_size
        shard_size = self.output_sizes[loaded_shard_id] // self.tp_size
        param_data = param_data.narrow(self.tp_dim, shard_offset, shard_size)

        # loaded_weight 是 HF 中该原始线性层的完整权重.
        # 沿输出维切成 tp_size 份后, 只取当前 rank 负责的输出通道.
        # 每个线性层的原始权重沿输出维度, 均分在各 TP 中
        loaded_weight = loaded_weight.chunk(self.tp_size, self.tp_dim)[self.tp_rank]
        param_data.copy_(loaded_weight)


class QKVParallelLinear(ColumnParallelLinear):
    """
    Attention 中使用的 QKV 合并 Column Parallel 线性层

    HuggingFace 原始权重通常是 q_proj、k_proj、v_proj 三个独立线性层.
    这里把三者合并成一个大权重: [Q, K, V] = x @ qkv_weight.T

    同时这个大权重仍然按输出维度做 tensor parallel 切分.
    对 Attention 来说, 输出维度可以理解为 head 维度展开后的结果,
    所以每个 rank 只负责一部分 Q heads 和一部分 KV heads.
    """

    def __init__(
        self,
        hidden_size: int,
        head_size: int,
        total_num_heads: int,
        total_num_kv_heads: int | None = None,
        bias: bool = False,
    ):
        tp_size = dist.get_world_size()
        total_num_kv_heads = total_num_kv_heads or total_num_heads
        self.head_size = head_size

        # 每个 TP rank 只保存并计算自己的 head 分片.
        # 例如 total_num_heads=32, tp_size=4 时, 每个 rank 负责 8 个 Q heads.
        self.num_heads = divide(total_num_heads, tp_size)
        self.num_kv_heads = divide(total_num_kv_heads, tp_size)

        # 完整 QKV 输出大小:
        #   Q: total_num_heads * head_size
        #   K: total_num_kv_heads * head_size
        #   V: total_num_kv_heads * head_size
        # super().__init__ 会再把这个 output_size 按 tp_size 切成局部输出大小.
        output_size = (total_num_heads + 2 * total_num_kv_heads) * self.head_size
        super().__init__(hidden_size, output_size, bias)

    def weight_loader(self, param: nn.Parameter, loaded_weight: torch.Tensor, loaded_shard_id: str):
        """
        将 HF 中单独的 q/k/v 权重加载到合并后的 qkv_proj 局部参数中

        loaded_shard_id 表示当前 loaded_weight 来自 q_proj、k_proj 还是 v_proj.
        param 是当前 rank 的合并参数, 布局为 [local_Q, local_K, local_V].
        """
        param_data = param.data
        assert loaded_shard_id in ["q", "k", "v"]

        # 先确定当前 q/k/v shard 在本 rank 合并参数中的写入位置.
        if loaded_shard_id == "q":
            shard_size = self.num_heads * self.head_size
            shard_offset = 0
        elif loaded_shard_id == "k":
            shard_size = self.num_kv_heads * self.head_size
            shard_offset = self.num_heads * self.head_size
        else:
            shard_size = self.num_kv_heads * self.head_size
            shard_offset = self.num_heads * self.head_size + self.num_kv_heads * self.head_size

        # param_data 是 [local_Q, local_K, local_V] 合并后的局部参数,
        # 这里先 narrow 到当前 q/k/v 对应的局部区域.
        param_data = param_data.narrow(self.tp_dim, shard_offset, shard_size)

        # loaded_weight 是 HF 中完整的 q/k/v 权重.
        # 沿输出维切成 tp_size 份, 只取当前 rank 对应的 head 分片.
        loaded_weight = loaded_weight.chunk(self.tp_size, self.tp_dim)[self.tp_rank]
        param_data.copy_(loaded_weight)


class RowParallelLinear(LinearBase):
    """
    按输入维度切分的并行线性层.

    完整 Linear 的权重形状是 [output_size, input_size].
    Row Parallel 会沿着 input_size 维度切分权重, 也就是每个 rank
    只保存一部分输入通道对应的权重列:
        local_weight: [output_size, input_size / tp_size]

    forward 时每个 rank 接收已经切分好的局部输入, 计算完整输出维度上的局部贡献.
    最后通过 all_reduce(sum) 把所有 rank 的局部贡献相加, 得到完整 Linear 输出.
    """

    def __init__(
        self,
        input_size: int,
        output_size: int,
        bias: bool = False,
    ):
        tp_size = dist.get_world_size()

        # 当前 rank 只持有 input_size / tp_size 个输入通道对应的权重列.
        # tp_dim=1 表示加载权重时沿 weight 的第 1 维切分.
        super().__init__(divide(input_size, tp_size), output_size, bias, 1)

    def weight_loader(self, param: nn.Parameter, loaded_weight: torch.Tensor):
        """从完整权重中切出当前 rank 负责的输入通道分片"""
        param_data = param.data

        # bias 不沿输入维切分, 每个 rank 都能加载完整 bias.
        # forward 时只让 rank 0 使用 bias, 避免 all_reduce 后重复相加.
        if param_data.ndim == 1:
            param_data.copy_(loaded_weight)
            return

        # param_data 是本 rank 的局部 weight, 其第 1 维大小就是单个 shard 的大小.
        shard_size = param_data.size(self.tp_dim)
        start_idx = self.tp_rank * shard_size

        # 从完整权重 loaded_weight 中取出当前 rank 负责的输入通道列.
        loaded_weight = loaded_weight.narrow(self.tp_dim, start_idx, shard_size)
        param_data.copy_(loaded_weight)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x 已经是当前 rank 对应的局部输入, shape 最后一维是 input_size / tp_size.
        # 每个 rank 先算出对完整输出的局部贡献.
        y = F.linear(x, self.weight, self.bias if self.tp_rank == 0 else None)

        # 矩阵乘法沿输入维度是求和关系:
        # full_y = x0 @ w0.T + x1 @ w1.T + ...
        # all_reduce(sum) 将所有 rank 的局部贡献相加为完整输出.
        if self.tp_size > 1:
            dist.all_reduce(y)
        return y


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
