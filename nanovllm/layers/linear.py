import torch
from torch import nn
import torch.nn.functional as F
import torch.distributed as dist


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