import torch
from torch import nn
import torch.nn.functional as F
import torch.distributed as dist

from nanovllm.utils.context import get_context


class VocabParallelEmbedding(nn.Module):
    """
    按词表维度切分的 Embedding 层

    每个 tensor parallel rank 只保存完整 vocab 的一段 embedding 权重.
    forward 时, 当前 rank 只处理落在自己 vocab 范围内的 token, 最后通过
    all_reduce 汇总所有 rank 的 embedding 结果.
    """

    def __init__(
        self,
        num_embeddings: int,
        embedding_dim: int,
    ):
        """
        Args:
            num_embeddings: 完整词表大小
            embedding_dim: 每个 token embedding 的隐藏维度
        """
        super().__init__()
        self.tp_rank = dist.get_rank()
        self.tp_size = dist.get_world_size()
        assert num_embeddings % self.tp_size == 0
        self.num_embeddings = num_embeddings
        self.num_embeddings_per_partition = self.num_embeddings // self.tp_size
        # 当前 rank 负责的 vocab 范围: [vocab_start_idx, vocab_end_idx)
        self.vocab_start_idx = self.num_embeddings_per_partition * self.tp_rank
        self.vocab_end_idx = self.vocab_start_idx + self.num_embeddings_per_partition
        self.weight = nn.Parameter(torch.empty(self.num_embeddings_per_partition, embedding_dim))
        self.weight.weight_loader = self.weight_loader

    def weight_loader(self, param: nn.Parameter, loaded_weight: torch.Tensor):
        """
        加载当前 rank 对应的 embedding 权重分片

        Args:
            param: 当前模块的 self.weight 参数
            loaded_weight: safetensors 中读取到的完整 embedding 权重
        """
        param_data = param.data
        shard_size = param_data.size(0)
        start_idx = self.tp_rank * shard_size

        # narrow(0, start_idx, shard_size) : loaded_weight[start_idx:start_idx + shard_size,]
        # 沿0维度进行切片, 从索引 start_idx开始, 切大小为 shard_size 的部分
        loaded_weight = loaded_weight.narrow(0, start_idx, shard_size) 
        param_data.copy_(loaded_weight)

    def forward(self, x: torch.Tensor):
        """
        根据 token id 查表得到 hidden states

        Args:
            x: 输入 token ids

        Returns:
            与输入 token ids 对应的 embedding hidden states
        """
        if self.tp_size > 1:
            # 掩码不属于当前 rank 的 token
            mask = (x >= self.vocab_start_idx) & (x < self.vocab_end_idx)

            # 全局 token id 转成局部 vocab id
            # 通过 mask 把不属于当前 rank 的位置置 0
            # 否则不属于当前 rank 的位置查表可能出现越界
            x = mask * (x - self.vocab_start_idx)
        y = F.embedding(x, self.weight)
        if self.tp_size > 1:
            y = mask.unsqueeze(1) * y # 非当前 rank 负责的 token 置零

            # 通过 all_reduce 对各个 rank 的结果求和, 合并所有 rank 的结果
            # 上述两次 mask 操作已经保证非当前 rank 处理的位置, 都为 0
            dist.all_reduce(y) 
        return y


class ParallelLMHead(VocabParallelEmbedding):
    """
    按词表维度切分的 LM Head

    该层复用 VocabParallelEmbedding 的权重形状和加载逻辑, 但 forward 中执行
    hidden states 到 vocab logits 的线性投影. 每个 tensor parallel rank 只计算
    自己负责的 vocab 分片 logits, 最后收集到 rank 0 并拼成完整 vocab logits.
    """

    def __init__(
        self,
        num_embeddings: int,
        embedding_dim: int,
        bias: bool = False,
    ):
        """
        Args:
            num_embeddings: 完整词表大小
            embedding_dim: hidden states 的隐藏维度
            bias: LM Head 是否使用 bias, 当前实现不支持 bias
        """
        assert not bias
        super().__init__(num_embeddings, embedding_dim)

    def forward(self, x: torch.Tensor):
        """
        将 hidden states 投影为 vocab logits

        Args:
            x: 模型输出的 hidden states

        Returns:
            rank 0 上返回完整 vocab logits; 其他 rank 在 tensor parallel 时返回 None.
        """
        context = get_context()
        if context.is_prefill:
            # 在 Prefill 阶段, 只取每段 Seq 的最后一个 Token 作为输出
            last_indices = context.cu_seqlens_q[1:] - 1
            x = x[last_indices].contiguous()
        logits = F.linear(x, self.weight)
        if self.tp_size > 1:
            # 每个 rank 只得到局部 vocab logits, 收集到 rank 0 后沿 vocab 维度拼接
            all_logits = [torch.empty_like(logits) for _ in range(self.tp_size)] if self.tp_rank == 0 else None
            dist.gather(logits, all_logits, 0)
            logits = torch.cat(all_logits, -1) if self.tp_rank == 0 else None
        return logits
