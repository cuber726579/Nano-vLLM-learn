import torch
from torch import nn


class RMSNorm(nn.Module):

    def __init__(
        self,
        hidden_size: int,
        eps: float = 1e-6,
    ) -> None:
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(hidden_size))

    @torch.compile
    def rms_forward(
        self,
        x: torch.Tensor,
    ) -> torch.Tensor:
        """
        执行不带残差相加的 RMSNorm, 
        在 Token-Embedding 之后, 
        第一个 Decoder Layer 的 Input Norm 执行一次.

        Args:
            x: 需要归一化的 hidden states

        返回与输入形状相同的归一化结果
        """
        orig_dtype = x.dtype
        x = x.float()
        var = x.pow(2).mean(dim=-1, keepdim=True)
        x.mul_(torch.rsqrt(var + self.eps))
        x = x.to(orig_dtype).mul_(self.weight)
        return x

    @torch.compile
    def add_rms_forward(
        self,
        x: torch.Tensor,
        residual: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        执行融合的残差相加和 RMSNorm

        Args:
            x: 前一个子模块输出的 hidden states, 没有跟 residual进行相加; 例如 attention 或 MLP 的输出.
            residual: 前面保存并持续传递的残差流, 会与 x 相加后更新.

        返回归一化后的 hidden states, 以及加完残差后保存下来的 residual,
        对应 decoder layer 中持续传递的残差流.
        """
        orig_dtype = x.dtype
        x = x.float().add_(residual.float())
        residual = x.to(orig_dtype)
        var = x.pow(2).mean(dim=-1, keepdim=True)
        x.mul_(torch.rsqrt(var + self.eps))
        x = x.to(orig_dtype).mul_(self.weight)
        return x, residual

    def forward(
        self,
        x: torch.Tensor,
        residual: torch.Tensor | None = None,
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        if residual is None:
            return self.rms_forward(x)
        else:
            return self.add_rms_forward(x, residual)
