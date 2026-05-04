import torch
from torch import nn
import torch.nn.functional as F


class SiluAndMul(nn.Module):
    """
    SwiGLU 激活层: SwiGLU(gate, up) = silu(gate) * up

    输入来自合并后的 gate_up_proj, 最后一维包含 gate 和 up 两个分支.
    forward 中先把二者切开, 再计算 silu(gate) * up.
    """

    @torch.compile
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x, y = x.chunk(2, -1) # Gate, Up
        return F.silu(x) * y
