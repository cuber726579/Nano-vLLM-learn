import torch
from torch import nn


class Sampler(nn.Module):

    @torch.compile
    def forward(self, logits: torch.Tensor, temperatures: torch.Tensor):
        # 将 logits 转成 float32 后按每条序列的 temperature 缩放
        # temperature 越大分布越平, 随机性越强; 越小分布越尖锐, logit 大的概率大
        logits = logits.float().div_(temperatures.unsqueeze(dim=1))

        probs = torch.softmax(logits, dim=-1) # 将缩放后的 logits 转成概率分布

        # 根据概率分布 probs 采样 token
        sample_tokens = probs.div_(torch.empty_like(probs).exponential_(1).clamp_min_(1e-10)).argmax(dim=-1)
        return sample_tokens
