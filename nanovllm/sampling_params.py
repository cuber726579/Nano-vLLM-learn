from dataclasses import dataclass


@dataclass(slots=True)
class SamplingParams: # 每条生成请求的采样参数，包含在 Sequence 中，由 Scheduler 传递给 ModelRunner
    temperature: float = 1.0 # 采样温度，值越大输出越随机；当前实现不允许 greedy sampling
    max_tokens: int = 64 # 单条序列最多新生成的 token 数，不包含 prompt token (prompt token + max_tokens <= config.max_model_len)
    ignore_eos: bool = False # 是否忽略结束符 token；为 True 时遇到 eos 也继续生成，直到达到 max_tokens

    def __post_init__(self):
        assert self.temperature > 1e-10, "greedy sampling is not permitted"
