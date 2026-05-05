import os
from dataclasses import dataclass
from transformers import AutoConfig


@dataclass(slots=True) # 标准库装饰器，自动生成初始化等方法，并限制实例只能使用已声明字段
class Config:
    model: str # 模型加载路径，用于加载 HuggingFace 模型配置 hf_config 和初始化 tokenizer
    max_num_batched_tokens: int = 16384 # Scheduler 中单次调度 batch 中允许处理的最大 token 数（调度budget）
    max_num_seqs: int = 512 # Scheduler 中单次调度 batch 中允许并发处理的最大序列数
    
    max_model_len: int = 4096 # 单条序列的最大上下文(Prompt+Generate) token 长度，受限于模型的最大上下文大小
    # prompt_tokens + max_tokens <= config.max_model_len

    gpu_memory_utilization: float = 0.9 # 最大显存利用率，包括模型占用和生成过程中 KV cache 的占用
    tensor_parallel_size: int = 1 # 张量并行的 GPU 数量
    enforce_eager: bool = False # 是否强制使用 eager 模式，开启后不使用 CUDA Graph 优化，每次 forward 都按普通 PyTorch 方式立即执行
    hf_config: AutoConfig | None = None # 根据 model 进行HuggingFace 模型配置，初始化后从 model 路径加载
    eos: int = -1 # 结束符 token id，标志一个 Sequence 的生成完成
    kvcache_block_size: int = 256 # PageAttention KV cache 的块大小，每个块可存放的 token 数
    num_kvcache_blocks: int = -1 # PageAttention 可分配的 KV cache 块数量，根据模型加载之后的剩余显存后计算得到

    def __post_init__(self):
        assert os.path.isdir(self.model)
        assert self.kvcache_block_size % 256 == 0
        assert 1 <= self.tensor_parallel_size <= 8
        self.hf_config = AutoConfig.from_pretrained(self.model)
        self.max_model_len = min(self.max_model_len, self.hf_config.max_position_embeddings)
        assert self.max_num_batched_tokens >= self.max_model_len