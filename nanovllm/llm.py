from nanovllm.engine.llm_engine import LLMEngine


class LLM(LLMEngine):
    """
    用户侧的高级推理入口, 对标工业级 vLLM 中面向 Python 用户的 LLM API.

    在本项目里, LLM 只是 LLMEngine 的薄封装: 初始化模型、tokenizer、
    scheduler、model runner 之后, 直接复用 LLMEngine.generate() 完成离线批量
    推理. 真正的调度、KV cache 管理、prefill/decode 执行都在 engine 目录中.

    工业级 vLLM 中, 这一层通常还会负责更多“产品化 API”能力, 例如:
      1. 统一的 generate/chat/encode/score 等高层接口;
      2. prompts、token ids、chat messages、multimodal inputs 等输入格式适配;
      3. SamplingParams、LoRA、guided decoding、structured output 等请求级参数校验;
      4. 同步/异步调用、流式输出、请求取消、超时和错误处理;
      5. tokenizer、chat template、输出 detokenize 与返回结构封装;
      6. 与底层 engine 隔离, 让用户无需关心 scheduler、worker、KV cache 等细节.

    简单说, LLM 是用户 API 门面; LLMEngine 是真正执行推理生命周期的核心.
    """
    pass
