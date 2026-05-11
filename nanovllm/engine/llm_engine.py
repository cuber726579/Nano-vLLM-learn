import atexit
from dataclasses import fields
from time import perf_counter
from tqdm.auto import tqdm
from transformers import AutoTokenizer
import torch.multiprocessing as mp

from nanovllm.config import Config
from nanovllm.sampling_params import SamplingParams
from nanovllm.engine.sequence import Sequence
from nanovllm.engine.scheduler import Scheduler
from nanovllm.engine.model_runner import ModelRunner


class LLMEngine:
    """LLM 推理引擎: 负责串起 Config, Tokenizer, Scheduler 和 ModelRunner"""

    def __init__(self, model, **kwargs):
        """初始化推理引擎, 并按 tensor_parallel_size 启动必要的模型执行进程"""
        # 只把 Config 中声明过的参数透传进去, 避免外部传入的无关 kwargs 影响配置初始化
        config_fields = {field.name for field in fields(Config)}
        config_kwargs = {k: v for k, v in kwargs.items() if k in config_fields}
        config = Config(model, **config_kwargs)
        # Sequence 使用类变量记录 KV cache block 大小, 后续每条请求都会按这个粒度管理 block_table
        Sequence.block_size = config.kvcache_block_size
        self.ps = []
        self.events = []
        ctx = mp.get_context("spawn")
        for i in range(1, config.tensor_parallel_size): # Sub Process
            # rank 1 到 N-1 放在子进程中运行, 用 Event 与主进程的 ModelRunner 做同步
            event = ctx.Event()
            process = ctx.Process(target=ModelRunner, args=(config, i, event))
            process.start()
            self.ps.append(process)
            self.events.append(event)
        self.model_runner = ModelRunner(config, 0, self.events) # Main Process

        # 加载 tokenizer, 确定 eos token id, 用于 Scheduler 结束判断
        self.tokenizer = AutoTokenizer.from_pretrained(config.model, use_fast=True)
        config.eos = self.tokenizer.eos_token_id

        self.scheduler = Scheduler(config)

        # 程序退出时自动释放 ModelRunner 和子进程, 避免 tensor parallel 子进程残留
        atexit.register(self.exit)

    def exit(self):
        """关闭模型执行器, 并等待所有 tensor parallel 子进程退出"""
        self.model_runner.call("exit")
        del self.model_runner
        for p in self.ps:
            p.join()

    def add_request(self, prompt: str | list[int], sampling_params: SamplingParams):
        """prompt 请求封装成 Sequence, 并加入 Scheduler 的等待队列"""
        if isinstance(prompt, str): # 即可以输入自然语言, 也可以输入 token id 列表
            prompt = self.tokenizer.encode(prompt)
        seq = Sequence(prompt, sampling_params)
        self.scheduler.add(seq)

    def step(self):
        """单轮调度和模型推理入口"""
        # Scheduler 决定本轮做 prefill 还是 decode, 以及具体调度哪些 Sequence
        seqs, is_prefill = self.scheduler.schedule()
        num_tokens = sum(seq.num_scheduled_tokens for seq in seqs) if is_prefill else -len(seqs) # 正负数区分单轮调度 prefill 和 decode 的 token 数量

        # ModelRunner 负责真正的模型 forward / sampling, 返回每条序列本轮采样出的单个 token
        token_ids = self.model_runner.call("run", seqs, is_prefill)

        # Scheduler 根据模型输出更新 Sequence 状态, 并释放已经完成序列的 KV cache
        self.scheduler.postprocess(seqs, token_ids, is_prefill)
        outputs = [(seq.seq_id, seq.completion_token_ids) for seq in seqs if seq.is_finished]
        return outputs, num_tokens # 返回本轮完成的请求及吞吐量统计用 token 数

    def is_finished(self):
        """判断当前引擎中所有请求是否都已经处理完成"""
        return self.scheduler.is_finished()

    def generate(
        self,
        prompts: list[str] | list[list[int]],
        sampling_params: SamplingParams | list[SamplingParams],
        use_tqdm: bool = True,
    ) -> list[str]:
        """
        顶层批量文本生成入口

        批量提交 prompts, 并持续 step 直到全部请求完成, 最后返回文本和 token id
        """
        pbar = tqdm(total=len(prompts), desc="Generating", dynamic_ncols=True, disable=not use_tqdm)
        # 支持所有 prompt 共用一份 SamplingParams, 也支持每条 prompt 使用独立参数
        if not isinstance(sampling_params, list):
            sampling_params = [sampling_params] * len(prompts)
        for prompt, sp in zip(prompts, sampling_params):
            self.add_request(prompt, sp)
        outputs = {}
        prefill_throughput = decode_throughput = 0.
        while not self.is_finished():
            t = perf_counter()
            output, num_tokens = self.step()
            # step 返回的 num_tokens 约定为：prefill 为正数, decode 为负数, 便于区分两类吞吐
            if num_tokens > 0:
                prefill_throughput = num_tokens / (perf_counter() - t)
            else:
                decode_throughput = -num_tokens / (perf_counter() - t)
            pbar.set_postfix({
                "Prefill": f"{int(prefill_throughput)}tok/s",
                "Decode": f"{int(decode_throughput)}tok/s",
            })
            for seq_id, token_ids in output:
                outputs[seq_id] = token_ids
                pbar.update(1)
        pbar.close()
        # Scheduler 内部完成顺序不一定等于请求提交顺序, 因此按 seq_id 排序恢复原始请求顺序
        outputs = [outputs[seq_id] for seq_id in sorted(outputs.keys())]
        outputs = [{"text": self.tokenizer.decode(token_ids), "token_ids": token_ids} for token_ids in outputs]
        return outputs
