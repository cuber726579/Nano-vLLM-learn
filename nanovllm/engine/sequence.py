from copy import copy
from enum import Enum, auto # 用于定义枚举类型和自动生成枚举值的函数
from itertools import count

from nanovllm.sampling_params import SamplingParams


class SequenceStatus(Enum): # 序列状态
    WAITING = auto() # 等待状态，等待被调度，存放在 Scheduler 的等待队列中
    RUNNING = auto() # 运行状态，正在被 Scheduler 调度，正在被处理
    FINISHED = auto() # 完成状态，已经完成，结果已经生成


class Sequence:
    # 类变量，所有实例共享的属性
    block_size = 256 # KV Cache在序列维度上的块大小，每个块能存放多少个 Token 的KV序列
    counter = count() # 计数器，用于给每个实例化的 Sequence 生成唯一的序列ID

    def __init__(self, token_ids: list[int], sampling_params = SamplingParams()):
        self.seq_id = next(Sequence.counter) # 序列ID，用于唯一标识每个实例化的 Sequence 对象
        self.status = SequenceStatus.WAITING # 序列状态，默认是等待状态，等待被调度
        self.token_ids = copy(token_ids) # 序列的 Token ID 列表，包括提示 Token 和新生成 Token
        self.last_token = token_ids[-1] # 序列的 Token ID 列表的最后一个元素
        self.num_tokens = len(self.token_ids) # 序列的 Token 数量
        self.num_prompt_tokens = len(token_ids) # 提示 Token 的数量
        self.num_cached_tokens = 0 # 已命中KV缓存的 Token 数量，这部分不需要重新 prefill
        self.num_scheduled_tokens = 0 # 当前调度轮计划处理的 Token 数量
        self.block_table = [] # 当前序列占用的 KV cache block 编号列表
        self.temperature = sampling_params.temperature # 采样温度，控制生成随机性
        self.max_tokens = sampling_params.max_tokens # 最多生成的 completion Token 数量
        self.ignore_eos = sampling_params.ignore_eos # 是否忽略 EOS Token 继续生成

    def __len__(self):
        return self.num_tokens

    def __getitem__(self, key):
        return self.token_ids[key]

    @property
    def is_finished(self):
        return self.status == SequenceStatus.FINISHED

    @property
    def num_completion_tokens(self):
        return self.num_tokens - self.num_prompt_tokens

    @property
    def prompt_token_ids(self):
        return self.token_ids[:self.num_prompt_tokens]

    @property
    def completion_token_ids(self):
        return self.token_ids[self.num_prompt_tokens:]

    @property
    def num_blocks(self):
        return (self.num_tokens + self.block_size - 1) // self.block_size

    @property
    def last_block_num_tokens(self):
        return self.num_tokens - (self.num_blocks - 1) * self.block_size

    def block(self, i):
        assert 0 <= i < self.num_blocks
        return self.token_ids[i*self.block_size: (i+1)*self.block_size]

    def append_token(self, token_id: int):
        self.token_ids.append(token_id)
        self.last_token = token_id
        self.num_tokens += 1

    def __getstate__(self):
        last_state = self.token_ids if self.num_completion_tokens == 0 or self.num_cached_tokens < self.num_tokens else self.last_token
        return (self.num_tokens, self.num_prompt_tokens, self.num_cached_tokens, self.num_scheduled_tokens, self.block_table, last_state)

    def __setstate__(self, state):
        self.num_tokens, self.num_prompt_tokens, self.num_cached_tokens, self.num_scheduled_tokens, self.block_table, last_state = state
        if isinstance(last_state, list):
            self.token_ids = last_state
            self.last_token = self.token_ids[-1]
        else:
            self.token_ids = []
            self.last_token = last_state

if __name__ == "__main__":
    status = SequenceStatus.WAITING
    print(status)
