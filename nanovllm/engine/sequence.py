from copy import copy
from enum import Enum, auto # 用于定义枚举类型和自动生成枚举值的函数
from itertools import count

from nanovllm.sampling_params import SamplingParams


class SequenceStatus(Enum): # 序列状态
    WAITING = auto() # 等待状态, 等待被调度进行prefill (没有完成 prefill) 或者是 decode 阶段被抢占
    RUNNING = auto() # 运行状态, 正在被 Scheduler 调度进行 decode
    FINISHED = auto() # 完成状态, 当生成到 EOS, 或者达到 max_tokens 限制


class Sequence:
    """
    表示一次生成请求中的单条序列, 保存 prompt token、已生成 token、
    调度状态、KV cache block 映射以及采样参数。

    Sequence 创建后默认处于 WAITING 状态, 随后由 Scheduler 调度为
    RUNNING, 并在遇到 EOS 或达到 max_tokens 后变为 FINISHED。
    调度、prefix caching、chunked prefill 和 decode 阶段都会通过
    num_cached_tokens、num_scheduled_tokens、block_table 等字段更新它的状态。
    """
    # 类变量, 所有实例共享的属性
    block_size = 256 # KV Cache在序列维度上的块大小, 每个块能存放多少个 Token 的KV序列
    counter = count() # 计数器, 用于给每个实例化的 Sequence 生成唯一的序列ID

    def __init__(self, token_ids: list[int], sampling_params = SamplingParams()):
        """
        Args:
            token_ids: 已编码(tokenize)好的 prompt token ID 列表, 至少需要包含一个 token
            sampling_params: 当前序列使用的采样参数, 包括 temperature、max_tokens 和 ignore_eos
        """
        self.seq_id = next(Sequence.counter) # 序列ID, 用于唯一标识每个实例化的 Sequence 对象
        self.status = SequenceStatus.WAITING # 序列状态, 默认是等待状态, 等待被调度
        self.token_ids = copy(token_ids) # 序列的 Token ID 列表, 包括提示 Token 和新生成 Token
        self.last_token = token_ids[-1] # 序列的 Token ID 列表的最后一个元素
        self.num_tokens = len(self.token_ids) # 序列的 Token 数量
        self.num_prompt_tokens = len(token_ids) # 提示 Token 的数量
        self.num_cached_tokens = 0 # 已命中KV缓存的 Token 数量, 这部分不需要重新 prefill
        self.num_scheduled_tokens = 0 # 当前调度轮 prefill/decode 处理的 Token 数量
        
        self.block_table = [] 
        # Paged Attention KV Cache 的块映射表, 记录虚拟块号到真实块号的映射关系
        # 真实块号对应 model_runner.kv_cache 的第三个维度的索引
        # seq.block_table = [7, 12]
        # 序列的第 0 个逻辑 block 存在 KV cache 的物理 block 7
        # 序列的第 1 个逻辑 block 存在 KV cache 的物理 block 12
        # block_table 在 Scheduler.schedule() 中, 通过 Scheduler.block_manager 进行分配
        # prefill 阶段在 block_manager.allocate 中根据 num_cached_tokens 和 block_size 来一次性分配多个 block
        # decode 阶段在 block_manager.may_append 中根据 len(sequence) 来追加分配 block

        self.temperature = sampling_params.temperature # 采样温度, 控制生成随机性
        self.max_tokens = sampling_params.max_tokens # 最多生成的 completion Token 数量
        self.ignore_eos = sampling_params.ignore_eos # 是否忽略 EOS Token 继续生成, 直到达到 max_tokens 限制

    def __len__(self):
        return self.num_tokens

    def __getitem__(self, key):
        return self.token_ids[key]

    @property
    def is_finished(self):
        """当前序列是否已经完成生成"""
        return self.status == SequenceStatus.FINISHED

    @property
    def num_completion_tokens(self):
        """新生成的 token 数量, 不包含 prompt token"""
        return self.num_tokens - self.num_prompt_tokens

    @property
    def prompt_token_ids(self):
        """原始 prompt 对应的 token ID 列表"""
        return self.token_ids[:self.num_prompt_tokens]

    @property
    def completion_token_ids(self):
        """新生成出来的 token ID 列表"""
        return self.token_ids[self.num_prompt_tokens:]

    @property
    def num_blocks(self):
        """当前序列按照 block_size 切分后需要占用的 block 数量"""
        # ceil(n, b) = (n + b - 1) // b
        return (self.num_tokens + self.block_size - 1) // self.block_size

    @property
    def last_block_num_tokens(self):
        """当前序列最后一个 block 中已经存放的 token 数量"""
        return self.num_tokens - (self.num_blocks - 1) * self.block_size

    def block(self, i):
        """返回第 i 个 block 对应的 tokens 切片"""
        assert 0 <= i < self.num_blocks
        # Slice token_ids by block index and block size.
        return self.token_ids[i*self.block_size: (i+1)*self.block_size]

    def append_token(self, token_id: int):
        """追加一个新生成的 token, 并同步更新序列状态。"""
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
