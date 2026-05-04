import os
from glob import glob
import torch
from torch import nn
from safetensors import safe_open


def default_weight_loader(param: nn.Parameter, loaded_weight: torch.Tensor):
    """
    默认权重加载逻辑

    当参数对象没有自定义 weight_loader 时, 直接把读取到的权重复制到目标参数中.
    """
    param.data.copy_(loaded_weight)


def load_model(model: nn.Module, path: str):
    """
    从 safetensors 文件加载模型权重

    加载时会根据模型的 packed_modules_mapping, 将 HF 原始权重中分开的
    q/k/v 或 gate/up 权重映射到项目中合并后的参数名. 找到目标参数后,
    优先调用参数对象上挂载的 weight_loader, 由具体模块决定是直接复制,
    按 tensor parallel 切片加载, 还是写入合并参数的指定分片.

    weight_loader 机制: 
        vLLM 里的 weight_loader 是一套“每个参数自己知道该怎么加载权重”的机制.
        不是简单 state_dict[name].copy_(), 因为 vLLM 的参数经常需要 Tensor Parallel, 
        QKV/Gate-Up 融合, 量化格式, MoE expert 分片去修改参数布局.
        因此模块会自定义 self.weight_loader 方法去重写参数加载的过程, 并挂在 weight.weight_loader 属性上.
        ```
            self.weight = nn.Parameter(torch.empty(output_size, input_size))
            self.weight.weight_loader = self.weight_loader
        ```

    其他函数解析:
        get_parameter(param_name): 
            torch.nn.Module 自带的方法, 根据参数名字 param_name 获取对应的 Parameter 对象
        getattr(obj, attr_name, default_value): 
            Python 内置函数, 用来根据字符串从对象 obj 上取属性 attr_name, 不存在使用默认值 default_value
        get_tensor(weight_name):
            safetensors 读取器对象的方法，根据权重名 weight_name 从 .safetensors 文件中取出对应的 tensor
    """
    packed_modules_mapping = getattr(model, "packed_modules_mapping", {})
    for file in glob(os.path.join(path, "*.safetensors")):
        with safe_open(file, "pt", "cpu") as f:
            for weight_name in f.keys():
                for k in packed_modules_mapping:
                    if k in weight_name:
                        v, shard_id = packed_modules_mapping[k]
                        param_name = weight_name.replace(k, v)
                        param = model.get_parameter(param_name)
                        weight_loader = getattr(param, "weight_loader")
                        weight_loader(param, f.get_tensor(weight_name), shard_id)
                        break
                else:
                    param = model.get_parameter(weight_name)
                    weight_loader = getattr(param, "weight_loader", default_weight_loader)
                    weight_loader(param, f.get_tensor(weight_name))
