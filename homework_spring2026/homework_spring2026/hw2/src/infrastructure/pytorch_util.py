"""PyTorch 通用工具。

本模块集中管理三类重复操作：构造多层感知机（MLP）、选择 CPU/GPU，以及在
NumPy 数组和 PyTorch 张量之间转换。策略网络与价值网络都依赖这里的函数，因而
设备选择必须在创建网络之前由 ``init_gpu`` 完成。
"""

from typing import Union

import torch
from torch import nn

# 激活函数既可以写成名称（如 ``"tanh"``），也可以直接传入 nn.Module 实例。
Activation = Union[str, nn.Module]


# 字符串到 PyTorch 激活层的映射，供 build_mlp 解析配置。
_str_to_activation = {
    'relu': nn.ReLU(),
    'tanh': nn.Tanh(),
    'leaky_relu': nn.LeakyReLU(),
    'sigmoid': nn.Sigmoid(),
    'selu': nn.SELU(),
    'softplus': nn.Softplus(),
    'identity': nn.Identity(),
}

# 全局计算设备。run.py 会在创建 PGAgent 前调用 init_gpu 对它赋值。
device = None


def build_mlp(
        input_size: int,
        output_size: int,
        n_layers: int,
        size: int,
        activation: Activation = 'tanh',
        output_activation: Activation = 'identity',
):
    """构造一个全连接前馈网络。

    网络形状为：``input_size -> n_layers × size -> output_size``。每个隐藏层后
    使用 ``activation``，输出层后使用 ``output_activation``。返回的
    ``nn.Sequential`` 已经移动到全局 ``device``。

    Args:
        input_size: 输入特征维度，例如 observation 的维度。
        output_size: 输出维度，例如动作数、动作维度或标量 value。
        n_layers: 隐藏层数量。
        size: 每个隐藏层的神经元数量。
        activation: 隐藏层激活函数或其名称。
        output_activation: 输出层激活函数或其名称。

    Returns:
        按指定结构组成并放到目标设备上的 ``nn.Sequential``。
    """
    # 若传入的是字符串，先解析成真正的激活层对象。
    if isinstance(activation, str):
        activation = _str_to_activation[activation]
    if isinstance(output_activation, str):
        output_activation = _str_to_activation[output_activation]
    # 逐层维护当前输入宽度：首层来自 input_size，后续层都来自 size。
    layers = []
    in_size = input_size
    for _ in range(n_layers):
        layers.append(nn.Linear(in_size, size))
        layers.append(activation)
        in_size = size

    # 最后一层把隐藏表示投影到调用方需要的输出维度。
    layers.append(nn.Linear(in_size, output_size))
    layers.append(output_activation)

    mlp = nn.Sequential(*layers)
    mlp.to(device)
    return mlp


def init_gpu(use_gpu=True, gpu_id=0):
    """选择全局计算设备；CUDA 不可用或被禁用时自动退回 CPU。"""
    global device
    if torch.cuda.is_available() and use_gpu:
        device = torch.device("cuda:" + str(gpu_id))
        print("Using GPU id {}".format(gpu_id))
    else:
        device = torch.device("cpu")
        print("Using CPU.")


def set_device(gpu_id):
    """切换当前 CUDA device；仅在多 GPU 场景下需要。"""
    torch.cuda.set_device(gpu_id)


def from_numpy(*args, **kwargs):
    """把 NumPy 数组转成 float32 张量，并移动到全局计算设备。"""
    return torch.from_numpy(*args, **kwargs).float().to(device)


def to_numpy(tensor):
    """停止梯度跟踪，把张量移到 CPU 后转换成 NumPy 数组。"""
    return tensor.to('cpu').detach().numpy()
