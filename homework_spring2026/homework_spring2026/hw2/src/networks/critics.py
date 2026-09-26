"""状态价值网络（critic / baseline）。

策略梯度本身不依赖 critic 才能成立，但从 Monte Carlo Q 估计中减去 ``V(s)``
可以显著降低方差。本作业的 critic 是一个输出单个标量的 MLP，以采样得到的
``q_values`` 为监督目标进行回归。
"""

import itertools
from torch import nn
from torch.nn import functional as F
from torch import optim

import numpy as np
import torch
from torch import distributions

from infrastructure import pytorch_util as ptu


class ValueCritic(nn.Module):
    """输入 observation、输出状态价值估计的神经网络。"""

    def __init__(
        self,
        ob_dim: int,
        n_layers: int,
        layer_size: int,
        learning_rate: float,
    ):
        super().__init__()

        # 对 batch 输入 (B, ob_dim)，底层网络输出 (B, 1)。forward 中通常要把
        # 最后一维压成 (B,)，以匹配一维 q_values。
        self.network = ptu.build_mlp(
            input_size=ob_dim,
            output_size=1,
            n_layers=n_layers,
            size=layer_size,
        ).to(ptu.device)

        # critic 有独立于 actor 的学习率和优化器。
        self.optimizer = optim.Adam(
            self.network.parameters(),
            learning_rate,
        )

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        """计算每个 observation 的标量价值估计 ``V_φ(s)``。"""
        # 只移除最后那个确定为 1 的维度；不能使用无参数 squeeze，否则 batch
        # size 为 1 时也会意外丢掉 batch 维。
        return self.network(obs).squeeze(-1)

    def update(self, obs: np.ndarray, q_values: np.ndarray) -> dict:
        """用 Monte Carlo Q 估计监督训练 value baseline 一步。"""
        # 数据转换规则与 actor 一致：float32，并移动到 ptu.device。
        obs = ptu.from_numpy(obs)
        q_values = ptu.from_numpy(q_values)

        values = self(obs)
        loss = F.mse_loss(values, q_values)

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        # PGAgent 会把该字段合并进每轮训练日志。
        return {
            "Baseline Loss": loss.item(),
        }
