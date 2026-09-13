"""Push-T 模仿学习策略的模型定义。

本模块规定了训练代码与评估代码共同依赖的策略接口，并为作业的两种策略
（直接回归动作块的 MSE 策略、通过常微分方程生成动作块的 Flow Matching
策略）保留了待实现骨架。动作块统一使用形状
``(batch_size, chunk_size, action_dim)``。
"""

from __future__ import annotations

import abc
from typing import Literal, TypeAlias

import torch
from torch import nn
from torch.nn import functional as F


def _build_mlp(
    input_dim: int,
    output_dim: int,
    hidden_dims: tuple[int, ...],
) -> nn.Sequential:
    """构造由 Linear + ReLU 隐藏层组成的简单 MLP。"""

    layers: list[nn.Module] = []
    previous_dim = input_dim
    for hidden_dim in hidden_dims:
        layers.extend((nn.Linear(previous_dim, hidden_dim), nn.ReLU()))
        previous_dim = hidden_dim
    # 输出层不加激活函数，因为标准化后的动作和速度都可以取任意实数。
    layers.append(nn.Linear(previous_dim, output_dim))
    return nn.Sequential(*layers)


class BasePolicy(nn.Module, metaclass=abc.ABCMeta):
    """动作分块策略的抽象基类。

    子类必须实现训练时使用的 ``compute_loss`` 和推理时使用的
    ``sample_actions``。统一接口使 ``train.py`` 和 ``evaluation.py`` 无需知道
    当前策略究竟是 MSE 还是 Flow Matching。
    """

    def __init__(self, state_dim: int, action_dim: int, chunk_size: int) -> None:
        super().__init__()
        # 保存维度元数据，供子类搭建网络、reshape 输出和采样时使用。
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.chunk_size = chunk_size

    @abc.abstractmethod
    def compute_loss(
        self, state: torch.Tensor, action_chunk: torch.Tensor
    ) -> torch.Tensor:
        """计算一个 batch 的训练损失，并返回可反向传播的标量张量。"""

    @abc.abstractmethod
    def sample_actions(
        self,
        state: torch.Tensor,
        *,
        num_steps: int = 10,  # only applicable for flow policy
    ) -> torch.Tensor:
        """生成动作块，输出形状为 ``(batch, chunk_size, action_dim)``。"""


class MSEPolicy(BasePolicy):
    """用一次前向传播直接预测完整动作块，并以 MSE 监督训练。"""

    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        chunk_size: int,
        hidden_dims: tuple[int, ...] = (128, 128),
    ) -> None:
        super().__init__(state_dim, action_dim, chunk_size)
        self.network = _build_mlp(
            input_dim=state_dim,
            output_dim=chunk_size * action_dim,
            hidden_dims=hidden_dims,
        )

    def compute_loss(
        self,
        state: torch.Tensor,
        action_chunk: torch.Tensor,
    ) -> torch.Tensor:
        prediction = self.sample_actions(state)
        return F.mse_loss(prediction, action_chunk)

    def sample_actions(
        self,
        state: torch.Tensor,
        *,
        num_steps: int = 10,
    ) -> torch.Tensor:
        # MSE 策略无需迭代去噪；num_steps 仅为兼容统一接口而保留。
        del num_steps
        flat_actions = self.network(state)
        return flat_actions.reshape(-1, self.chunk_size, self.action_dim)


class FlowMatchingPolicy(BasePolicy):
    """学习条件速度场，将高斯噪声逐步运输为专家动作块。"""

    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        chunk_size: int,
        hidden_dims: tuple[int, ...] = (128, 128),
    ) -> None:
        super().__init__(state_dim, action_dim, chunk_size)
        self.flat_action_dim = chunk_size * action_dim
        # 输入由状态、展平后的 A_tau，以及每个样本的标量时间 tau 拼接而成。
        self.velocity_network = _build_mlp(
            input_dim=state_dim + self.flat_action_dim + 1,
            output_dim=self.flat_action_dim,
            hidden_dims=hidden_dims,
        )

    def _predict_velocity(
        self,
        state: torch.Tensor,
        action_chunk: torch.Tensor,
        tau: torch.Tensor,
    ) -> torch.Tensor:
        """预测条件速度场，并将展平输出还原为动作块形状。"""

        batch_size = state.shape[0]
        flat_chunk = action_chunk.reshape(batch_size, self.flat_action_dim)
        network_input = torch.cat((state, flat_chunk, tau), dim=-1)
        flat_velocity = self.velocity_network(network_input)
        return flat_velocity.reshape(batch_size, self.chunk_size, self.action_dim)

    def compute_loss(
        self,
        state: torch.Tensor,
        action_chunk: torch.Tensor,
    ) -> torch.Tensor:
        batch_size = state.shape[0]
        noise = torch.randn_like(action_chunk)
        tau = torch.rand(
            batch_size,
            1,
            device=action_chunk.device,
            dtype=action_chunk.dtype,
        )
        # 广播 tau，使一个样本内的整个动作块共享同一个流时间。
        tau_chunk = tau.reshape(batch_size, 1, 1)
        interpolated_chunk = tau_chunk * action_chunk + (1.0 - tau_chunk) * noise
        target_velocity = action_chunk - noise
        predicted_velocity = self._predict_velocity(state, interpolated_chunk, tau)
        return F.mse_loss(predicted_velocity, target_velocity)

    def sample_actions(
        self,
        state: torch.Tensor,
        *,
        num_steps: int = 10,
    ) -> torch.Tensor:
        if num_steps <= 0:
            raise ValueError("num_steps must be positive")

        batch_size = state.shape[0]
        action_chunk = torch.randn(
            batch_size,
            self.chunk_size,
            self.action_dim,
            device=state.device,
            dtype=state.dtype,
        )
        step_size = 1.0 / num_steps
        # 左端点 Euler 法：tau 依次为 0, 1/n, ..., (n-1)/n。
        for step in range(num_steps):
            tau = torch.full(
                (batch_size, 1),
                step * step_size,
                device=state.device,
                dtype=state.dtype,
            )
            velocity = self._predict_velocity(state, action_chunk, tau)
            action_chunk = action_chunk + step_size * velocity
        return action_chunk


# 限制命令行可接受的策略名称，也为静态类型检查提供信息。
PolicyType: TypeAlias = Literal["mse", "flow"]


def build_policy(
    policy_type: PolicyType,
    *,
    state_dim: int,
    action_dim: int,
    chunk_size: int,
    hidden_dims: tuple[int, ...] = (128, 128),
) -> BasePolicy:
    """按配置创建策略；这是训练入口选择具体模型的唯一工厂函数。"""
    if policy_type == "mse":
        return MSEPolicy(
            state_dim=state_dim,
            action_dim=action_dim,
            chunk_size=chunk_size,
            hidden_dims=hidden_dims,
        )
    if policy_type == "flow":
        return FlowMatchingPolicy(
            state_dim=state_dim,
            action_dim=action_dim,
            chunk_size=chunk_size,
            hidden_dims=hidden_dims,
        )
    raise ValueError(f"Unknown policy type: {policy_type}")
