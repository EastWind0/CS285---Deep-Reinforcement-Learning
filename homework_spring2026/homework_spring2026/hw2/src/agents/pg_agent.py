"""策略梯度智能体。

``PGAgent`` 是作业算法部分的组织中心：它接收若干条完整轨迹，计算 Monte Carlo
Q 值和 advantage，然后更新 actor；启用 baseline 时还会更新 critic；启用
``gae_lambda`` 时则用 GAE 取代简单的 ``Q - V`` 优势估计。
"""

from typing import Optional, Sequence
import numpy as np
import torch

from networks.critics import ValueCritic
from networks.policies import MLPPolicyPG
from infrastructure import pytorch_util as ptu
from torch import nn


class PGAgent(nn.Module):
    """封装 actor、可选 critic 以及策略梯度训练流程。"""

    def __init__(
        self,
        ob_dim: int,
        ac_dim: int,
        discrete: bool,
        n_layers: int,
        layer_size: int,
        gamma: float,
        learning_rate: float,
        use_baseline: bool,
        use_reward_to_go: bool,
        baseline_learning_rate: Optional[float],
        baseline_gradient_steps: Optional[int],
        gae_lambda: Optional[float],
        normalize_advantages: bool,
    ):
        super().__init__()

        # actor 始终存在：输入状态，输出离散或连续动作分布。
        self.actor = MLPPolicyPG(
            ac_dim, ob_dim, discrete, n_layers, layer_size, learning_rate
        )

        # critic 是可选的状态价值基线；不开启时保持为 None。
        if use_baseline:
            self.critic = ValueCritic(
                ob_dim, n_layers, layer_size, baseline_learning_rate
            )
            # 一次 actor 更新后，critic 可连续更新多步以跟上不断变化的回报目标。
            self.baseline_gradient_steps = baseline_gradient_steps
        else:
            self.critic = None

        # gamma 控制远期奖励权重；另外三个开关决定 Q/advantage 的计算方式。
        self.gamma = gamma
        self.use_reward_to_go = use_reward_to_go
        self.gae_lambda = gae_lambda
        self.normalize_advantages = normalize_advantages

    def update(
        self,
        obs: Sequence[np.ndarray],
        actions: Sequence[np.ndarray],
        rewards: Sequence[np.ndarray],
        terminals: Sequence[np.ndarray],
    ) -> dict:
        """使用一批完整轨迹更新 actor，并按需更新 critic。

        每个输入都是“NumPy 数组的列表”，列表中的一个数组对应一条轨迹。这里的
        batch size 指所有轨迹长度之和，而不是轨迹条数。计算 return 时必须保留
        轨迹边界；完成逐轨迹计算后才能拼平并向量化网络更新。
        """

        # 第 1 步：仅根据每条轨迹的奖励序列，为每个 (s_t, a_t) 估计 Q 值。
        q_values: Sequence[np.ndarray] = self._calculate_q_vals(rewards)

        # 沿时间维拼接各条轨迹。注意 Q 必须在此之前逐轨迹计算，否则 return
        # 会错误地跨 episode 累积。
        obs = np.concatenate(obs, axis=0)
        actions = np.concatenate(actions, axis=0)
        rewards = np.concatenate(rewards, axis=0)
        terminals = np.concatenate(terminals, axis=0)
        q_values = np.concatenate(q_values, axis=0)

        # 第 2 步：根据 Q、可选 value baseline 和可选 GAE 计算每个样本的优势。
        advantages: np.ndarray = self._estimate_advantage(
            obs, rewards, q_values, terminals
        )

        # 第 3 步：使用全部 (s_t, a_t, A_t) 对 actor 做一次策略梯度更新。
        info: dict = self.actor.update(obs, actions, advantages)

        # 第 4 步：如果启用了 baseline，以 (s_t, q_t) 为监督数据训练 critic。
        if self.critic is not None:
            # 每一步都使用当前完整 batch；记录最后一步 loss，表示这一轮 critic
            # 更新结束后的拟合状态。
            critic_info = {}
            for _ in range(self.baseline_gradient_steps):
                critic_info = self.critic.update(obs, q_values)

            info.update(critic_info)

        return info

    def _discounted_return(self, rewards: Sequence[float]) -> Sequence[float]:
        """计算整条轨迹的折扣回报，并复制到轨迹的每个时间步。

        输入 ``[r_0, ..., r_T]``；每个输出位置都应等于
        ``sum(gamma**t * r_t, t=0..T)``。这是 trajectory-centric estimator，
        因为求和范围不依赖当前时间步，所以同一轨迹的全部输出完全相同。
        """
        discounted_return = sum(
            (self.gamma ** t) * reward for t, reward in enumerate(rewards)
        )
        return np.full(len(rewards), discounted_return, dtype=np.float32)

    def _discounted_reward_to_go(self, rewards: Sequence[float]) -> Sequence[float]:
        """计算每个时间步各自的 discounted reward-to-go。

        第 ``t`` 项为 ``sum(gamma**(t'-t) * r_t', t'=t..T)``。可从后向前
        使用递推 ``G_t = r_t + gamma * G_{t+1}``，时间复杂度为 O(T)。
        """
        discounted_reward_to_go = np.zeros(len(rewards), dtype=np.float32)
        running_return = 0.0
        for t in reversed(range(len(rewards))):
            running_return = rewards[t] + self.gamma * running_return
            discounted_reward_to_go[t] = running_return
        return discounted_reward_to_go

    def _calculate_q_vals(self, rewards: Sequence[np.ndarray]) -> Sequence[np.ndarray]:
        """逐条轨迹计算 Q 的 Monte Carlo 估计，并保留原轨迹结构。"""

        if not self.use_reward_to_go:
            # 情况 1：忽略当前时间步；同一条轨迹的每个动作都乘整轨迹折扣回报。
            # Q(s_t, a_t) = sum_{t'=0}^T gamma^t' r_{t'}。
            q_values = [self._discounted_return(path_rewards) for path_rewards in rewards]
        else:
            # 情况 2：动作只对当前及之后的奖励负责。
            # Q(s_t, a_t) = sum_{t'=t}^T gamma^(t'-t) r_{t'}。
            q_values = [
                self._discounted_reward_to_go(path_rewards)
                for path_rewards in rewards
            ]

        return q_values

    def _estimate_advantage(
        self,
        obs: np.ndarray,
        rewards: np.ndarray,
        q_values: np.ndarray,
        terminals: np.ndarray,
    ) -> np.ndarray:
        """计算 advantage，并按配置选择 baseline、GAE 和归一化。

        进入本函数时轨迹已拼平；``terminals`` 负责在 GAE 递推中重新标出边界。
        ``q_values``、``rewards``、``terminals`` 都是一维数组。
        """
        if self.critic is None:
            advantages = q_values.copy()
        else:
            # advantage 是训练 actor 的目标，不应反向传播进 critic；转成 NumPy
            # 同时切断计算图。
            with torch.no_grad():
                values = ptu.to_numpy(self.critic(ptu.from_numpy(obs)))
            assert values.shape == q_values.shape

            if self.gae_lambda is None:
                advantages = q_values - values
            else:
                batch_size = obs.shape[0]

                # 在末尾追加一个虚拟 V_{T+1}=0 和 A_{T+1}=0，统一递推边界写法。
                values = np.append(values, [0])
                advantages = np.zeros(batch_size + 1)

                # 从最后一个时间步反向计算：先求 TD residual δ_t，再累积
                # A_t = δ_t + gamma * lambda * (1-d_t) * A_{t+1}。
                for i in reversed(range(batch_size)):
                    nonterminal = 1.0 - terminals[i]
                    delta = (
                        rewards[i]
                        + self.gamma * values[i + 1] * nonterminal
                        - values[i]
                    )
                    advantages[i] = (
                        delta
                        + self.gamma
                        * self.gae_lambda
                        * nonterminal
                        * advantages[i + 1]
                    )

                # 删除仅为简化边界处理而追加的虚拟项。
                advantages = advantages[:-1]

        if self.normalize_advantages:
            advantages = (advantages - advantages.mean()) / (
                advantages.std() + 1e-8
            )

        return advantages
