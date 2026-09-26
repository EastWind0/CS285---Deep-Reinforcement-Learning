"""策略网络（actor）。

同一个 MLPPolicy 同时支持离散与连续动作空间：离散策略输出 Categorical 分布的
logits；连续策略输出高斯分布的均值，并额外学习一个与状态无关的 log standard
deviation。``MLPPolicyPG`` 在此基础上实现策略梯度更新。
"""

import itertools
from torch import nn
from torch.nn import functional as F
import torch.distributions as D
from torch import optim

import numpy as np
import torch
from torch import distributions

from infrastructure import pytorch_util as ptu


class MLPPolicy(nn.Module):
    """把 observation 映射成动作概率分布的 MLP 策略基类。

    本类负责网络结构、前向传播与动作采样。不同算法的更新规则不同，因此
    ``update`` 留给子类实现。
    """

    def __init__(
        self,
        ac_dim: int,
        ob_dim: int,
        discrete: bool,
        n_layers: int,
        layer_size: int,
        learning_rate: float,
    ):
        super().__init__()

        if discrete:
            # 离散动作：每个动作对应一个未归一化 logit，之后交给 Categorical。
            self.logits_net = ptu.build_mlp(
                input_size=ob_dim,
                output_size=ac_dim,
                n_layers=n_layers,
                size=layer_size,
            ).to(ptu.device)
            parameters = self.logits_net.parameters()
        else:
            # 连续动作：网络预测正态分布均值，输出维度等于动作维度。
            self.mean_net = ptu.build_mlp(
                input_size=ob_dim,
                output_size=ac_dim,
                n_layers=n_layers,
                size=layer_size,
            ).to(ptu.device)
            # 使用 log 标准差可保证 exp(logstd) 始终为正；设为 Parameter 后会与
            # mean_net 的权重一起被优化。这里的方差不依赖 observation。
            self.logstd = nn.Parameter(
                torch.zeros(ac_dim, dtype=torch.float32, device=ptu.device)
            )
            # chain 让一个 Adam optimizer 同时管理 logstd 和均值网络参数。
            parameters = itertools.chain([self.logstd], self.mean_net.parameters())

        # actor 只创建一个优化器，每个 PG iteration 通常更新一次。
        self.optimizer = optim.Adam(
            parameters,
            learning_rate,
        )

        self.discrete = discrete

    @torch.no_grad()
    def get_action(self, obs: np.ndarray) -> np.ndarray:
        """根据单个 observation 随机采样一个动作并返回 NumPy 数组。

        ``@torch.no_grad`` 表示与环境交互时不建立反向传播计算图；训练所需的
        log-probability 会在 ``update`` 中重新计算。
        """
        # 单个 observation 补上 batch 维，使网络始终接收 (B, ob_dim)。
        if obs.ndim == 1:
            obs = obs[None]
        obs_tensor = ptu.from_numpy(obs)

        # forward 返回动作分布；sample 保留探索随机性。这里只处理一个状态，
        # 因此取第 0 项，离散环境得到 NumPy 标量，连续环境得到动作向量。
        action_distribution = self(obs_tensor)
        action = ptu.to_numpy(action_distribution.sample())[0]

        return action

    def forward(self, obs: torch.FloatTensor):
        """根据一批 observation 构造可微分的动作分布。

        推荐直接返回 ``torch.distributions.Distribution``。这样采样阶段可调用
        ``sample``，训练阶段可调用 ``log_prob``，而分布参数仍连接着网络计算图。
        """
        if self.discrete:
            # logits 不必手工 softmax；Categorical 会在内部完成稳定的归一化。
            logits = self.logits_net(obs)
            return distributions.Categorical(logits=logits)
        else:
            # logstd 是与状态无关的可学习参数；指数变换保证标准差始终为正，
            # 广播后与每个 batch 样本的 mean 共同参数化各动作维的 Normal。
            mean = self.mean_net(obs)
            std = torch.exp(self.logstd)
            return distributions.Normal(mean, std)

    def update(self, obs: np.ndarray, actions: np.ndarray, *args, **kwargs) -> dict:
        """用一个 batch 做一次梯度更新；具体目标函数由子类定义。

        基类只规定接口，不知道该使用策略梯度、行为克隆还是其他算法。
        """
        raise NotImplementedError


class MLPPolicyPG(MLPPolicy):
    """使用 REINFORCE/Policy Gradient 目标更新的策略子类。"""

    def update(
        self,
        obs: np.ndarray,
        actions: np.ndarray,
        advantages: np.ndarray,
    ) -> dict:
        """执行一次策略梯度 actor 更新。

        三个数组的首维都应为当前 batch 的总时间步数。目标是最小化
        ``-mean(log π(a_t|s_t) * advantage_t)``。
        """
        # 统一转成 float32 并移动到 ptu.device，随后才能参与网络运算。
        obs = ptu.from_numpy(obs)
        actions = ptu.from_numpy(actions)
        advantages = ptu.from_numpy(advantages)

        action_distribution = self(obs)
        if self.discrete:
            # 轨迹为了统一存储曾把动作转为 float32，这里恢复为类别索引。
            log_probs = action_distribution.log_prob(actions.long())
        else:
            # 各动作维条件独立，联合动作的 log probability 是各维之和。
            log_probs = action_distribution.log_prob(actions).sum(dim=-1)

        # PyTorch optimizer 执行梯度下降，所以对需要最大化的 PG 目标取负号。
        loss = -(log_probs * advantages).mean()

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        # 返回 Python 标量，供 Logger 同时写入 CSV 和 W&B。
        return {
            "Actor Loss": loss.item(),
        }
