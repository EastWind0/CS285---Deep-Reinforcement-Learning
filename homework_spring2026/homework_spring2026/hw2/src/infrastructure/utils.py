"""环境交互、轨迹采样与评估指标工具。

策略梯度是 on-policy 算法，每次训练迭代都要调用本模块，用当前 actor 重新采集
完整 episode。轨迹以“字段到 NumPy 数组”的字典表示；上层代码既可保留轨迹
边界计算 return，也可把同名字段拼成一个 batch 进行向量化训练。
"""

from collections import OrderedDict
import numpy as np
import copy
from networks.policies import MLPPolicy
import gym
import cv2
from infrastructure import pytorch_util as ptu
from typing import Dict, Tuple, List

############################################
############################################


def sample_trajectory(
    env: gym.Env, policy: MLPPolicy, max_length: int, render: bool = False
) -> Dict[str, np.ndarray]:
    """让策略在环境中运行，采集一条完整或达到长度上限的轨迹。

    Args:
        env: Gym 环境实例。
        policy: 提供 ``get_action`` 的策略网络。
        max_length: 单条轨迹允许的最大时间步数。
        render: 是否额外保存每一步的 RGB 帧，用于生成视频。

    Returns:
        字典中每个数组的首维都是轨迹长度；``image_obs`` 只在 render=True 时
        包含帧，否则是空数组。
    """
    # reset 获得 episode 的初始 observation；后续始终用最新的 ob 决策。
    ob = env.reset()
    # 分字段暂存每个时间步的数据，episode 结束后统一转为 NumPy 数组。
    obs, acs, rewards, next_obs, terminals, image_obs = [], [], [], [], [], []
    steps = 0
    while True:
        # 可选渲染：MuJoCo 老接口从 sim 渲染，其他 Gym 环境走 env.render。
        if render:
            if hasattr(env, "sim"):
                img = env.sim.render(camera_name="track", height=500, width=500)[::-1]
            else:
                img = env.render(mode="single_rgb_array")
            image_obs.append(
                cv2.resize(img, dsize=(250, 250), interpolation=cv2.INTER_CUBIC)
            )

        # 使用最新 observation 调用策略。get_action 对单个状态返回一个可直接
        # 传给 Gym 的离散标量或连续动作向量。
        ac = policy.get_action(ob)

        # 项目固定 Gym 0.25.2 并使用旧 step API，返回四元组
        # (next_observation, reward, done, info)。
        next_ob, rew, done, info = env.step(ac)

        # rollout 既可能由环境主动终止，也可能达到调用方规定的长度上限。
        # 二者都标成 terminal，确保后续 GAE 不会跨越两条轨迹递推。
        steps += 1
        rollout_done = done or steps >= max_length

        # 记录转移 (s_t, a_t, r_t, s_{t+1}, terminal_t)。这里的 terminal 使用
        # rollout_done，使 GAE 在环境终止和人为长度上限处都不会跨轨迹递推。
        obs.append(ob)
        acs.append(ac)
        rewards.append(rew)
        next_obs.append(next_ob)
        terminals.append(rollout_done)

        ob = next_ob  # 下一轮以 s_{t+1} 作为最新状态。

        # 当前 episode 完成后退出循环并封装整条轨迹。
        if rollout_done:
            break

    return {
        "observation": np.array(obs, dtype=np.float32),
        "image_obs": np.array(image_obs, dtype=np.uint8),
        "reward": np.array(rewards, dtype=np.float32),
        "action": np.array(acs, dtype=np.float32),
        "next_observation": np.array(next_obs, dtype=np.float32),
        "terminal": np.array(terminals, dtype=np.float32),
    }


def sample_trajectories(
    env: gym.Env,
    policy: MLPPolicy,
    min_timesteps_per_batch: int,
    max_length: int,
    render: bool = False,
) -> Tuple[List[Dict[str, np.ndarray]], int]:
    """持续采集完整轨迹，直到累计步数达到 batch 下限。

    最后一条轨迹不会为了精确满足下限而被切断，所以返回步数可能略大于
    ``min_timesteps_per_batch``。
    """
    timesteps_this_batch = 0
    trajs = []
    while timesteps_this_batch < min_timesteps_per_batch:
        # 每次追加一条完整 rollout，并累计它的实际长度。
        traj = sample_trajectory(env, policy, max_length, render)
        trajs.append(traj)

        # 按 reward 数组长度统计这条轨迹贡献的环境时间步数。
        timesteps_this_batch += get_traj_length(traj)
    return trajs, timesteps_this_batch


def sample_n_trajectories(
    env: gym.Env, policy: MLPPolicy, ntraj: int, max_length: int, render: bool = False
):
    """固定采集 ``ntraj`` 条轨迹，主要供视频记录使用。"""
    trajs = []
    for _ in range(ntraj):
        # 此函数只关心轨迹条数，不按累计时间步提前停止。
        traj = sample_trajectory(env, policy, max_length, render)
        trajs.append(traj)
    return trajs


def compute_metrics(trajs, eval_trajs):
    """汇总训练轨迹与评估轨迹的 return、长度统计量。"""

    # 一条轨迹的 return 是其中所有即时奖励之和；这里只用于评估和展示，
    # 不等同于 PGAgent 内按 gamma 计算的训练目标。
    train_returns = [traj["reward"].sum() for traj in trajs]
    eval_returns = [eval_traj["reward"].sum() for eval_traj in eval_trajs]

    # episode 长度有助于判断性能提升是否来自存活更久或更快结束任务。
    train_ep_lens = [len(traj["reward"]) for traj in trajs]
    eval_ep_lens = [len(eval_traj["reward"]) for eval_traj in eval_trajs]

    # 使用 OrderedDict 保持控制台、CSV 和 W&B 中字段顺序稳定。
    logs = OrderedDict()
    logs["Eval_AverageReturn"] = np.mean(eval_returns)
    logs["Eval_StdReturn"] = np.std(eval_returns)
    logs["Eval_MaxReturn"] = np.max(eval_returns)
    logs["Eval_MinReturn"] = np.min(eval_returns)
    logs["Eval_AverageEpLen"] = np.mean(eval_ep_lens)

    logs["Train_AverageReturn"] = np.mean(train_returns)
    logs["Train_StdReturn"] = np.std(train_returns)
    logs["Train_MaxReturn"] = np.max(train_returns)
    logs["Train_MinReturn"] = np.min(train_returns)
    logs["Train_AverageEpLen"] = np.mean(train_ep_lens)

    return logs


def convert_listofrollouts(trajs):
    """把轨迹字典列表整理成训练常用的独立数组。

    observation/action/next_observation/terminal 会沿时间维拼接。奖励同时返回
    拼接版和保留 episode 边界的列表版：前者方便向量化，后者用于逐轨迹计算
    discounted return。
    """
    observations = np.concatenate([traj["observation"] for traj in trajs])
    actions = np.concatenate([traj["action"] for traj in trajs])
    next_observations = np.concatenate([traj["next_observation"] for traj in trajs])
    terminals = np.concatenate([traj["terminal"] for traj in trajs])
    concatenated_rewards = np.concatenate([traj["reward"] for traj in trajs])
    unconcatenated_rewards = [traj["reward"] for traj in trajs]
    return (
        observations,
        actions,
        next_observations,
        terminals,
        concatenated_rewards,
        unconcatenated_rewards,
    )


def get_traj_length(traj):
    """返回一条轨迹包含的环境时间步数。"""
    return len(traj["reward"])
