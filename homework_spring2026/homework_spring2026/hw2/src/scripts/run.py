"""HW2 本地训练入口。

本脚本解析命令行参数、创建 Gym 环境与 PGAgent，并执行“采样 → 更新 → 评估 →
记录”的 on-policy 训练循环。作业文档中的所有 ``uv run src/scripts/run.py ...``
命令最终都会进入这里。
"""

import argparse
import os
from datetime import datetime
import time

import gym
import numpy as np
import torch
import tqdm

from agents.pg_agent import PGAgent
from infrastructure import utils
from infrastructure import pytorch_util as ptu
from infrastructure.log_utils import setup_wandb, Logger, dump_log

# 每次记录视频时最多并排保存两条评估轨迹，避免日志体积过大。
MAX_NVIDEO = 2


def run_training_loop(logger, args):
    """执行完整策略梯度训练循环，并在结束时保存日志和模型。"""
    # 固定 NumPy/PyTorch 随机源，便于相同 seed 下复现实验。
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    # 必须先选设备再创建网络，因为 build_mlp 会立即把网络移动到 ptu.device。
    ptu.init_gpu(use_gpu=not args.no_gpu, gpu_id=args.which_gpu)

    # 创建指定 Gym 环境；训练默认不实时渲染，以免拖慢采样。
    env = gym.make(args.env_name, render_mode=None)
    # 策略网络需要知道应构造 Categorical 还是 Normal 分布。
    discrete = isinstance(env.action_space, gym.spaces.Discrete)

    # 命令行可覆盖 episode 最大长度，否则使用环境注册时的默认上限。
    max_ep_len = args.ep_len or env.spec.max_episode_steps

    # observation 是向量；离散动作维度为动作类别数，连续动作为向量长度。
    ob_dim = env.observation_space.shape[0]
    ac_dim = env.action_space.n if discrete else env.action_space.shape[0]

    # 根据仿真时间步或环境元数据计算视频帧率，只影响视频播放速度。
    if hasattr(env, "model"):
        fps = 1 / env.dt
    else:
        fps = env.env.metadata["render_fps"]

    # 把全部算法开关和超参数传给智能体；actor/critic 都在其构造函数中创建。
    agent = PGAgent(
        ob_dim,
        ac_dim,
        discrete,
        n_layers=args.n_layers,
        layer_size=args.layer_size,
        gamma=args.discount,
        learning_rate=args.learning_rate,
        use_baseline=args.use_baseline,
        use_reward_to_go=args.use_reward_to_go,
        normalize_advantages=args.normalize_advantages,
        baseline_learning_rate=args.baseline_learning_rate,
        baseline_gradient_steps=args.baseline_gradient_steps,
        gae_lambda=args.gae_lambda,
    )

    # total_envsteps 是作业绘图规定的横轴，累计的是训练采样步数。
    total_envsteps = 0
    start_time = time.time()

    # 每次 iteration 都必须重新用当前策略收集新数据，再做 on-policy 更新。
    for itr in range(args.n_iter):
        print(f"\n********** Iteration {itr} ************")
        trajs, envsteps_this_batch = utils.sample_trajectories(
            env,
            agent.actor,
            args.batch_size,
            max_ep_len,
        )
        total_envsteps += envsteps_this_batch

        # trajs 是“轨迹字典”的列表。这里把它转置成“字段 -> 数组列表”，使
        # PGAgent.update 能在计算 return 时继续识别每条 episode 的边界。
        trajs_dict = {k: [traj[k] for traj in trajs] for k in trajs[0]}

        train_info: dict = agent.update(
            trajs_dict["observation"],
            trajs_dict["action"],
            trajs_dict["reward"],
            trajs_dict["terminal"],
        )

        if itr % args.scalar_log_freq == 0:
            # 评估也通过实际 rollout 完成，但不会执行梯度更新。
            print("\nCollecting data for eval...")
            eval_trajs, eval_envsteps_this_batch = utils.sample_trajectories(
                env, agent.actor, args.eval_batch_size, max_ep_len
            )

            # 同时报告本轮训练 batch 和独立评估 batch 的 return/episode length。
            logs = utils.compute_metrics(trajs, eval_trajs)
            # 合并网络 loss、累计环境步数和墙钟时间。
            logs.update(train_info)
            logs["Train_EnvstepsSoFar"] = total_envsteps
            logs["TimeSinceStart"] = time.time() - start_time
            if itr == 0:
                logs["Initial_DataCollection_AverageReturn"] = logs[
                    "Train_AverageReturn"
                ]

            # 先打印到终端，再由 Logger 写 CSV 并上传 W&B。
            for key, value in logs.items():
                print("{} : {}".format(key, value))
            logger.log(logs, itr)
            print("Done logging...\n\n", flush=True)

        if args.video_log_freq != -1 and itr % args.video_log_freq == 0:
            # 视频 rollout 会保存 RGB 帧，成本明显高于普通评估，所以默认关闭。
            print("\nCollecting video rollouts...")
            eval_video_trajs = utils.sample_n_trajectories(
                env, agent.actor, MAX_NVIDEO, max_ep_len, render=True
            )

            logger.log_trajs_as_videos(
                eval_video_trajs,
                itr,
                fps=fps,
                max_videos_to_save=MAX_NVIDEO,
                video_title="eval_rollouts",
            )

    # 全部迭代结束后写 flags.json、log.pkl 和 agent.pt。
    dump_log(agent, logger, args, args.save_dir)


def setup_arguments(args=None):
    """定义并解析本地与 Modal 共用的命令行参数。"""
    parser = argparse.ArgumentParser()
    # 实验身份与训练总轮数。
    parser.add_argument("--env_name", type=str, default='CartPole-v0')
    parser.add_argument("--exp_name", type=str, default='exp')
    parser.add_argument("--n_iter", "-n", type=int, default=200)

    # 策略梯度变体：reward-to-go、baseline、GAE、优势归一化。
    parser.add_argument("--use_reward_to_go", "-rtg", action="store_true")
    parser.add_argument("--use_baseline", action="store_true")
    parser.add_argument("--baseline_learning_rate", "-blr", type=float, default=5e-3)
    parser.add_argument("--baseline_gradient_steps", "-bgs", type=int, default=5)
    parser.add_argument("--gae_lambda", type=float, default=None)
    parser.add_argument("--normalize_advantages", "-na", action="store_true")
    # batch size 按环境时间步计数，而非 episode 数量。
    parser.add_argument(
        "--batch_size", "-b", type=int, default=1000
    )  # steps collected per train iteration
    parser.add_argument(
        "--eval_batch_size", "-eb", type=int, default=400
    )  # steps collected per eval iteration

    # 优化与 MLP 结构参数。
    parser.add_argument("--discount", type=float, default=1.0)
    parser.add_argument("--learning_rate", "-lr", type=float, default=5e-3)
    parser.add_argument("--n_layers", "-l", type=int, default=2)
    parser.add_argument("--layer_size", "-s", type=int, default=64)

    # 运行控制：episode 上限、随机种子、设备和日志频率。
    parser.add_argument(
        "--ep_len", type=int
    )  # students shouldn't change this away from env's default
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--no_gpu", "-ngpu", action="store_true")
    parser.add_argument("--which_gpu", "-gpu_id", default=0)
    parser.add_argument("--video_log_freq", type=int, default=-1)
    parser.add_argument("--scalar_log_freq", type=int, default=1)

    args = parser.parse_args(args=args)

    return args


def main(args):
    """创建实验日志上下文，然后启动训练。"""
    # 自动评分器要求本地实验统一写入 exp/，不要更改这个目录名。
    logdir_prefix = "exp"  # Keep for autograder

    # 目录名编码环境、实验名、seed 和时间，避免多个 run 相互覆盖。
    exp_name = f"{args.env_name}_{args.exp_name}_sd{args.seed}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"

    # W&B 用相同实验名和完整参数配置，便于在线比较曲线。
    config = vars(args)
    setup_wandb(project='cs285_hw2', name=exp_name, config=config)
    args.save_dir = os.path.join(logdir_prefix, exp_name)
    os.makedirs(args.save_dir, exist_ok=True)
    logger = Logger(os.path.join(args.save_dir, 'log.csv'))

    run_training_loop(logger, args)


if __name__ == "__main__":
    # 只有作为脚本直接执行时才解析真实命令行；Modal 会复用上面的函数。
    args = setup_arguments()
    main(args)
