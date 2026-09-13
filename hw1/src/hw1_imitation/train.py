"""Push-T 模仿策略的本地训练入口、配置解析和实验生命周期管理。"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import torch
import tyro
import wandb
from torch.utils.data import DataLoader

from hw1_imitation.data import (
    Normalizer,
    PushtChunkDataset,
    download_pusht,
    load_pusht_zarr,
)
from hw1_imitation.evaluation import Logger, evaluate_policy
from hw1_imitation.model import PolicyType, build_policy

# 每次运行会在 exp/ 下创建带随机种子和时间戳的独立实验目录。
LOGDIR_PREFIX = "exp"


@dataclass
class TrainConfig:
    """可由 tyro 自动映射为命令行参数的训练配置。"""
    # The path to download the Push-T dataset to.
    data_dir: Path = Path("data")

    # The policy type -- either MSE or flow.
    policy_type: PolicyType = "mse"
    # The number of denoising steps to use for the flow policy (has no effect for the MSE policy).
    flow_num_steps: int = 10
    # The action chunk size.
    chunk_size: int = 8

    batch_size: int = 128
    lr: float = 3e-4
    weight_decay: float = 0.0
    hidden_dims: tuple[int, ...] = (256, 256, 256)
    # The number of epochs to train for.
    num_epochs: int = 400
    # How often to run evaluation, measured in training steps.
    eval_interval: int = 10_000
    num_video_episodes: int = 5
    video_size: tuple[int, int] = (256, 256)
    # How often to log training metrics, measured in training steps.
    log_interval: int = 100
    # Random seed.
    seed: int = 42
    # WandB project name.
    wandb_project: str = "hw1-imitation"
    # Experiment name suffix for logging and WandB.
    exp_name: str | None = None


def parse_train_config(
    args: list[str] | None = None,
    *,
    defaults: TrainConfig | None = None,
    description: str = "Train a Push-T MLP policy.",
) -> TrainConfig:
    """解析命令行，并允许 Modal 入口传入一套不同的默认值。"""
    defaults = defaults or TrainConfig()
    return tyro.cli(
        TrainConfig,
        args=args,
        default=defaults,
        description=description,
    )


def set_seed(seed: int) -> None:
    """同步 NumPy、CPU PyTorch 和所有 CUDA 设备的随机种子。"""
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def config_to_dict(config: TrainConfig) -> dict[str, Any]:
    """把 dataclass 转为 WandB 可序列化的字典，特别处理 Path。"""
    data = asdict(config)
    for key, value in data.items():
        if isinstance(value, Path):
            data[key] = str(value)
    return data


def run_training(config: TrainConfig) -> None:
    """组装数据、模型和日志器，并执行待补全的核心训练流程。"""
    set_seed(config.seed)
    # 自动优先使用 CUDA；本作业的小型 MLP 在 CPU 上也能较快训练。
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # 首次运行会下载并解压数据；后续运行复用本地缓存。
    zarr_path = download_pusht(config.data_dir)
    states, actions, episode_ends = load_pusht_zarr(zarr_path)
    normalizer = Normalizer.from_data(states, actions)

    # 把逐时间步轨迹转换成 state -> 长度 K 的 action chunk 监督样本。
    dataset = PushtChunkDataset(
        states,
        actions,
        episode_ends,
        chunk_size=config.chunk_size,
        normalizer=normalizer,
    )

    # drop_last=True 保证每个训练 batch 大小一致。
    loader = DataLoader(
        dataset,
        batch_size=config.batch_size,
        shuffle=True,
        drop_last=True,
    )

    # 工厂函数根据 policy_type 创建 MSE 或 Flow Matching 策略。
    model = build_policy(
        config.policy_type,
        state_dim=states.shape[1],
        action_dim=actions.shape[1],
        chunk_size=config.chunk_size,
        hidden_dims=config.hidden_dims,
    ).to(device)

    # 时间戳避免覆盖旧实验，exp_name 后缀便于标记超参数或用途。
    exp_name = f"seed_{config.seed}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    if config.exp_name is not None:
        exp_name += f"_{config.exp_name}"
    log_dir = Path(LOGDIR_PREFIX) / exp_name
    # WandB 保存在线指标；Logger 同时维护提交所需的本地 log.csv。
    wandb.init(
        project=config.wandb_project, config=config_to_dict(config), name=exp_name
    )
    logger = Logger(log_dir)

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=config.lr,
        weight_decay=config.weight_decay,
    )
    global_step = 0
    last_eval_step: int | None = None

    for epoch in range(config.num_epochs):
        model.train()
        for state, action_chunk in loader:
            state = state.to(device)
            action_chunk = action_chunk.to(device)

            optimizer.zero_grad(set_to_none=True)
            loss = model.compute_loss(state, action_chunk)
            loss.backward()
            optimizer.step()
            global_step += 1

            if global_step == 1 or global_step % config.log_interval == 0:
                # 首条记录预留 eval 列，保证提交用 CSV 同时包含 loss 和 reward。
                logger.log(
                    {
                        "train/loss": float(loss.detach().cpu()),
                        "train/epoch": epoch + 1,
                        "eval/mean_reward": float("nan"),
                    },
                    step=global_step,
                )

            if global_step % config.eval_interval == 0:
                evaluate_policy(
                    model=model,
                    normalizer=normalizer,
                    device=device,
                    chunk_size=config.chunk_size,
                    video_size=config.video_size,
                    num_video_episodes=config.num_video_episodes,
                    flow_num_steps=config.flow_num_steps,
                    step=global_step,
                    logger=logger,
                )
                last_eval_step = global_step
                model.train()

    # 即使短测试未达到 eval_interval，也要产生一次评分、视频和 checkpoint。
    if global_step == 0:
        raise RuntimeError("Training produced zero optimizer steps")
    if last_eval_step != global_step:
        evaluate_policy(
            model=model,
            normalizer=normalizer,
            device=device,
            chunk_size=config.chunk_size,
            video_size=config.video_size,
            num_video_episodes=config.num_video_episodes,
            flow_num_steps=config.flow_num_steps,
            step=global_step,
            logger=logger,
        )

    # 必须执行：结束 WandB run，并把完整运行目录复制进 exp 供评分。
    logger.dump_for_grading()


def main() -> None:
    """命令行入口：解析参数后启动一次完整实验。"""
    config = parse_train_config()
    run_training(config)


if __name__ == "__main__":
    main()
