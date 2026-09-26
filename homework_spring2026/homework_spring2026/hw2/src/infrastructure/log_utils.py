"""训练日志、模型持久化与视频记录工具。

标量指标会同时写入本地 ``log.csv`` 和 Weights & Biases；训练结束后还会保存
``flags.json``、``log.pkl`` 与 ``agent.pt``。视频函数负责把多条 rollout 的帧
补齐、拼成网格并包装成 W&B 可上传的对象。
"""

import copy
import json
import os
import pickle
import tempfile
from datetime import datetime

import absl.flags as flags
import ml_collections
import numpy as np
import torch
from torch import nn
import wandb
from PIL import Image, ImageEnhance


class Logger:
    """同时维护本地 CSV 与 W&B 的轻量日志器。"""

    def __init__(self, path):
        # CSV 输出位置；父目录由 run.py 提前创建。
        self.path = path
        # 第一条日志决定 CSV 的列顺序，后续各行沿用相同表头。
        self.header = None
        self.file = None
        # 图片、视频、直方图不适合序列化进 CSV，只上传到 W&B。
        self.disallowed_types = (wandb.Image, wandb.Video, wandb.Histogram)
        # 在内存中保留所有行，训练结束时一并写入 log.pkl。
        self.rows = []

    def log(self, row, step):
        """记录某一训练轮的指标，并立即刷新到磁盘。"""
        # 把迭代编号显式加入日志；W&B 也使用它作为 step。
        row['step'] = step
        if self.file is None:
            # 首次写入时创建文件，并以第一行可序列化字段生成固定表头。
            self.file = open(self.path, 'w')
            if self.header is None:
                self.header = [k for k, v in row.items() if not isinstance(v, self.disallowed_types)]
                self.file.write(','.join(self.header) + '\n')
            filtered_row = {k: v for k, v in row.items() if not isinstance(v, self.disallowed_types)}
            self.file.write(','.join([str(filtered_row.get(k, '')) for k in self.header]) + '\n')
        else:
            # 后续日志严格依照首次生成的表头写列，缺失字段写为空字符串。
            filtered_row = {k: v for k, v in row.items() if not isinstance(v, self.disallowed_types)}
            self.file.write(','.join([str(filtered_row.get(k, '')) for k in self.header]) + '\n')
        # 每轮 flush，避免训练中断时缓冲区中的日志丢失。
        self.file.flush()

        # 本地记录与在线仪表板保持相同的迭代编号。
        wandb.log(row, step=step)
        self.rows.append(copy.deepcopy(row))

    def log_trajs_as_videos(self, trajs, step, max_videos_to_save=2, fps=10, video_title='video'):
        """从若干轨迹中取渲染帧，拼成一个网格视频并上传 W&B。"""
        videos = [traj['image_obs'] for traj in trajs][:max_videos_to_save]
        video = get_wandb_video(videos, fps=fps)
        wandb.log({video_title: video}, step=step)

    def close(self):
        """若 CSV 已打开，则关闭文件句柄。"""
        if self.file is not None:
            self.file.close()


def remove_functions(obj):
    """递归移除配置中的可调用对象，使其可以被 JSON 序列化。"""
    if isinstance(obj, dict):
        return {
            k: remove_functions(v)
            for k, v in obj.items()
            if not callable(v)
        }
    elif isinstance(obj, list):
        return [remove_functions(v) for v in obj if not callable(v)]
    elif callable(obj):
        return None
    else:
        return obj


def dump_log(agent: nn.Module, logger: Logger, args, save_dir: str):
    """在训练结束时保存参数、完整日志、配置和模型权重。

    输出文件分别服务于不同用途：``flags.json`` 便于阅读实验配置，``log.pkl``
    保留 Python 结构化数据，``agent.pt`` 保存 PyTorch ``state_dict``。
    """
    cur_time = datetime.now().strftime('%Y%m%d_%H%M%S')
    config = vars(args)
    config = remove_functions(config)

    # hash 可用于粗略检查日志或配置是否发生变化；它不是加密签名。
    data = {
        'log': logger.rows,
        'log_hash': hash(json.dumps(str(logger.rows), sort_keys=True)),
        'config': config,
        'config_hash': hash(json.dumps(str(config), sort_keys=True)),
        'time': cur_time,
    }

    # 三个文件名是作业提交结构的一部分，不应随意更改。
    with open(os.path.join(save_dir, 'flags.json'), 'w') as f:
        json.dump(config, f)
    with open(os.path.join(save_dir, f'log.pkl'), 'wb') as f:
        pickle.dump(data, f)

    torch.save(agent.state_dict(), os.path.join(save_dir, 'agent.pt'))


def get_flag_dict():
    """将 absl flags 转成普通字典，兼容 ConfigDict 类型。"""
    flag_dict = {k: getattr(flags.FLAGS, k) for k in flags.FLAGS if '.' not in k}
    for k in flag_dict:
        if isinstance(flag_dict[k], ml_collections.ConfigDict):
            flag_dict[k] = flag_dict[k].to_dict()
    return flag_dict


def setup_wandb(
    entity=None,
    project='project',
    group=None,
    name=None,
    mode='online',
    config=None,
):
    """初始化一个 W&B run，并返回其句柄。

    临时目录用于存放 W&B 的运行文件，避免污染作业目录；真正需要提交的本地日志
    仍写入 ``exp/``。若没有联网需求，可以由调用方把 ``mode`` 改为 ``offline``。
    """
    # 允许命令行通过 WANDB_MODE=offline/disabled 覆盖默认在线模式；这样批量
    # 实验不依赖网络，同时不改变普通运行时的默认行为。
    mode = os.environ.get("WANDB_MODE", mode)

    # W&B 的内部缓存放在系统临时目录，不进入 Git 或作业压缩包。
    wandb_output_dir = tempfile.mkdtemp()
    tags = [group] if group is not None else None

    # group 同时用作 tag，方便在 W&B 页面筛选同组实验。
    init_kwargs = dict(
        config=config,
        project=project,
        entity=entity,
        tags=tags,
        group=group,
        dir=wandb_output_dir,
        name=name,
        settings=wandb.Settings(
            start_method='thread',
            _disable_stats=False,
        ),
        mode=mode,
        save_code=True,
    )

    run = wandb.init(**init_kwargs)

    return run


def reshape_video(v, n_cols=None):
    """把多段同长度视频排列成网格，并转换到 W&B 要求的通道顺序。

    输入为 ``(n, t, h, w, c)``（或单段 ``(t, h, w, c)``），输出为
    ``(t, c, n_rows*h, n_cols*w)``。
    """
    if v.ndim == 4:
        # 单段视频补上 video 数量维。
        v = v[None,]

    _, t, h, w, c = v.shape

    if n_cols is None:
        # 取接近平方形的列数，使拼接后的画面不过分狭长。
        n_cols = np.ceil(np.sqrt(v.shape[0])).astype(int)
    if v.shape[0] % n_cols != 0:
        # 视频数无法整除列数时，用全零视频补齐最后一行。
        len_addition = n_cols - v.shape[0] % n_cols
        v = np.concatenate((v, np.zeros(shape=(len_addition, t, h, w, c))), axis=0)
    n_rows = v.shape[0] // n_cols

    # 先排成 (行, 列, 时间, 高, 宽, 通道)，再把行列网格展平成一幅大画面。
    v = np.reshape(v, newshape=(n_rows, n_cols, t, h, w, c))
    v = np.transpose(v, axes=(2, 5, 0, 3, 1, 4))
    v = np.reshape(v, newshape=(t, c, n_rows * h, n_cols * w))

    return v


def get_wandb_video(renders=None, n_cols=None, fps=15):
    """把若干条 rollout 的 RGB 帧转换成一个 W&B 视频对象。

    不同 episode 的长度可能不同。函数用各自最后一帧的暗化版本补齐较短视频，
    再为每格添加黑色边框并调用 :func:`reshape_video` 拼成网格。

    Args:
        renders: 视频列表，每项形状为 ``(t, h, w, c)`` 且类型为 uint8。
        n_cols: 网格列数；为 None 时自动选择接近平方形的布局。
        fps: 输出视频帧率。
    """
    # 以最长轨迹为基准，将所有视频补到相同帧数。
    max_length = max([len(render) for render in renders])
    for i, render in enumerate(renders):
        assert render.dtype == np.uint8

        # 用变暗的最后一帧作 padding，让观众容易看出该 episode 已结束。
        final_frame = render[-1]
        final_image = Image.fromarray(final_frame)
        enhancer = ImageEnhance.Brightness(final_image)
        final_image = enhancer.enhance(0.5)
        final_frame = np.array(final_image)

        pad = np.repeat(final_frame[np.newaxis, ...], max_length - len(render), axis=0)
        renders[i] = np.concatenate([render, pad], axis=0)

        # 每个小视频外加 1 像素黑边，拼接后更容易区分。
        renders[i] = np.pad(renders[i], ((0, 0), (1, 1), (1, 1), (0, 0)), mode='constant', constant_values=0)
    renders = np.array(renders)  # (n, t, h, w, c)

    renders = reshape_video(renders, n_cols)  # (t, c, nr * h, nc * w)

    return wandb.Video(renders, fps=fps, format='mp4')
