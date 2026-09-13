# HW1 代码结构与模块作用

## 1. 项目目标

本项目在 Push-T 环境中训练动作分块（action chunking）模仿学习策略。策略每次观察当前 5 维状态，不只预测一个动作，而是预测连续 `chunk_size` 个二维动作，并在环境中开环执行完整动作块。

需要完成两类策略：

- **MSEPolicy**：一次前向传播直接回归动作块。
- **FlowMatchingPolicy**：学习条件速度场，在推理时用 Euler 积分把高斯噪声逐渐变成动作块。

## 2. 目录结构

```text
hw1/
├── .python-version        # uv 使用的 Python 版本约束
├── pyproject.toml         # 项目元数据及直接依赖
├── uv.lock                # 完整、可复现的依赖锁文件
├── README.md              # 官方环境安装与运行说明
├── CODE_STRUCTURE_ZH.md   # 本文档
└── src/hw1_imitation/
    ├── __init__.py        # Python 包声明
    ├── data.py            # 数据下载、读取、标准化和动作块 Dataset
    ├── model.py           # 策略抽象接口、MSE 与 Flow Matching 模型
    ├── train.py           # 配置、训练入口和统一训练循环
    ├── evaluation.py      # 环境评估、CSV/WandB 日志、视频和 checkpoint
    └── modal_train.py     # 可选的 Modal 云端训练入口
```

首次运行后还会出现：

```text
.venv/                    # uv 创建的项目虚拟环境
data/                     # 自动下载并解压的 Push-T 数据
exp/                      # 每次实验的日志、视频和 WandB 运行文件
```

## 3. 整体调用流程

```text
命令行
  ↓
train.py: parse_train_config()
  ↓
train.py: run_training()
  ├── data.py: 下载/读取专家轨迹、创建 Normalizer 和 Dataset
  ├── model.py: build_policy() 创建 MSE 或 Flow 策略
  ├── 执行 Adam 优化循环并写入训练 loss
  ├── evaluation.py: evaluate_policy() 在 Push-T 中跑 100 个 episode
  │     ├── 调用 model.sample_actions() 取得动作块
  │     ├── 记录 mean reward 与 rollout 视频
  │     └── 保存并上传模型 checkpoint
  └── Logger.dump_for_grading() 整理提交所需 WandB 文件
```

## 4. 各文件详解

### `data.py`

这是数据管线，通常不需要修改。

- `download_pusht()`：按需下载并解压专家数据；已有缓存时直接复用。
- `load_pusht_zarr()`：从 Zarr 读取 `states`、`actions` 和 `episode_ends`。
- `Normalizer`：分别保存状态和动作的逐特征均值、标准差。训练在标准化空间中进行，评估时再反标准化动作。
- `build_valid_indices()`：寻找完整动作块的合法起点，确保一个动作块不会越过 episode 边界。
- `PushtChunkDataset`：把逐时间步轨迹转换为 `(state_t, actions[t:t+K])`。

主要张量形状：

| 数据         | 形状                                     |
| ------------ | ---------------------------------------- |
| 单个状态     | `(state_dim,)`，Push-T 中为 5 维       |
| 单个动作     | `(action_dim,)`，Push-T 中为 2 维      |
| 单个动作块   | `(chunk_size, action_dim)`             |
| batch 状态   | `(batch_size, state_dim)`              |
| batch 动作块 | `(batch_size, chunk_size, action_dim)` |

### `model.py`

这是作业的主要实现位置之一。

- `BasePolicy`：规定所有策略必须支持 `compute_loss()` 和 `sample_actions()`。
- `MSEPolicy`：网络输入当前状态，输出展平后的整个动作块，再 reshape 回三维张量；训练目标是专家动作块的 MSE。
- `FlowMatchingPolicy`：网络以状态、插值动作块 `A_tau` 和时间 `tau` 为条件预测速度；采样时通过多次 Euler 更新生成动作块。
- `build_policy()`：根据 `policy_type` 创建具体模型，让训练和评估逻辑保持通用。

### `train.py`

这是本地训练的主入口，也是另一处主要 TODO。

- `TrainConfig`：集中声明数据目录、策略类型、动作块长度、batch size、学习率、网络结构、训练轮数、评估频率和 WandB 配置。
- `parse_train_config()`：使用 Tyro 自动把 dataclass 字段变为命令行参数。
- `set_seed()`：设置 NumPy 和 PyTorch 随机种子。
- `run_training()`：建立数据管线、模型、实验名、WandB 和本地 Logger，使用 Adam 执行优化，定期记录损失并调用评估，结束时保证至少完成一次评估。
- `main()`：标准命令行入口。

训练循环当前负责：Adam 优化、前向损失、反向传播、定期记录 loss、定期调用 `evaluate_policy()`、评估后恢复训练模式，以及结束时整理评分文件。

### `evaluation.py`

这是固定的评估和实验记录基础设施，通常不修改。

- `Logger.log()`：把普通数值同时写入 `log.csv` 和 WandB；视频等对象只写 WandB。
- `evaluate_policy()`：固定运行 100 个 episode，并用固定 episode seed 提高不同 checkpoint 的可比性。
- 策略每次生成完整动作块，环境连续执行其中动作，动作块耗尽后才重新观察并调用策略。
- 指标 `eval/mean_reward` 是每个 episode 的最大 reward 再取平均。
- 前若干 episode 会被编码为 MP4。
- 每次评估都会保存 `.pkl` checkpoint 并记录 WandB artifact。
- `dump_for_grading()`：结束 WandB run，将完整本地 WandB 目录复制到本次 `exp` 目录。

### `modal_train.py`

这是可选云端入口，不影响本地完成作业。

- 用 `uv.lock` 构建 Modal 容器镜像。
- 上传项目时根据 `.gitignore` 排除不需要的内容。
- 将数据、WandB 文件和 checkpoint 写入持久化 Modal Volume。
- 最终仍调用 `train.py` 的同一个 `run_training()`，因此本地与远端共享训练逻辑。

### `__init__.py`

用于把目录声明为 `hw1_imitation` Python 包。目前不导出额外对象。

## 5. 当前实现状态

以下三个原始作业 TODO 已经补齐：

1. `model.py` 中的 `MSEPolicy`：MLP 动作块回归与 MSE 损失。
2. `model.py` 中的 `FlowMatchingPolicy`：流匹配损失与 Euler 采样。
3. `train.py` 中的主训练循环：Adam、日志、定期及最终评估。

`evaluation.py` 中捕获 `FileNotFoundError` 后的 `pass` 是正常的容错逻辑，不是待实现代码。

## 6. 环境和常用命令

项目要求 Python 3.11 及以上，并通过 `uv.lock` 固定依赖。请始终在本目录运行：

```powershell
uv sync --frozen
uv run src/hw1_imitation/train.py --help
```

在真正训练前还需要登录 WandB：

```powershell
uv run wandb login
```

基础训练命令形式为：

```powershell
uv run src/hw1_imitation/train.py --policy-type mse --exp-name mse
uv run src/hw1_imitation/train.py --policy-type flow --exp-name flow
```

模型和训练循环已经可以运行。正式训练仍需使用足够的 epoch，并根据评估结果选择最佳实验；单轮 smoke run 仅用于验证端到端流程。

## 7. 不应混淆的边界

- 数据是专家离线轨迹；训练阶段不是强化学习在线采样。
- Action chunking 是一次预测多个未来动作，不等同于循环神经网络。
- MSE 策略直接生成动作；Flow Matching 策略必须从噪声迭代积分。
- 模型在标准化动作空间输出；真实环境执行前必须反标准化并裁剪。
- `evaluate_policy()` 会执行昂贵的 100-episode 评估，不适合每个训练 step 调用。
- `logger.dump_for_grading()` 是提交格式要求的一部分，不能删除。
