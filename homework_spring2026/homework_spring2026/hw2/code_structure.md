# HW2 代码结构与运行流程

本文解释 `hw2` 项目的目录结构、各文件职责、核心对象之间的调用关系、数据形状和训练流程。作业的题目要求与实验命令另见 `task_description.md`。

当前版本已经完成全部 starter-code 空缺，并完成四类环境共 19 个本地训练 run。
代码实现说明以本文为准；实验参数、数值结果和曲线见 `report.md`。

## 0. 当前完成状态

| 模块 | 已完成内容 |
| --- | --- |
| 轨迹采样 | 当前策略动作采样、`env.step`、环境/长度终止、轨迹封装 |
| Actor | Categorical/Normal 分布、动作采样、策略梯度 loss 与更新 |
| Critic | 标量状态价值预测、Monte Carlo Q 监督的 MSE 更新 |
| Return | trajectory-centric return 与 reward-to-go |
| Advantage | 无 baseline、`Q - V`、GAE、优势归一化 |
| 训练入口 | on-policy 采样、agent 更新、独立评估与日志落盘 |
| 实验产物 | CartPole、HalfCheetah、LunarLander、InvertedPendulum 曲线与报告 |

## 1. 项目结构

```text
hw2/
├── .python-version          # uv 使用的 Python 主版本：3.10
├── .gitignore               # 不纳入 Git 的环境、日志等文件
├── pyproject.toml           # 项目元数据、Python 依赖、打包和 uv 配置
├── uv.lock                  # uv 解析出的精确依赖版本与文件哈希
├── README.md                # 官方简要安装/运行说明
├── task_description.md      # 作业 PDF 的中文翻译与任务清单
├── code_structure.md        # 本文：源码结构和数据流说明
├── report.md                # 已完成实验的结果、曲线与分析
├── report_assets/           # 报告引用的六张实验对比图
├── exp/                     # 本地训练日志与模型（被 .gitignore 排除）
└── src/
    ├── agents/
    │   ├── __init__.py
    │   └── pg_agent.py      # PG 算法中枢：return、advantage、actor/critic 更新
    ├── networks/
    │   ├── __init__.py
    │   ├── policies.py      # 离散/连续 actor 与策略梯度 loss
    │   └── critics.py       # value baseline 与回归 loss
    ├── infrastructure/
    │   ├── __init__.py
    │   ├── utils.py         # 环境交互、轨迹采样、指标统计
    │   ├── pytorch_util.py  # MLP、设备及 NumPy/Tensor 转换
    │   └── log_utils.py     # CSV、W&B、视频、模型与配置保存
    └── scripts/
        ├── __init__.py
        ├── run.py           # 本地训练入口和命令行参数
        └── modal_run.py     # Modal 云端运行入口
```

## 2. 从命令到训练的一条完整调用链

运行：

```powershell
uv run src/scripts/run.py --env_name CartPole-v0 -n 100 -b 1000 --exp_name cartpole
```

会发生以下调用：

```text
scripts/run.py
│
├─ setup_arguments()
│  └─ 解析环境、batch、网络、baseline、GAE、日志等参数
│
├─ main(args)
│  ├─ setup_wandb()                   infrastructure/log_utils.py
│  ├─ 创建 exp/<实验名>/log.csv
│  └─ run_training_loop(logger, args)
│
└─ run_training_loop()
   ├─ ptu.init_gpu()                  infrastructure/pytorch_util.py
   ├─ gym.make()
   ├─ PGAgent(...)                    agents/pg_agent.py
   │  ├─ MLPPolicyPG(...)             networks/policies.py
   │  └─ ValueCritic(...)（可选）      networks/critics.py
   │
   └─ 每轮 iteration
      ├─ sample_trajectories()        infrastructure/utils.py
      │  └─ sample_trajectory()
      │     ├─ policy.get_action(ob)
      │     └─ env.step(action)
      ├─ PGAgent.update()
      │  ├─ _calculate_q_vals()
      │  ├─ _estimate_advantage()
      │  ├─ actor.update()
      │  └─ critic.update()（可选、多步）
      ├─ 再采一批 eval trajectories
      ├─ compute_metrics()
      └─ Logger.log()

训练结束
└─ dump_log()
   ├─ flags.json
   ├─ log.pkl
   └─ agent.pt
```

最重要的理解是：`run.py` 负责组织过程，`utils.py` 负责获得数据，`PGAgent` 负责把数据变成训练目标，`policies.py` 和 `critics.py` 负责真正的神经网络计算与参数更新。

## 3. 各文件的详细作用

### 3.1 `pyproject.toml`

这是 Python 项目的配置入口：

- `[project]`：项目名、版本、Python 版本要求和依赖。
- `[build-system]`：用 setuptools 把 `src/` 中的包以 editable project 的方式安装。
- `[tool.setuptools]`：声明源码根目录是 `src`，因此代码可以直接写 `from agents...`。
- `[tool.uv]`：本机项目路径较长，启用 uv centralized environment，避免 Windows 安装 PyTorch 时超过路径长度限制。

Windows 使用带 CPython 3.10 预编译 wheel 的 `Box2D 2.3.10`；Linux/Modal 仍使用作业原本的 `box2d-py 2.3.5`。二者都提供 Gym LunarLander 所需的 `Box2D` Python 模块。

### 3.2 `uv.lock`

记录 119 个解析后的包及各平台可用安装文件。不要手工编辑；修改 `pyproject.toml` 后由 `uv sync` 或 `uv lock` 更新。它保证其他机器安装到相同依赖版本。

### 3.3 `src/scripts/run.py`

这是日常使用的主入口，职责包括：

- 解析命令行参数；
- 固定随机种子；
- 选择 CPU/GPU；
- 创建 Gym 环境，读取 observation/action 维度；
- 创建 `PGAgent`；
- 逐轮采样、训练、评估与记录；
- 训练结束时保存模型和日志。

其中 `total_envsteps` 只累加训练采样步数，是作业规定的学习曲线横轴 `Train_EnvstepsSoFar`。eval 采样不加到这个值中。

本文件中的训练循环已经补齐，不再含作业占位逻辑。每轮会：

1. 调用 `sample_trajectories`，用当前 actor 收集至少 `batch_size` 个时间步；
2. 把轨迹列表转置成“字段 → NumPy 数组列表”的 `trajs_dict`；
3. 将 observation、action、reward 和 terminal 交给 `agent.update`；
4. 重新采集独立的 eval trajectories，并记录回报、loss、环境步数和运行时间。

`trajs_dict` 仍保留每条 episode 的数组边界。这一点不能省略：return 必须先逐
episode 计算，随后才能把各轨迹拼成一个网络训练 batch。

### 3.4 `src/scripts/modal_run.py`

这是可选的云端入口，不包含另一套算法。它负责：

- 创建包含依赖的 Debian 容器镜像；
- 安装渲染和 Box2D 编译所需系统库；
- 上传未被 `.gitignore` 排除的项目文件；
- 配置 T4、CPU、内存和一小时 timeout；
- 把 `/root/exp` 挂载成持久化卷；
- 最终仍调用 `run.py` 的 `setup_arguments` 和 `main`。

HW2 模型和环境较小，官方建议优先在本地 CPU 上运行。

### 3.5 `src/infrastructure/utils.py`

#### `sample_trajectory`

与环境交互直至 episode 结束或达到 `max_length`。每一步保存：

```text
(observation, action, reward, next_observation, terminal)
```

如果 `render=True`，还会保存 `image_obs` 供视频日志使用。这是最底层的数据来源；动作必须来自当前 policy，不能混入旧策略数据。

#### `sample_trajectories`

不断采样完整 episode，直至累计时间步不少于指定 batch size。因为不切断最后一条轨迹，真实 batch 可能比 `-b` 稍大。

#### `sample_n_trajectories`

固定采样指定数量的 episode。当前主要用于每次最多两条的视频 rollout。

#### `compute_metrics`

计算训练与评估轨迹的 return 均值、标准差、最大/最小值和平均 episode 长度。这些值只用于监测表现，不参与反向传播。

#### `convert_listofrollouts`

把多个轨迹的同类数据沿时间维拼接，同时保留未拼接的 rewards。后者很重要，因为 return 计算不能跨 episode。

这是 starter code 提供的通用整理函数。当前 `run.py` 没有直接调用它，而是构造
`trajs_dict` 后交给 `PGAgent.update`，再由 agent 在正确的时机完成拼接。两种
写法目的相同；当前写法让“先逐轨迹算 return，后拼平 batch”的顺序更直观。

### 3.6 `src/infrastructure/pytorch_util.py`

#### `build_mlp`

统一创建策略和价值网络使用的 MLP：

```text
input
→ [Linear → Tanh] × n_layers
→ Linear
→ output activation（默认 Identity）
```

#### `init_gpu`

设置模块级全局变量 `device`。若 CUDA 可用且未指定 `-ngpu`，使用指定 GPU；否则使用 CPU。必须先调用它，再创建网络。

#### `from_numpy` / `to_numpy`

- `from_numpy`：NumPy → float32 tensor → 当前 device。
- `to_numpy`：tensor → detach → CPU → NumPy。

### 3.7 `src/infrastructure/log_utils.py`

#### `Logger`

- 将标量写入本地 `log.csv`；
- 同时调用 `wandb.log`；
- 把每一行保存在内存中，训练结束后写入 pickle；
- 图片、视频和直方图只传给 W&B，不写 CSV。

#### `setup_wandb`

创建 W&B run，项目名固定为 `cs285_hw2`。它使用系统临时目录存放 W&B 内部文件，所以不会把大量 W&B 缓存放进作业目录。

默认 mode 是 `online`，但实现会优先读取环境变量 `WANDB_MODE`。批量训练时可
使用 `WANDB_MODE=offline`，这样本地 CSV、模型和曲线仍会正常生成，只是不立即
上传网络；也可设为 `disabled` 完全关闭 W&B 记录。

#### `dump_log`

训练结束时写出：

| 文件           | 内容                                   |
| -------------- | -------------------------------------- |
| `flags.json` | 本次实验的完整参数                     |
| `log.csv`    | 每轮标量指标，适合绘图                 |
| `log.pkl`    | 日志、配置和简单 hash 的 Python 序列化 |
| `agent.pt`   | 整个`PGAgent` 的 `state_dict`      |

#### 视频函数

`get_wandb_video` 会把不同长度的视频补到相同帧数，并用暗化的最终帧表示已经结束；`reshape_video` 再把多段视频拼成网格并转换为 W&B 所需的 `(time, channel, height, width)`。

### 3.8 `src/networks/policies.py`

#### `MLPPolicy`

它对离散和连续动作使用不同参数化：

| 动作空间 | 网络输出                                             | 分布                 | 采样结果                |
| -------- | ---------------------------------------------------- | -------------------- | ----------------------- |
| 离散     | 每个动作的 logits，形状`(B, ac_dim)`               | `Categorical`      | 动作编号，形状`(B,)`  |
| 连续     | 均值`(B, ac_dim)`，另有可学习 `logstd (ac_dim,)` | 独立维度的`Normal` | 动作向量`(B, ac_dim)` |

`forward` 应返回分布对象，而不是直接返回动作。这样：

- 与环境交互时，`get_action` 调用 `distribution.sample()`；
- 训练时，`update` 调用 `distribution.log_prob(actions)`，梯度可以回到网络参数。

#### `MLPPolicyPG`

实现 actor 的核心损失：

```text
Actor Loss = -mean(log π(a_t | s_t) * advantage_t)
```

连续动作的 `Normal.log_prob` 会返回每个动作维度一个值，需要先沿动作维求和，得到联合动作的 log probability。

离散动作在轨迹中统一保存为 `float32`；传入 `Categorical.log_prob` 前会显式
转换为 `long` 类别索引。连续动作保持 `float32` 向量。

### 3.9 `src/networks/critics.py`

`ValueCritic` 是一个 `ob_dim → 1` 的 MLP，估计 `V(s)`。训练目标是 PGAgent 根据真实轨迹算出的 Monte Carlo `q_values`：

```text
Baseline Loss = mean((V(s_t) - q_t)²)
```

底层网络输出 `(B, 1)`，而 `q_values` 是 `(B,)`；`forward` 应明确压掉最后的单例维度，避免 PyTorch 广播造成错误的 `(B, B)` loss。

### 3.10 `src/agents/pg_agent.py`

这是本作业最核心的算法文件。

#### `_discounted_return`

一条轨迹只计算一个折扣总回报，然后复制到每个时间步：

```text
G = r_0 + γr_1 + γ²r_2 + ...
output = [G, G, ..., G]
```

#### `_discounted_reward_to_go`

每个时间步只计算从当前位置开始的回报：

```text
G_t = r_t + γG_{t+1}
```

适合从轨迹尾部向前递推，复杂度为 `O(T)`。

#### `_calculate_q_vals`

根据 `use_reward_to_go` 选择上述方法。输入仍然是按 episode 分开的 rewards 列表，输出也保留相同边界。

#### `_estimate_advantage`

有三条分支：

```text
无 baseline              A_t = Q_t
有 baseline、无 GAE      A_t = Q_t - V(s_t)
有 baseline、有 GAE       A_t = δ_t + γλ(1-d_t)A_{t+1}
```

GAE 的 TD residual 为：

```text
δ_t = r_t + γ(1-d_t)V(s_{t+1}) - V(s_t)
```

`d_t` 是 terminal。它同时截断 next value 和 next advantage，防止拼平后的多个 episode 互相污染。

#### `update`

更新顺序是：

1. 仍按轨迹计算 Q。
2. 把所有字段拼成时间步 batch。
3. 计算 advantage。
4. actor 更新一次。
5. 若存在 critic，则用同一 batch 更新 `baseline_gradient_steps` 次。

## 4. 关键数据结构与形状

设：

- `E`：当前 batch 中 episode 数量；
- `T_i`：第 `i` 条轨迹长度；
- `B = Σ_i T_i`：总时间步数；
- `ob_dim`：状态维度；
- `ac_dim`：动作维度或离散动作类别数。

### 单条轨迹字典

| 字段                 | 离散动作环境                | 连续动作环境      |
| -------------------- | --------------------------- | ----------------- |
| `observation`      | `(T_i, ob_dim)`           | `(T_i, ob_dim)` |
| `action`           | `(T_i,)`                  | `(T_i, ac_dim)` |
| `reward`           | `(T_i,)`                  | `(T_i,)`        |
| `next_observation` | `(T_i, ob_dim)`           | `(T_i, ob_dim)` |
| `terminal`         | `(T_i,)`                  | `(T_i,)`        |
| `image_obs`        | `(T_i, 250, 250, 3)` 或空 | 同左              |

### `PGAgent.update` 入口

```text
obs       = [array(T_1, ob_dim), ..., array(T_E, ob_dim)]
actions   = [array(T_1, ...),    ..., array(T_E, ...)]
rewards   = [array(T_1),         ..., array(T_E)]
terminals = [array(T_1),         ..., array(T_E)]
```

先保留 list-of-arrays 计算 return，再拼成：

```text
obs        (B, ob_dim)
actions    (B,) 或 (B, ac_dim)
rewards    (B,)
terminals  (B,)
q_values   (B,)
advantages (B,)
```

如果误把 episode 提前拼平，discounted return 或 GAE 会跨越轨迹边界，这是本作业最容易出现且不一定立即报错的逻辑 bug。

## 5. 训练与评估的区别

| 项目                            | Train rollout | Eval rollout                  |
| ------------------------------- | ------------- | ----------------------------- |
| 使用的策略                      | 当前 actor    | 当前 actor                    |
| 是否随机采样动作                | 是            | 是，代码没有切换成均值/argmax |
| 是否用于梯度更新                | 是            | 否                            |
| 是否计入`Train_EnvstepsSoFar` | 是            | 否                            |
| 日志前缀                        | `Train_...` | `Eval_...`                  |

训练与评估都从相同的概率策略采样，因此评估曲线本身也会有随机波动。

## 6. 输出目录

每次实验创建：

```text
exp/<env_name>_<exp_name>_sd<seed>_<date>_<time>/
├── log.csv
├── flags.json
├── log.pkl
└── agent.pt
```

其中目录名前缀是自动评分器识别实验的依据。不要手动修改 `run.py` 中的 `logdir_prefix = "exp"`，也不要随意改变作业规定的 `--exp_name`。

`exp/` 已被 `.gitignore` 排除，因为 19 个训练 run 包含重复模型和日志，不适合
直接提交到仓库。需要复现实验时，应使用 `report.md` 中记录的完整命令重新运行。

项目级实验产物为：

```text
report.md
└─ 引用 report_assets/ 中的图片

report_assets/
├── cartpole_small_batch.png
├── cartpole_large_batch.png
├── halfcheetah_eval_return.png
├── halfcheetah_baseline_loss.png
├── lunarlander_gae.png
└── inverted_pendulum.png
```

`report.md` 汇总每组配置的峰值、最终回报、首次达标环境步数、完整命令和实验
结论；`code_structure.md` 只解释代码职责与数据流，避免把结果分析重复维护两份。

## 7. 环境使用方法

依赖已经由 uv 安装。进入本目录后直接运行：

```powershell
uv run python --version
uv run python -c "import torch, gym, Box2D, mujoco; print(torch.__version__)"
```

运行训练仍使用作业原命令：

```powershell
uv run src/scripts/run.py ...
```

通常不需要手工执行 `Activate.ps1`。如果希望 VS Code 使用正确解释器，选择本项目 `.venv` 指向的 Python；uv 的集中式环境功能会维护这个入口。

W&B 登录仅用于在线曲线和视频，不需要 Berkeley 账号。可使用个人 W&B 账号的 API key：

```powershell
uv run wandb login
```

若只需要本地日志，可在当前 PowerShell 会话设置：

```powershell
$env:WANDB_MODE = "offline"
uv run src/scripts/run.py ...
```

## 8. 关键实现约束与易错点

1. **必须保持 on-policy。** Actor 更新后，下一轮必须重新与环境交互，不能把旧
   batch 当作 replay buffer 重复训练 actor。
2. **return 不能跨 episode。** Rewards 要保持 list-of-arrays，先逐轨迹计算 Q，
   再拼成 `(B,)`。
3. **GAE 边界由 terminal 控制。** 环境自然结束和达到 `max_ep_len` 都会记作
   terminal，使 `V(s_{t+1})` 和 `A_{t+1}` 在边界处被截断。
4. **GAE 实际依赖 critic。** 只有启用 `--use_baseline` 时，`--gae_lambda`
   才会进入 GAE 分支；否则 advantage 直接等于 Q。
5. **评估仍然随机。** Eval rollout 也调用 `distribution.sample()`，没有改成
   离散 argmax 或连续均值动作，因此单轮 eval return 会明显波动。
6. **训练步数不含评估。** `Train_EnvstepsSoFar` 只累计训练 rollout，符合题目
   对样本效率横轴的要求。
7. **模型文件只是 state_dict。** 恢复 `agent.pt` 时，需要先根据同目录
   `flags.json` 构造结构一致的 `PGAgent`，再调用 `load_state_dict`。
8. **代码面向 Gym 0.25.2。** `reset()` 返回 observation，`step()` 返回四元组；
   若升级到新版 Gym/Gymnasium，需要同时适配 reset/step API，不能只改依赖版本。

## 9. 验证状态

当前实现已经完成以下本地检查：

- 12 个 `src/*.py` 文件均可编译和导入；
- discounted return、reward-to-go 和跨 episode 的 GAE 结果通过数值检查；
- 离散 actor 能完成一次有限 loss 的反向更新；
- CartPole 短 rollout 能正确产生 terminal；
- 四个 Gym 环境均能创建；
- 四类正式实验均已完成，性能结果见 `report.md`。

## 10. 推荐阅读顺序

1. `scripts/run.py`：先建立对整个训练循环的全局认识。
2. `infrastructure/utils.py`：理解轨迹是怎样产生和存储的。
3. `networks/policies.py`：理解策略分布、动作采样和 actor loss。
4. `agents/pg_agent.py`：理解 return、advantage、GAE 以及总体更新顺序。
5. `networks/critics.py`：加入 value baseline。
6. `infrastructure/log_utils.py`：最后理解结果如何落盘和可视化。
7. `scripts/modal_run.py`：只有需要云端运行时再读。
