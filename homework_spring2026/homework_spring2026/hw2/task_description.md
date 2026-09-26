# CS 185/285 HW2：策略梯度（Policy Gradients）

> 本文是 `HW_material/hw2.pdf` 的中文翻译与任务导读，并结合本目录中的 starter code 标出了实际需要完成的位置。它不是答案或实验报告，而是一份实施清单。

## 1. 作业概览

- 课程：Berkeley CS 185/285 Deep Reinforcement Learning, Decision Making, and Control，Spring 2026
- 作业：Assignment 2 — Policy Gradients
- 原文截止时间：2026 年 2 月 25 日 23:59
- 目标：实现最基础的策略梯度算法，并实验比较下列方差缩减方法：
  - reward-to-go（从当前时刻开始的回报）；
  - advantage normalization（优势归一化）；
  - 神经网络 value baseline（价值函数基线）；
  - GAE-λ（Generalized Advantage Estimation，广义优势估计）。
- 计算资源：单次训练通常只需几秒到约 10 分钟。作业建议使用本地笔记本 CPU；这些环境和网络较小，CPU 往往比云端 GPU 更快，也可以把云端额度留给后续作业。
- 主要环境：`CartPole-v0`、`HalfCheetah-v4`、`LunarLander-v2`、`InvertedPendulum-v4`。
- 日志：代码会同时在本地 `exp/` 中保存结果，并通过 Weights & Biases（W&B）记录曲线。

## 2. 最终要完成什么

从实现到提交，可以把作业分成以下四个阶段：

1. 补齐数据采样、策略网络、策略梯度损失、回报计算和训练循环。
2. 在 CartPole 上比较两种回报估计、优势归一化和 batch size。
3. 实现 value baseline 与 GAE，并分别在 HalfCheetah 和 LunarLander 上实验。
4. 在 InvertedPendulum 上调参，提高样本效率；整理代码、日志、曲线和文字分析。

这是一种 **on-policy** 算法：每次更新后策略已经改变，因此下一轮必须用新策略重新采样，不能反复把旧 batch 当作普通监督学习数据使用。

## 3. 理论内容翻译与解释

### 3.1 策略梯度

强化学习的目标是找到参数 `θ`，使策略生成的轨迹具有尽可能高的期望累计奖励：

$$
J(\theta)=\mathbb{E}_{\tau\sim\pi_\theta(\tau)}[r(\tau)].
$$

长度为 `H` 的轨迹记作

$$
\tau=(s_0,a_0,\ldots,s_{H-1},a_{H-1}),
$$

其总奖励为

$$
r(\tau)=\sum_{t=0}^{H-1}r(s_t,a_t).
$$

利用 log-derivative trick，可将目标的梯度写成：

$$
\nabla_\theta J(\theta)
=\mathbb{E}_{\tau\sim\pi_\theta}
[\nabla_\theta\log\pi_\theta(\tau)\,r(\tau)].
$$

用 `N` 条采样轨迹近似这个期望：

$$
\nabla_\theta J(\theta)\approx
\frac{1}{N}\sum_{i=1}^{N}\sum_{t=0}^{H-1}
\nabla_\theta\log\pi_\theta(a_t^i\mid s_t^i)\,r(\tau^i).
$$

实现中通常最小化负号后的损失：

$$
L_{actor}=-\frac{1}{B}\sum_{t=1}^{B}
\log\pi_\theta(a_t\mid s_t)\,\hat A_t.
$$

如果某个动作的优势 `A_t` 为正，梯度更新会提高该动作在该状态下的概率；若为负，则会降低其概率。

### 3.2 Reward-to-go：只让动作负责未来

轨迹早期的动作不可能影响过去已经发生的奖励。利用这一因果性，可以把每个动作的权重从“整条轨迹回报”改为“从当前时刻起的回报”：

$$
\hat Q_t^{RTG}=\sum_{t'=t}^{H-1}\gamma^{t'-t}r_{t'}.
$$

这样去掉了与当前动作无关的过去奖励，通常能降低梯度估计的方差，所以实践中一般优先使用 reward-to-go。

作业要求实现并比较两种估计器：

- trajectory-centric return：

  $$
  \hat Q_t=\sum_{t'=0}^{H-1}\gamma^{t'}r_{t'},
  $$

  同一条轨迹内所有时间步得到同一个值。

- reward-to-go：

  $$
  \hat Q_t=\sum_{t'=t}^{H-1}\gamma^{t'-t}r_{t'}.
  $$

  不同时间步拥有不同的未来回报。

### 3.3 两种折扣方式为何不同

把折扣放在整条轨迹的绝对时间上，会让较晚发生的奖励天然更小，因此策略逐渐不重视轨迹后段的状态。reward-to-go 使用相对于当前时刻的指数 `t'-t`，无论当前处在轨迹的早期还是晚期，都一致地偏好更快到来的奖励。实际算法几乎总使用后一种相对折扣方式。

### 3.4 Value baseline

从回报中减去一个不依赖当前采样动作的 baseline，不会改变策略梯度期望，却能降低方差。本作业用神经网络近似状态价值：

$$
V_\phi^\pi(s_t)\approx
\mathbb{E}_{\pi_\theta}\left[
\sum_{t'=t}^{H-1}\gamma^{t'-t}r_{t'}\mid s_t
\right].
$$

优势估计变成：

$$
\hat A_t=\hat Q_t-V_\phi(s_t).
$$

critic/value 网络用 Monte Carlo 的 `q_values` 作为回归目标，通常使用均方误差：

$$
L_{critic}=\frac{1}{B}\sum_t(V_\phi(s_t)-\hat Q_t)^2.
$$

每次 actor 更新对应多次 critic 更新，次数由 `--baseline_gradient_steps` 控制。

### 3.5 GAE-λ

一步 TD 残差为：

$$
\delta_t=r_t+\gamma V_\phi(s_{t+1})-V_\phi(s_t).
$$

终止时间步没有下一状态的 bootstrap 值，因此：

$$
\delta_{H-1}=r_{H-1}-V_\phi(s_{H-1}).
$$

有限时域下，GAE 可写为 TD 残差的加权和：

$$
\hat A_t^{GAE}=\sum_{t'=t}^{H-1}(\gamma\lambda)^{t'-t}\delta_{t'}.
$$

实际实现无需显式构造所有 n-step return，而应从后向前递推：

$$
\hat A_t=\delta_t+\gamma\lambda(1-d_t)\hat A_{t+1},
$$

其中 `d_t` 表示该时间步是否是轨迹终点。`(1-d_t)` 防止优势值跨越 episode 边界传播。

- `λ=0`：退化为一步 TD 优势，方差低、偏差高，更依赖 value 网络是否准确。
- `λ=1`：在有限轨迹中接近 Monte Carlo 优势，偏差低、方差高。
- 增大 `λ`：通常减小偏差、增大方差；减小 `λ` 则相反。

### 3.6 优势归一化

对一个 batch 内的优势做：

$$
\hat A_t\leftarrow\frac{\hat A_t-\mu_A}{\sigma_A+\epsilon},
\qquad \epsilon\approx10^{-8}.
$$

归一化严格来说会因为依赖当前 batch 而引入偏差；极端情况下 batch size 为 1，归一化后的优势恒为 0。不过在 batch 足够大时，这种偏差通常很小，而且该技巧往往显著改善优化稳定性。

## 4. Starter code 中需要补齐的内容

### 4.1 `src/infrastructure/utils.py`

在 `sample_trajectory` 中完成一条 rollout：

1. 用最新 observation 调用 `policy.get_action(ob)`。
2. 把动作送入 `env.step(ac)`，取得 `next_ob`、`reward`、`done` 和 `info`。
3. 同时考虑环境终止和 `max_length`，计算 `rollout_done`。
4. 正确保存 observation、action、reward、next observation 和 terminal。

`sample_trajectories` 已负责不断采样完整轨迹，直到累计时间步不少于 batch size。注意：由于最后一条轨迹要完整采完，实际时间步数可能略大于指定的 `-b`。

### 4.2 `src/networks/policies.py`

补齐 `MLPPolicy` 与 `MLPPolicyPG`：

- `get_action`：把单个 NumPy observation 转成 tensor，经 `forward` 得到动作分布并采样，再转回 NumPy。
- 离散动作：网络输出 logits，用 `Categorical(logits=...)` 构造分布。
- 连续动作：网络输出均值，`logstd` 是可学习参数；由均值和标准差构造多维正态分布。
- actor 更新：计算被采样动作的 `log_prob`，乘以 advantage，取负平均；然后执行标准的 `zero_grad → backward → step`。
- 连续动作的 `log_prob` 会在动作维度上产生多个值，需要按动作维求和，得到每个样本一个标量 log-probability。

### 4.3 `src/networks/critics.py`

补齐 `ValueCritic`：

- `forward`：输入 observation，输出形状与一维 `q_values` 匹配的 value，通常需要去掉最后一个长度为 1 的维度。
- `update`：以 `q_values` 为监督信号计算 MSE，并完成一次 optimizer step。

### 4.4 `src/agents/pg_agent.py`

需要完成：

- `_discounted_return`：整条轨迹的折扣回报，复制到该轨迹的每个时间步。
- `_discounted_reward_to_go`：为每个时间步计算从该步开始的折扣回报；从后向前递推更高效。
- `_calculate_q_vals`：按 `use_reward_to_go` 选择上述估计器，并逐条轨迹处理。
- `update`：先保留每条轨迹的 rewards 来计算 Q，再把 observations、actions、rewards、terminals 和 q-values 沿时间维拼成一个 batch。
- `_estimate_advantage`：
  - 无 critic：`advantage = q_values`；
  - 有 critic、无 GAE：`advantage = q_values - V(s)`；
  - 有 GAE：计算 TD residual，并从 batch 末尾向前递推，使用 terminal 阻断不同轨迹；
  - 若启用 `-na`，最后做优势归一化。
- actor 每轮更新一次。
- 如果有 critic，则按 `baseline_gradient_steps` 更新多次，并返回最后一次或约定的 critic 日志。

### 4.5 `src/scripts/run.py`

在训练循环中：

1. 用 `utils.sample_trajectories` 收集至少 `args.batch_size` 个训练时间步，并更新 `total_envsteps`。
2. 将轨迹字典整理成 `PGAgent.update` 所需的多个 list-of-arrays。
3. 调用 agent 的 `update` 并取得训练指标。
4. 已有代码会定期收集 eval 轨迹、写本地日志、上传 W&B，并可选记录视频。

## 5. 实验 1：CartPole — 回报估计、归一化与 batch size

需要运行 8 组实验。为便于在 PowerShell 中直接复制，下面都写成单行命令。

### 小 batch：`b=1000`

```powershell
uv run src/scripts/run.py --env_name CartPole-v0 -n 100 -b 1000 --exp_name cartpole
uv run src/scripts/run.py --env_name CartPole-v0 -n 100 -b 1000 -rtg --exp_name cartpole_rtg
uv run src/scripts/run.py --env_name CartPole-v0 -n 100 -b 1000 -na --exp_name cartpole_na
uv run src/scripts/run.py --env_name CartPole-v0 -n 100 -b 1000 -rtg -na --exp_name cartpole_rtg_na
```

### 大 batch：`b=4000`

```powershell
uv run src/scripts/run.py --env_name CartPole-v0 -n 100 -b 4000 --exp_name cartpole_lb
uv run src/scripts/run.py --env_name CartPole-v0 -n 100 -b 4000 -rtg --exp_name cartpole_lb_rtg
uv run src/scripts/run.py --env_name CartPole-v0 -n 100 -b 4000 -na --exp_name cartpole_lb_na
uv run src/scripts/run.py --env_name CartPole-v0 -n 100 -b 4000 -rtg -na --exp_name cartpole_lb_rtg_na
```

参数含义：

- `-n 100`：训练 100 次 policy-gradient iteration。
- `-b`：每轮至少采集的 state-action 数量，而不是 episode 数量。
- `-rtg`：启用 reward-to-go；不写时使用整轨迹回报。
- `-na`：把一个 batch 内的 advantage 归一化为均值 0、标准差 1。
- `--exp_name`：日志目录名的一部分，不要随意更改规定的前缀。

交付要求：

- 提交全部 8 组日志。
- 小 batch 与大 batch 各自表现最好的配置都应收敛到最大 return `200`。
- 画两张图：
  - 四组小 batch 的 `average return vs. environment steps`；
  - 四组大 batch 的同类曲线。
- 横轴必须使用日志中的 `Train_EnvstepsSoFar`，不能使用训练 iteration。
- 简要回答：
  1. 不做优势归一化时，整轨迹估计与 reward-to-go 哪个表现更好？
  2. 为什么实践中通常偏好其中一种估计？
  3. 优势归一化是否有帮助？
  4. batch size 是否造成明显影响？
- 报告实际使用的完整命令，包括所有偏离默认值的参数。

## 6. 实验 2：HalfCheetah — 神经网络 baseline

先运行以下两组：

```powershell
# 无 baseline
uv run src/scripts/run.py --env_name HalfCheetah-v4 -n 100 -b 5000 -eb 3000 -rtg --discount 0.95 -lr 0.01 --exp_name cheetah

# 有 baseline
uv run src/scripts/run.py --env_name HalfCheetah-v4 -n 100 -b 5000 -eb 3000 -rtg --discount 0.95 -lr 0.01 --use_baseline -blr 0.01 -bgs 5 --exp_name cheetah_baseline
```

这里故意没有添加 `-na`，因为优势归一化很强，可能掩盖 baseline 在简单环境上的作用。

交付要求：

- 提交上述两组日志。
- baseline 版本训练结束时平均 return 应超过 `300`。
- 随机种子会造成波动；正确实现偶尔也会失败，但一般不应需要超过 2–3 次尝试。
- 画 baseline loss 学习曲线和 eval return 学习曲线。
- 额外进行一组实验：降低 `-bgs` 或 `-blr`，分析它如何影响：
  1. baseline 的 loss 曲线；
  2. policy 的表现。
- 报告完整命令和修改过的参数。

可选：加入 `-na` 观察提升；同时设置 `--video_log_freq 10`，在 W&B 中观看 HalfCheetah 的运动视频。

## 7. 实验 3：LunarLander — GAE-λ

用同一套超参数搜索：

```text
λ ∈ {0, 0.95, 0.98, 0.99, 1}
```

对每个 `λ` 运行：

```powershell
uv run src/scripts/run.py --env_name LunarLander-v2 --ep_len 1000 --discount 0.99 -n 200 -b 2000 -eb 2000 -l 3 -s 128 -lr 0.001 --use_reward_to_go --use_baseline --gae_lambda <lambda> --exp_name lunar_lander_lambda<lambda>
```

把 `<lambda>` 同时替换成数值，例如 `0.95`。除 `λ` 外，不允许修改 batch size、learning rate 等其他超参数。

交付要求：

- 提交 5 组 λ 的日志。
- 最佳实验在训练过程中至少一次达到 average return `>150`。
- 用一张图比较全部 LunarLander 学习曲线，并文字说明 λ 对表现的影响。
- 用一两句话解释 `λ=0` 和 `λ=1` 各自对应什么，以及这和观察到的任务表现有何关系。
- 报告每组完整命令。

## 8. 实验 4：InvertedPendulum — 超参数与样本效率

先运行默认配置作为基准：

```powershell
uv run src/scripts/run.py --env_name InvertedPendulum-v4 -n 100 -b 5000 -eb 1000 --exp_name pendulum
```

然后调参，目标不是减少 iteration 数，而是减少真实环境交互步数：在 `100,000` 个 environment steps 以内至少一次达到最大 average return `1000`。

可调整：

- discount factor；
- 网络层数与宽度；
- batch size；
- learning rate；
- 是否使用 reward-to-go；
- 是否做优势归一化；
- 是否使用 GAE 以及 λ。

所有实验名都必须以 `pendulum` 开头，以便自动识别。

交付要求：

- 提交性能最好的一组日志；该组须在 100K environment steps 内至少一次达到 average return `1000`。
- 给出最佳超参数和完整命令，简述调参时哪些参数最重要。
- 在同一张图中比较最佳配置与默认配置，横轴使用 environment steps。
- 正式 RL 研究通常应报告多个随机种子的均值和不确定性；本作业为减少计算量，只要求提交最佳 run。

## 9. 命令行参数速查

| 参数 | 默认值 | 含义 |
| --- | ---: | --- |
| `--env_name` | `CartPole-v0` | Gym 环境名 |
| `--exp_name` | `exp` | 实验名，会进入日志目录名 |
| `-n`, `--n_iter` | `200` | policy-gradient 更新轮数 |
| `-b`, `--batch_size` | `1000` | 每轮至少采样的训练时间步数 |
| `-eb`, `--eval_batch_size` | `400` | 每次评估至少采样的时间步数 |
| `-rtg` | 关闭 | 使用 reward-to-go |
| `-na` | 关闭 | 优势归一化 |
| `--use_baseline` | 关闭 | 启用 value baseline |
| `-blr` | `5e-3` | baseline 学习率 |
| `-bgs` | `5` | 每轮 actor 更新对应的 baseline 梯度步数 |
| `--gae_lambda` | `None` | GAE 的 λ；未设置时不用 GAE |
| `--discount` | `1.0` | 折扣因子 γ |
| `-lr` | `5e-3` | actor 学习率 |
| `-l` | `2` | MLP 隐藏层数 |
| `-s` | `64` | 每个隐藏层宽度 |
| `--ep_len` | 环境默认值 | 单个 episode 最大长度 |
| `--seed` | `1` | 随机种子 |
| `-ngpu` | 关闭 | 强制不用 GPU |
| `--video_log_freq` | `-1` | 视频记录频率；`-1` 表示关闭 |
| `--scalar_log_freq` | `1` | 标量日志频率 |

## 10. 曲线与报告内容清单

报告至少应包含：

- CartPole：2 张图（小 batch、大 batch）和 4 个简答问题。
- HalfCheetah：baseline loss 曲线、eval return 曲线，以及降低 `-bgs` 或 `-blr` 后的影响分析。
- LunarLander：5 个 λ 的对比图、λ 影响分析、`λ=0/1` 解释。
- InvertedPendulum：最佳配置与默认配置的 return 对比图、最佳超参数和调参分析。
- 每部分实际运行的完整命令。

绘图时尤其注意：作业多次强调横轴要用环境交互步数，而不是训练轮数。对应的日志字段是：

```text
Train_EnvstepsSoFar
```

## 11. 提交格式

提交物分为两份：

1. 在 Gradescope 的 **HW2 Code** 上传包含代码和实验日志的 `submit.zip`。
2. 在 Gradescope 的 **HW2 Report** 上传实验报告 PDF。

`submit.zip` 解压后，`exp/` 和 `src/` 必须直接位于根目录，不能再套一层父目录；压缩包必须小于 100 MB。根目录同时保留 `pyproject.toml`、`uv.lock` 和 `README.md`。

示意结构：

```text
submit.zip
├── exp/
│   ├── CartPole-v0_cartpole_sd1_<date>_<time>/
│   │   ├── agent.pt
│   │   ├── flags.json
│   │   ├── log.csv
│   │   └── log.pkl
│   ├── CartPole-v0_cartpole_rtg_sd1_<date>_<time>/
│   ├── CartPole-v0_cartpole_na_sd1_<date>_<time>/
│   ├── CartPole-v0_cartpole_rtg_na_sd1_<date>_<time>/
│   ├── CartPole-v0_cartpole_lb_sd1_<date>_<time>/
│   ├── CartPole-v0_cartpole_lb_rtg_sd1_<date>_<time>/
│   ├── CartPole-v0_cartpole_lb_na_sd1_<date>_<time>/
│   ├── CartPole-v0_cartpole_lb_rtg_na_sd1_<date>_<time>/
│   ├── HalfCheetah-v4_cheetah_sd1_<date>_<time>/
│   ├── HalfCheetah-v4_cheetah_baseline_sd1_<date>_<time>/
│   ├── LunarLander-v2_lunar_lander_lambda0_sd1_<date>_<time>/
│   ├── LunarLander-v2_lunar_lander_lambda0.95_sd1_<date>_<time>/
│   ├── LunarLander-v2_lunar_lander_lambda0.98_sd1_<date>_<time>/
│   ├── LunarLander-v2_lunar_lander_lambda0.99_sd1_<date>_<time>/
│   ├── LunarLander-v2_lunar_lander_lambda1_sd1_<date>_<time>/
│   └── InvertedPendulum-v4_pendulum..._sd1_<date>_<time>/
├── src/
│   ├── agents/
│   ├── infrastructure/
│   ├── networks/
│   └── scripts/
├── pyproject.toml
├── uv.lock
└── README.md
```

目录名由 `run.py` 自动生成为：

```text
<env_name>_<exp_name>_sd<seed>_<YYYYMMDD>_<HHMMSS>
```

因此不要手工重命名规定的实验前缀，也不要把多个 run 混到同一目录。

## 12. 推荐执行顺序

1. 先完成 `utils.py` 和 `policies.py`，确认能采样轨迹且 actor loss 可以反向传播。
2. 完成 `pg_agent.py` 中两种 return 和无 baseline 的 advantage。
3. 接通 `run.py`，先用很小的 `-n`、`-b` 做 smoke test。
4. 跑 8 组 CartPole，确认正确性并完成第一组分析。
5. 实现 critic 和 baseline advantage，跑 HalfCheetah。
6. 实现 GAE，跑 5 组 LunarLander。
7. 最后在 InvertedPendulum 上调参，避免在实现尚未验证时浪费时间。
8. 从 `log.csv` 或 W&B 导出曲线，写报告。
9. 按规定目录检查日志与代码，再压缩并确保小于 100 MB。

## 13. 最终自检

- [ ] 所有 `TODO` 都已完成，且没有遗留 `pass`/`None` 占位。
- [ ] 离散与连续策略都能正确采样并计算 log-probability。
- [ ] reward-to-go 的折扣指数从当前时刻重新从 0 开始。
- [ ] GAE 在 terminal 处截断，没有跨 episode 传播。
- [ ] actor 和 critic 的 loss 都是标量，且各自 optimizer 更新正确。
- [ ] CartPole 的 8 组日志齐全，最好配置达到 200。
- [ ] HalfCheetah baseline 版本训练末期平均 return 超过 300。
- [ ] LunarLander 的 5 个 λ 都已运行，最好结果曾超过 150。
- [ ] InvertedPendulum 最佳结果在 100K environment steps 内达到 1000。
- [ ] 所有图的横轴使用 `Train_EnvstepsSoFar`。
- [ ] 报告记录了实际完整命令与要求的文字分析。
- [ ] `submit.zip` 根目录结构正确且小于 100 MB。

