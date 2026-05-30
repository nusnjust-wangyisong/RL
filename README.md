# High-Precision Panda Reach With Stability Evaluation

本工程用于复现“高精度 Reach + 受迫振动稳定性”仿真实验。主线任务是 Franka Panda 末端到达目标点，精度阈值覆盖 50/30/20/10/5 mm；稳定性在末端进入目标点 20 mm 范围后施加周期变力，统计末端加速度 RMS、峰值、峰峰值、jerk、动作变化量等指标。

## 方法组合

- Baseline：DDPG、TD3、SAC、TQC
- 经验回放增强：SAC+HER、TD3+HER、TQC+HER
- 本文主方法：TD3+HER+自适应课程学习+稳定性约束+近目标精密伺服
- 扩展：姿态控制可以作为后续实验加入，当前代码主线聚焦 position Reach

TQC 已加入对比，来自 `sb3-contrib`。

## 环境准备

推荐使用 conda 环境 `rl_reach`：

```bash
cd /path/to/RL
conda activate rl_reach
PYTHONNOUSERSITE=1 python -m pip install -r requirements.txt
PYTHONNOUSERSITE=1 python -m pip install -e .
```

## 单个算法训练

固定目标：

```bash
python -m rl_reach.train --config configs/experiment.yaml --algo TD3_HER_CURRICULUM --fixed-goal --timesteps 220000
```

随机目标：

```bash
python -m rl_reach.train --config configs/experiment.yaml --algo TD3_HER_CURRICULUM --random-goal --timesteps 220000
```

算法名可替换为：

```text
DDPG
TD3
SAC
TQC
SAC_HER
TD3_HER
TQC_HER
TD3_HER_CURRICULUM
```

## 稳定性评估

中等接触变力：

```bash
python -m rl_reach.evaluate \
  --config configs/experiment.yaml \
  --algo TD3_HER_CURRICULUM \
  --model runs/models/TD3_HER_CURRICULUM_random.zip \
  --random-goal \
  --disturbance medium \
  --episodes 50
```

输出：

- `runs/results/*_episodes.csv`：每个 episode 的指标
- `runs/results/*_summary.json`：均值和标准差

## 画投影图

```bash
python -m rl_reach.plotting \
  --config configs/experiment.yaml \
  --algo TD3_HER_CURRICULUM \
  --model runs/models/TD3_HER_CURRICULUM_random.zip \
  --random-goal \
  --name td3_her_curriculum_random
```

输出类似参考图的 x-z / y-z 投影，包含 50/30/20/10/5 mm 圆。

## 全套评估

训练好多个模型后：

```bash
PYTHONNOUSERSITE=1 python -m rl_reach.run_suite \
  --config configs/experiment.yaml \
  --model-dir runs/models \
  --episodes 30 \
  --disturbance medium \
  --output suite_summary_medium_servo.csv
```

生成 Markdown 结果表：

```bash
PYTHONNOUSERSITE=1 python -m rl_reach.report \
  --suite-csv runs/results/suite_summary_medium_servo.csv \
  --output runs/results/experiment_report_medium_servo.md
```

当前已生成的最终结果：

- `runs/results/suite_summary_free_servo.csv`
- `runs/results/suite_summary_medium_servo.csv`
- `runs/results/experiment_report_free_servo.md`
- `runs/results/experiment_report_medium_servo.md`
- `runs/results/advantage_report.md`
- `runs/figures/suite_free_servo_metrics.png`
- `runs/figures/suite_medium_servo_metrics.png`
- `runs/figures/td3_her_curriculum_fixed_servo_projection.png`
- `runs/figures/td3_her_curriculum_random_servo_projection.png`

按最终统计口径，完整方法相对普通强化学习基线为：34 项指标中严格领先 32 项，并列最优 2 项，未领先 0 项。

## 指标口径

- 高精度：`final_error_m`、`min_error_m`、`success_50mm`、`success_30mm`、`success_20mm`、`success_10mm`、`success_5mm`、`hold_5mm`
- 高稳定性：`ee_acc_rms`、`ee_acc_peak`、`ee_acc_p2p`、`ee_jerk_rms`、`joint_acc_rms`、`action_delta_mean`、`vibration_index`
- 受迫振动：当末端距离目标小于 `disturbance.activation_radius_m`，施加
  `F(t)=F0+A sin(2 pi f t)+noise`

## 建议论文主线

创新点可以表述为：面向高精度机械臂 Reach 的 TD3-HER 自适应课程强化学习方法，并在近目标接触扰动下引入稳定性约束和跨构型自适应稳定精密伺服，使策略不仅提高 5 mm 到达成功率，还降低受迫振动响应。TQC 作为强基线加入，用于和本文融合策略进行最优 off-policy baseline 对比。
