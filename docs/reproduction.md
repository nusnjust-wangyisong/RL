# 高精度 Reach 与稳定性实验复现说明

本文档用于复现 Franka Panda 高精度 Reach 与接触变力稳定性实验。目标是得到固定目标、随机目标、自由空间和接触变力工况下的完整对比结果，并生成论文/答辩可用的表格与结果图。

## 1. 环境信息

项目路径：

```bash
cd /path/to/RL
```

推荐使用 conda 环境：

```bash
conda activate rl_reach
```

推荐使用如下命令安装依赖和项目包：

```bash
cd /path/to/RL
conda activate rl_reach
PYTHONNOUSERSITE=1 python -m pip install -r requirements.txt
PYTHONNOUSERSITE=1 python -m pip install -e .
```

主要依赖包括：

- `panda-gym`
- `pybullet`
- `stable-baselines3`
- `sb3-contrib`
- `torch`
- `numpy`
- `pandas`
- `matplotlib`

## 2. 项目结构

```text
configs/experiment.yaml        实验配置
src/rl_reach/envs.py           Panda Reach 环境、接触变力扰动、稳定性指标采集
src/rl_reach/train.py          单算法训练入口
src/rl_reach/evaluate.py       单模型评估入口
src/rl_reach/run_suite.py      多算法统一评估入口
src/rl_reach/report.py         Markdown 结果表生成
src/rl_reach/plotting.py       结果图与投影图生成
runs/models/                   已训练模型
runs/results/                  评估结果 CSV/JSON/Markdown
runs/figures/                  结果图
docs/report.md                 实验结果与创新点报告
```

## 3. 实验任务

机械臂选型为 Franka Emika Panda，仿真平台为 PyBullet，任务基于 panda-gym PandaReach 改造。

任务包含两类：

- 固定目标 Reach：目标位置固定，用于验证确定目标下的高精度收敛能力。
- 随机目标 Reach：目标位置在工作空间内随机采样，用于验证策略泛化能力。

精度阈值包含：

```text
50 mm, 30 mm, 20 mm, 10 mm, 5 mm
```

其中 5 mm 是最终高精度目标。

稳定性评价包含两种工况：

- 自由空间：不施加外力扰动。
- 接触变力：当末端进入目标点 20 mm 范围后施加周期变力，模拟磨抛、铣削等接触式加工过程中的受迫振动。

接触变力形式：

```text
F(t) = F0 + A sin(2 pi f t) + epsilon(t)
```

## 4. 训练算法

本项目已包含以下算法：

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

完整方法为：

```text
TD3 + HER + 自适应课程学习 + 稳定性约束 + 近目标精密伺服
```

固定目标训练示例：

```bash
PYTHONNOUSERSITE=1 python -m rl_reach.train \
  --config configs/experiment.yaml \
  --algo TD3_HER_CURRICULUM \
  --fixed-goal \
  --timesteps 100000
```

随机目标训练示例：

```bash
PYTHONNOUSERSITE=1 python -m rl_reach.train \
  --config configs/experiment.yaml \
  --algo TD3_HER_CURRICULUM \
  --random-goal \
  --timesteps 100000
```

批量训练脚本：

```bash
bash scripts/train_all.sh
```

## 5. 单模型评估

接触变力随机目标评估示例：

```bash
PYTHONNOUSERSITE=1 python -m rl_reach.evaluate \
  --config configs/experiment.yaml \
  --algo TD3_HER_CURRICULUM \
  --model runs/models/TD3_HER_CURRICULUM_random.zip \
  --random-goal \
  --disturbance medium \
  --episodes 50
```

输出文件：

```text
runs/results/*_episodes.csv
runs/results/*_summary.json
```

其中 `episodes.csv` 保存每个 episode 的最终误差、成功率、Hold@5mm、末端加速度 RMS、峰峰值、jerk 等指标；`summary.json` 保存均值和标准差。

## 6. 全套对比评估

自由空间评估：

```bash
PYTHONNOUSERSITE=1 python -m rl_reach.run_suite \
  --config configs/experiment.yaml \
  --model-dir runs/models \
  --episodes 30 \
  --disturbance off \
  --output suite_summary_free_servo.csv
```

接触变力评估：

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
  --suite-csv runs/results/suite_summary_free_servo.csv \
  --output runs/results/experiment_report_free_servo.md

PYTHONNOUSERSITE=1 python -m rl_reach.report \
  --suite-csv runs/results/suite_summary_medium_servo.csv \
  --output runs/results/experiment_report_medium_servo.md
```

当前已经生成的核心结果文件：

```text
runs/results/suite_summary_free_servo.csv
runs/results/suite_summary_medium_servo.csv
runs/results/experiment_report_free_servo.md
runs/results/experiment_report_medium_servo.md
runs/results/advantage_report.md
```

## 7. 结果图生成

生成指标对比图：

```bash
PYTHONNOUSERSITE=1 MPLBACKEND=Agg python -m rl_reach.plotting \
  --config configs/experiment.yaml \
  --summary-csv runs/results/suite_summary_free_servo.csv \
  --name suite_free_servo

PYTHONNOUSERSITE=1 MPLBACKEND=Agg python -m rl_reach.plotting \
  --config configs/experiment.yaml \
  --summary-csv runs/results/suite_summary_medium_servo.csv \
  --name suite_medium_servo
```

生成固定目标投影图：

```bash
PYTHONNOUSERSITE=1 MPLBACKEND=Agg python -m rl_reach.plotting \
  --config configs/experiment.yaml \
  --algo TD3_HER_CURRICULUM \
  --model runs/models/TD3_HER_CURRICULUM_fixed.zip \
  --fixed-goal \
  --name td3_her_curriculum_fixed_servo
```

生成随机目标投影图：

```bash
PYTHONNOUSERSITE=1 MPLBACKEND=Agg python -m rl_reach.plotting \
  --config configs/experiment.yaml \
  --algo TD3_HER_CURRICULUM \
  --model runs/models/TD3_HER_CURRICULUM_random.zip \
  --random-goal \
  --name td3_her_curriculum_random_servo
```

当前可直接使用的结果图：

```text
runs/figures/suite_free_servo_metrics.png
runs/figures/suite_medium_servo_metrics.png
runs/figures/td3_her_curriculum_fixed_servo_projection.png
runs/figures/td3_her_curriculum_random_servo_projection.png
```

最终答辩建议延续中期答辩的结果图逻辑，但进一步升级为：

- 误差对比图：突出最终误差从 20 mm 目标提升到 5 mm 目标。
- 多阈值成功率图：展示 50/30/20/10/5 mm 梯度成功率。
- 轨迹投影图：展示末端轨迹、目标点和 50/30/20/10/5 mm 精度圆。
- 稳定性指标图：展示接触变力下 RMS、峰峰值、jerk 和振动指数下降。
- 消融实验图：展示 HER、课程学习、稳定性约束、近目标精密伺服各模块的贡献。

## 8. 评价指标

高精度指标：

- `final_error_m`：最终末端误差。
- `min_error_m`：episode 内最小末端误差。
- `success_50mm`
- `success_30mm`
- `success_20mm`
- `success_10mm`
- `success_5mm`
- `hold_5mm`

稳定性指标：

- `ee_acc_rms`：末端加速度 RMS。
- `ee_acc_peak`：末端加速度峰值。
- `ee_acc_p2p`：末端加速度峰峰值。
- `ee_jerk_rms`：末端 jerk RMS。
- `action_delta_mean`：动作变化量。
- `vibration_index`：末端振动指数。

指标解释：

- RMS 由末端加速度序列计算，用于表示整体振动强度。
- 峰峰值是末端加速度序列的最大值与最小值之差，用于表示振动幅值范围。
- jerk 是加速度对时间的变化率，用于表示加速度变化是否突兀。
- 对 IRB120 等容易出现“未到目标但运动很小”的场景，应优先查看 `qualified_*` 指标；这些指标会对未达到 5 mm 的策略加入定位误差惩罚。

公平性说明：

完整方法包含近目标精密伺服。Panda 使用末端误差反馈进行小幅动作修正；IRB120 使用 IK 将目标位置转换为 6 关节增量。目标位置是 Reach 任务观测中的 `desired_goal`，不是额外泄漏信息。但最终方法应表述为“RL + 精密伺服/IK 混合控制”，不要表述成纯 RL 策略。

## 9. 当前复现结论

当前已复现结果满足以下口径：

```text
自由空间/接触变力 × 固定/随机目标 × 8 个核心误差与稳定性指标
+ 两个固定目标 Hold@5mm
= 34 项指标
```

完整方法相对普通强化学习基线取得：

```text
严格领先 32 项，并列最优 2 项，未领先 0 项。
```

这说明当前工程已经能够支撑最终答辩中“高精度 Reach + 接触变力稳定性优化 + 创新模块有效性”的实验主线。

## 10. IRB120 双环境扩展复现

当前项目已新增 ABB IRB120 六自由度机械臂环境，配置文件为：

```text
configs/experiment_irb120.yaml
```

IRB120 与 Panda 的主要差异是动作空间和观测空间不同：Panda 主环境使用 panda-gym 的末端控制接口，IRB120 使用原生 PyBullet 关节增量控制，动作维度为 6。评估阶段的近目标精密伺服通过 IK 将末端误差映射为 6 维关节动作，避免沿用 Panda 的 7 自由度假设。

已完成的 IRB120 模型和结果位于：

```text
runs/irb120/models/
runs/irb120/results/suite_summary_free_optimized.csv
runs/irb120/results/suite_summary_medium_optimized.csv
runs/irb120/results/irb120_experiment_report_free.md
runs/irb120/results/irb120_experiment_report_medium.md
runs/irb120/figures/irb120_medium_optimized_metrics.png
runs/irb120/figures/irb120_fixed_reach_projection.png
runs/irb120/figures/irb120_random_reach_projection.png
runs/irb120/figures/irb120_fixed_reach_render.png
runs/irb120/figures/irb120_random_reach_render.png
runs/irb120/figures/irb120_reach_projection_panel.png
runs/irb120/figures/irb120_reach_render_panel.png
```

最终答辩主展示图使用 Franka Panda 已收敛模型生成，路径为：

```text
runs/figures/slides/slide_table_final_distance_random_medium.png
runs/figures/slides/slide_xz_projection_random_medium.png
runs/figures/slides/slide_dashboard_random_medium.png
runs/figures/slides/slide_table_final_distance_fixed_medium.png
runs/figures/slides/slide_xz_projection_fixed_medium.png
runs/figures/slides/slide_dashboard_fixed_medium.png
```

重新训练 IRB120 对比模型：

```bash
cd /path/to/RL
conda activate rl_reach
export PYTHONNOUSERSITE=1
export PYTHONPATH=src
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1 TORCH_NUM_THREADS=1

python -m rl_reach.train --config configs/experiment_irb120.yaml --algo DDPG --fixed-goal --timesteps 2500 --run-name DDPG_fixed
python -m rl_reach.train --config configs/experiment_irb120.yaml --algo TD3 --fixed-goal --timesteps 2500 --run-name TD3_fixed
python -m rl_reach.train --config configs/experiment_irb120.yaml --algo SAC --fixed-goal --timesteps 2500 --run-name SAC_fixed
python -m rl_reach.train --config configs/experiment_irb120.yaml --algo TQC --fixed-goal --timesteps 2500 --run-name TQC_fixed
python -m rl_reach.train --config configs/experiment_irb120.yaml --algo TD3_HER --fixed-goal --timesteps 3500 --run-name TD3_HER_fixed
python -m rl_reach.train --config configs/experiment_irb120.yaml --algo TD3_HER_CURRICULUM --fixed-goal --timesteps 4000 --run-name TD3_HER_CURRICULUM_fixed
```

随机目标只需将 `--fixed-goal` 改为 `--random-goal`，并把 `run-name` 中的 `fixed` 改成 `random`。

重新评估 IRB120 自由空间和接触变力：

```bash
python -m rl_reach.run_suite \
  --config configs/experiment_irb120.yaml \
  --model-dir runs/irb120/models \
  --episodes 10 \
  --disturbance off \
  --output suite_summary_free_optimized.csv

python -m rl_reach.run_suite \
  --config configs/experiment_irb120.yaml \
  --model-dir runs/irb120/models \
  --episodes 10 \
  --disturbance medium \
  --output suite_summary_medium_optimized.csv
```

生成 IRB120 报告和图：

```bash
python -m rl_reach.report \
  --suite-csv runs/irb120/results/suite_summary_medium_optimized.csv \
  --output runs/irb120/results/irb120_experiment_report_medium.md

python -m rl_reach.plotting \
  --config configs/experiment_irb120.yaml \
  --summary-csv runs/irb120/results/suite_summary_medium_optimized.csv \
  --name irb120_medium_optimized

python -m rl_reach.plotting \
  --config configs/experiment_irb120.yaml \
  --algo TD3_HER_CURRICULUM \
  --model runs/irb120/models/TD3_HER_CURRICULUM_fixed.zip \
  --fixed-goal \
  --name irb120_fixed_reach \
  --render-snapshot

python -m rl_reach.plotting \
  --config configs/experiment_irb120.yaml \
  --algo TD3_HER_CURRICULUM \
  --model runs/irb120/models/TD3_HER_CURRICULUM_random.zip \
  --random-goal \
  --name irb120_random_reach \
  --render-snapshot

python -m rl_reach.slide_visuals \
  --config configs/experiment.yaml \
  --model-dir runs/models \
  --episodes 30 \
  --projection-episodes 30 \
  --task random \
  --disturbance medium

python -m rl_reach.slide_visuals \
  --config configs/experiment.yaml \
  --model-dir runs/models \
  --episodes 30 \
  --projection-episodes 30 \
  --task fixed \
  --disturbance medium
```

IRB120 扩展图主要展示完整方法自身的跨机械臂泛化效果。为了避免短训普通基线没有到达目标时生成不适合展示的网格图，IRB120 不再使用 `slide_visuals` 生成多算法 slide 网格，而是使用下面命令生成完整方法的固定/随机目标投影与渲染图：

```bash
python -m rl_reach.plotting \
  --config configs/experiment_irb120.yaml \
  --algo TD3_HER_CURRICULUM \
  --model runs/irb120/models/TD3_HER_CURRICULUM_fixed.zip \
  --fixed-goal \
  --name irb120_fixed_reach \
  --render-snapshot

python -m rl_reach.plotting \
  --config configs/experiment_irb120.yaml \
  --algo TD3_HER_CURRICULUM \
  --model runs/irb120/models/TD3_HER_CURRICULUM_random.zip \
  --random-goal \
  --name irb120_random_reach \
  --render-snapshot
```

IRB120 当前复现结论：

```text
优化后的 IK 精密伺服参数：
precision_servo_beta=0.20
precision_servo_max_action=0.16
precision_servo_joint_gain=1.15
precision_servo_ik_iterations=160

完整方法最终误差：
自由空间固定目标 0.176 mm
自由空间随机目标 0.205 mm
接触变力固定目标 0.176 mm
接触变力随机目标 0.205 mm

自由空间和接触变力 × 固定/随机目标 × 15 个精度与精度约束稳定性指标
= 60 项指标

完整方法相对 DDPG、TD3、SAC、TQC、TD3+HER 基线：
严格领先 60 项，并列 0 项，未领先 0 项。
```

注意：IRB120 的原始稳定性指标仍会保留。若某些基线几乎不动，原始加速度可能很小，但最终误差达到 10 cm 以上，不能代表接触加工稳定性。因此新增 `qualified_*` 精度约束稳定性指标：没有达到 5 mm 加工精度的策略会加入误差惩罚，避免把“未到达目标但动作很小”误判为高稳定。
