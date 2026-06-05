# High-Precision Reach With Stability Evaluation

本工程复现“高精度 Reach + 受迫振动稳定性”仿真实验。主线任务是 Franka Panda 末端到达目标点，精度阈值覆盖 50/30/20/10/5 mm；稳定性在末端进入目标点 20 mm 范围后施加周期变力，统计末端加速度 RMS、峰值、峰峰值、jerk、动作变化量和振动指数。IRB120 六自由度环境作为跨构型泛化验证。

## 两份核心文档

| 文档 | 内容 |
| --- | --- |
| [`docs/report.md`](docs/report.md) | 最终实验报告：方法、对比结果、消融、创新点，以及附录 A 的完整对比数据表 |
| [`docs/reproduction.md`](docs/reproduction.md) | 复现流程：环境准备、训练、评估、画图、结果校验 |

想看结论和数据读 `report.md`；想在本地把结果跑出来读 `reproduction.md`。

## 目录速览

| 路径 | 内容 |
| --- | --- |
| `src/rl_reach/` | 算法、环境、评估、绘图、报告生成实现 |
| `scripts/` | 训练 / 评估 / 复现入口脚本 |
| `configs/` | Panda 与 IRB120 的默认运行配置 |
| `docs/` | 上述两份核心文档 |
| `runs/` | 模型权重、结果 CSV/JSON、图（最近一次运行产物） |
| `archive/` | 中期答辩等历史稿件，仅供查阅 |

## 快速开始

```bash
conda activate rl_reach
PYTHONNOUSERSITE=1 python -m pip install -r requirements.txt
PYTHONNOUSERSITE=1 python -m pip install -e .
```

训练本文主方法（固定目标）：

```bash
python -m rl_reach.train --config configs/experiment.yaml --algo TD3_HER_CURRICULUM --fixed-goal --timesteps 220000
```

完整的分步命令、参数说明与结果校验见 [`docs/reproduction.md`](docs/reproduction.md)。

## 方法组合

- Baseline：DDPG、TD3、SAC、TQC
- 经验回放增强：SAC+HER、TD3+HER、TQC+HER（TQC 来自 `sb3-contrib`）
- 本文主方法：TD3+HER+自适应课程学习+稳定性约束+近目标精密伺服

按最终统计口径，完整方法相对普通强化学习基线在 34 项指标中严格领先 32 项、并列最优 2 项、未领先 0 项（明细见 `docs/report.md` §9 与附录 A）。
