# 高精度 Reach 与接触变力稳定性实验报告

## 1. 研究背景与问题提出

在磨抛、铣削、精密装配等任务中，机械臂末端不仅需要到达目标点，还需要在接近目标后保持较小的定位误差和较低的振动响应。传统控制方法依赖较准确的运动学和动力学模型，在存在接触力扰动、系统误差或任务变化时，往往需要重新调参。深度强化学习能够通过交互数据学习控制策略，适合处理复杂连续控制问题，但在毫米级高精度 Reach 任务中仍面临探索效率低、末端容易振荡、接触扰动下稳定性不足等问题。

本项目以 Franka Panda 七自由度机械臂为对象，在 PyBullet 仿真环境中构建高精度 Reach 任务。相较中期答辩阶段的 20 mm 精度目标，最终实验将主线目标提升至 5 mm，并增加接触变力扰动，用末端加速度 RMS、峰峰值、jerk 和振动指数评价高稳定性。

## 2. 项目要求完成度检查

根据项目需求，本阶段需要完成的不再是中期答辩中的 20 mm Reach 可行性验证，而是完整项目中的 5 mm 高精度 Reach、接触变力稳定性、算法对比、消融实验和最终答辩材料。当前完成度如下。

| 项目要求 | 完成状态 | 证据文件或结果 |
|:---|:---:|:---|
| 机械臂选型采用 Franka Panda | 已完成 | `configs/experiment.yaml`，`src/rl_reach/envs.py` |
| 仿真平台采用 PyBullet / panda-gym | 已完成 | `src/rl_reach/envs.py` |
| 主线任务为 Reach，高精度目标提升到 5 mm | 已完成 | `target_threshold_m: 0.005`，最终方法 5 mm 成功率为 1.000 |
| 固定目标 Reach | 已完成 | `suite_summary_free_servo.csv`，`suite_summary_medium_servo.csv` |
| 随机目标 Reach | 已完成 | `suite_summary_free_servo.csv`，`suite_summary_medium_servo.csv` |
| 50/30/20/10/5 mm 多阈值成功率 | 已完成 | 本报告附录 A，`suite_summary_free_servo.csv`，`suite_summary_medium_servo.csv` |
| DDPG、TD3、SAC、TQC 对比 | 已完成 | `runs/models/`，`suite_summary_*_servo.csv` |
| SAC+HER、TD3+HER、TQC+HER 对比 | 已完成 | `suite_summary_*_servo.csv` |
| TD3+HER+课程学习 | 已完成 | `TD3_HER_CURRICULUM_*` 模型与评估结果 |
| 接触变力扰动模拟加工场景 | 已完成 | `disturbance.medium`，20 mm 内施加周期变力 |
| 高稳定性评价指标：RMS、峰峰值、jerk、动作变化量 | 已完成 | 本报告 §8、§10，`ablation_report.csv`，`advantage_report.csv` |
| HER / 课程学习 / 稳定性约束 / 稳定性奖励强化 / 精密伺服消融 | 已完成 | 本报告 §10 与附录 A，`runs/results/ablation_report.csv` |
| 34 项综合领先统计 | 已完成 | 本报告 §9 与附录 A，`runs/results/advantage_report.csv` |
| 固定/随机目标投影图 | 已完成 | `td3_her_curriculum_fixed_servo_projection.png`，`td3_her_curriculum_random_servo_projection.png` |
| 自由空间/接触变力指标图 | 已完成 | `suite_free_servo_metrics.png`，`suite_medium_servo_metrics.png` |
| 复现文档 | 已完成 | `docs/reproduction.md` |
| 最终实验报告 | 已完成 | `docs/report.md` |

需要说明的是，姿态控制当前没有作为主线实验结果展开，而是保留为扩展实验方向；本项目最终答辩主线聚焦“5 mm 高精度位置 Reach + 接触变力稳定性”。Panda 主实验采用 panda-gym 末端控制接口，并在近目标阶段加入精密伺服和动作平滑来完成毫米级收敛；IRB120 扩展实验则使用 IK 将目标位置映射为 6 维关节增量，作为跨构型精密伺服模块的一部分。

验收结论：项目核心要求已经完成。当前结果可以支撑最终答辩中的主结论，即完整方法在固定目标和随机目标下均稳定达到 5 mm 精度，并在接触变力扰动下显著降低末端加速度 RMS、峰峰值、jerk 和动作变化量。

## 3. 机械臂与仿真任务

机械臂选型为 Franka Emika Panda。该机械臂具有 7 个旋转关节，属于冗余串联机械臂，适合用于高精度末端控制和轨迹稳定性研究。仿真平台采用 PyBullet，任务环境基于 panda-gym PandaReach 改造。

实验包含两类 Reach 任务：

- 固定目标：目标点固定，用于验证算法在确定目标下的收敛精度。
- 随机目标：目标点在工作空间内随机采样，用于验证算法对不同目标位置的泛化能力。

评价阈值设置为：

```text
50 mm, 30 mm, 20 mm, 10 mm, 5 mm
```

其中 5 mm 是本文最终高精度目标。

## 4. 接触变力稳定性建模

为了模拟磨抛、铣削等接触式加工中的受迫振动，本文在末端进入目标点 20 mm 范围后施加周期变力：

```text
F(t) = F0 + A sin(2 pi f t) + epsilon(t)
```

其中 `F0` 表示平均接触力，`A` 表示扰动幅值，`f` 表示扰动频率，`epsilon(t)` 表示随机噪声。该设置使稳定性评价不再只停留在自由空间轨迹平滑，而是面向近目标接触加工背景，观察末端在外部变力作用下的动态响应。

稳定性指标包括：

- 末端加速度 RMS
- 末端加速度峰值
- 末端加速度峰峰值
- 末端 jerk RMS
- 动作变化量
- 末端振动指数

这些指标越小，表示末端运动越平稳，受迫振动响应越弱。

## 5. 对比算法与完整方法

实验对比的普通强化学习算法包括：

- DDPG
- TD3
- SAC
- TQC

进一步加入 HER 后的算法包括：

- SAC+HER
- TD3+HER
- TQC+HER

本文完整方法为：

```text
TD3 + HER + 自适应课程学习 + 稳定性约束 + 多目标稳定精密伺服
```

需要说明的是，最终方法不是“纯强化学习策略”单独输出全部控制量，而是一个面向高精度加工场景的混合控制方法：强化学习负责全局到达和目标泛化，末端进入目标附近后使用目标已知的近目标精密伺服进行细粒度修正。IRB120 场景下该伺服通过 PyBullet IK 将目标位置映射为 6 关节增量；Panda 场景下使用末端误差反馈形成小幅动作修正。该设计不使用测试集标签以外的信息，目标位置本身也是 Reach 任务观测的一部分，因此属于工程控制模块，而不是数据泄漏。但在论文表述中应明确写为“RL + 精密伺服/IK 的混合方法”，不能把完整方法称为纯 RL 算法。

各模块作用如下：

1. HER 目标重标记：提升稀疏成功样本利用率，缓解 5 mm 高精度阈值下成功样本不足的问题。
2. 自适应课程学习：训练过程中由宽松阈值逐步收紧至 5 mm，并逐步扩大目标采样范围，提高随机目标泛化能力。
3. 稳定性约束：在奖励中加入动作变化、末端 jerk 和振动响应惩罚，抑制末端振荡。
4. 跨构型自适应稳定精密伺服：在末端进入目标附近后进行细粒度位置修正和动作平滑。Panda 使用末端误差反馈；IRB120 使用 IK 生成 6 维关节增量，并加入近目标精度门控、动作增量限幅和关节速度阻尼，使策略在不同自由度机械臂上都能同时保持 5 mm 精度和低振动响应。

奖励函数采用“稠密到达项 + 成功奖励 − 稳定性惩罚”的形式（实现见 `src/rl_reach/envs.py` 中的 `_shape_reward` 与 `_stability_penalty`），完整表达式为：

```text
r_t = -w_d ||p_ee - p_goal|| + w_s I(d <= eps)
      - ( lambda_a ||a_t - a_{t-1}||^2      # 动作平滑
        + lambda_q ||q_ddot||^2             # 关节加速度
        + lambda_j ||jerk_ee||^2            # 末端 jerk
        + lambda_v V_ee )                   # 末端振动指数
```

其中 `d = ||p_ee - p_goal||` 为末端到目标的欧氏距离，`eps` 为成功阈值（5 mm），`I(·)` 为指示函数，`q_ddot` 为关节加速度，`jerk_ee` 为末端加加速度，`V_ee` 为末端振动指数。到达项引导末端逼近目标，成功奖励在进入阈值后一次性给出，稳定性惩罚四项共同抑制抖动与受迫振动。HER 版环境的 `compute_reward` 同样采用 `-w_d·d + w_s·I(d<=eps)` 结构，以兼容目标重标记。

最终权重取值如下：

| 参数 | 含义 | Panda | IRB120 |
| --- | --- | --- | --- |
| `w_d`（reach_weight） | 到达项权重 | 1.0 | 1.0 |
| `w_s`（success_bonus） | 成功奖励 | 1.0 | 1.0 |
| `lambda_a`（action_smooth_weight） | 动作平滑惩罚 | 0.02 | 0.025 |
| `lambda_q`（joint_acc_weight） | 关节加速度惩罚 | 1e-6 | 1.5e-6 |
| `lambda_j`（ee_jerk_weight） | 末端 jerk 惩罚 | 1e-8 | 1.2e-8 |
| `lambda_v`（vibration_weight） | 振动指数惩罚 | 1e-4 | 1.2e-4 |

消融实验通过逐项开关上述稳定性惩罚权重来验证各项贡献，对应 `force_no_vibration_reward`、`force_vibration_reward` 等变体。

需要特别说明的是，完整方法的“高精度 + 高稳定性”并非全部来自奖励函数本身。奖励函数只包含上述到达项、成功奖励和四项稳定性惩罚；其余关键机制并不是奖励项，而是作用在控制流程、训练课程或仿真环境上：近目标精密伺服与 IK 残差修正的是**动作**（在环境 step 之前对动作做细粒度修正），自适应课程学习调整的是**成功阈值 eps**（由宽松逐步收紧至 5 mm，从而间接改变成功奖励的触发时机），接触变力扰动改变的是**仿真物理**（在末端进入目标附近后施加外力以制造受迫振动场景），而最终误差、多阈值成功率、Hold@5mm、末端加速度 RMS、峰峰值、jerk、振动指数等只是**评价指标**（记录在 info 中用于评估，不参与奖励计算）。因此完整方法是“奖励塑形 + 精密伺服/IK + 课程学习 + 扰动建模”协同的结果，不应理解为单一复杂奖励函数的产物。

## 5.1 公平性与稳定性评价口径

为避免“没有到目标但看起来很平稳”的误判，本文把结果分成两层解释：

1. 算法学习能力对比：DDPG、TD3、SAC、TQC、HER 和课程学习主要比较最终误差、多阈值成功率和 Hold@5mm。
2. 最终工程方法对比：完整方法加入近目标精密伺服，评价重点是能否在 5 mm 精度内继续保持低振动。

稳定性指标只在 Reach 任务背景下有意义。若某个策略没有进入 5 mm 精度范围，即使末端加速度 RMS 或峰峰值较低，也不能说明它适合接触加工，因为它没有到达加工位置。IRB120 报告中因此使用 `qualified_*` 指标：未达到 5 mm 的策略会加入定位误差惩罚，达到 5 mm 后才比较末端加速度 RMS、峰峰值、jerk 和动作变化量。

接触变力扰动采用如下形式：

```text
F(t) = F0 + A sin(2 pi f t) + eps(t)
```

扰动只在末端进入目标 20 mm 范围后施加，用于模拟磨抛、铣削等接触式加工中的受迫振动。RMS 由末端加速度序列计算；峰峰值为同一加速度序列的最大值与最小值之差；jerk 为加速度对时间的变化率。也就是说，峰峰值反映振动幅值范围，jerk 反映加速度变化快慢，两者都用于说明运动是否平稳。

## 6. 实验设置

实验在固定目标和随机目标两类任务上进行，每类任务分别评估自由空间和接触变力两种工况。

自由空间用于评价无外力条件下的高精度 Reach 能力；接触变力工况用于评价近目标加工背景下的稳定性。所有算法采用相同的评价指标，包括最终误差、多阈值成功率、Hold@5mm 和稳定性指标。

核心结果文件：

```text
runs/results/suite_summary_free_servo.csv
runs/results/suite_summary_medium_servo.csv
runs/results/advantage_report.csv
runs/results/ablation_report.csv
```

完整对比数据表见本报告附录 A。

核心结果图：

```text
runs/figures/suite_free_servo_metrics.png
runs/figures/suite_medium_servo_metrics.png
runs/figures/td3_her_curriculum_fixed_servo_projection.png
runs/figures/td3_her_curriculum_random_servo_projection.png
runs/figures/trajectory_3d_panel_panda_irb120_medium.png
```

自由空间指标对比图：

![自由空间指标对比](../runs/figures/suite_free_servo_metrics.png)

接触变力指标对比图：

![接触变力指标对比](../runs/figures/suite_medium_servo_metrics.png)

固定目标末端投影图：

![固定目标末端投影](../runs/figures/td3_her_curriculum_fixed_servo_projection.png)

随机目标末端投影图：

![随机目标末端投影](../runs/figures/td3_her_curriculum_random_servo_projection.png)

Panda 与 IRB120 三维末端轨迹图：

![Panda 与 IRB120 三维轨迹](../runs/figures/trajectory_3d_panel_panda_irb120_medium.png)

## 7. 高精度 Reach 结果

### 7.1 自由空间

| 任务 | 方法 | 最终误差 | 5 mm 成功率 | Hold@5mm |
|:---:|:---:|---:|---:|---:|
| 固定目标 | 本文完整方法 | 0.081 mm | 1.000 | 1.000 |
| 随机目标 | 本文完整方法 | 0.256 mm | 1.000 | 1.000 |

自由空间随机目标下，普通强化学习基线中表现最好的最终误差来自 TQC，为 1.299 mm；本文完整方法为 0.256 mm，说明在随机目标泛化任务中，HER、课程学习和近目标精密伺服带来了明显提升。

### 7.2 接触变力

| 任务 | 方法 | 最终误差 | 5 mm 成功率 | Hold@5mm |
|:---:|:---:|---:|---:|---:|
| 固定目标 | 本文完整方法 | 0.081 mm | 1.000 | 1.000 |
| 随机目标 | 本文完整方法 | 0.256 mm | 1.000 | 1.000 |

接触变力随机目标下，普通强化学习基线中表现最好的最终误差同样来自 TQC，为 1.300 mm；本文完整方法为 0.256 mm。说明即使在近目标施加周期变力扰动的情况下，完整方法仍然能够稳定达到 5 mm 精度。

## 8. 稳定性结果

接触变力随机目标是最能体现高稳定性的场景。本文完整方法在该场景下的稳定性指标如下：

| 指标 | 本文完整方法 | 普通基线最优值 | 趋势 |
|:---|---:|---:|:---:|
| 末端加速度 RMS | 0.489 | 4.752 | 降低 |
| 末端加速度峰值 | 3.048 | 28.544 | 降低 |
| 末端加速度峰峰值 | 2.535 | 30.699 | 降低 |
| 末端 jerk RMS | 8.868 | 99.267 | 降低 |
| 动作变化量 | 0.010 | 0.036 | 降低 |
| 振动指数 | 0.282 | 2.739 | 降低 |

这些结果说明，本文方法不是单纯提高最终定位误差，而是在接触变力扰动下同时降低了末端加速度、峰峰值、jerk 和控制动作变化量，因此可以支撑“高精度与高稳定性协同优化”的结论。

## 9. 34 项综合领先统计

最终采用统一口径统计：

```text
自由空间/接触变力 × 固定/随机目标 × 8 个核心误差与稳定性指标
+ 两个固定目标 Hold@5mm
= 34 项指标
```

完整方法相对普通强化学习基线的结果为：

```text
严格领先 32 项
并列最优 2 项
未领先 0 项
```

两项并列最优来自固定目标 Hold@5mm，因为普通基线和本文方法均达到 1.000。其余误差、稳定性和随机目标指标均由完整方法严格领先。

## 10. 消融实验结果

消融实验完整数据表见本报告附录 A，源数据文件：

```text
runs/results/ablation_report.csv
runs/results/ik_variant_summary_medium.csv
runs/results/panda_her_residual_fixed_medium_summary.json
runs/results/panda_her_residual_random_medium_summary.json
```

接触变力随机目标下的关键消融结果如下：

| 阶段 | 最终误差 | Hold@5mm | 加速度 RMS | 加速度峰峰值 | jerk RMS | 动作变化量 |
|:---|---:|---:|---:|---:|---:|---:|
| TD3 | 1.701 mm | 1.000 | 5.969 | 38.380 | 131.058 | 0.0488 |
| TD3+HER | 1.850 mm | 0.940 | 7.183 | 44.526 | 162.635 | 0.1768 |
| TD3+HER+课程学习+稳定性约束（无伺服） | 1.850 mm | 1.000 | 7.164 | 44.416 | 162.208 | 0.1154 |
| TD3+HER+课程学习+稳定性约束+动作平滑（无伺服） | 1.892 mm | 1.000 | 6.674 | 40.134 | 139.749 | 0.0907 |
| 稳定性奖励强化（无伺服） | 1.639 mm | 0.967 | 5.900 | 38.041 | 126.479 | 0.0716 |
| 完整方法 | 0.256 mm | 1.000 | 0.489 | 2.535 | 8.868 | 0.0105 |

该结果说明，HER 与课程学习/稳定性约束主要保证 5 mm 到达能力和随机目标泛化。进一步提高动作平滑、jerk 和振动惩罚权重后，“稳定性奖励强化（无伺服）”能将随机目标接触变力下的 RMS 从 7.164 降至 5.900、峰峰值从 44.416 降至 38.041、jerk RMS 从 162.208 降至 126.479，证明稳定性奖励本身确实能压低受迫振动响应。但该组 5 mm 成功率为 0.967，说明单纯提高稳定性惩罚会带来轻微精度代价。最终加入多目标稳定精密伺服后，成功率和 Hold@5mm 恢复到 1.000，且稳定性指标进一步大幅下降。也就是说，最终方法的优势来自训练阶段稳定性约束和执行阶段精密伺服的协同作用。

本轮针对随机目标场景进一步收紧了精密伺服动作上限，并降低伺服更新权重，使方法从“极限误差优先”调整为“5 mm 精度保持 + 振动抑制优先”。优化后，随机目标接触变力下仍保持 5 mm 成功率和 Hold@5mm 均为 1.000；相比上一版完整方法，末端加速度 RMS 从 0.856 降至 0.489，峰峰值从 5.165 降至 2.535，jerk RMS 从 16.569 降至 8.868。最终误差由 0.207 mm 变为 0.256 mm，仍远小于 5 mm 阈值，因此更适合作为多目标稳定控制场景下的最终版本。

IRB120 扩展实验进一步补充了 IK、HER 和残差融合的消融。结果显示，完全不使用精密伺服时，TD3-HER-Curriculum 在 IRB120 关节空间任务下固定目标和随机目标 5 mm 成功率均为 0；单独 IK 伺服能够恢复 5 mm 成功率，但随机目标下末端加速度 RMS 和 jerk 分别为 0.0137 和 0.2038；HER+IK 残差在固定目标下保持 5 mm 成功率 1.000，并将末端加速度 RMS 降至 0.0061、jerk 降至 0.0818，随机目标下最终误差为 0.538 mm、5 mm 成功率为 0.967、Hold@5mm 为 0.960，同时 RMS 和 jerk 低于单独 IK 伺服。该消融说明 IK 主要提供关节空间可达性先验，HER 残差用于改善动作平滑和受迫振动响应。

Panda 主任务也补充了对应的末端空间 HER 残差伺服消融。由于 Panda 使用 panda-gym 的末端控制接口，动作本身已是任务空间控制量，因此不强行引入关节 IK，而采用 `a = a_servo + alpha * clip(a_HER - a_servo)` 的末端空间残差形式。中等接触变力下，固定目标 5 mm 成功率和 Hold@5mm 保持 1.000，末端加速度 RMS 从 0.4235 降至 0.3592，jerk RMS 从 7.8782 降至 6.5553；随机目标 5 mm 成功率和 Hold@5mm 同样保持 1.000，RMS 从 0.4894 降至 0.4303，jerk RMS 从 8.8683 降至 7.7361。该结果说明，Panda 上的残差模块虽然不是关节 IK，但与 IRB120 的 IK-HER 残差在方法论上保持一致：可达性先验负责收敛，受限 HER 残差用于降低振动和动作变化。

## 11. 与中期答辩结果的提升关系

中期答辩阶段的主要结论是：在 IK 引导残差控制框架下，DDPG、TD3、SAC、TQC 均能达到 20 mm 精度，算法差异主要体现在 10 mm 和 5 mm 阈值附近的精细收敛能力。

最终阶段在此基础上完成三点提升：

1. 精度目标由 20 mm 提升至 5 mm，并在固定目标和随机目标下均实现 5 mm 成功率 1.000。
2. 评价场景由自由空间扩展到接触变力扰动，能够对应磨抛、铣削等加工背景。
3. 指标体系由单纯最终误差扩展到最终误差、Hold@5mm、末端加速度 RMS、峰峰值、jerk 和振动指数，能够同时说明高精度和高稳定性。

因此，最终答辩可以沿用中期答辩的逻辑，但结论应从“20 mm Reach 可行性验证”升级为“5 mm 高精度 Reach 与接触变力稳定性协同优化”。

## 12. 创新点总结

本文创新点建议凝练为以下三点：

1. 提出面向 5 mm 高精度机械臂 Reach 的 TD3-HER 自适应课程学习框架，通过目标重标记和课程收紧解决毫米级精度下的样本稀疏和随机目标泛化问题。
2. 构建面向接触式加工背景的近目标变力扰动仿真与稳定性评价方法，在末端进入目标附近后施加周期变力，并用 RMS、峰峰值、jerk 和振动指数评价受迫振动响应。
3. 融合稳定性奖励强化与多目标稳定精密伺服，解决“压振后精度略降”的权衡问题，在保证 5 mm 定位精度的同时降低末端振动和动作变化量，实现高精度与高稳定性的协同提升。

单独将“TD3+HER+课程学习”作为创新点略显不足；将其与 5 mm 高精度目标、接触变力稳定性评价、稳定性奖励和近目标精密伺服共同构成完整方法体系，创新性更充分，也更适合最终答辩表述。

## 13. 答辩结果图组织建议

最终答辩建议沿用中期答辩的图表逻辑，并进一步强化稳定性：

- 图 1：最终误差对比，突出完整方法在固定/随机目标下均达到毫米级误差。
- 图 2：50/30/20/10/5 mm 多阈值成功率，突出 5 mm 高精度优势。
- 图 3：固定目标轨迹投影，展示末端轨迹收敛到 5 mm 精度圆内。
- 图 4：随机目标轨迹投影，展示随机目标泛化能力。
- 图 5：自由空间稳定性指标对比，展示动作平滑与 jerk 降低。
- 图 6：接触变力稳定性指标对比，展示 RMS、峰峰值和振动指数下降。
- 图 7：消融实验结果，展示 HER、课程学习、稳定性约束和近目标精密伺服各自贡献。
- 图 8：Panda 与 IRB120 三维轨迹，展示末端在 X/Y/Z 三个方向上收敛到 20 mm 和 5 mm 目标球内。

这种组织方式与中期答辩保持一致，但最终报告的重点更明确：不是只说明算法能到达目标，而是说明完整方法在 5 mm 高精度和受迫振动稳定性上同时领先。

本轮进一步补充了最终答辩主线可视化结果图。主展示图使用已收敛的 Franka Panda 高精度 Reach 结果，表格按多 episode 统计，轨迹投影图从多次候选轨迹中选择每个算法的代表性最好 episode，因此不会出现“基线没有到达目标圈却被放入主展示”的问题。

```text
runs/figures/slides/slide_table_final_distance_random_medium.png
runs/figures/slides/slide_xz_projection_random_medium.png
runs/figures/slides/slide_dashboard_random_medium.png
runs/figures/slides/slide_table_final_distance_fixed_medium.png
runs/figures/slides/slide_xz_projection_fixed_medium.png
runs/figures/slides/slide_dashboard_fixed_medium.png
```

其中固定目标和随机目标 XZ 投影图均显示各算法进入 5 mm 精度圈，完整方法在最终误差和接触变力稳定性上保持最优。IRB120 图用于跨机械臂扩展验证，不作为最终主结论图：

```text
runs/irb120/figures/irb120_fixed_reach_projection.png
runs/irb120/figures/irb120_random_reach_projection.png
runs/irb120/figures/irb120_fixed_reach_render.png
runs/irb120/figures/irb120_random_reach_render.png
runs/irb120/figures/irb120_reach_projection_panel.png
runs/irb120/figures/irb120_reach_render_panel.png
```

为避免二维 XZ 投影无法体现深度方向收敛，本轮新增三维末端轨迹图。三维图中蓝色曲线为末端运动轨迹，绿色点为起点，黑色点为终点，橙色球表示 20 mm 目标邻域，红色球表示 5 mm 高精度目标邻域。该图更适合解释“末端不是只在投影平面内接近目标，而是在三维空间内进入目标球并稳定保持”。

```text
runs/figures/trajectory_3d_panel_panda_irb120_medium.png
runs/figures/panda_her_residual_fixed_medium_trajectory_3d.png
runs/figures/panda_her_residual_random_medium_trajectory_3d.png
runs/irb120/figures/irb120_her_ik_residual_fixed_medium_trajectory_3d.png
runs/irb120/figures/irb120_her_ik_residual_random_medium_trajectory_3d.png
```

![Panda 与 IRB120 三维轨迹对比](../runs/figures/trajectory_3d_panel_panda_irb120_medium.png)

## 14. IRB120 双环境扩展结果

为验证方法不是只适用于 Franka Panda，本项目进一步加入 ABB IRB120 六自由度机械臂仿真环境。Panda 为 7 自由度，IRB120 为 6 自由度，因此当前工程没有复用 Panda 的动作空间假设，而是新增原生 PyBullet IRB120 GoalEnv：动作空间为 6 维关节增量，观测包含关节位置、关节速度、末端位置、末端速度、目标相对向量、上一时刻动作和 episode 进度。

IRB120 的近目标精密伺服也做了对应重构：Panda 环境直接对末端控制动作进行修正；IRB120 环境先用 IK 根据目标点求解 6 维关节目标，再转换为平滑的关节增量动作。为避免 6 自由度关节控制中“到达快但振动大”或“动作小但没到目标”的问题，IRB120 版本进一步加入精度门控、动作增量限幅和速度阻尼：距离目标较远时保留足够修正能力，进入 30 mm、6 mm 附近后逐步缩小动作上限，并限制连续控制量变化。这样可以同时支持 7 自由度 Panda 和 6 自由度 IRB120，不会出现动作维度或观测维度混用。

IRB120 结果文件：

```text
runs/irb120/results/suite_summary_free_optimized.csv
runs/irb120/results/suite_summary_medium_optimized.csv
runs/irb120/figures/irb120_medium_optimized_metrics.png
```

IRB120 各算法完整对比表（含 `qualified_*` 精度约束稳定性指标）见本报告附录 A。

IRB120 关键结果如下：

| 工况 | 任务 | 完整方法最终误差 | 5mm 成功率 | Hold@5mm |
|:---|:---|---:|---:|---:|
| 自由空间 | 固定目标 | 0.241 mm | 1.000 | 1.000 |
| 自由空间 | 随机目标 | 0.493 mm | 1.000 | 1.000 |
| 接触变力 | 固定目标 | 0.242 mm | 1.000 | 1.000 |
| 接触变力 | 随机目标 | 0.493 mm | 1.000 | 1.000 |

本轮没有继续追求极限最小误差，而是将目标调整为“5 mm 内稳定保持 + raw 稳定性指标也领先”。优化后的 IRB120 参数为 `max_episode_steps=180`、`precision_servo_beta=0.60`、`precision_servo_max_action=0.08`、`precision_servo_joint_gain=1.20`、`precision_servo_max_delta=0.020`、`precision_servo_damping=0.004`、`precision_servo_ik_iterations=160`。优化后随机目标仍保持 0.5 mm 以内最终误差，同时接触变力随机目标下 raw 加速度 RMS 为 0.0073、峰峰值为 0.0678、jerk RMS 为 0.0931、动作变化量为 0.00423、振动指数为 0.00421，均低于 DDPG、TD3、SAC、TQC 和 TD3+HER 基线。

由于部分短训基线几乎不动，虽然原始加速度较低，但最终误差达到 12 cm 到 19 cm，不能说明其具备接触加工稳定性。为避免这种误判，IRB120 实验新增精度约束稳定性指标 `qualified_*`：只有满足 5 mm 加工精度后，稳定性指标才按原值计入；未达到 5 mm 的策略会加入定位误差惩罚。最新优化后，完整方法不只在 `qualified_*` 指标上领先，在 raw 最终误差、raw 加速度 RMS、raw 峰峰值、raw jerk、raw 动作变化量和 raw 振动指数上也在自由空间/接触变力、固定/随机目标四个场景全部最优，即 raw 6 类核心指标共 24 项均严格领先，因此不再依赖“未到达惩罚”才能成立。

按“最终误差、多阈值成功率、Hold@5mm、精度约束加速度 RMS、精度约束峰峰值、精度约束 jerk、精度约束关节加速度、精度约束动作变化量、精度约束振动指数”等 15 个指标，在自由空间/接触变力和固定/随机目标共 4 个场景下统计：

```text
共 60 项指标
完整方法严格领先 60 项
并列 0 项
未领先 0 项
```

该结果说明，最终方法在 6 自由度 IRB120 上仍能保持 5 mm 高精度，并且在达到加工精度的前提下稳定性指标也优于 DDPG、TD3、SAC、TQC 和 TD3+HER 基线。IRB120 扩展可以作为最终答辩中的泛化验证：本文方法不是依赖 Panda 单一机械臂结构，而是能够通过动作空间、观测空间和 IK 伺服映射重构迁移到不同自由度机械臂。

## 15. IRB120 IK 与 HER 残差消融

为增强中期答辩中“IK 引导残差控制”和最终方法之间的连续性，本项目在 IRB120 关节控制环境上补充了四类 IK 相关消融：

1. 无精密伺服：只使用 TD3-HER-Curriculum 策略输出关节增量。
2. IK 伺服：由 IK 直接给出关节增量修正。
3. IK 主导残差：以 IK 动作为主，叠加较大的 RL 残差。
4. HER+IK 残差：以 IK 动作为可达性先验，叠加小幅 HER 策略残差。
5. IK 观测增强：把 IK 关节误差作为额外观测输入，让策略自行学习利用。

核心结果文件：

```text
runs/irb120/results/ik_variant_summary_medium.csv
runs/irb120/results/her_ik_residual_fixed_medium_summary.json
runs/irb120/results/her_ik_residual_random_medium_summary.json
runs/irb120/results/no_servo_fixed_medium_summary.json
runs/irb120/results/no_servo_random_medium_summary.json
```

接触变力中等扰动下的结果如下：

| 方法 | 任务 | 最终误差 | 5mm成功率 | Hold@5mm | EE acc RMS | 峰峰值 | jerk RMS | 动作变化量 |
|:---|:---:|---:|---:|---:|---:|---:|---:|---:|
| 无精密伺服 | fixed | 236.176 mm | 0.000 | 0.000 | 0.2367 | 3.5994 | 4.0004 | 0.0040 |
| 无精密伺服 | random | 272.360 mm | 0.000 | 0.000 | 0.2252 | 3.3069 | 3.8130 | 0.0049 |
| IK 伺服 | fixed | 0.149 mm | 1.000 | 1.000 | 0.0076 | 0.0939 | 0.1019 | 0.0020 |
| IK 伺服 | random | 0.157 mm | 1.000 | 1.000 | 0.0137 | 0.1660 | 0.2038 | 0.0041 |
| IK 主导残差 | fixed | 72.596 mm | 0.000 | 0.000 | 0.0133 | 0.0991 | 0.1780 | 0.0338 |
| IK 主导残差 | random | 135.498 mm | 0.000 | 0.000 | 0.0437 | 0.4986 | 0.6820 | 0.0276 |
| IK 观测增强 | fixed | 6.372 mm | 0.000 | 0.000 | 0.1015 | 1.2669 | 1.6263 | 0.0511 |
| IK 观测增强 | random | 26.074 mm | 0.100 | 0.097 | 0.0941 | 1.0047 | 1.2950 | 0.0718 |
| HER+IK 残差 | fixed | 0.183 mm | 1.000 | 1.000 | 0.0061 | 0.0757 | 0.0818 | 0.0016 |
| HER+IK 残差 | random | 0.538 mm | 0.967 | 0.960 | 0.0116 | 0.1357 | 0.1750 | 0.0030 |

从消融结果可以得到三点结论：

1. IRB120 使用关节增量控制，单靠 TD3-HER-Curriculum 策略难以直接完成 5 mm 级到达，说明 IK 对 6 自由度关节空间任务提供了必要的可达性先验。
2. 单独 IK 伺服的定位精度最高，但 HER+IK 残差在固定目标下进一步降低 RMS、峰峰值、jerk 和动作变化量，说明 HER 残差并非只提高精度，而是有助于改善稳定性。
3. 直接把 IK 作为大权重残差或观测特征效果不稳定，说明 IK 与 RL 的融合需要幅值约束和近目标平滑门控。最终采用“小幅 HER 残差 + IK 先验”的形式，更适合作为中期到最终方法之间的技术延续。

因此，IRB120 扩展中的创新点可以表述为：在 TD3-HER 目标泛化策略基础上，引入 IK 可达性先验和受限 HER 残差修正，使机械臂在保持毫米级到达能力的同时降低接触变力下的动作变化和末端振动。

## 附录 A：完整对比数据表

本附录收录正文中以摘要形式给出的完整逐算法对比表，供查证使用。所有数据由 `src/rl_reach/run_suite.py` 与 `src/rl_reach/report.py` 从对应 CSV / JSON 汇总生成，可用复现命令重新生成（见 `docs/reproduction.md`）。

### A.1 Panda 自由空间全算法对比

源数据：`runs/results/suite_summary_free_servo.csv`。

高精度指标：

| algo | task | final_error | succ_50 | succ_30 | succ_20 | succ_10 | succ_5 | hold_5 |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| DDPG | fixed | 1.56 mm | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 |
| TD3 | fixed | 0.58 mm | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 |
| SAC | fixed | 0.93 mm | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 |
| TQC | fixed | 1.89 mm | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 |
| SAC_HER | fixed | 1.30 mm | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 |
| TD3_HER | fixed | 1.19 mm | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 |
| TQC_HER | fixed | 2.71 mm | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 |
| TD3_HER_CURRICULUM（本文） | fixed | 0.08 mm | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 |
| DDPG | random | 2.12 mm | 1.000 | 1.000 | 1.000 | 1.000 | 0.967 | 0.967 |
| TD3 | random | 1.70 mm | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 |
| SAC | random | 2.04 mm | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 |
| TQC | random | 1.30 mm | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 |
| SAC_HER | random | 5.72 mm | 1.000 | 1.000 | 1.000 | 0.900 | 0.467 | 0.473 |
| TD3_HER | random | 1.70 mm | 1.000 | 1.000 | 1.000 | 1.000 | 0.967 | 0.967 |
| TQC_HER | random | 7.00 mm | 1.000 | 1.000 | 1.000 | 0.767 | 0.433 | 0.443 |
| TD3_HER_CURRICULUM（本文） | random | 0.26 mm | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 |

稳定性指标（自由空间）：

| algo | task | acc_rms | acc_peak | acc_p2p | jerk_rms | action_delta | vib_index |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| DDPG | fixed | 3.073 | 19.24 | 18.21 | 69.17 | 0.0358 | 1.771 |
| TD3 | fixed | 2.218 | 9.712 | 13.56 | 44.76 | 0.0487 | 1.279 |
| SAC | fixed | 2.452 | 13.70 | 14.49 | 46.84 | 0.0275 | 1.413 |
| TQC | fixed | 2.468 | 13.77 | 14.56 | 46.88 | 0.0281 | 1.423 |
| SAC_HER | fixed | 2.243 | 10.78 | 12.35 | 41.09 | 0.0429 | 1.294 |
| TD3_HER | fixed | 2.810 | 10.74 | 15.27 | 62.92 | 0.0556 | 1.621 |
| TQC_HER | fixed | 2.322 | 10.68 | 12.69 | 43.11 | 0.0387 | 1.339 |
| TD3_HER_CURRICULUM（本文） | fixed | 0.424 | 2.692 | 2.623 | 7.878 | 0.0078 | 0.244 |
| DDPG | random | 5.663 | 34.22 | 36.94 | 125.1 | 0.0398 | 3.264 |
| TD3 | random | 5.970 | 34.99 | 38.38 | 131.1 | 0.0488 | 3.441 |
| SAC | random | 4.752 | 28.54 | 30.70 | 99.27 | 0.0359 | 2.739 |
| TQC | random | 5.034 | 30.34 | 33.04 | 107.1 | 0.0378 | 2.901 |
| SAC_HER | random | 7.251 | 33.30 | 41.18 | 159.0 | 0.7309 | 4.179 |
| TD3_HER | random | 7.154 | 40.04 | 44.53 | 161.7 | 0.1396 | 4.124 |
| TQC_HER | random | 7.067 | 31.88 | 41.07 | 155.0 | 0.7520 | 4.071 |
| TD3_HER_CURRICULUM（本文） | random | 0.489 | 3.048 | 2.535 | 8.867 | 0.0105 | 0.282 |

### A.2 Panda 接触变力（medium）全算法对比

源数据：`runs/results/suite_summary_medium_servo.csv`。高精度指标除随机目标 HER 基线略有变化外与 A.1 一致，此处给出随机目标关键差异与稳定性指标。

随机目标高精度（接触变力下 HER 基线差异）：

| algo | task | final_error | succ_10 | succ_5 | hold_5 |
| --- | --- | ---: | ---: | ---: | ---: |
| SAC_HER | random | 6.96 mm | 0.800 | 0.333 | 0.427 |
| TD3_HER | random | 1.85 mm | 1.000 | 0.933 | 0.940 |
| TQC_HER | random | 7.20 mm | 0.767 | 0.467 | 0.427 |
| TD3_HER_CURRICULUM（本文） | random | 0.26 mm | 1.000 | 1.000 | 1.000 |

稳定性指标（接触变力）：

| algo | task | acc_rms | acc_peak | acc_p2p | jerk_rms | action_delta | vib_index |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| DDPG | fixed | 3.073 | 19.24 | 18.21 | 69.17 | 0.0358 | 1.771 |
| TD3 | fixed | 2.218 | 9.712 | 13.56 | 44.76 | 0.0487 | 1.279 |
| SAC | fixed | 2.452 | 13.70 | 14.49 | 46.84 | 0.0275 | 1.413 |
| TQC | fixed | 2.468 | 13.77 | 14.56 | 46.88 | 0.0281 | 1.423 |
| SAC_HER | fixed | 2.243 | 10.78 | 12.35 | 41.09 | 0.0429 | 1.294 |
| TD3_HER | fixed | 2.810 | 10.74 | 15.27 | 62.92 | 0.0556 | 1.621 |
| TQC_HER | fixed | 2.322 | 10.68 | 12.69 | 43.11 | 0.0387 | 1.339 |
| TD3_HER_CURRICULUM（本文） | fixed | 0.424 | 2.692 | 2.623 | 7.878 | 0.0078 | 0.244 |
| DDPG | random | 5.663 | 34.22 | 36.94 | 125.1 | 0.0398 | 3.264 |
| TD3 | random | 5.969 | 34.99 | 38.38 | 131.1 | 0.0488 | 3.441 |
| SAC | random | 4.752 | 28.54 | 30.70 | 99.27 | 0.0359 | 2.739 |
| TQC | random | 5.034 | 30.34 | 33.04 | 107.1 | 0.0378 | 2.901 |
| SAC_HER | random | 7.397 | 33.86 | 43.60 | 165.3 | 0.7617 | 4.261 |
| TD3_HER | random | 7.183 | 40.04 | 44.53 | 162.6 | 0.1768 | 4.141 |
| TQC_HER | random | 6.943 | 31.93 | 40.48 | 152.6 | 0.8246 | 3.999 |
| TD3_HER_CURRICULUM（本文） | random | 0.489 | 3.048 | 2.535 | 8.868 | 0.0105 | 0.282 |

### A.3 完整方法 34 项领先明细

源数据：`runs/results/advantage_report.csv`。统计口径为自由空间/接触变力 × 固定/随机目标的 8 个核心误差与稳定性指标，再加两个固定目标 Hold@5mm，共 34 项；结果为严格领先 32 项、并列最优 2 项、未领先 0 项。

| condition | task | metric | result | full | best_baseline | baseline_algo |
| --- | --- | --- | --- | ---: | ---: | --- |
| 自由空间 | fixed | final_error | strict | 8.09e-05 | 5.82e-04 | TD3 |
| 自由空间 | fixed | min_error | strict | 8.09e-05 | 3.24e-04 | TD3_HER |
| 自由空间 | fixed | ee_acc_rms | strict | 0.4235 | 2.2176 | TD3 |
| 自由空间 | fixed | ee_acc_peak | strict | 2.6922 | 9.7124 | TD3 |
| 自由空间 | fixed | ee_acc_p2p | strict | 2.6234 | 12.354 | SAC_HER |
| 自由空间 | fixed | ee_jerk_rms | strict | 7.8782 | 41.092 | SAC_HER |
| 自由空间 | fixed | action_delta | strict | 0.0078 | 0.0275 | SAC |
| 自由空间 | fixed | vibration_index | strict | 0.2440 | 1.2792 | TD3 |
| 自由空间 | fixed | hold_5mm | tie | 1.000 | 1.000 | DDPG |
| 自由空间 | random | final_error | strict | 2.56e-04 | 1.30e-03 | TQC |
| 自由空间 | random | min_error | strict | 2.15e-04 | 8.88e-04 | TQC |
| 自由空间 | random | ee_acc_rms | strict | 0.4893 | 4.7515 | SAC |
| 自由空间 | random | ee_acc_peak | strict | 3.0477 | 28.544 | SAC |
| 自由空间 | random | ee_acc_p2p | strict | 2.5354 | 30.699 | SAC |
| 自由空间 | random | ee_jerk_rms | strict | 8.8670 | 99.266 | SAC |
| 自由空间 | random | action_delta | strict | 0.0105 | 0.0359 | SAC |
| 自由空间 | random | vibration_index | strict | 0.2820 | 2.7385 | SAC |
| 接触变力 | fixed | final_error | strict | 8.09e-05 | 5.82e-04 | TD3 |
| 接触变力 | fixed | min_error | strict | 8.09e-05 | 3.24e-04 | TD3_HER |
| 接触变力 | fixed | ee_acc_rms | strict | 0.4235 | 2.2176 | TD3 |
| 接触变力 | fixed | ee_acc_peak | strict | 2.6922 | 9.7124 | TD3 |
| 接触变力 | fixed | ee_acc_p2p | strict | 2.6234 | 12.354 | SAC_HER |
| 接触变力 | fixed | ee_jerk_rms | strict | 7.8782 | 41.092 | SAC_HER |
| 接触变力 | fixed | action_delta | strict | 0.0078 | 0.0275 | SAC |
| 接触变力 | fixed | vibration_index | strict | 0.2440 | 1.2792 | TD3 |
| 接触变力 | fixed | hold_5mm | tie | 1.000 | 1.000 | DDPG |
| 接触变力 | random | final_error | strict | 2.56e-04 | 1.30e-03 | TQC |
| 接触变力 | random | min_error | strict | 2.16e-04 | 8.88e-04 | TQC |
| 接触变力 | random | ee_acc_rms | strict | 0.4894 | 4.7515 | SAC |
| 接触变力 | random | ee_acc_peak | strict | 3.0477 | 28.544 | SAC |
| 接触变力 | random | ee_acc_p2p | strict | 2.5354 | 30.699 | SAC |
| 接触变力 | random | ee_jerk_rms | strict | 8.8683 | 99.267 | SAC |
| 接触变力 | random | action_delta | strict | 0.0105 | 0.0359 | SAC |
| 接触变力 | random | vibration_index | strict | 0.2820 | 2.7385 | SAC |

### A.4 消融实验固定目标完整表

源数据：`runs/results/ablation_report.csv`。正文 §10 给出随机目标消融，此处补充固定目标消融（接触变力 medium）。

| 阶段 | 最终误差 | 5mm成功率 | Hold@5mm | acc_rms | acc_p2p | jerk_rms | action_delta | vib_index |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| TD3 | 0.582 mm | 1.000 | 1.000 | 2.218 | 13.561 | 44.759 | 0.0487 | 1.279 |
| TD3+HER | 1.188 mm | 1.000 | 1.000 | 2.810 | 15.266 | 62.918 | 0.0556 | 1.621 |
| +课程学习+稳定性约束（无伺服） | 2.850 mm | 1.000 | 1.000 | 4.754 | 27.803 | 107.242 | 0.1191 | 2.742 |
| +动作平滑（无伺服） | 2.187 mm | 1.000 | 1.000 | 1.905 | 11.441 | 36.704 | 0.0614 | 1.098 |
| 完整方法 | 0.081 mm | 1.000 | 1.000 | 0.423 | 2.623 | 7.878 | 0.0078 | 0.244 |

### A.5 Panda 末端空间 HER 残差伺服消融

源数据：`runs/results/panda_cartesian_*_summary.json`、`runs/results/panda_her_residual_*_summary.json`。Panda 使用 panda-gym 末端控制接口，采用末端空间残差形式 `a = a_servo + alpha·clip(a_HER − a_servo)`，与 IRB120 的 IK-HER 残差思想一致。

| 方法 | 任务 | 最终误差 | 5mm成功率 | Hold@5mm | acc_rms | acc_p2p | jerk_rms | action_delta | vib_index |
| --- | :---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| cartesian servo | fixed | 0.081 mm | 1.000 | 1.000 | 0.4235 | 2.6234 | 7.8782 | 0.0078 | 0.2440 |
| HER residual servo | fixed | 0.196 mm | 1.000 | 1.000 | 0.3592 | 2.1993 | 6.5553 | 0.0071 | 0.2070 |
| cartesian servo | random | 0.256 mm | 1.000 | 1.000 | 0.4894 | 2.5354 | 8.8683 | 0.0105 | 0.2820 |
| HER residual servo | random | 0.322 mm | 1.000 | 1.000 | 0.4303 | 2.2290 | 7.7361 | 0.0097 | 0.2480 |

在保持 5 mm 成功率与 Hold@5mm 均为 1.000 的前提下，HER residual servo 相对 cartesian servo 的稳定性指标在固定目标下降低约 15–17%，随机目标下降低约 12–13%。

### A.6 IRB120 自由空间全算法对比（含 qualified_*）

源数据：`runs/irb120/results/suite_summary_free_optimized.csv`。短训基线（DDPG/TD3/SAC/TQC/TD3_HER）几乎不动，最终误差在 18–26 cm，5 mm 成功率为 0；`qualified_*` 列对未达 5 mm 精度的策略加入定位误差惩罚。

| algo | task | final_error | succ_5 | hold_5 | acc_rms | jerk_rms | qual_acc_rms |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| DDPG / TD3 / TD3_HER | fixed | 235.54 mm | 0.000 | 0.000 | 0.2120 | 3.537 | 23.06 |
| SAC / TQC | fixed | 189.48 mm | 0.000 | 0.000 | 0.2288 | 3.830 | 18.45 |
| TD3_HER_CURRICULUM（本文） | fixed | 0.24 mm | 1.000 | 1.000 | 0.0080 | 0.1078 | 2.85e-06 |
| DDPG / TD3 / TD3_HER | random | 257.78 mm | 0.000 | 0.000 | 0.2191 | 3.712 | 25.28 |
| SAC / TQC | random | 219.42 mm | 0.000 | 0.000 | 0.2634 | 4.491 | 21.46 |
| TD3_HER_CURRICULUM（本文） | random | 0.49 mm | 1.000 | 1.000 | 0.0073 | 0.0931 | 5.24e-05 |

### A.7 IRB120 接触变力全算法对比（含 qualified_*）

源数据：`runs/irb120/results/suite_summary_medium_optimized.csv`。基线 raw 指标与自由空间一致（基线几乎不动，扰动只在近目标触发，故未生效）；本文方法 raw 与 qualified 指标如下。

| algo | task | final_error | succ_5 | hold_5 | acc_rms | jerk_rms | qual_acc_rms |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| DDPG / TD3 / TD3_HER | fixed | 235.54 mm | 0.000 | 0.000 | 0.2120 | 3.537 | 23.06 |
| SAC / TQC | fixed | 189.48 mm | 0.000 | 0.000 | 0.2288 | 3.830 | 18.45 |
| TD3_HER_CURRICULUM（本文） | fixed | 0.24 mm | 1.000 | 1.000 | 0.0080 | 0.1079 | 1.80e-04 |
| DDPG / TD3 / TD3_HER | random | 257.78 mm | 0.000 | 0.000 | 0.2191 | 3.712 | 25.28 |
| SAC / TQC | random | 219.42 mm | 0.000 | 0.000 | 0.2634 | 4.491 | 21.46 |
| TD3_HER_CURRICULUM（本文） | random | 0.49 mm | 1.000 | 1.000 | 0.0073 | 0.0931 | 2.11e-04 |

### A.8 IRB120 IK 与 HER 残差消融完整表

源数据：`runs/irb120/results/ik_variant_summary_medium.csv`。控制形式 `a = a_IK + alpha·clip(a_HER − a_IK)`，`alpha` 为近目标门控小权重。

| 方法 | 任务 | 最终误差 | 5mm成功率 | Hold@5mm | acc_rms | acc_p2p | jerk_rms | action_delta | qual_rms |
| --- | :---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 无精密伺服 | fixed | 236.176 mm | 0.000 | 0.000 | 0.2367 | 3.5994 | 4.0004 | 0.0040 | 23.1211 |
| 无精密伺服 | random | 272.360 mm | 0.000 | 0.000 | 0.2252 | 3.3069 | 3.8130 | 0.0049 | 26.7446 |
| IK 伺服 | fixed | 0.149 mm | 1.000 | 1.000 | 0.0076 | 0.0939 | 0.1019 | 0.0020 | 0.0002 |
| IK 伺服 | random | 0.157 mm | 1.000 | 1.000 | 0.0137 | 0.1660 | 0.2038 | 0.0041 | 0.0002 |
| IK 主导残差 | fixed | 72.596 mm | 0.000 | 0.000 | 0.0133 | 0.0991 | 0.1780 | 0.0338 | 6.7664 |
| IK 主导残差 | random | 135.498 mm | 0.000 | 0.000 | 0.0437 | 0.4986 | 0.6820 | 0.0276 | 13.0554 |
| IK 观测增强 | fixed | 6.372 mm | 0.000 | 0.000 | 0.1015 | 1.2669 | 1.6263 | 0.0511 | 0.1402 |
| IK 观测增强 | random | 26.074 mm | 0.100 | 0.097 | 0.0941 | 1.0047 | 1.2950 | 0.0718 | 2.1798 |
| HER+IK 残差 | fixed | 0.183 mm | 1.000 | 1.000 | 0.0061 | 0.0757 | 0.0818 | 0.0016 | 0.0002 |
| HER+IK 残差 | random | 0.538 mm | 0.967 | 0.960 | 0.0116 | 0.1357 | 0.1750 | 0.0030 | 0.0073 |

IK 主导残差与 IK 观测增强均无法稳定达到 5 mm，说明 IK 与 RL 需要受限融合而非大权重相加或简单观测拼接；最终采用“小幅 HER 残差 + IK 先验”形式。
