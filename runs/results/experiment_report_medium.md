# 高精度 Reach 与稳定性实验结果

## 高精度指标

| algo | task | disturbance | final_error_m_mean | success_50mm_mean | success_30mm_mean | success_20mm_mean | success_10mm_mean | success_5mm_mean | hold_5mm_mean |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| DDPG | fixed | medium | 1.56 mm | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 |
| TD3 | fixed | medium | 0.58 mm | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 |
| SAC | fixed | medium | 0.93 mm | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 |
| TQC | fixed | medium | 1.89 mm | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 |
| SAC_HER | fixed | medium | 1.30 mm | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 |
| TD3_HER | fixed | medium | 1.19 mm | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 |
| TQC_HER | fixed | medium | 2.71 mm | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 |
| TD3_HER_CURRICULUM | fixed | medium | 2.19 mm | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 |
| DDPG | random | medium | 2.12 mm | 1.000 | 1.000 | 1.000 | 1.000 | 0.967 | 0.967 |
| TD3 | random | medium | 1.70 mm | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 |
| SAC | random | medium | 2.04 mm | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 |
| TQC | random | medium | 1.30 mm | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 |
| SAC_HER | random | medium | 6.96 mm | 1.000 | 1.000 | 1.000 | 0.800 | 0.333 | 0.427 |
| TD3_HER | random | medium | 1.85 mm | 1.000 | 1.000 | 1.000 | 1.000 | 0.933 | 0.940 |
| TQC_HER | random | medium | 7.20 mm | 1.000 | 1.000 | 1.000 | 0.767 | 0.467 | 0.427 |
| TD3_HER_CURRICULUM | random | medium | 1.89 mm | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 |

## 稳定性指标

| algo | task | disturbance | ee_acc_rms_mean | ee_acc_peak_mean | ee_acc_p2p_mean | ee_jerk_rms_mean | joint_acc_rms_mean | action_delta_mean_mean | vibration_index_mean |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| DDPG | fixed | medium | 3.073 | 19.24 | 18.21 | 69.17 | 0 | 0.03584 | 1.771 |
| TD3 | fixed | medium | 2.218 | 9.712 | 13.56 | 44.76 | 0 | 0.04865 | 1.279 |
| SAC | fixed | medium | 2.452 | 13.7 | 14.49 | 46.84 | 0 | 0.0275 | 1.413 |
| TQC | fixed | medium | 2.468 | 13.77 | 14.56 | 46.88 | 0 | 0.02812 | 1.423 |
| SAC_HER | fixed | medium | 2.243 | 10.78 | 12.35 | 41.09 | 0 | 0.04285 | 1.294 |
| TD3_HER | fixed | medium | 2.81 | 10.74 | 15.27 | 62.92 | 0 | 0.05558 | 1.621 |
| TQC_HER | fixed | medium | 2.322 | 10.68 | 12.69 | 43.11 | 0 | 0.03871 | 1.339 |
| TD3_HER_CURRICULUM | fixed | medium | 1.905 | 9.42 | 11.44 | 36.7 | 0 | 0.0614 | 1.098 |
| DDPG | random | medium | 5.663 | 34.22 | 36.94 | 125.1 | 0 | 0.0398 | 3.264 |
| TD3 | random | medium | 5.969 | 34.99 | 38.38 | 131.1 | 0 | 0.04877 | 3.441 |
| SAC | random | medium | 4.752 | 28.54 | 30.7 | 99.27 | 0 | 0.03593 | 2.739 |
| TQC | random | medium | 5.034 | 30.34 | 33.04 | 107.1 | 0 | 0.03775 | 2.901 |
| SAC_HER | random | medium | 7.397 | 33.86 | 43.6 | 165.3 | 0 | 0.7617 | 4.261 |
| TD3_HER | random | medium | 7.183 | 40.04 | 44.53 | 162.6 | 0 | 0.1768 | 4.141 |
| TQC_HER | random | medium | 6.943 | 31.93 | 40.48 | 152.6 | 0 | 0.8246 | 3.999 |
| TD3_HER_CURRICULUM | random | medium | 6.674 | 39.19 | 40.13 | 139.7 | 0 | 0.09072 | 3.847 |

## 结果分析模板

从高精度指标看，重点比较 `success_5mm_mean`、`hold_5mm_mean` 和 `final_error_m_mean`。若本文方法在固定目标和随机目标下均达到较高 5 mm 成功率，同时最终误差最低或接近最低，说明 HER 与课程学习对高精度 Reach 有明显帮助。

从稳定性指标看，重点比较 `ee_acc_rms_mean`、`ee_acc_p2p_mean` 和 `ee_jerk_rms_mean`。在接触变力扰动下这些指标越低，说明末端受迫振动越弱，运动越平稳。