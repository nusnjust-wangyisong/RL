# 完整方法相对普通基线领先统计

统计口径：自由空间/接触变力 × 固定/随机目标的 8 个核心误差与稳定性指标，再加两个固定目标 Hold@5mm。共 34 项。
结果：严格领先 32 项，并列最优 2 项，未领先 0 项。

| condition | task | metric | result | full | best_baseline | best_baseline_algo |
| --- | --- | --- | --- | ---: | ---: | --- |
| 自由空间 | fixed | final_error_m_mean | strict | 1.94474e-05 | 0.000582277 | TD3 |
| 自由空间 | fixed | min_error_m_mean | strict | 1.94474e-05 | 0.000323961 | TD3_HER |
| 自由空间 | fixed | ee_acc_rms_mean | strict | 0.69135 | 2.21759 | TD3 |
| 自由空间 | fixed | ee_acc_peak_mean | strict | 4.43758 | 9.71238 | TD3 |
| 自由空间 | fixed | ee_acc_p2p_mean | strict | 4.47663 | 12.3541 | SAC_HER |
| 自由空间 | fixed | ee_jerk_rms_mean | strict | 13.6557 | 41.0919 | SAC_HER |
| 自由空间 | fixed | action_delta_mean_mean | strict | 0.0107542 | 0.0275018 | SAC |
| 自由空间 | fixed | vibration_index_mean | strict | 0.39836 | 1.27915 | TD3 |
| 自由空间 | fixed | hold_5mm_mean | tie | 1 | 1 | DDPG |
| 自由空间 | random | final_error_m_mean | strict | 0.000206534 | 0.00129922 | TQC |
| 自由空间 | random | min_error_m_mean | strict | 0.000173956 | 0.00088846 | TQC |
| 自由空间 | random | ee_acc_rms_mean | strict | 0.856354 | 4.75151 | SAC |
| 自由空间 | random | ee_acc_peak_mean | strict | 5.37331 | 28.5438 | SAC |
| 自由空间 | random | ee_acc_p2p_mean | strict | 5.16503 | 30.6987 | SAC |
| 自由空间 | random | ee_jerk_rms_mean | strict | 16.571 | 99.2661 | SAC |
| 自由空间 | random | action_delta_mean_mean | strict | 0.0142691 | 0.0359293 | SAC |
| 自由空间 | random | vibration_index_mean | strict | 0.493479 | 2.73851 | SAC |
| 接触变力 | fixed | final_error_m_mean | strict | 1.9445e-05 | 0.000582277 | TD3 |
| 接触变力 | fixed | min_error_m_mean | strict | 1.9445e-05 | 0.000323961 | TD3_HER |
| 接触变力 | fixed | ee_acc_rms_mean | strict | 0.691345 | 2.2176 | TD3 |
| 接触变力 | fixed | ee_acc_peak_mean | strict | 4.43758 | 9.71238 | TD3 |
| 接触变力 | fixed | ee_acc_p2p_mean | strict | 4.47663 | 12.3541 | SAC_HER |
| 接触变力 | fixed | ee_jerk_rms_mean | strict | 13.6557 | 41.092 | SAC_HER |
| 接触变力 | fixed | action_delta_mean_mean | strict | 0.0107541 | 0.0275018 | SAC |
| 接触变力 | fixed | vibration_index_mean | strict | 0.398357 | 1.27915 | TD3 |
| 接触变力 | fixed | hold_5mm_mean | tie | 1 | 1 | DDPG |
| 接触变力 | random | final_error_m_mean | strict | 0.000207307 | 0.00129962 | TQC |
| 接触变力 | random | min_error_m_mean | strict | 0.00017834 | 0.000888187 | TQC |
| 接触变力 | random | ee_acc_rms_mean | strict | 0.856242 | 4.75154 | SAC |
| 接触变力 | random | ee_acc_peak_mean | strict | 5.37331 | 28.5438 | SAC |
| 接触变力 | random | ee_acc_p2p_mean | strict | 5.16466 | 30.6987 | SAC |
| 接触变力 | random | ee_jerk_rms_mean | strict | 16.5688 | 99.2671 | SAC |
| 接触变力 | random | action_delta_mean_mean | strict | 0.0142688 | 0.0359296 | SAC |
| 接触变力 | random | vibration_index_mean | strict | 0.493414 | 2.73853 | SAC |