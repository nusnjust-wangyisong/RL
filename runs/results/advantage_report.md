# 完整方法相对普通基线领先统计

统计口径：自由空间/接触变力 × 固定/随机目标的 8 个核心误差与稳定性指标，再加两个固定目标 Hold@5mm。共 34 项。
结果：严格领先 32 项，并列最优 2 项，未领先 0 项。

| condition | task | metric | result | full | best_baseline | best_baseline_algo |
| --- | --- | --- | --- | ---: | ---: | --- |
| 自由空间 | fixed | final_error_m_mean | strict | 8.0944e-05 | 0.000582277 | TD3 |
| 自由空间 | fixed | min_error_m_mean | strict | 8.0944e-05 | 0.000323961 | TD3_HER |
| 自由空间 | fixed | ee_acc_rms_mean | strict | 0.423485 | 2.21759 | TD3 |
| 自由空间 | fixed | ee_acc_peak_mean | strict | 2.69218 | 9.71238 | TD3 |
| 自由空间 | fixed | ee_acc_p2p_mean | strict | 2.62337 | 12.3541 | SAC_HER |
| 自由空间 | fixed | ee_jerk_rms_mean | strict | 7.87818 | 41.0919 | SAC_HER |
| 自由空间 | fixed | action_delta_mean_mean | strict | 0.00780807 | 0.0275018 | SAC |
| 自由空间 | fixed | vibration_index_mean | strict | 0.244024 | 1.27915 | TD3 |
| 自由空间 | fixed | hold_5mm_mean | tie | 1 | 1 | DDPG |
| 自由空间 | random | final_error_m_mean | strict | 0.000256155 | 0.00129922 | TQC |
| 自由空间 | random | min_error_m_mean | strict | 0.000215249 | 0.00088846 | TQC |
| 自由空间 | random | ee_acc_rms_mean | strict | 0.489313 | 4.75151 | SAC |
| 自由空间 | random | ee_acc_peak_mean | strict | 3.04771 | 28.5438 | SAC |
| 自由空间 | random | ee_acc_p2p_mean | strict | 2.53544 | 30.6987 | SAC |
| 自由空间 | random | ee_jerk_rms_mean | strict | 8.86703 | 99.2661 | SAC |
| 自由空间 | random | action_delta_mean_mean | strict | 0.0104717 | 0.0359293 | SAC |
| 自由空间 | random | vibration_index_mean | strict | 0.281977 | 2.73851 | SAC |
| 接触变力 | fixed | final_error_m_mean | strict | 8.09457e-05 | 0.000582277 | TD3 |
| 接触变力 | fixed | min_error_m_mean | strict | 8.09457e-05 | 0.000323961 | TD3_HER |
| 接触变力 | fixed | ee_acc_rms_mean | strict | 0.423484 | 2.2176 | TD3 |
| 接触变力 | fixed | ee_acc_peak_mean | strict | 2.69218 | 9.71238 | TD3 |
| 接触变力 | fixed | ee_acc_p2p_mean | strict | 2.62337 | 12.3541 | SAC_HER |
| 接触变力 | fixed | ee_jerk_rms_mean | strict | 7.87818 | 41.092 | SAC_HER |
| 接触变力 | fixed | action_delta_mean_mean | strict | 0.00780807 | 0.0275018 | SAC |
| 接触变力 | fixed | vibration_index_mean | strict | 0.244024 | 1.27915 | TD3 |
| 接触变力 | fixed | hold_5mm_mean | tie | 1 | 1 | DDPG |
| 接触变力 | random | final_error_m_mean | strict | 0.00025613 | 0.00129962 | TQC |
| 接触变力 | random | min_error_m_mean | strict | 0.000215621 | 0.000888187 | TQC |
| 接触变力 | random | ee_acc_rms_mean | strict | 0.489376 | 4.75154 | SAC |
| 接触变力 | random | ee_acc_peak_mean | strict | 3.04771 | 28.5438 | SAC |
| 接触变力 | random | ee_acc_p2p_mean | strict | 2.53541 | 30.6987 | SAC |
| 接触变力 | random | ee_jerk_rms_mean | strict | 8.8683 | 99.2671 | SAC |
| 接触变力 | random | action_delta_mean_mean | strict | 0.0104715 | 0.0359296 | SAC |
| 接触变力 | random | vibration_index_mean | strict | 0.282013 | 2.73853 | SAC |
