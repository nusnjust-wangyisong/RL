from __future__ import annotations

import argparse
import csv
from pathlib import Path


PRECISION_KEYS = [
    "final_error_m_mean",
    "success_50mm_mean",
    "success_30mm_mean",
    "success_20mm_mean",
    "success_10mm_mean",
    "success_5mm_mean",
    "hold_5mm_mean",
]

STABILITY_KEYS = [
    "ee_acc_rms_mean",
    "ee_acc_peak_mean",
    "ee_acc_p2p_mean",
    "ee_jerk_rms_mean",
    "joint_acc_rms_mean",
    "action_delta_mean_mean",
    "vibration_index_mean",
    "qualified_ee_acc_rms_mean",
    "qualified_ee_acc_p2p_mean",
    "qualified_ee_jerk_rms_mean",
    "qualified_joint_acc_rms_mean",
    "qualified_action_delta_mean_mean",
    "qualified_vibration_index_mean",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate markdown tables from suite_summary.csv.")
    parser.add_argument("--suite-csv", default="runs/results/suite_summary.csv")
    parser.add_argument("--output", default="runs/results/experiment_report.md")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    rows = read_rows(Path(args.suite_csv))
    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(render_report(rows), encoding="utf-8")
    print(f"Saved report: {out}")


def read_rows(path: Path) -> list[dict[str, str]]:
    with path.open("r", newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def render_report(rows: list[dict[str, str]]) -> str:
    lines = [
        "# 高精度 Reach 与稳定性实验结果",
        "",
        "## 高精度指标",
        "",
        render_table(rows, ["algo", "task", "disturbance", *PRECISION_KEYS], precision=True),
        "",
        "## 稳定性指标",
        "",
        render_table(rows, ["algo", "task", "disturbance", *STABILITY_KEYS], precision=False),
        "",
        "## 结果分析模板",
        "",
        "从高精度指标看，重点比较 `success_5mm_mean`、`hold_5mm_mean` 和 `final_error_m_mean`。"
        "若本文方法在固定目标和随机目标下均达到较高 5 mm 成功率，同时最终误差最低或接近最低，"
        "说明 HER 与课程学习对高精度 Reach 有明显帮助。",
        "",
        "从稳定性指标看，重点比较 `ee_acc_rms_mean`、`ee_acc_p2p_mean` 和 `ee_jerk_rms_mean`。"
        "在接触变力扰动下这些指标越低，说明末端受迫振动越弱，运动越平稳。",
        "",
        "`qualified_*` 指标是在最终 5 mm 精度约束下计算的稳定性评价：若策略没有到达加工精度，"
        "会加入定位误差惩罚，避免“机械臂几乎不动所以振动小”的无效稳定性结论。",
    ]
    return "\n".join(lines)


def render_table(rows: list[dict[str, str]], keys: list[str], *, precision: bool) -> str:
    if not rows:
        return "暂无结果。"
    header = "| " + " | ".join(keys) + " |"
    sep = "| " + " | ".join(["---"] * len(keys)) + " |"
    body = []
    for row in rows:
        values = [format_value(row.get(key, ""), key, precision=precision) for key in keys]
        body.append("| " + " | ".join(values) + " |")
    return "\n".join([header, sep, *body])


def format_value(value: str, key: str, *, precision: bool) -> str:
    if key in {"algo", "task", "disturbance"}:
        return value
    try:
        number = float(value)
    except (TypeError, ValueError):
        return value
    if key == "final_error_m_mean":
        return f"{number * 1000:.2f} mm"
    if key.startswith("success_") or key.startswith("hold_"):
        return f"{number:.3f}"
    return f"{number:.4g}"


if __name__ == "__main__":
    main()
