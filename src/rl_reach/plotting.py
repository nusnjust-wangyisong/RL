from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any

import numpy as np

from rl_reach.algorithms import load_model
from rl_reach.config import deep_update, ensure_run_dirs, load_config
from rl_reach.envs import make_env, unwrap_high_precision


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Create trajectory projection and metric comparison figures.")
    parser.add_argument("--config", default="configs/experiment.yaml")
    parser.add_argument("--algo", default=None)
    parser.add_argument("--model", default=None)
    parser.add_argument("--fixed-goal", action="store_true")
    parser.add_argument("--random-goal", action="store_true")
    parser.add_argument("--summary-csv", default=None)
    parser.add_argument("--name", default="trajectory")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    cfg = load_config(args.config)
    if args.fixed_goal:
        cfg = deep_update(cfg, {"env": {"fixed_goal": True}})
    if args.random_goal:
        cfg = deep_update(cfg, {"env": {"fixed_goal": False}})
    paths = ensure_run_dirs(cfg)
    if args.summary_csv:
        plot_suite_csv(Path(args.summary_csv), paths["figure_dir"] / f"{args.name}_metrics.png")
    if args.model and args.algo:
        plot_policy_projection(cfg, args.algo, Path(args.model), paths["figure_dir"] / f"{args.name}_projection.png")


def plot_policy_projection(cfg: dict[str, Any], algo: str, model_path: Path, out_path: Path) -> None:
    import matplotlib.pyplot as plt
    from matplotlib.patches import Circle
    from stable_baselines3.common.monitor import Monitor

    env = Monitor(make_env(cfg, seed=int(cfg.get("seed", 0)) + 2026)())
    model = load_model(algo, str(model_path), env, cfg)
    obs, _ = env.reset()
    done = False
    while not done:
        action, _ = model.predict(obs, deterministic=True)
        obs, _reward, terminated, truncated, _info = env.step(action)
        done = bool(terminated or truncated)
    wrapper = unwrap_high_precision(env)
    if wrapper is None:
        raise RuntimeError("HighPrecisionReachWrapper not found.")
    trace = wrapper.get_episode_trace().arrays()
    ee = trace["ee_pos"]
    goal = trace["goals"][-1]
    thresholds = cfg.get("env", {}).get("thresholds_m", [0.05, 0.03, 0.02, 0.01, 0.005])

    fig, axes = plt.subplots(2, 1, figsize=(5.4, 8.2), constrained_layout=True)
    views = [(0, 2, "x (m)", "z (m)"), (1, 2, "y (m)", "z (m)")]
    colors = {0.05: "tab:green", 0.03: "tab:blue", 0.02: "tab:purple", 0.01: "tab:red", 0.005: "tab:orange"}
    for ax, (i, j, xlabel, ylabel) in zip(axes, views):
        ax.plot(ee[:, i], ee[:, j], color="0.25", linewidth=1.2, label="trajectory")
        ax.scatter([goal[i]], [goal[j]], color="tab:green", s=36, label="goal", zorder=3)
        ax.scatter([ee[-1, i]], [ee[-1, j]], marker="x", color="tab:red", s=54, label="end effector", zorder=4)
        for threshold in thresholds:
            ax.add_patch(
                Circle(
                    (goal[i], goal[j]),
                    radius=threshold,
                    fill=False,
                    linestyle="--",
                    linewidth=0.8,
                    color=colors.get(float(threshold), "0.5"),
                    label=f"{int(threshold * 1000)} mm",
                )
            )
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
        ax.set_aspect("equal", adjustable="box")
        ax.grid(True, alpha=0.25)
    final_error = float(np.linalg.norm(ee[-1] - goal))
    axes[0].set_title(f"{algo} | distance to target: {final_error * 1000:.1f} mm")
    handles, labels = axes[0].get_legend_handles_labels()
    by_label = dict(zip(labels, handles))
    axes[0].legend(by_label.values(), by_label.keys(), ncol=4, fontsize=8, loc="upper center")
    fig.savefig(out_path, dpi=180)
    plt.close(fig)
    env.close()
    print(f"Saved projection figure: {out_path}")


def plot_suite_csv(csv_path: Path, out_path: Path) -> None:
    import matplotlib.pyplot as plt

    rows = read_csv(csv_path)
    if not rows:
        return
    algos = [row["algo"] for row in rows]
    error = [float(row.get("final_error_m_mean", 0.0)) * 1000 for row in rows]
    acc = [float(row.get("ee_acc_rms_mean", 0.0)) for row in rows]
    jerk = [float(row.get("ee_jerk_rms_mean", 0.0)) for row in rows]

    x = np.arange(len(algos))
    fig, axes = plt.subplots(3, 1, figsize=(10, 8), constrained_layout=True)
    for ax, values, ylabel in [
        (axes[0], error, "final error (mm)"),
        (axes[1], acc, "EE acc RMS"),
        (axes[2], jerk, "EE jerk RMS"),
    ]:
        ax.bar(x, values, color="#4c78a8")
        ax.set_ylabel(ylabel)
        ax.grid(axis="y", alpha=0.25)
    axes[-1].set_xticks(x, algos, rotation=30, ha="right")
    fig.savefig(out_path, dpi=180)
    plt.close(fig)
    print(f"Saved metric figure: {out_path}")


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


if __name__ == "__main__":
    main()
