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
from rl_reach.evaluate import filter_action_if_needed, precision_servo_if_needed


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Create trajectory projection and metric comparison figures.")
    parser.add_argument("--config", default="configs/experiment.yaml")
    parser.add_argument("--algo", default=None)
    parser.add_argument("--model", default=None)
    parser.add_argument("--fixed-goal", action="store_true")
    parser.add_argument("--random-goal", action="store_true")
    parser.add_argument("--summary-csv", default=None)
    parser.add_argument("--name", default="trajectory")
    parser.add_argument("--render-snapshot", action="store_true")
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
        if args.render_snapshot:
            plot_policy_snapshot(cfg, args.algo, Path(args.model), paths["figure_dir"] / f"{args.name}_render.png")


def plot_policy_projection(cfg: dict[str, Any], algo: str, model_path: Path, out_path: Path) -> None:
    import matplotlib.pyplot as plt
    from matplotlib.patches import Circle

    trace, _frame = rollout_policy(cfg, algo, model_path, render_mode=None)
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
    print(f"Saved projection figure: {out_path}")


def plot_policy_snapshot(cfg: dict[str, Any], algo: str, model_path: Path, out_path: Path) -> None:
    import matplotlib.pyplot as plt

    cfg = deep_update(cfg, {"irb120": {"target_visual_radius": 0.035}})
    trace, frame = rollout_policy(cfg, algo, model_path, render_mode="rgb_array")
    if frame is None:
        raise RuntimeError("Environment did not return an RGB frame.")
    ee = trace["ee_pos"]
    goal = trace["goals"][-1]
    final_error = float(np.linalg.norm(ee[-1] - goal))

    fig, ax = plt.subplots(figsize=(8.4, 5.8), constrained_layout=True)
    ax.imshow(frame)
    ax.set_axis_off()
    mode = "fixed target" if cfg.get("env", {}).get("fixed_goal") else "random target"
    ax.set_title(f"{algo} | {mode} | final error {final_error * 1000:.2f} mm")
    fig.savefig(out_path, dpi=180)
    plt.close(fig)
    print(f"Saved render snapshot: {out_path}")


def rollout_policy(
    cfg: dict[str, Any],
    algo: str,
    model_path: Path,
    *,
    render_mode: str | None,
    reset_seed: int | None = None,
) -> tuple[dict[str, np.ndarray], np.ndarray | None]:
    from stable_baselines3.common.monitor import Monitor

    cfg = deep_update(cfg, {"env": {"terminate_on_success": False}})
    env = Monitor(make_env(cfg, render_mode=render_mode, seed=int(cfg.get("seed", 0)) + 2026)())
    model = load_model(algo, str(model_path), env, cfg)
    if reset_seed is None:
        obs, _ = env.reset()
    else:
        obs, _ = env.reset(seed=reset_seed)
    done = False
    prev_action = None
    frame: np.ndarray | None = env.render() if render_mode == "rgb_array" else None
    while not done:
        action, _ = model.predict(obs, deterministic=True)
        wrapper = unwrap_high_precision(env)
        action = precision_servo_if_needed(action, obs, prev_action, algo, cfg, wrapper)
        action = filter_action_if_needed(action, prev_action, algo, cfg)
        prev_action = np.asarray(action, dtype=float).copy()
        obs, _reward, terminated, truncated, _info = env.step(action)
        if render_mode == "rgb_array":
            current_frame = env.render()
            if current_frame is not None:
                frame = np.asarray(current_frame)
        done = bool(terminated or truncated)
    wrapper = unwrap_high_precision(env)
    if wrapper is None:
        raise RuntimeError("HighPrecisionReachWrapper not found.")
    trace = wrapper.get_episode_trace().arrays()
    env.close()
    if frame is not None and frame.ndim == 3 and frame.shape[-1] == 4:
        frame = frame[..., :3]
    return trace, frame


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
