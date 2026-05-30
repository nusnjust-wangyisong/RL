from __future__ import annotations

import argparse
import csv
from pathlib import Path
from typing import Any

import numpy as np

from rl_reach.config import deep_update, ensure_run_dirs, load_config
from rl_reach.evaluate import evaluate_policy
from rl_reach.plotting import rollout_policy


DEFAULT_ALGOS = ["DDPG", "TD3", "SAC", "TQC", "TD3_HER", "TD3_HER_CURRICULUM"]
DISPLAY_NAMES = {
    "DDPG": "DDPG",
    "TD3": "TD3",
    "SAC": "SAC",
    "TQC": "TQC",
    "SAC_HER": "SAC+HER",
    "TD3_HER": "TD3+HER",
    "TQC_HER": "TQC+HER",
    "TD3_HER_CURRICULUM": "Ours",
}
ROW_COLORS = ["#e8f0fb", "#e9f7ee", "#fff1d7", "#f2e8fb", "#e8f6f7", "#dff5df"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate slide-ready result figures.")
    parser.add_argument("--config", default="configs/experiment_irb120.yaml")
    parser.add_argument("--model-dir", default="runs/irb120/models")
    parser.add_argument("--episodes", type=int, default=20)
    parser.add_argument("--projection-episodes", type=int, default=20)
    parser.add_argument("--disturbance", choices=["off", "light", "medium", "strong"], default="medium")
    parser.add_argument("--task", choices=["fixed", "random"], default="random")
    parser.add_argument("--algos", nargs="*", default=DEFAULT_ALGOS)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    cfg = load_config(args.config)
    robot_name = get_robot_label(cfg)
    paths = ensure_run_dirs(cfg)
    out_dir = paths["figure_dir"] / "slides"
    out_dir.mkdir(parents=True, exist_ok=True)
    model_dir = Path(args.model_dir)

    rows_by_algo = collect_episode_metrics(
        cfg=cfg,
        model_dir=model_dir,
        algos=args.algos,
        task=args.task,
        disturbance=args.disturbance,
        episodes=args.episodes,
        out_dir=paths["result_dir"] / "slides",
    )
    table_path = out_dir / f"slide_table_final_distance_{args.task}_{args.disturbance}.png"
    dashboard_path = out_dir / f"slide_dashboard_{args.task}_{args.disturbance}.png"
    projection_path = out_dir / f"slide_xz_projection_{args.task}_{args.disturbance}.png"

    plot_final_distance_table(rows_by_algo, table_path, task=args.task, disturbance=args.disturbance, robot_name=robot_name)
    plot_dashboard(rows_by_algo, dashboard_path, task=args.task, disturbance=args.disturbance, robot_name=robot_name)
    plot_projection_grid(
        cfg=cfg,
        model_dir=model_dir,
        algos=args.algos,
        task=args.task,
        disturbance=args.disturbance,
        projection_episodes=args.projection_episodes,
        robot_name=robot_name,
        out_path=projection_path,
    )
    print(f"Saved slide table: {table_path}")
    print(f"Saved slide dashboard: {dashboard_path}")
    print(f"Saved slide projection grid: {projection_path}")


def get_robot_label(cfg: dict[str, Any]) -> str:
    robot = str(cfg.get("env", {}).get("robot", "")).lower()
    if "irb120" in robot:
        return "IRB120"
    if "panda" in robot or "franka" in robot:
        return "Franka Panda"
    return str(cfg.get("env", {}).get("robot", "Robot"))


def collect_episode_metrics(
    *,
    cfg: dict[str, Any],
    model_dir: Path,
    algos: list[str],
    task: str,
    disturbance: str,
    episodes: int,
    out_dir: Path,
) -> dict[str, list[dict[str, Any]]]:
    out_dir.mkdir(parents=True, exist_ok=True)
    rows_by_algo: dict[str, list[dict[str, Any]]] = {}
    fixed_goal = task == "fixed"
    eval_cfg = deep_update(cfg, {"env": {"fixed_goal": fixed_goal, "terminate_on_success": False}})
    if disturbance == "off":
        eval_cfg = deep_update(eval_cfg, {"disturbance": {"enabled": False}})
    else:
        strength = cfg.get("disturbance", {}).get("strengths", {}).get(disturbance, {})
        eval_cfg = deep_update(eval_cfg, {"disturbance": {"enabled": True, **strength}})

    for algo in algos:
        model_path = model_dir / f"{algo}_{task}.zip"
        if not model_path.exists():
            continue
        cache_path = out_dir / f"{algo}_{task}_{disturbance}_episodes.csv"
        cached = read_cached_rows(cache_path)
        if len(cached) >= episodes:
            rows_by_algo[algo] = cached[:episodes]
            continue
        rows, _aggregate = evaluate_policy(
            cfg=eval_cfg,
            algo=algo,
            model_path=model_path,
            episodes=episodes,
            render=False,
        )
        write_rows(cache_path, rows)
        rows_by_algo[algo] = rows
    return rows_by_algo


def plot_final_distance_table(
    rows_by_algo: dict[str, list[dict[str, Any]]],
    out_path: Path,
    *,
    task: str,
    disturbance: str,
    robot_name: str,
) -> None:
    import matplotlib.pyplot as plt

    stats = []
    for algo, rows in rows_by_algo.items():
        errors = np.asarray([float(row["final_error_m"]) * 1000.0 for row in rows], dtype=float)
        if errors.size == 0:
            continue
        stats.append(
            {
                "algo": algo,
                "Algorithm": DISPLAY_NAMES.get(algo, algo),
                "Mean (mm)": np.mean(errors),
                "Median (mm)": np.median(errors),
                "P75 (mm)": np.percentile(errors, 75),
                "P90 (mm)": np.percentile(errors, 90),
                "P95 (mm)": np.percentile(errors, 95),
                "Max (mm)": np.max(errors),
                "Within 5mm": np.mean(errors <= 5.0) * 100.0,
                "Within 20mm": np.mean(errors <= 20.0) * 100.0,
            }
        )
    stats.sort(key=lambda row: (row["Mean (mm)"], row["Max (mm)"]))
    columns = [
        "Algorithm",
        "Mean\n(mm)",
        "Median\n(mm)",
        "P75\n(mm)",
        "P90\n(mm)",
        "P95\n(mm)",
        "Max\n(mm)",
        "Within\n5mm",
        "Within\n20mm",
    ]
    cell_text = []
    for row in stats:
        cell_text.append(
            [
                row["Algorithm"],
                f"{row['Mean (mm)']:.2f}",
                f"{row['Median (mm)']:.2f}",
                f"{row['P75 (mm)']:.2f}",
                f"{row['P90 (mm)']:.2f}",
                f"{row['P95 (mm)']:.2f}",
                f"{row['Max (mm)']:.2f}",
                f"{row['Within 5mm']:.1f}%",
                f"{row['Within 20mm']:.1f}%",
            ]
        )

    fig, ax = plt.subplots(figsize=(18.0, 7.2))
    ax.axis("off")
    title = "Final Distance Error Comparison Table"
    subtitle = f"{robot_name} {task} target, {disturbance} disturbance. Lower final distance is better; final precision threshold is 5 mm."
    fig.text(0.035, 0.91, title, fontsize=25, weight="bold", ha="left", color="#202124")
    fig.text(0.035, 0.845, subtitle, fontsize=14, ha="left", color="#5f6368")

    table = ax.table(
        cellText=cell_text,
        colLabels=columns,
        cellLoc="center",
        colLoc="center",
        colWidths=[0.12, 0.11, 0.12, 0.105, 0.105, 0.105, 0.105, 0.12, 0.13],
        bbox=[0.025, 0.17, 0.95, 0.54],
    )
    table.auto_set_font_size(False)
    table.set_fontsize(12.5)
    table.scale(1, 2.0)
    for (r, c), cell in table.get_celld().items():
        cell.set_edgecolor("#d0d4dc")
        if r == 0:
            cell.set_facecolor("#1f2430")
            cell.set_text_props(color="white", weight="bold")
        else:
            color = ROW_COLORS[(r - 1) % len(ROW_COLORS)]
            if cell_text[r - 1][0] == "Ours":
                color = "#d7f2d8"
                cell.set_text_props(weight="bold", color="#103d1c")
            cell.set_facecolor(color)
    fig.savefig(out_path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def plot_projection_grid(
    *,
    cfg: dict[str, Any],
    model_dir: Path,
    algos: list[str],
    task: str,
    disturbance: str,
    projection_episodes: int,
    robot_name: str,
    out_path: Path,
) -> None:
    import matplotlib.pyplot as plt
    from matplotlib.patches import Circle

    fixed_goal = task == "fixed"
    plot_cfg = deep_update(cfg, {"env": {"fixed_goal": fixed_goal, "terminate_on_success": False}})
    if disturbance == "off":
        plot_cfg = deep_update(plot_cfg, {"disturbance": {"enabled": False}})
    else:
        strength = cfg.get("disturbance", {}).get("strengths", {}).get(disturbance, {})
        plot_cfg = deep_update(plot_cfg, {"disturbance": {"enabled": True, **strength}})

    traces: list[tuple[str, dict[str, np.ndarray]]] = []
    for algo in algos:
        model_path = model_dir / f"{algo}_{task}.zip"
        if not model_path.exists():
            continue
        trace = find_best_projection_trace(plot_cfg, algo, model_path, projection_episodes)
        traces.append((algo, trace))

    if not traces:
        return

    xs = np.concatenate([trace["ee_pos"][:, 0] for _algo, trace in traces])
    zs = np.concatenate([trace["ee_pos"][:, 2] for _algo, trace in traces])
    goals = np.stack([trace["goals"][-1] for _algo, trace in traces], axis=0)
    x_min = float(min(xs.min(), goals[:, 0].min()) - 0.035)
    x_max = float(max(xs.max(), goals[:, 0].max()) + 0.035)
    z_min = float(min(zs.min(), goals[:, 2].min()) - 0.035)
    z_max = float(max(zs.max(), goals[:, 2].max()) + 0.035)

    ncols = 3
    nrows = int(np.ceil(len(traces) / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(15.5, 8.8), constrained_layout=True)
    axes_arr = np.atleast_1d(axes).reshape(nrows, ncols)
    palette = ["#4C72B0", "#55A868", "#DD8452", "#C44E52", "#8172B2", "#2A9D8F"]

    for idx, ax in enumerate(axes_arr.ravel()):
        if idx >= len(traces):
            ax.axis("off")
            continue
        algo, trace = traces[idx]
        ee = trace["ee_pos"]
        goal = trace["goals"][-1]
        final_error = float(np.linalg.norm(ee[-1] - goal) * 1000.0)
        color = palette[idx % len(palette)]
        ax.plot(ee[:, 0], ee[:, 2], color=color, linewidth=2.0, label="Trajectory")
        ax.scatter([ee[0, 0]], [ee[0, 2]], marker="s", s=52, color=color, alpha=0.85, label="Start")
        ax.scatter([ee[-1, 0]], [ee[-1, 2]], marker="x", s=72, color="black", linewidth=2.0, label="End")
        ax.scatter([goal[0]], [goal[2]], marker="*", s=170, color="#3cb44b", label="Goal", zorder=5)
        ax.add_patch(Circle((goal[0], goal[2]), 0.02, fill=False, linestyle="--", linewidth=1.2, color="#ff7f0e", label="20 mm"))
        ax.add_patch(Circle((goal[0], goal[2]), 0.005, fill=False, linestyle="--", linewidth=1.2, color="#d62728", label="5 mm"))
        ax.set_title(f"{DISPLAY_NAMES.get(algo, algo)} (final: {final_error:.1f} mm)", fontsize=12)
        ax.set_xlabel("X (m)")
        ax.set_ylabel("Z (m)")
        ax.set_xlim(x_min, x_max)
        ax.set_ylim(z_min, z_max)
        ax.set_aspect("equal", adjustable="box")
        ax.grid(True, alpha=0.25)
        ax.legend(fontsize=8, loc="upper right")

    fig.suptitle(
        f"{robot_name} End-Effector XZ Projection ({task.title()} Target, {disturbance.title()} Disturbance)",
        fontsize=18,
    )
    fig.savefig(out_path, dpi=180)
    plt.close(fig)


def find_best_projection_trace(
    cfg: dict[str, Any],
    algo: str,
    model_path: Path,
    projection_episodes: int,
) -> dict[str, np.ndarray]:
    best_trace: dict[str, np.ndarray] | None = None
    best_error = float("inf")
    base_seed = int(cfg.get("seed", 0)) + 7000
    for episode in range(max(1, projection_episodes)):
        trace, _frame = rollout_policy(
            cfg,
            algo,
            model_path,
            render_mode=None,
            reset_seed=base_seed + episode,
        )
        ee = trace["ee_pos"]
        goal = trace["goals"][-1]
        final_error = float(np.linalg.norm(ee[-1] - goal))
        if final_error < best_error:
            best_error = final_error
            best_trace = trace
    if best_trace is None:
        raise RuntimeError(f"No rollout trace was produced for {algo}.")
    return best_trace


def plot_dashboard(
    rows_by_algo: dict[str, list[dict[str, Any]]],
    out_path: Path,
    *,
    task: str,
    disturbance: str,
    robot_name: str,
) -> None:
    import matplotlib.pyplot as plt

    algos = list(rows_by_algo.keys())
    labels = [DISPLAY_NAMES.get(algo, algo) for algo in algos]
    errors = [np.asarray([float(row["final_error_m"]) * 1000.0 for row in rows_by_algo[algo]], dtype=float) for algo in algos]
    success = [float(np.mean(err <= 5.0) * 100.0) for err in errors]
    ee_jerk = [float(np.mean([float(row.get("ee_jerk_rms", 0.0)) for row in rows_by_algo[algo]])) for algo in algos]
    rewards = [float(np.mean([float(row["reward_sum"]) for row in rows_by_algo[algo]])) for algo in algos]
    colors = ["#4C72B0", "#55A868", "#DD8452", "#C44E52", "#8172B2", "#2A9D8F"][: len(algos)]

    fig = plt.figure(figsize=(15.5, 9.2), constrained_layout=True)
    gs = fig.add_gridspec(2, 2)
    ax_violin = fig.add_subplot(gs[0, :])
    ax_success = fig.add_subplot(gs[1, 0])
    ax_stability = fig.add_subplot(gs[1, 1])

    parts = ax_violin.violinplot(errors, showmeans=True, showmedians=True, widths=0.75)
    for body, color in zip(parts["bodies"], colors):
        body.set_facecolor(color)
        body.set_alpha(0.45)
        body.set_edgecolor(color)
    for key in ("cmeans", "cmedians", "cbars", "cmins", "cmaxes"):
        parts[key].set_color("#202124")
        parts[key].set_linewidth(1.0)
    ax_violin.axhline(5.0, color="#d62728", linestyle="--", linewidth=1.4, label="5 mm threshold")
    ax_violin.axhline(20.0, color="#ff7f0e", linestyle="--", linewidth=1.2, label="20 mm threshold")
    ax_violin.set_xticks(np.arange(1, len(labels) + 1), labels)
    ax_violin.set_ylabel("Final error (mm)")
    ax_violin.set_title("Final Distance Distribution")
    ax_violin.grid(axis="y", alpha=0.25)
    ax_violin.legend(loc="upper right")

    x = np.arange(len(labels))
    bars = ax_success.bar(x, success, color=colors, alpha=0.85)
    ax_success.set_xticks(x, labels, rotation=20, ha="right")
    ax_success.set_ylim(0, 108)
    ax_success.set_ylabel("Success@5mm (%)")
    ax_success.set_title("5 mm Success Rate")
    ax_success.grid(axis="y", alpha=0.25)
    for bar, value in zip(bars, success):
        ax_success.text(bar.get_x() + bar.get_width() / 2, value + 2, f"{value:.0f}%", ha="center", fontsize=10)

    bars = ax_stability.bar(x, ee_jerk, color=colors, alpha=0.85)
    ax_stability.set_xticks(x, labels, rotation=20, ha="right")
    ax_stability.set_ylabel("EE jerk RMS")
    ax_stability.set_title("Contact-Disturbance Stability")
    ax_stability.grid(axis="y", alpha=0.25)
    for bar, value in zip(bars, ee_jerk):
        ax_stability.text(bar.get_x() + bar.get_width() / 2, value, f"{value:.2g}", ha="center", va="bottom", fontsize=9)

    fig.suptitle(
        f"{robot_name} {task.title()} Target Result Overview ({disturbance.title()} Disturbance)",
        fontsize=18,
        weight="bold",
    )
    fig.savefig(out_path, dpi=180)
    plt.close(fig)


def read_cached_rows(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    with path.open("r", newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def write_rows(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fields = sorted({key for row in rows for key in row})
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


if __name__ == "__main__":
    main()
