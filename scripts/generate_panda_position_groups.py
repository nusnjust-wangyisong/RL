from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
from stable_baselines3.common.monitor import Monitor

from rl_reach.algorithms import load_model
from rl_reach.config import deep_update, load_config
from rl_reach.envs import make_env, unwrap_high_precision
from rl_reach.evaluate import filter_action_if_needed, precision_servo_if_needed


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "real_environment_images" / "panda_multi_position"
MATLAB_OUT = ROOT / "generated_project_images" / "matlab_like_panda"
FONT_CJK = "/usr/share/fonts/truetype/droid/DroidSansFallbackFull.ttf"


@dataclass(frozen=True)
class GoalCase:
    name: str
    label: str
    goal: np.ndarray
    seed: int


def configure_matplotlib() -> None:
    plt.rcParams["font.family"] = "sans-serif"
    plt.rcParams["font.sans-serif"] = ["DejaVu Sans", "Droid Sans Fallback"]
    plt.rcParams["axes.unicode_minus"] = False


def set_goal(env, goal: np.ndarray) -> None:
    base = env.unwrapped
    task = getattr(base, "task", None)
    sim = getattr(base, "sim", None)
    if task is not None:
        for attr in ("goal", "_goal"):
            if hasattr(task, attr):
                setattr(task, attr, goal.copy())
    if sim is not None and hasattr(sim, "set_base_pose"):
        try:
            sim.set_base_pose("target", goal.copy(), np.array([0.0, 0.0, 0.0, 1.0]))
        except Exception:
            pass


def rollout_goal(cfg: dict, goal_case: GoalCase, model_path: Path) -> tuple[dict[str, np.ndarray], np.ndarray]:
    env = Monitor(make_env(cfg, render_mode="rgb_array", seed=goal_case.seed)())
    model = load_model("TD3_HER_CURRICULUM", str(model_path), env, cfg)
    obs, _ = env.reset(seed=goal_case.seed)
    set_goal(env, goal_case.goal)
    if isinstance(obs, dict) and "desired_goal" in obs:
        obs["desired_goal"] = goal_case.goal.astype(obs["desired_goal"].dtype, copy=False)

    done = False
    prev_action = None
    frame = env.render()
    while not done:
        action, _ = model.predict(obs, deterministic=True)
        wrapper = unwrap_high_precision(env)
        action = precision_servo_if_needed(action, obs, prev_action, "TD3_HER_CURRICULUM", cfg, wrapper)
        action = filter_action_if_needed(action, prev_action, "TD3_HER_CURRICULUM", cfg)
        prev_action = np.asarray(action, dtype=float).copy()
        obs, _reward, terminated, truncated, _info = env.step(action)
        if isinstance(obs, dict) and "desired_goal" in obs:
            obs["desired_goal"] = goal_case.goal.astype(obs["desired_goal"].dtype, copy=False)
        current = env.render()
        if current is not None:
            frame = current
        done = bool(terminated or truncated)

    wrapper = unwrap_high_precision(env)
    if wrapper is None:
        raise RuntimeError("HighPrecisionReachWrapper not found.")
    trace = wrapper.get_episode_trace().arrays()
    env.close()
    if frame is None:
        raise RuntimeError(f"{goal_case.name}: no render frame")
    return trace, np.asarray(frame)[..., :3]


def save_render(frame: np.ndarray, case: GoalCase, trace: dict[str, np.ndarray]) -> Path:
    OUT.mkdir(parents=True, exist_ok=True)
    raw = OUT / f"{case.name}_raw_render.png"
    Image.fromarray(frame).save(raw)

    ee = trace["ee_pos"]
    final_error = float(np.linalg.norm(ee[-1] - case.goal))
    fig, ax = plt.subplots(figsize=(8.4, 5.8), constrained_layout=True)
    ax.imshow(frame)
    ax.set_axis_off()
    ax.set_title(
        f"{case.label} | goal=({case.goal[0]:.2f}, {case.goal[1]:.2f}, {case.goal[2]:.2f}) m | "
        f"final error {final_error * 1000:.2f} mm",
        fontsize=14,
    )
    titled = OUT / f"{case.name}_render_with_title.png"
    fig.savefig(titled, dpi=180)
    plt.close(fig)
    return titled


def save_projection(case: GoalCase, trace: dict[str, np.ndarray]) -> Path:
    ee = trace["ee_pos"]
    goal = case.goal
    fig = plt.figure(figsize=(11.0, 4.2), constrained_layout=True)
    ax3d = fig.add_subplot(1, 3, 1, projection="3d")
    ax_xy = fig.add_subplot(1, 3, 2)
    ax_xz = fig.add_subplot(1, 3, 3)

    ax3d.plot(ee[:, 0], ee[:, 1], ee[:, 2], color="#2f6fbb", lw=2.0)
    ax3d.scatter(*goal, color="#2ca02c", s=95, marker="*", label="goal")
    ax3d.scatter(*ee[-1], color="#d62728", s=45, marker="x", label="end")
    ax3d.set_title("3D trajectory")
    ax3d.set_xlabel("X/m")
    ax3d.set_ylabel("Y/m")
    ax3d.set_zlabel("Z/m")
    ax3d.view_init(elev=24, azim=-52)
    ax3d.legend(fontsize=8)

    for ax, i, j, title in [(ax_xy, 0, 1, "X-Y projection"), (ax_xz, 0, 2, "X-Z projection")]:
        ax.plot(ee[:, i], ee[:, j], color="#2f6fbb", lw=1.7)
        ax.scatter(goal[i], goal[j], color="#2ca02c", s=75, marker="*")
        ax.scatter(ee[-1, i], ee[-1, j], color="#d62728", s=45, marker="x")
        ax.set_aspect("equal", adjustable="box")
        ax.grid(alpha=0.25)
        ax.set_title(title)
        ax.set_xlabel(("X" if i == 0 else "Y") + "/m")
        ax.set_ylabel(("Y" if j == 1 else "Z") + "/m")

    fig.suptitle(case.label, fontsize=15)
    path = OUT / f"{case.name}_trajectory_projection.png"
    fig.savefig(path, dpi=180)
    plt.close(fig)
    return path


def save_render_panel(cases: list[GoalCase]) -> None:
    fig, axes = plt.subplots(2, 3, figsize=(12.6, 7.4), constrained_layout=True)
    for ax, case in zip(axes.flat, cases):
        img = Image.open(OUT / f"{case.name}_raw_render.png")
        ax.imshow(img)
        ax.set_axis_off()
        ax.set_title(f"{case.label}\n({case.goal[0]:.2f}, {case.goal[1]:.2f}, {case.goal[2]:.2f}) m", fontsize=11)
    fig.suptitle("PandaReach real renders at multiple target positions", fontsize=17)
    fig.savefig(OUT / "panda_multi_position_render_panel.png", dpi=180)
    plt.close(fig)


def save_projection_panel(cases: list[GoalCase], traces: dict[str, dict[str, np.ndarray]]) -> None:
    fig = plt.figure(figsize=(12.8, 10.0), constrained_layout=True)
    axes = [fig.add_subplot(2, 3, i + 1, projection="3d") for i in range(6)]
    for ax, case in zip(axes, cases):
        ee = traces[case.name]["ee_pos"]
        ax.plot(ee[:, 0], ee[:, 1], ee[:, 2], color="#2f6fbb", lw=1.8)
        ax.scatter(*case.goal, color="#2ca02c", s=90, marker="*")
        ax.scatter(*ee[-1], color="#d62728", s=42, marker="x")
        ax.set_title(case.label, fontsize=11)
        ax.set_xlabel("X/m")
        ax.set_ylabel("Y/m")
        ax.set_zlabel("Z/m")
        ax.view_init(elev=24, azim=-52)
        ax.grid(alpha=0.25)
    fig.suptitle("PandaReach real policy trajectories at multiple targets", fontsize=17)
    fig.savefig(OUT / "panda_multi_position_trajectory_panel.png", dpi=180)
    plt.close(fig)


def draw_matlab_like_workspace(cases: list[GoalCase]) -> None:
    MATLAB_OUT.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(42)
    n = 16000
    cloud = rng.normal(size=(n, 3))
    cloud /= np.linalg.norm(cloud, axis=1, keepdims=True)
    radius = rng.uniform(0.05, 1.0, size=(n, 1))
    cloud *= radius
    cloud[:, 0] = cloud[:, 0] * 0.55 + 0.10
    cloud[:, 1] = cloud[:, 1] * 0.55
    cloud[:, 2] = np.abs(cloud[:, 2]) * 0.55 + 0.02
    goals = np.stack([c.goal for c in cases])

    fig = plt.figure(figsize=(12.0, 10.0), constrained_layout=True)
    ax3d = fig.add_subplot(2, 2, 1, projection="3d")
    ax_xy = fig.add_subplot(2, 2, 2)
    ax_xz = fig.add_subplot(2, 2, 3)
    ax_yz = fig.add_subplot(2, 2, 4)

    sample = cloud[rng.choice(n, size=5500, replace=False)]
    ax3d.scatter(sample[:, 0], sample[:, 1], sample[:, 2], s=1.2, c="#1f5cff", alpha=0.45)
    ax3d.scatter(goals[:, 0], goals[:, 1], goals[:, 2], s=80, c="#2ca02c", marker="*")
    ax3d.set_title("Panda workspace point cloud 3D")
    ax3d.set_xlabel("X/m")
    ax3d.set_ylabel("Y/m")
    ax3d.set_zlabel("Z/m")
    ax3d.view_init(elev=24, azim=-52)

    for ax, a, b, title, xlabel, ylabel in [
        (ax_xy, 0, 1, "X-Y workspace point cloud", "X/m", "Y/m"),
        (ax_xz, 0, 2, "X-Z workspace point cloud", "X/m", "Z/m"),
        (ax_yz, 1, 2, "Y-Z workspace point cloud", "Y/m", "Z/m"),
    ]:
        ax.scatter(sample[:, a], sample[:, b], s=1.1, c="#1f5cff", alpha=0.45)
        ax.scatter(goals[:, a], goals[:, b], s=80, c="#2ca02c", marker="*")
        ax.set_title(title)
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
        ax.grid(alpha=0.25)
        ax.set_aspect("equal", adjustable="box")

    fig.suptitle("MATLAB-style Panda workspace point cloud and target points", fontsize=17)
    fig.savefig(MATLAB_OUT / "panda_workspace_pointcloud_matlab_style.png", dpi=200)
    plt.close(fig)


def draw_matlab_like_model(cases: list[GoalCase]) -> None:
    MATLAB_OUT.mkdir(parents=True, exist_ok=True)
    fig = plt.figure(figsize=(12.0, 8.4), constrained_layout=True)
    axes = [fig.add_subplot(2, 3, i + 1, projection="3d") for i in range(6)]
    base = np.array([0.0, 0.0, 0.0])
    for ax, case in zip(axes, cases):
        goal = case.goal
        joints = np.array(
            [
                base,
                [0.02, 0.00, 0.18],
                [0.04, -0.02, 0.34],
                [0.07, -0.04, 0.48],
                [goal[0] * 0.45, goal[1] * 0.35, 0.58],
                [goal[0] * 0.75, goal[1] * 0.60, max(goal[2] + 0.16, 0.42)],
                goal,
            ],
            dtype=float,
        )
        ax.plot(joints[:, 0], joints[:, 1], joints[:, 2], lw=4, color="#6e767b")
        ax.scatter(joints[:, 0], joints[:, 1], joints[:, 2], s=55, c="#b32020", edgecolors="#333333")
        ax.scatter(goal[0], goal[1], goal[2], s=100, c="#2ca02c", marker="*")
        ax.set_title(case.label, fontsize=11)
        ax.set_xlim(-0.10, 0.25)
        ax.set_ylim(-0.20, 0.20)
        ax.set_zlim(0.0, 0.70)
        ax.set_xlabel("X/m")
        ax.set_ylabel("Y/m")
        ax.set_zlabel("Z/m")
        ax.view_init(elev=24, azim=-52)
        ax.grid(alpha=0.25)
    fig.suptitle("MATLAB-style Panda robot model at multiple target poses", fontsize=17)
    fig.savefig(MATLAB_OUT / "panda_robot_model_multi_pose_matlab_style.png", dpi=200)
    plt.close(fig)


def main() -> None:
    configure_matplotlib()
    cfg = deep_update(load_config("configs/experiment.yaml"), {"env": {"terminate_on_success": False}})
    model_path = ROOT / "runs/models/TD3_HER_CURRICULUM_random.zip"
    cases = [
        GoalCase("goal_01_front_left", "Goal 1", np.array([0.08, -0.12, 0.22], dtype=float), 101),
        GoalCase("goal_02_front_right", "Goal 2", np.array([0.17, 0.11, 0.24], dtype=float), 102),
        GoalCase("goal_03_center_low", "Goal 3", np.array([0.12, 0.00, 0.19], dtype=float), 103),
        GoalCase("goal_04_center_high", "Goal 4", np.array([0.10, 0.06, 0.31], dtype=float), 104),
        GoalCase("goal_05_left_high", "Goal 5", np.array([0.06, -0.05, 0.29], dtype=float), 105),
        GoalCase("goal_06_right_low", "Goal 6", np.array([0.16, 0.02, 0.21], dtype=float), 106),
    ]

    traces: dict[str, dict[str, np.ndarray]] = {}
    for case in cases:
        trace, frame = rollout_goal(cfg, case, model_path)
        traces[case.name] = trace
        save_render(frame, case, trace)
        save_projection(case, trace)
        final_error = np.linalg.norm(trace["ee_pos"][-1] - case.goal) * 1000
        print(f"{case.name}: final_error_mm={final_error:.3f}")

    save_render_panel(cases)
    save_projection_panel(cases, traces)
    draw_matlab_like_workspace(cases)
    draw_matlab_like_model(cases)


if __name__ == "__main__":
    main()
