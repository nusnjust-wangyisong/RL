from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np


@dataclass
class EpisodeTrace:
    ee_pos: list[np.ndarray] = field(default_factory=list)
    ee_vel: list[np.ndarray] = field(default_factory=list)
    joint_pos: list[np.ndarray] = field(default_factory=list)
    joint_vel: list[np.ndarray] = field(default_factory=list)
    actions: list[np.ndarray] = field(default_factory=list)
    goals: list[np.ndarray] = field(default_factory=list)
    rewards: list[float] = field(default_factory=list)
    distances: list[float] = field(default_factory=list)
    disturbances: list[np.ndarray] = field(default_factory=list)

    def append(
        self,
        *,
        ee_pos: np.ndarray | None = None,
        ee_vel: np.ndarray | None = None,
        joint_pos: np.ndarray | None = None,
        joint_vel: np.ndarray | None = None,
        action: np.ndarray | None = None,
        goal: np.ndarray | None = None,
        reward: float | None = None,
        distance: float | None = None,
        disturbance: np.ndarray | None = None,
    ) -> None:
        if ee_pos is not None:
            self.ee_pos.append(np.asarray(ee_pos, dtype=float).copy())
        if ee_vel is not None:
            self.ee_vel.append(np.asarray(ee_vel, dtype=float).copy())
        if joint_pos is not None:
            self.joint_pos.append(np.asarray(joint_pos, dtype=float).copy())
        if joint_vel is not None:
            self.joint_vel.append(np.asarray(joint_vel, dtype=float).copy())
        if action is not None:
            self.actions.append(np.asarray(action, dtype=float).copy())
        if goal is not None:
            self.goals.append(np.asarray(goal, dtype=float).copy())
        if reward is not None:
            self.rewards.append(float(reward))
        if distance is not None:
            self.distances.append(float(distance))
        if disturbance is not None:
            self.disturbances.append(np.asarray(disturbance, dtype=float).copy())

    def arrays(self) -> dict[str, np.ndarray]:
        return {
            "ee_pos": _stack(self.ee_pos),
            "ee_vel": _stack(self.ee_vel),
            "joint_pos": _stack(self.joint_pos),
            "joint_vel": _stack(self.joint_vel),
            "actions": _stack(self.actions),
            "goals": _stack(self.goals),
            "rewards": np.asarray(self.rewards, dtype=float),
            "distances": np.asarray(self.distances, dtype=float),
            "disturbances": _stack(self.disturbances),
        }


def _stack(values: list[np.ndarray]) -> np.ndarray:
    if not values:
        return np.empty((0,), dtype=float)
    return np.stack(values, axis=0)


def finite_difference(values: np.ndarray, dt: float) -> np.ndarray:
    values = np.asarray(values, dtype=float)
    if values.shape[0] < 2:
        return np.zeros_like(values)
    return np.gradient(values, dt, axis=0)


def rms(values: np.ndarray) -> float:
    values = np.asarray(values, dtype=float)
    if values.size == 0:
        return 0.0
    return float(np.sqrt(np.mean(np.square(values))))


def peak(values: np.ndarray) -> float:
    values = np.asarray(values, dtype=float)
    if values.size == 0:
        return 0.0
    return float(np.max(np.linalg.norm(np.atleast_2d(values), axis=-1)))


def peak_to_peak(values: np.ndarray) -> float:
    values = np.asarray(values, dtype=float)
    if values.size == 0:
        return 0.0
    if values.ndim == 1:
        return float(np.ptp(values))
    return float(np.max(np.ptp(values, axis=0)))


def action_delta_mean(actions: np.ndarray) -> float:
    actions = np.asarray(actions, dtype=float)
    if actions.shape[0] < 2:
        return 0.0
    return float(np.mean(np.linalg.norm(np.diff(actions, axis=0), axis=1)))


def summarize_episode(
    trace: EpisodeTrace | dict[str, np.ndarray],
    *,
    dt: float,
    thresholds_m: list[float],
    hold_threshold_m: float = 0.005,
    hold_window: int = 10,
) -> dict[str, Any]:
    data = trace.arrays() if isinstance(trace, EpisodeTrace) else trace
    distances = np.asarray(data.get("distances", []), dtype=float)
    ee_pos = np.asarray(data.get("ee_pos", []), dtype=float)
    ee_vel = np.asarray(data.get("ee_vel", []), dtype=float)
    joint_vel = np.asarray(data.get("joint_vel", []), dtype=float)
    actions = np.asarray(data.get("actions", []), dtype=float)

    if distances.size == 0 and ee_pos.size and np.asarray(data.get("goals", [])).size:
        goals = np.asarray(data["goals"], dtype=float)
        distances = np.linalg.norm(ee_pos - goals, axis=1)

    ee_acc = finite_difference(ee_vel, dt) if ee_vel.size else finite_difference(ee_pos, dt)
    ee_jerk = finite_difference(ee_acc, dt)
    joint_acc = finite_difference(joint_vel, dt) if joint_vel.size else np.empty((0,), dtype=float)
    final_error = float(distances[-1]) if distances.size else float("nan")
    min_error = float(np.min(distances)) if distances.size else float("nan")
    hold_ee_acc = _tail_window(ee_acc, hold_window)
    hold_ee_jerk = _tail_window(ee_jerk, hold_window)
    hold_joint_acc = _tail_window(joint_acc, hold_window)
    hold_actions = _tail_window(actions, hold_window)
    miss_penalty = max(final_error - hold_threshold_m, 0.0) if np.isfinite(final_error) else 1.0
    precision_penalty = 100.0 * miss_penalty

    # 目标邻域误差波动：末端稳定保持阶段（末 hold_window 步）定位误差的标准差，
    # 隔离收敛沉降段、只刻画“稳定保持”时的抖动幅度（与 hold_5mm 口径一致、互补）。
    hold_distances = _tail_window(distances, hold_window)
    neighborhood_error_std = float(np.std(hold_distances)) if hold_distances.size else float("nan")

    summary: dict[str, Any] = {
        "final_error_m": final_error,
        "min_error_m": min_error,
        "mean_error_m": float(np.mean(distances)) if distances.size else float("nan"),
        "neighborhood_error_std_m": neighborhood_error_std,
        "reward_sum": float(np.sum(data.get("rewards", []))),
        "ee_acc_rms": rms(np.linalg.norm(np.atleast_2d(ee_acc), axis=1)) if ee_acc.size else 0.0,
        "ee_acc_peak": peak(ee_acc),
        "ee_acc_p2p": peak_to_peak(ee_acc),
        "ee_jerk_rms": rms(np.linalg.norm(np.atleast_2d(ee_jerk), axis=1)) if ee_jerk.size else 0.0,
        "joint_acc_rms": rms(joint_acc),
        "action_delta_mean": action_delta_mean(actions),
        "vibration_index": vibration_index(ee_acc),
        "hold_5mm": hold_rate(distances, hold_threshold_m, hold_window),
        "hold_ee_acc_rms": rms(np.linalg.norm(np.atleast_2d(hold_ee_acc), axis=1)) if hold_ee_acc.size else 0.0,
        "hold_ee_acc_p2p": peak_to_peak(hold_ee_acc),
        "hold_ee_jerk_rms": rms(np.linalg.norm(np.atleast_2d(hold_ee_jerk), axis=1)) if hold_ee_jerk.size else 0.0,
        "hold_joint_acc_rms": rms(hold_joint_acc),
        "hold_action_delta_mean": action_delta_mean(hold_actions),
        "hold_vibration_index": vibration_index(hold_ee_acc),
    }
    summary.update(
        {
            "qualified_ee_acc_rms": summary["hold_ee_acc_rms"] + precision_penalty,
            "qualified_ee_acc_p2p": summary["hold_ee_acc_p2p"] + precision_penalty,
            "qualified_ee_jerk_rms": summary["hold_ee_jerk_rms"] + precision_penalty,
            "qualified_joint_acc_rms": summary["hold_joint_acc_rms"] + precision_penalty,
            "qualified_action_delta_mean": summary["hold_action_delta_mean"] + precision_penalty,
            "qualified_vibration_index": summary["hold_vibration_index"] + precision_penalty,
        }
    )

    for threshold in thresholds_m:
        key = f"success_{int(round(threshold * 1000))}mm"
        summary[key] = float(final_error <= threshold) if np.isfinite(final_error) else 0.0
    return summary


def _tail_window(values: np.ndarray, window: int) -> np.ndarray:
    values = np.asarray(values, dtype=float)
    if values.size == 0:
        return values
    return values[-max(int(window), 1) :]


def hold_rate(distances: np.ndarray, threshold: float, window: int) -> float:
    distances = np.asarray(distances, dtype=float)
    if distances.size == 0:
        return 0.0
    if distances.size < window:
        return float(np.mean(distances <= threshold))
    tail = distances[-window:]
    return float(np.mean(tail <= threshold))


def vibration_index(ee_acc: np.ndarray) -> float:
    ee_acc = np.asarray(ee_acc, dtype=float)
    if ee_acc.size == 0:
        return 0.0
    if ee_acc.ndim == 1:
        centered = ee_acc - np.mean(ee_acc)
    else:
        centered = ee_acc - np.mean(ee_acc, axis=0, keepdims=True)
    return rms(centered)


def aggregate_summaries(rows: list[dict[str, Any]]) -> dict[str, float]:
    if not rows:
        return {}
    keys = sorted({k for row in rows for k, v in row.items() if isinstance(v, (int, float, np.floating))})
    out: dict[str, float] = {}
    for key in keys:
        values = np.asarray([row[key] for row in rows if key in row], dtype=float)
        values = values[np.isfinite(values)]
        if values.size:
            out[f"{key}_mean"] = float(np.mean(values))
            out[f"{key}_std"] = float(np.std(values))
    return out
