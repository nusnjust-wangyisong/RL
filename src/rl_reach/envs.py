from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any

import numpy as np

from rl_reach.metrics import EpisodeTrace, vibration_index

try:  # Keep metric-only imports usable before optional simulation deps are installed.
    import gymnasium as gym
except Exception:  # pragma: no cover - dependency guard
    gym = None


@dataclass
class DisturbanceState:
    force: np.ndarray
    active: bool


class _FallbackWrapper:
    def __init__(self, env: Any):
        self.env = env

    @property
    def unwrapped(self) -> Any:
        return getattr(self.env, "unwrapped", self.env)

    def __getattr__(self, name: str) -> Any:
        return getattr(self.env, name)


class HighPrecisionReachWrapper(gym.Wrapper if gym is not None else _FallbackWrapper):
    """GoalEnv wrapper adding precision thresholds, stability rewards, and contact-like force.

    The wrapper intentionally uses duck typing so it remains compatible with minor
    panda-gym API differences across versions.
    """

    def __init__(self, env: Any, cfg: dict[str, Any]):
        if gym is None:  # pragma: no cover - dependency guard
            raise RuntimeError("gymnasium is required to create the Reach wrapper.")
        super().__init__(env)
        self.cfg = cfg
        self.env_cfg = cfg.get("env", {})
        self.reward_cfg = cfg.get("reward", {})
        self.disturbance_cfg = cfg.get("disturbance", {})
        self.thresholds_m = list(self.env_cfg.get("thresholds_m", [0.05, 0.03, 0.02, 0.01, 0.005]))
        self.target_threshold_m = float(self.env_cfg.get("target_threshold_m", 0.005))
        self.goal_range_scale = 1.0
        self.step_count = 0
        self.dt = self._resolve_dt()
        self.trace = EpisodeTrace()
        self._prev_action: np.ndarray | None = None
        self._prev_ee_vel: np.ndarray | None = None
        self._prev_ee_acc: np.ndarray | None = None
        self._prev_joint_vel: np.ndarray | None = None
        self._rng = np.random.default_rng(cfg.get("seed", None))
        self._handles = self._resolve_pybullet_handles()
        self.set_precision_threshold(self.target_threshold_m)

    def reset(self, **kwargs: Any):  # type: ignore[override]
        self.step_count = 0
        self.trace = EpisodeTrace()
        self._prev_action = None
        self._prev_ee_vel = None
        self._prev_ee_acc = None
        self._prev_joint_vel = None
        obs, info = self.env.reset(**kwargs)
        self._maybe_set_fixed_goal(obs)
        self._maybe_scale_goal(obs)
        self._record(obs, action=None, reward=0.0, disturbance=np.zeros(3))
        return obs, info

    def step(self, action: np.ndarray):  # type: ignore[override]
        self.step_count += 1
        action = np.asarray(action, dtype=float)

        if self.reward_cfg.get("precision_servo", False):
            action = self._precision_servo_action(action)

        disturbance = self._compute_disturbance()
        if disturbance.active:
            self._apply_external_force(disturbance.force)

        obs, reward, terminated, truncated, info = self.env.step(action)
        distance = self._distance(obs)
        stability_penalty = self._stability_penalty(action)
        shaped_reward = self._shape_reward(float(reward), distance, stability_penalty)

        info = dict(info)
        info["distance_m"] = distance
        info["is_success_5mm"] = bool(distance <= 0.005)
        info["is_success"] = bool(distance <= self.target_threshold_m)
        info["stability_penalty"] = stability_penalty
        info["disturbance_force"] = disturbance.force
        info["disturbance_active"] = disturbance.active
        if not self.env_cfg.get("terminate_on_success", True) and terminated:
            terminated = False

        self._record(obs, action=action, reward=shaped_reward, disturbance=disturbance.force)
        self._prev_action = action.copy()
        return obs, shaped_reward, terminated, truncated, info

    def compute_reward(self, achieved_goal: np.ndarray, desired_goal: np.ndarray, info: Any) -> np.ndarray:
        distance = np.linalg.norm(np.asarray(achieved_goal) - np.asarray(desired_goal), axis=-1)
        if self.reward_cfg.get("dense", True):
            reward = -float(self.reward_cfg.get("reach_weight", 1.0)) * distance
            reward = np.asarray(reward)
        else:
            reward = -(distance > self.target_threshold_m).astype(float)
        reward = reward + float(self.reward_cfg.get("success_bonus", 0.0)) * (distance <= self.target_threshold_m)
        return reward

    def set_precision_threshold(self, threshold_m: float) -> None:
        self.target_threshold_m = float(threshold_m)
        task = getattr(self.unwrapped, "task", None)
        if task is not None and hasattr(task, "distance_threshold"):
            task.distance_threshold = float(threshold_m)

    def set_goal_range_scale(self, scale: float) -> None:
        self.goal_range_scale = float(scale)

    def get_episode_trace(self) -> EpisodeTrace:
        return self.trace

    def _shape_reward(self, env_reward: float, distance: float, stability_penalty: float) -> float:
        if self.reward_cfg.get("dense", True):
            reward = -float(self.reward_cfg.get("reach_weight", 1.0)) * distance
        else:
            reward = env_reward
        if distance <= self.target_threshold_m:
            reward += float(self.reward_cfg.get("success_bonus", 0.0))
        return float(reward - stability_penalty)

    def _stability_penalty(self, action: np.ndarray) -> float:
        ee_vel = self._get_ee_velocity()
        joint_vel = self._get_joint_velocity()
        penalty = 0.0

        if self._prev_action is not None:
            penalty += float(self.reward_cfg.get("action_smooth_weight", 0.0)) * float(
                np.sum(np.square(action - self._prev_action))
            )

        if self._prev_joint_vel is not None and joint_vel.size:
            joint_acc = (joint_vel - self._prev_joint_vel) / max(self.dt, 1e-6)
            penalty += float(self.reward_cfg.get("joint_acc_weight", 0.0)) * float(np.sum(np.square(joint_acc)))
        if joint_vel.size:
            self._prev_joint_vel = joint_vel.copy()

        if self._prev_ee_vel is not None and ee_vel.size:
            ee_acc = (ee_vel - self._prev_ee_vel) / max(self.dt, 1e-6)
            if self._prev_ee_acc is not None:
                ee_jerk = (ee_acc - self._prev_ee_acc) / max(self.dt, 1e-6)
                penalty += float(self.reward_cfg.get("ee_jerk_weight", 0.0)) * float(np.sum(np.square(ee_jerk)))
            penalty += float(self.reward_cfg.get("vibration_weight", 0.0)) * vibration_index(ee_acc)
            self._prev_ee_acc = ee_acc.copy()
        if ee_vel.size:
            self._prev_ee_vel = ee_vel.copy()
        return penalty

    def _compute_disturbance(self) -> DisturbanceState:
        if not self.disturbance_cfg.get("enabled", False):
            return DisturbanceState(force=np.zeros(3), active=False)
        if self.step_count < int(self.disturbance_cfg.get("start_after_steps", 0)):
            return DisturbanceState(force=np.zeros(3), active=False)

        distance = self._current_distance()
        activation_radius = float(self.disturbance_cfg.get("activation_radius_m", 0.02))
        active = bool(distance <= activation_radius)
        if not active:
            return DisturbanceState(force=np.zeros(3), active=False)

        direction = np.asarray(self.disturbance_cfg.get("direction", [0.0, 0.0, 1.0]), dtype=float)
        norm = np.linalg.norm(direction)
        direction = direction / norm if norm > 1e-12 else np.array([0.0, 0.0, 1.0])
        t = self.step_count * self.dt
        base = float(self.disturbance_cfg.get("base_force_n", 0.0))
        amplitude = float(self.disturbance_cfg.get("amplitude_n", 5.0))
        frequency = float(self.disturbance_cfg.get("frequency_hz", 20.0))
        noise_std = float(self.disturbance_cfg.get("noise_std_n", 0.0))
        scalar = base + amplitude * math.sin(2.0 * math.pi * frequency * t)
        scalar += float(self._rng.normal(0.0, noise_std))
        return DisturbanceState(force=direction * scalar, active=True)

    def _apply_external_force(self, force: np.ndarray) -> None:
        handles = self._handles
        if not handles:
            return
        client = handles.get("client")
        body_id = handles.get("body_id")
        link_id = handles.get("ee_link_id")
        if client is None or body_id is None or link_id is None:
            return
        try:
            pos = self._get_ee_position()
            client.applyExternalForce(
                objectUniqueId=body_id,
                linkIndex=link_id,
                forceObj=np.asarray(force, dtype=float).tolist(),
                posObj=np.asarray(pos, dtype=float).tolist(),
                flags=client.WORLD_FRAME,
            )
        except Exception:
            return

    def _record(
        self,
        obs: Any,
        *,
        action: np.ndarray | None,
        reward: float,
        disturbance: np.ndarray,
    ) -> None:
        ee_pos = self._get_ee_position(obs)
        ee_vel = self._get_ee_velocity()
        joint_pos = self._get_joint_position()
        joint_vel = self._get_joint_velocity()
        goal = self._desired_goal(obs)
        distance = float(np.linalg.norm(ee_pos - goal)) if ee_pos.size and goal.size else self._distance(obs)
        self.trace.append(
            ee_pos=ee_pos,
            ee_vel=ee_vel,
            joint_pos=joint_pos,
            joint_vel=joint_vel,
            action=action,
            goal=goal,
            reward=reward,
            distance=distance,
            disturbance=disturbance,
        )

    def _precision_servo_action(self, action: np.ndarray) -> np.ndarray:
        distance = self._current_distance()
        radius = float(self.reward_cfg.get("precision_radius_m", 0.02))
        if distance > radius:
            return action
        ee_pos = self._get_ee_position()
        goal = self._current_goal()
        if ee_pos.size < 3 or goal.size < 3 or action.size < 3:
            return action
        servo_gain = float(self.reward_cfg.get("precision_servo_gain", 0.35))
        guided = action.copy()
        guided[:3] = (1.0 - servo_gain) * guided[:3] + servo_gain * np.clip(goal[:3] - ee_pos[:3], -1.0, 1.0)
        return guided

    def _maybe_set_fixed_goal(self, obs: Any) -> None:
        if not self.env_cfg.get("fixed_goal", False):
            return
        goal = np.asarray(self.env_cfg.get("fixed_goal_position", [0.12, 0.0, 0.25]), dtype=float)
        task = getattr(self.unwrapped, "task", None)
        if task is not None:
            for attr in ("goal", "_goal"):
                if hasattr(task, attr):
                    try:
                        setattr(task, attr, goal.copy())
                    except Exception:
                        pass
        if isinstance(obs, dict) and "desired_goal" in obs:
            obs["desired_goal"] = goal.astype(obs["desired_goal"].dtype, copy=False)

    def _maybe_scale_goal(self, obs: Any) -> None:
        if self.env_cfg.get("fixed_goal", False) or self.goal_range_scale >= 0.999:
            return
        goal = self._desired_goal(obs)
        if goal.size < 3:
            return
        ranges = self.env_cfg.get("random_goal_range", {})
        center = np.asarray(
            [
                np.mean(ranges.get("x", [goal[0], goal[0]])),
                np.mean(ranges.get("y", [goal[1], goal[1]])),
                np.mean(ranges.get("z", [goal[2], goal[2]])),
            ],
            dtype=float,
        )
        scaled = center + self.goal_range_scale * (goal[:3] - center)
        task = getattr(self.unwrapped, "task", None)
        if task is not None:
            for attr in ("goal", "_goal"):
                if hasattr(task, attr):
                    try:
                        setattr(task, attr, scaled.copy())
                    except Exception:
                        pass
        if isinstance(obs, dict) and "desired_goal" in obs:
            obs["desired_goal"] = scaled.astype(obs["desired_goal"].dtype, copy=False)

    def _distance(self, obs: Any) -> float:
        achieved = self._achieved_goal(obs)
        desired = self._desired_goal(obs)
        if achieved.size and desired.size:
            return float(np.linalg.norm(achieved - desired))
        return self._current_distance()

    def _current_distance(self) -> float:
        ee_pos = self._get_ee_position()
        goal = self._current_goal()
        if ee_pos.size and goal.size:
            return float(np.linalg.norm(ee_pos - goal))
        return float("inf")

    def _current_goal(self) -> np.ndarray:
        task = getattr(self.unwrapped, "task", None)
        for owner in (task, self.unwrapped):
            for attr in ("goal", "_goal"):
                if owner is not None and hasattr(owner, attr):
                    try:
                        return np.asarray(getattr(owner, attr), dtype=float).reshape(-1)[:3]
                    except Exception:
                        pass
        return np.empty((0,), dtype=float)

    def _achieved_goal(self, obs: Any) -> np.ndarray:
        if isinstance(obs, dict) and "achieved_goal" in obs:
            return np.asarray(obs["achieved_goal"], dtype=float).reshape(-1)[:3]
        return self._get_ee_position(obs)

    def _desired_goal(self, obs: Any) -> np.ndarray:
        if isinstance(obs, dict) and "desired_goal" in obs:
            return np.asarray(obs["desired_goal"], dtype=float).reshape(-1)[:3]
        return self._current_goal()

    def _get_ee_position(self, obs: Any | None = None) -> np.ndarray:
        robot = getattr(self.unwrapped, "robot", None)
        for name in ("get_ee_position", "get_end_effector_position"):
            if robot is not None and hasattr(robot, name):
                try:
                    return np.asarray(getattr(robot, name)(), dtype=float).reshape(-1)[:3]
                except Exception:
                    pass
        if isinstance(obs, dict) and "achieved_goal" in obs:
            return np.asarray(obs["achieved_goal"], dtype=float).reshape(-1)[:3]
        handles = self._handles
        client = handles.get("client") if handles else None
        body_id = handles.get("body_id") if handles else None
        link_id = handles.get("ee_link_id") if handles else None
        if client is not None and body_id is not None and link_id is not None:
            try:
                return np.asarray(client.getLinkState(body_id, link_id, computeLinkVelocity=1)[0], dtype=float)
            except Exception:
                pass
        return np.empty((0,), dtype=float)

    def _get_ee_velocity(self) -> np.ndarray:
        robot = getattr(self.unwrapped, "robot", None)
        for name in ("get_ee_velocity", "get_end_effector_velocity"):
            if robot is not None and hasattr(robot, name):
                try:
                    vel = getattr(robot, name)()
                    if isinstance(vel, tuple):
                        vel = vel[0]
                    return np.asarray(vel, dtype=float).reshape(-1)[:3]
                except Exception:
                    pass
        handles = self._handles
        client = handles.get("client") if handles else None
        body_id = handles.get("body_id") if handles else None
        link_id = handles.get("ee_link_id") if handles else None
        if client is not None and body_id is not None and link_id is not None:
            try:
                return np.asarray(client.getLinkState(body_id, link_id, computeLinkVelocity=1)[6], dtype=float)
            except Exception:
                pass
        pos = self._get_ee_position()
        arr = self.trace.arrays().get("ee_pos", np.empty((0,)))
        if pos.size and arr.ndim == 2 and arr.shape[0] > 0:
            return (pos - arr[-1]) / max(self.dt, 1e-6)
        return np.zeros(3)

    def _get_joint_position(self) -> np.ndarray:
        robot = getattr(self.unwrapped, "robot", None)
        for name in ("get_joint_angles", "get_joint_position", "get_joint_positions"):
            if robot is not None and hasattr(robot, name):
                try:
                    return np.asarray(getattr(robot, name)(), dtype=float).reshape(-1)
                except Exception:
                    pass
        return np.empty((0,), dtype=float)

    def _get_joint_velocity(self) -> np.ndarray:
        robot = getattr(self.unwrapped, "robot", None)
        for name in ("get_joint_velocities", "get_joint_velocity"):
            if robot is not None and hasattr(robot, name):
                try:
                    return np.asarray(getattr(robot, name)(), dtype=float).reshape(-1)
                except Exception:
                    pass
        return np.empty((0,), dtype=float)

    def _resolve_dt(self) -> float:
        for owner in (getattr(self, "unwrapped", None), getattr(getattr(self, "unwrapped", None), "sim", None)):
            for attr in ("dt", "timestep", "time_step"):
                if owner is not None and hasattr(owner, attr):
                    try:
                        return float(getattr(owner, attr))
                    except Exception:
                        pass
        return 1.0 / 240.0

    def _resolve_pybullet_handles(self) -> dict[str, Any]:
        sim = getattr(self.unwrapped, "sim", None)
        robot = getattr(self.unwrapped, "robot", None)
        client = None
        for attr in ("physics_client", "client", "p"):
            if sim is not None and hasattr(sim, attr):
                client = getattr(sim, attr)
                break
        if client is None:
            return {}

        body_id = None
        body_name = getattr(robot, "body_name", "panda")
        for attr in ("bodies_idx", "_bodies_idx", "body_idx", "body_id"):
            if sim is not None and hasattr(sim, attr):
                value = getattr(sim, attr)
                if isinstance(value, dict):
                    body_id = value.get(body_name) or value.get("panda")
                else:
                    body_id = value
                if body_id is not None:
                    break
        if body_id is None and robot is not None:
            for attr in ("body_idx", "body_id", "robot_id"):
                if hasattr(robot, attr):
                    body_id = getattr(robot, attr)
                    break

        ee_link_id = None
        if robot is not None:
            for attr in ("ee_link", "ee_link_id", "end_effector_link"):
                if hasattr(robot, attr):
                    ee_link_id = getattr(robot, attr)
                    break
        if ee_link_id is None:
            ee_link_id = 11
        return {"client": client, "body_id": body_id, "ee_link_id": ee_link_id}


def make_env(cfg: dict[str, Any], *, render_mode: str | None = None, rank: int = 0, seed: int | None = None):
    def _init():
        try:
            import gymnasium as gym
            import panda_gym  # noqa: F401
        except Exception as exc:  # pragma: no cover - dependency guard
            raise RuntimeError(
                "Missing simulation dependencies. Install them with `python -m pip install -r requirements.txt`."
            ) from exc

        env_cfg = cfg.get("env", {})
        kwargs: dict[str, Any] = {}
        mode = render_mode if render_mode is not None else env_cfg.get("render_mode")
        if mode is not None:
            kwargs["render_mode"] = mode
        if env_cfg.get("control_type"):
            kwargs["control_type"] = env_cfg["control_type"]
        env = gym.make(env_cfg.get("id", "PandaReach-v3"), **kwargs)
        env = HighPrecisionReachWrapper(env, cfg)
        max_steps = env_cfg.get("max_episode_steps")
        if max_steps:
            env = gym.wrappers.TimeLimit(env, max_episode_steps=int(max_steps))
        env.reset(seed=(seed if seed is not None else cfg.get("seed", 0)) + rank)
        return env

    return _init


def unwrap_high_precision(env: Any) -> HighPrecisionReachWrapper | None:
    current = env
    visited = set()
    while current is not None and id(current) not in visited:
        visited.add(id(current))
        if isinstance(current, HighPrecisionReachWrapper):
            return current
        current = getattr(current, "env", None)
    return None
