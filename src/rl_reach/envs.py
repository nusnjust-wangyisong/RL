from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path
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
        self._prev_deflection = np.zeros(3)
        self._dist_scale = 1.0
        self._dist_random_dir = False
        self._episode_dir: np.ndarray | None = None
        self._rng = np.random.default_rng(cfg.get("seed", None))
        self._handles = self._resolve_pybullet_handles()
        self.set_precision_threshold(self.target_threshold_m)

    def set_disturbance_scale(self, scale: float) -> None:
        """扰动感知课程：缩放扰动幅值（训练期由 0 渐增到 1）。"""
        self._dist_scale = float(scale)

    def set_disturbance_random_direction(self, flag: bool) -> None:
        """扰动感知课程：训练期是否每个 episode 随机化扰动方向。"""
        self._dist_random_dir = bool(flag)

    def _maybe_sample_dir(self) -> None:
        if self._dist_random_dir:
            v = self._rng.normal(size=3)
            n = np.linalg.norm(v)
            self._episode_dir = v / n if n > 1e-9 else np.array([0.0, 0.0, 1.0])
        else:
            self._episode_dir = None

    def reset(self, **kwargs: Any):  # type: ignore[override]
        self.step_count = 0
        self.trace = EpisodeTrace()
        self._prev_action = None
        self._prev_ee_vel = None
        self._prev_ee_acc = None
        self._prev_joint_vel = None
        self._prev_deflection = np.zeros(3)
        self._maybe_sample_dir()
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

        # 接触柔度扰动：接触力经末端等效刚度产生柔性偏移 Δx=F/k，
        # 因 panda-gym 为运动学/位置驱动（外力无效），按柔度模型将偏移增量注入末端指令，
        # 使末端产生真实振荡偏移，控制器据此感知并修正。
        disturbance = self._compute_disturbance()
        if disturbance.active and disturbance.force.size:
            stiffness = float(self.disturbance_cfg.get("compliance_stiffness_n_per_m", 3300.0))
            scale = float(self.disturbance_cfg.get("compliance_action_scale_m", 0.0388))
            deflection = disturbance.force / max(stiffness, 1e-9)
            delta = deflection - self._prev_deflection
            self._prev_deflection = deflection
            if action.size >= 3 and scale > 1e-9:
                action = action.copy()
                action[:3] = action[:3] + delta[:3] / scale
        else:
            self._prev_deflection = np.zeros(3)

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

    def precision_servo_action(
        self,
        action: np.ndarray,
        obs: Any,
        prev_action: np.ndarray | None,
        eval_cfg: dict[str, Any],
    ) -> np.ndarray:
        if not isinstance(obs, dict) or "achieved_goal" not in obs or "desired_goal" not in obs:
            return action
        ee = np.asarray(obs["achieved_goal"], dtype=float).reshape(-1)
        goal = np.asarray(obs["desired_goal"], dtype=float).reshape(-1)
        if ee.size < 3 or goal.size < 3 or action.size < 3:
            return action
        gain = float(eval_cfg.get("precision_servo_gain", 10.0))
        beta = float(np.clip(float(eval_cfg.get("precision_servo_beta", 0.35)), 0.0, 1.0))
        max_action = float(eval_cfg.get("precision_servo_max_action", 0.4))
        rl_action = np.asarray(action, dtype=float).copy()
        servo = rl_action.copy()
        servo[:3] = np.clip(gain * (goal[:3] - ee[:3]), -max_action, max_action)
        mode = str(eval_cfg.get("precision_servo_mode", "cartesian")).lower()
        if mode in {"her_cartesian_residual", "her_ik_residual"}:
            max_action = float(eval_cfg.get("precision_servo_residual_max_action", max_action))
            servo[:3] = np.clip(gain * (goal[:3] - ee[:3]), -max_action, max_action)
            residual_scale = float(eval_cfg.get("precision_servo_residual_scale", 0.08))
            radius = float(eval_cfg.get("precision_servo_residual_radius_m", 0.05))
            min_alpha = float(eval_cfg.get("precision_servo_residual_min_alpha", 0.005))
            max_alpha = float(eval_cfg.get("precision_servo_residual_max_alpha", 0.03))
            distance = float(np.linalg.norm(goal[:3] - ee[:3]))
            closeness = 1.0 - np.clip(distance / max(radius, 1e-9), 0.0, 1.0)
            smooth = closeness * closeness * (3.0 - 2.0 * closeness)
            residual_weight = min_alpha + (max_alpha - min_alpha) * smooth
            residual = np.clip(rl_action - servo, -residual_scale, residual_scale)
            servo = servo + residual_weight * residual
        # 主动阻尼：末端速度反馈，提高闭环等效阻尼比 ζ，抑制受迫振动的动态放大
        vel_damp = float(eval_cfg.get("precision_servo_vel_damping", 0.0))
        if vel_damp > 0.0:
            v_ee = self._get_ee_velocity()
            if v_ee.size >= 3:
                servo[:3] = np.clip(servo[:3] - vel_damp * v_ee[:3], -max_action, max_action)
        if prev_action is not None:
            servo = beta * servo + (1.0 - beta) * np.asarray(prev_action, dtype=float)
        return np.clip(servo, -1.0, 1.0)

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

        if self._episode_dir is not None:
            direction = self._episode_dir
        else:
            direction = np.asarray(self.disturbance_cfg.get("direction", [0.0, 0.0, 1.0]), dtype=float)
            norm = np.linalg.norm(direction)
            direction = direction / norm if norm > 1e-12 else np.array([0.0, 0.0, 1.0])
        t = self.step_count * self.dt
        base = float(self.disturbance_cfg.get("base_force_n", 0.0))
        amplitude = float(self.disturbance_cfg.get("amplitude_n", 5.0)) * self._dist_scale
        frequency = float(self.disturbance_cfg.get("frequency_hz", 20.0))
        noise_std = float(self.disturbance_cfg.get("noise_std_n", 0.0)) * self._dist_scale
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


class IRB120ReachEnv(gym.Env if gym is not None else object):
    """PyBullet GoalEnv for ABB IRB120 high-precision Reach experiments.

    Panda-gym exposes an end-effector action interface for Panda. IRB120 is kept
    as a native joint-space environment because the robot has six revolute DOF,
    and the action/observation spaces must not inherit Panda's seven-joint or
    end-effector-control assumptions.
    """

    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 30}

    def __init__(self, cfg: dict[str, Any], *, render_mode: str | None = None, seed: int | None = None):
        if gym is None:  # pragma: no cover - dependency guard
            raise RuntimeError("gymnasium is required to create IRB120ReachEnv.")
        try:
            import pybullet as pybullet
            import pybullet_data
        except Exception as exc:  # pragma: no cover - dependency guard
            raise RuntimeError("pybullet is required to create IRB120ReachEnv.") from exc

        self.p = pybullet
        self.pybullet_data = pybullet_data
        self.cfg = cfg
        self.env_cfg = cfg.get("env", {})
        self.robot_cfg = cfg.get("irb120", {})
        self.reward_cfg = cfg.get("reward", {})
        self.disturbance_cfg = cfg.get("disturbance", {})
        self.render_mode = render_mode if render_mode is not None else self.env_cfg.get("render_mode")
        self.thresholds_m = list(self.env_cfg.get("thresholds_m", [0.05, 0.03, 0.02, 0.01, 0.005]))
        self.target_threshold_m = float(self.env_cfg.get("target_threshold_m", 0.005))
        self.max_episode_steps = int(self.env_cfg.get("max_episode_steps", 80))
        self.goal_range_scale = 1.0
        self.step_count = 0
        self.physics_dt = float(self.robot_cfg.get("physics_dt", 1.0 / 240.0))
        self.control_substeps = int(self.robot_cfg.get("control_substeps", 12))
        self.dt = self.physics_dt * self.control_substeps
        self.dof = 6
        self.joint_ids = list(range(self.dof))
        self.ee_link_id = int(self.robot_cfg.get("ee_link_id", 6))
        self.include_ik_observation = bool(self.env_cfg.get("include_ik_observation", False))
        self.action_scale_rad = float(self.robot_cfg.get("action_scale_rad", 0.04))
        self.joint_force = float(self.robot_cfg.get("joint_force", 180.0))
        self._rng = np.random.default_rng(seed if seed is not None else cfg.get("seed", None))
        self.trace = EpisodeTrace()
        self._prev_action: np.ndarray | None = None
        self._servo_prev_action: np.ndarray | None = None
        self._prev_ee_vel: np.ndarray | None = None
        self._prev_ee_acc: np.ndarray | None = None
        self._prev_joint_vel: np.ndarray | None = None
        self._dist_scale = 1.0
        self._dist_random_dir = False
        self._episode_dir: np.ndarray | None = None
        self.goal = np.asarray(self.env_cfg.get("fixed_goal_position", [0.32, 0.0, 0.35]), dtype=float)
        self.prev_action = np.zeros(self.dof, dtype=np.float32)

        self.action_space = gym.spaces.Box(-1.0, 1.0, shape=(self.dof,), dtype=np.float32)
        obs_dim = self.dof + self.dof + 3 + 3 + 3 + self.dof + 1
        if self.include_ik_observation:
            obs_dim += self.dof
        self.observation_space = gym.spaces.Dict(
            {
                "observation": gym.spaces.Box(-np.inf, np.inf, shape=(obs_dim,), dtype=np.float32),
                "achieved_goal": gym.spaces.Box(-np.inf, np.inf, shape=(3,), dtype=np.float32),
                "desired_goal": gym.spaces.Box(-np.inf, np.inf, shape=(3,), dtype=np.float32),
            }
        )

        mode = self.p.GUI if self.render_mode == "human" else self.p.DIRECT
        self.client_id = self.p.connect(mode)
        self.p.setAdditionalSearchPath(self.pybullet_data.getDataPath(), physicsClientId=self.client_id)
        self.p.setTimeStep(self.physics_dt, physicsClientId=self.client_id)
        self.p.setGravity(0.0, 0.0, -9.81, physicsClientId=self.client_id)
        self.robot_id: int | None = None
        self.target_visual_id: int | None = None
        self.lower_limits = np.zeros(self.dof, dtype=float)
        self.upper_limits = np.zeros(self.dof, dtype=float)
        self.joint_ranges = np.zeros(self.dof, dtype=float)
        self.rest_q = np.asarray(self.robot_cfg.get("rest_q", [0.0, 0.25, 0.65, 0.0, 0.2, 0.0]), dtype=float)
        self._load_world()

    def reset(self, *, seed: int | None = None, options: dict[str, Any] | None = None):  # type: ignore[override]
        if seed is not None:
            self._rng = np.random.default_rng(seed)
        self.step_count = 0
        self.trace = EpisodeTrace()
        self._prev_action = None
        self._servo_prev_action = None
        self._prev_ee_vel = None
        self._prev_ee_acc = None
        self._prev_joint_vel = None
        self.prev_action = np.zeros(self.dof, dtype=np.float32)
        if self.robot_id is None:
            self._load_world()
        for joint_id, q in zip(self.joint_ids, self.rest_q):
            self.p.resetJointState(self.robot_id, joint_id, float(q), targetVelocity=0.0, physicsClientId=self.client_id)
        self.goal = self._sample_goal()
        self._maybe_sample_dir()
        self._update_target_visual()
        obs = self._get_obs()
        self._record(obs, action=None, reward=0.0, disturbance=np.zeros(3))
        return obs, {}

    def step(self, action: np.ndarray):  # type: ignore[override]
        self.step_count += 1
        action = np.clip(np.asarray(action, dtype=float).reshape(self.dof), -1.0, 1.0)
        if self.reward_cfg.get("precision_servo", False):
            action = self.precision_servo_action(action, self._get_obs(), self._prev_action, self.reward_cfg)

        q = self._get_joint_position()
        target_q = np.clip(q + self.action_scale_rad * action, self.lower_limits, self.upper_limits)
        self.p.setJointMotorControlArray(
            self.robot_id,
            self.joint_ids,
            self.p.POSITION_CONTROL,
            targetPositions=target_q.tolist(),
            forces=[self.joint_force] * self.dof,
            positionGains=[float(self.robot_cfg.get("position_gain", 0.18))] * self.dof,
            velocityGains=[float(self.robot_cfg.get("velocity_gain", 0.70))] * self.dof,
            physicsClientId=self.client_id,
        )

        # 接触变力扰动：按物理子步（240 Hz）重新计算并施加，避免周期力被控制步采样混叠。
        disturbance = self._compute_disturbance()
        base_t = (self.step_count - 1) * self.dt
        for i in range(self.control_substeps):
            if disturbance.active:
                t_sub = base_t + (i + 1) * self.physics_dt
                self._apply_external_force(self._disturbance_force_at(t_sub))
            self.p.stepSimulation(physicsClientId=self.client_id)

        self.prev_action = action.astype(np.float32)
        obs = self._get_obs()
        distance = self._distance(obs)
        stability_penalty = self._stability_penalty(action)
        reward = self._shape_reward(distance, stability_penalty)
        terminated = bool(distance <= self.target_threshold_m and self.env_cfg.get("terminate_on_success", True))
        truncated = bool(self.step_count >= self.max_episode_steps)
        info = {
            "distance_m": distance,
            "is_success_5mm": bool(distance <= 0.005),
            "is_success": bool(distance <= self.target_threshold_m),
            "stability_penalty": stability_penalty,
            "disturbance_force": disturbance.force,
            "disturbance_active": disturbance.active,
        }
        self._record(obs, action=action, reward=reward, disturbance=disturbance.force)
        self._prev_action = action.copy()
        return obs, reward, terminated, truncated, info

    def close(self) -> None:
        if getattr(self, "client_id", None) is not None and self.p.isConnected(self.client_id):
            self.p.disconnect(self.client_id)

    def render(self):
        if self.render_mode == "rgb_array":
            width, height = 640, 480
            view = self.p.computeViewMatrix(cameraEyePosition=[0.85, -0.85, 0.65], cameraTargetPosition=[0.28, 0, 0.3], cameraUpVector=[0, 0, 1])
            proj = self.p.computeProjectionMatrixFOV(fov=55, aspect=width / height, nearVal=0.01, farVal=3.0)
            image = self.p.getCameraImage(width, height, view, proj, physicsClientId=self.client_id)
            return np.asarray(image[2], dtype=np.uint8)
        return None

    def compute_reward(self, achieved_goal: np.ndarray, desired_goal: np.ndarray, info: Any) -> np.ndarray:
        distance = np.linalg.norm(np.asarray(achieved_goal) - np.asarray(desired_goal), axis=-1)
        if self.reward_cfg.get("dense", True):
            reward = -float(self.reward_cfg.get("reach_weight", 1.0)) * distance
        else:
            reward = -(distance > self.target_threshold_m).astype(float)
        return np.asarray(reward + float(self.reward_cfg.get("success_bonus", 0.0)) * (distance <= self.target_threshold_m))

    def set_precision_threshold(self, threshold_m: float) -> None:
        self.target_threshold_m = float(threshold_m)

    def set_goal_range_scale(self, scale: float) -> None:
        self.goal_range_scale = float(scale)

    def set_disturbance_scale(self, scale: float) -> None:
        self._dist_scale = float(scale)

    def set_disturbance_random_direction(self, flag: bool) -> None:
        self._dist_random_dir = bool(flag)

    def _maybe_sample_dir(self) -> None:
        if self._dist_random_dir:
            v = self._rng.normal(size=3)
            n = np.linalg.norm(v)
            self._episode_dir = v / n if n > 1e-9 else np.array([0.0, 0.0, 1.0])
        else:
            self._episode_dir = None

    def get_episode_trace(self) -> EpisodeTrace:
        return self.trace

    def precision_servo_action(
        self,
        action: np.ndarray,
        obs: Any,
        prev_action: np.ndarray | None,
        eval_cfg: dict[str, Any],
    ) -> np.ndarray:
        if not isinstance(obs, dict) or "desired_goal" not in obs:
            return action
        goal = np.asarray(obs["desired_goal"], dtype=float).reshape(-1)[:3]
        if goal.size < 3:
            return action
        q = self._get_joint_position()
        ee = np.asarray(obs.get("achieved_goal", self._get_ee_position()), dtype=float).reshape(-1)[:3]
        distance = float(np.linalg.norm(goal - ee)) if ee.size >= 3 else self._current_distance()
        servo = self._ik_delta_action(goal, q, distance, eval_cfg)
        beta = float(np.clip(float(eval_cfg.get("precision_servo_beta", 0.35)), 0.0, 1.0))
        gain = float(eval_cfg.get("precision_servo_joint_gain", 1.0))
        mode = str(eval_cfg.get("precision_servo_mode", "ik_blend")).lower()
        rl_action = np.asarray(action, dtype=float).copy()
        if mode == "ik_servo":
            guided = gain * servo
        elif mode == "ik_residual":
            residual_scale = float(eval_cfg.get("precision_servo_residual_scale", 0.25))
            guided = gain * servo + residual_scale * rl_action
        elif mode == "her_ik_residual":
            residual_scale = float(eval_cfg.get("precision_servo_residual_scale", 0.25))
            radius = float(eval_cfg.get("precision_servo_residual_radius_m", 0.05))
            min_alpha = float(eval_cfg.get("precision_servo_residual_min_alpha", 0.08))
            max_alpha = float(eval_cfg.get("precision_servo_residual_max_alpha", 0.30))
            closeness = 1.0 - np.clip(distance / max(radius, 1e-9), 0.0, 1.0)
            smooth = closeness * closeness * (3.0 - 2.0 * closeness)
            residual_weight = max_alpha - (max_alpha - min_alpha) * smooth
            residual = np.clip(rl_action - servo, -residual_scale, residual_scale)
            guided = servo + residual_weight * residual
        else:
            guided = (1.0 - gain) * rl_action + gain * servo
        damping = float(eval_cfg.get("precision_servo_damping", 0.0))
        if damping > 0.0:
            joint_vel = self._get_joint_velocity()
            if joint_vel.size >= self.dof:
                guided -= damping * joint_vel[: self.dof] / max(self.action_scale_rad, 1e-9)
        if prev_action is not None:
            guided = beta * guided + (1.0 - beta) * np.asarray(prev_action, dtype=float)
        max_delta = float(eval_cfg.get("precision_servo_max_delta", 0.0))
        if max_delta > 0.0 and self._servo_prev_action is not None:
            guided = self._servo_prev_action + np.clip(guided - self._servo_prev_action, -max_delta, max_delta)
        guided = np.clip(guided, -1.0, 1.0)
        self._servo_prev_action = guided.copy()
        return guided

    def _ik_delta_action(
        self,
        goal: np.ndarray,
        q: np.ndarray | None = None,
        distance: float | None = None,
        cfg: dict[str, Any] | None = None,
    ) -> np.ndarray:
        cfg = cfg or {}
        if q is None:
            q = self._get_joint_position()
        ik = self.p.calculateInverseKinematics(
            self.robot_id,
            self.ee_link_id,
            np.asarray(goal, dtype=float).reshape(-1)[:3].tolist(),
            lowerLimits=self.lower_limits.tolist(),
            upperLimits=self.upper_limits.tolist(),
            jointRanges=self.joint_ranges.tolist(),
            restPoses=np.asarray(q, dtype=float).reshape(-1)[: self.dof].tolist(),
            maxNumIterations=int(cfg.get("precision_servo_ik_iterations", 80)),
            residualThreshold=float(cfg.get("precision_servo_ik_residual", 1e-5)),
            physicsClientId=self.client_id,
        )
        target_q = np.asarray(ik[: self.dof], dtype=float)
        max_action = float(cfg.get("precision_servo_max_action", 0.35))
        if distance is not None:
            hold_radius = float(cfg.get("precision_servo_hold_radius_m", 0.006))
            near_radius = float(cfg.get("precision_servo_near_radius_m", 0.030))
            near_scale = float(cfg.get("precision_servo_near_scale", 0.65))
            hold_scale = float(cfg.get("precision_servo_hold_scale", 0.35))
            if distance <= hold_radius:
                max_action *= hold_scale
            elif distance <= near_radius:
                ratio = (distance - hold_radius) / max(near_radius - hold_radius, 1e-9)
                smooth = ratio * ratio * (3.0 - 2.0 * ratio)
                max_action *= hold_scale + (near_scale - hold_scale) * smooth
        return np.clip((target_q - q) / max(self.action_scale_rad, 1e-9), -max_action, max_action)

    def _load_world(self) -> None:
        self.p.resetSimulation(physicsClientId=self.client_id)
        self.p.setTimeStep(self.physics_dt, physicsClientId=self.client_id)
        self.p.setGravity(0.0, 0.0, -9.81, physicsClientId=self.client_id)
        if self.robot_cfg.get("load_plane", True):
            self.p.loadURDF("plane.urdf", physicsClientId=self.client_id)
        urdf_path = Path(self.robot_cfg.get("urdf_path", "assets/irb120/model.urdf"))
        if not urdf_path.is_absolute():
            urdf_path = Path.cwd() / urdf_path
        self.robot_id = self.p.loadURDF(str(urdf_path), [0, 0, 0], useFixedBase=True, physicsClientId=self.client_id)
        for idx, joint_id in enumerate(self.joint_ids):
            info = self.p.getJointInfo(self.robot_id, joint_id, physicsClientId=self.client_id)
            self.lower_limits[idx] = float(info[8])
            self.upper_limits[idx] = float(info[9])
            self.joint_ranges[idx] = max(float(info[9] - info[8]), 1e-6)
        self.rest_q = np.clip(self.rest_q, self.lower_limits, self.upper_limits)
        radius = float(self.robot_cfg.get("target_visual_radius", 0.006))
        visual = self.p.createVisualShape(
            self.p.GEOM_SPHERE,
            radius=radius,
            rgbaColor=[0.1, 0.8, 0.2, 0.65],
            physicsClientId=self.client_id,
        )
        self.target_visual_id = self.p.createMultiBody(
            baseMass=0,
            baseVisualShapeIndex=visual,
            basePosition=self.goal.tolist(),
            physicsClientId=self.client_id,
        )

    def _sample_goal(self) -> np.ndarray:
        if self.env_cfg.get("fixed_goal", False):
            return np.asarray(self.env_cfg.get("fixed_goal_position", [0.32, 0.0, 0.35]), dtype=float)
        ranges = self.env_cfg.get("random_goal_range", {})
        lows = np.asarray(
            [
                ranges.get("x", [0.22, 0.42])[0],
                ranges.get("y", [-0.18, 0.18])[0],
                ranges.get("z", [0.24, 0.48])[0],
            ],
            dtype=float,
        )
        highs = np.asarray(
            [
                ranges.get("x", [0.22, 0.42])[1],
                ranges.get("y", [-0.18, 0.18])[1],
                ranges.get("z", [0.24, 0.48])[1],
            ],
            dtype=float,
        )
        center = (lows + highs) / 2.0
        lows = center + self.goal_range_scale * (lows - center)
        highs = center + self.goal_range_scale * (highs - center)
        return self._rng.uniform(lows, highs).astype(float)

    def _update_target_visual(self) -> None:
        if self.target_visual_id is not None:
            self.p.resetBasePositionAndOrientation(
                self.target_visual_id,
                self.goal.tolist(),
                [0, 0, 0, 1],
                physicsClientId=self.client_id,
            )

    def _get_obs(self) -> dict[str, np.ndarray]:
        q = self._get_joint_position()
        qd = self._get_joint_velocity()
        ee = self._get_ee_position()
        ee_vel = self._get_ee_velocity()
        rel = self.goal - ee
        progress = np.asarray([self.step_count / max(self.max_episode_steps, 1)], dtype=float)
        parts = [q, qd, ee, ee_vel, rel, self.prev_action.astype(float), progress]
        if self.include_ik_observation:
            parts.append(self._ik_delta_action(self.goal, q=q, distance=float(np.linalg.norm(rel)), cfg=self.reward_cfg))
        obs = np.concatenate(parts)
        return {
            "observation": obs.astype(np.float32),
            "achieved_goal": ee.astype(np.float32),
            "desired_goal": self.goal.astype(np.float32),
        }

    def _shape_reward(self, distance: float, stability_penalty: float) -> float:
        if self.reward_cfg.get("dense", True):
            reward = -float(self.reward_cfg.get("reach_weight", 1.0)) * distance
        else:
            reward = -float(distance > self.target_threshold_m)
        if distance <= self.target_threshold_m:
            reward += float(self.reward_cfg.get("success_bonus", 0.0))
        return float(reward - stability_penalty)

    def _stability_penalty(self, action: np.ndarray) -> float:
        ee_vel = self._get_ee_velocity()
        joint_vel = self._get_joint_velocity()
        penalty = 0.0
        if self._prev_action is not None:
            penalty += float(self.reward_cfg.get("action_smooth_weight", 0.0)) * float(np.sum(np.square(action - self._prev_action)))
        if self._prev_joint_vel is not None:
            joint_acc = (joint_vel - self._prev_joint_vel) / max(self.dt, 1e-6)
            penalty += float(self.reward_cfg.get("joint_acc_weight", 0.0)) * float(np.sum(np.square(joint_acc)))
        self._prev_joint_vel = joint_vel.copy()
        if self._prev_ee_vel is not None:
            ee_acc = (ee_vel - self._prev_ee_vel) / max(self.dt, 1e-6)
            if self._prev_ee_acc is not None:
                ee_jerk = (ee_acc - self._prev_ee_acc) / max(self.dt, 1e-6)
                penalty += float(self.reward_cfg.get("ee_jerk_weight", 0.0)) * float(np.sum(np.square(ee_jerk)))
            penalty += float(self.reward_cfg.get("vibration_weight", 0.0)) * vibration_index(ee_acc)
            self._prev_ee_acc = ee_acc.copy()
        self._prev_ee_vel = ee_vel.copy()
        return penalty

    def _compute_disturbance(self) -> DisturbanceState:
        if not self.disturbance_cfg.get("enabled", False):
            return DisturbanceState(force=np.zeros(3), active=False)
        if self.step_count < int(self.disturbance_cfg.get("start_after_steps", 0)):
            return DisturbanceState(force=np.zeros(3), active=False)
        active = bool(self._current_distance() <= float(self.disturbance_cfg.get("activation_radius_m", 0.02)))
        if not active:
            return DisturbanceState(force=np.zeros(3), active=False)
        direction = np.asarray(self.disturbance_cfg.get("direction", [0.0, 0.0, 1.0]), dtype=float)
        direction = direction / max(np.linalg.norm(direction), 1e-12)
        t = self.step_count * self.dt
        scalar = float(self.disturbance_cfg.get("base_force_n", 0.0))
        scalar += float(self.disturbance_cfg.get("amplitude_n", 5.0)) * math.sin(
            2.0 * math.pi * float(self.disturbance_cfg.get("frequency_hz", 20.0)) * t
        )
        scalar += float(self._rng.normal(0.0, float(self.disturbance_cfg.get("noise_std_n", 0.0))))
        return DisturbanceState(force=direction * scalar, active=True)

    def _disturbance_force_at(self, t: float) -> np.ndarray:
        """在给定物理时刻 t（按子步推进）计算接触变力，避免控制步采样混叠。"""
        if self._episode_dir is not None:
            direction = self._episode_dir
        else:
            direction = np.asarray(self.disturbance_cfg.get("direction", [0.0, 0.0, 1.0]), dtype=float)
            direction = direction / max(np.linalg.norm(direction), 1e-12)
        scalar = float(self.disturbance_cfg.get("base_force_n", 0.0))
        scalar += float(self.disturbance_cfg.get("amplitude_n", 5.0)) * self._dist_scale * math.sin(
            2.0 * math.pi * float(self.disturbance_cfg.get("frequency_hz", 20.0)) * t
        )
        scalar += float(self._rng.normal(0.0, float(self.disturbance_cfg.get("noise_std_n", 0.0)) * self._dist_scale))
        return direction * scalar

    def _apply_external_force(self, force: np.ndarray) -> None:
        self.p.applyExternalForce(
            objectUniqueId=self.robot_id,
            linkIndex=self.ee_link_id,
            forceObj=np.asarray(force, dtype=float).tolist(),
            posObj=self._get_ee_position().tolist(),
            flags=self.p.WORLD_FRAME,
            physicsClientId=self.client_id,
        )

    def _record(self, obs: Any, *, action: np.ndarray | None, reward: float, disturbance: np.ndarray) -> None:
        ee_pos = self._get_ee_position()
        ee_vel = self._get_ee_velocity()
        joint_pos = self._get_joint_position()
        joint_vel = self._get_joint_velocity()
        distance = self._distance(obs)
        self.trace.append(
            ee_pos=ee_pos,
            ee_vel=ee_vel,
            joint_pos=joint_pos,
            joint_vel=joint_vel,
            action=action,
            goal=self.goal,
            reward=reward,
            distance=distance,
            disturbance=disturbance,
        )

    def _distance(self, obs: Any) -> float:
        achieved = np.asarray(obs["achieved_goal"], dtype=float).reshape(-1)[:3]
        desired = np.asarray(obs["desired_goal"], dtype=float).reshape(-1)[:3]
        return float(np.linalg.norm(achieved - desired))

    def _current_distance(self) -> float:
        return float(np.linalg.norm(self._get_ee_position() - self.goal))

    def _get_ee_position(self) -> np.ndarray:
        return np.asarray(
            self.p.getLinkState(self.robot_id, self.ee_link_id, computeLinkVelocity=1, physicsClientId=self.client_id)[0],
            dtype=float,
        )

    def _get_ee_velocity(self) -> np.ndarray:
        return np.asarray(
            self.p.getLinkState(self.robot_id, self.ee_link_id, computeLinkVelocity=1, physicsClientId=self.client_id)[6],
            dtype=float,
        )

    def _get_joint_position(self) -> np.ndarray:
        return np.asarray(
            [self.p.getJointState(self.robot_id, joint_id, physicsClientId=self.client_id)[0] for joint_id in self.joint_ids],
            dtype=float,
        )

    def _get_joint_velocity(self) -> np.ndarray:
        return np.asarray(
            [self.p.getJointState(self.robot_id, joint_id, physicsClientId=self.client_id)[1] for joint_id in self.joint_ids],
            dtype=float,
        )


def make_env(cfg: dict[str, Any], *, render_mode: str | None = None, rank: int = 0, seed: int | None = None):
    def _init():
        try:
            import gymnasium as gym
        except Exception as exc:  # pragma: no cover - dependency guard
            raise RuntimeError(
                "Missing simulation dependencies. Install them with `python -m pip install -r requirements.txt`."
            ) from exc

        env_cfg = cfg.get("env", {})
        robot = str(env_cfg.get("robot", "franka_panda")).lower()
        mode = render_mode if render_mode is not None else env_cfg.get("render_mode")
        if robot in {"irb120", "abb_irb120", "abb120"} or str(env_cfg.get("id", "")).lower().startswith("irb120"):
            env = IRB120ReachEnv(cfg, render_mode=mode, seed=(seed if seed is not None else cfg.get("seed", 0)) + rank)
            env.reset(seed=(seed if seed is not None else cfg.get("seed", 0)) + rank)
            return env

        try:
            import panda_gym  # noqa: F401
        except Exception as exc:  # pragma: no cover - dependency guard
            raise RuntimeError(
                "panda-gym is required for Panda environments. Install dependencies with `python -m pip install -r requirements.txt`."
            ) from exc

        kwargs: dict[str, Any] = {}
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


def unwrap_high_precision(env: Any) -> Any | None:
    current = env
    visited = set()
    while current is not None and id(current) not in visited:
        visited.add(id(current))
        if isinstance(current, (HighPrecisionReachWrapper, IRB120ReachEnv)):
            return current
        current = getattr(current, "env", None)
    return None
