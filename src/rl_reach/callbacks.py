from __future__ import annotations

from typing import Any

import numpy as np

from rl_reach.envs import unwrap_high_precision


class CurriculumCallback:
    """Threshold and goal-range scheduler for high-precision reaching."""

    def __init__(self, curriculum_cfg: dict[str, Any]):
        try:
            from stable_baselines3.common.callbacks import BaseCallback
        except Exception as exc:  # pragma: no cover - dependency guard
            raise RuntimeError("stable-baselines3 is required for CurriculumCallback.") from exc

        class _Callback(BaseCallback):
            def __init__(self, outer: "CurriculumCallback"):
                super().__init__()
                self.outer = outer

            def _on_training_start(self) -> None:
                self.outer.apply(self.training_env, 0)

            def _on_step(self) -> bool:
                self.outer.apply(self.training_env, self.num_timesteps)
                return True

        self.enabled = bool(curriculum_cfg.get("enabled", False))
        self.stages = list(curriculum_cfg.get("stages", []))
        self._last_stage = -1
        self.callback = _Callback(self)

    def apply(self, vec_env: Any, num_timesteps: int) -> None:
        if not self.enabled or not self.stages:
            return
        cumulative = 0
        stage_idx = len(self.stages) - 1
        for idx, stage in enumerate(self.stages):
            cumulative += int(stage.get("steps", 0))
            if num_timesteps <= cumulative:
                stage_idx = idx
                break
        if stage_idx == self._last_stage:
            return
        self._last_stage = stage_idx
        stage = self.stages[stage_idx]
        threshold = float(stage.get("threshold_m", 0.005))
        goal_scale = float(stage.get("goal_range_scale", 1.0))

        envs = getattr(vec_env, "envs", [vec_env])
        for env in envs:
            wrapper = unwrap_high_precision(env)
            if wrapper is not None:
                wrapper.set_precision_threshold(threshold)
                wrapper.set_goal_range_scale(goal_scale)


def make_callback_list(callbacks: list[Any]):
    try:
        from stable_baselines3.common.callbacks import CallbackList
    except Exception as exc:  # pragma: no cover - dependency guard
        raise RuntimeError("stable-baselines3 is required for callbacks.") from exc
    return CallbackList(callbacks)


def linear_schedule(initial_value: float):
    def schedule(progress_remaining: float) -> float:
        return float(progress_remaining) * initial_value

    return schedule


def set_global_seeds(seed: int) -> None:
    np.random.seed(seed)
    try:
        import torch

        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
    except Exception:
        pass
