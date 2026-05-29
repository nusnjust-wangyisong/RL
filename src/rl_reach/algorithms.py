from __future__ import annotations

from typing import Any

import numpy as np


def build_model(algo_name: str, env: Any, cfg: dict[str, Any]):
    try:
        from stable_baselines3 import DDPG, HerReplayBuffer, SAC, TD3
        from stable_baselines3.common.noise import NormalActionNoise
        from sb3_contrib import TQC
    except Exception as exc:  # pragma: no cover - dependency guard
        raise RuntimeError(
            "Training dependencies are missing. Install `requirements.txt` before running algorithms."
        ) from exc

    train_cfg = cfg.get("train", {})
    normalized = algo_name.upper()
    use_her = normalized.endswith("_HER") or "_HER_" in normalized
    base_name = normalized.replace("_HER", "").replace("_CURRICULUM", "")
    algo_cls = {"DDPG": DDPG, "TD3": TD3, "SAC": SAC, "TQC": TQC}[base_name]

    action_noise = None
    if base_name in {"DDPG", "TD3"}:
        n_actions = int(np.prod(env.action_space.shape))
        action_noise = NormalActionNoise(mean=np.zeros(n_actions), sigma=0.1 * np.ones(n_actions))

    kwargs: dict[str, Any] = {
        "policy": "MultiInputPolicy",
        "env": env,
        "learning_rate": float(train_cfg.get("learning_rate", 3e-4)),
        "buffer_size": int(train_cfg.get("buffer_size", 1_000_000)),
        "learning_starts": int(train_cfg.get("learning_starts", 5000)),
        "batch_size": int(train_cfg.get("batch_size", 256)),
        "tau": float(train_cfg.get("tau", 0.05)),
        "gamma": float(train_cfg.get("gamma", 0.95)),
        "train_freq": int(train_cfg.get("train_freq", 1)),
        "gradient_steps": int(train_cfg.get("gradient_steps", 1)),
        "device": cfg.get("device", "auto"),
        "verbose": int(train_cfg.get("verbose", 0)),
        "seed": int(cfg.get("seed", 0)),
    }
    if action_noise is not None:
        kwargs["action_noise"] = action_noise
    if use_her:
        kwargs["replay_buffer_class"] = HerReplayBuffer
        kwargs["replay_buffer_kwargs"] = {
            "n_sampled_goal": int(train_cfg.get("her_n_sampled_goal", 4)),
            "goal_selection_strategy": "future",
        }
    if base_name == "TQC":
        kwargs.setdefault("top_quantiles_to_drop_per_net", 2)
    return algo_cls(**kwargs)


def load_model(algo_name: str, model_path: str, env: Any, cfg: dict[str, Any]):
    try:
        from stable_baselines3 import DDPG, SAC, TD3
        from sb3_contrib import TQC
    except Exception as exc:  # pragma: no cover - dependency guard
        raise RuntimeError("Training dependencies are missing.") from exc
    base_name = algo_name.upper().replace("_HER", "").replace("_CURRICULUM", "")
    algo_cls = {"DDPG": DDPG, "TD3": TD3, "SAC": SAC, "TQC": TQC}[base_name]
    return algo_cls.load(model_path, env=env, device=cfg.get("device", "auto"))
