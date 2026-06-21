from __future__ import annotations

import argparse
from pathlib import Path

from rl_reach.algorithms import build_model
from rl_reach.callbacks import CurriculumCallback, make_callback_list, set_global_seeds
from rl_reach.config import deep_update, ensure_run_dirs, load_config
from rl_reach.envs import make_env


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train high-precision Panda Reach policies.")
    parser.add_argument("--config", default="configs/experiment.yaml")
    parser.add_argument("--algo", default="TD3_HER_CURRICULUM")
    parser.add_argument("--timesteps", type=int, default=None)
    parser.add_argument("--fixed-goal", action="store_true")
    parser.add_argument("--random-goal", action="store_true")
    parser.add_argument("--no-disturbance", action="store_true")
    parser.add_argument("--precision-servo-mode", default=None)
    parser.add_argument("--train-precision-servo", action="store_true")
    parser.add_argument("--include-ik-observation", action="store_true")
    parser.add_argument("--run-name", default=None)
    parser.add_argument("--progress", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    cfg = load_config(args.config)
    if args.fixed_goal:
        cfg = deep_update(cfg, {"env": {"fixed_goal": True}})
    if args.random_goal:
        cfg = deep_update(cfg, {"env": {"fixed_goal": False}})
    if args.no_disturbance:
        cfg = deep_update(cfg, {"disturbance": {"enabled": False}})
    if args.precision_servo_mode:
        cfg = deep_update(
            cfg,
            {
                "eval": {"precision_servo_mode": args.precision_servo_mode},
                "reward": {"precision_servo_mode": args.precision_servo_mode},
            },
        )
    if args.train_precision_servo:
        servo_train_cfg = {
            key: value
            for key, value in cfg.get("eval", {}).items()
            if key.startswith("precision_servo_")
        }
        cfg = deep_update(cfg, {"reward": {"precision_servo": True, **servo_train_cfg}})
    if args.include_ik_observation:
        cfg = deep_update(cfg, {"env": {"include_ik_observation": True}})

    algo = args.algo.upper()
    run_name = args.run_name or f"{algo}_{'fixed' if cfg['env'].get('fixed_goal') else 'random'}"
    paths = ensure_run_dirs(cfg)
    set_global_seeds(int(cfg.get("seed", 0)))

    try:
        from stable_baselines3.common.monitor import Monitor
        from stable_baselines3.common.vec_env import DummyVecEnv
    except Exception as exc:
        raise RuntimeError("Install dependencies first: python -m pip install -r requirements.txt") from exc

    env_fn = make_env(cfg, seed=int(cfg.get("seed", 0)))
    env = DummyVecEnv([lambda: Monitor(env_fn())])
    model = build_model(algo, env, cfg)

    callbacks = []
    if "CURRICULUM" in algo and cfg.get("curriculum", {}).get("enabled", False):
        callbacks.append(CurriculumCallback(cfg["curriculum"]).callback)
    callback = make_callback_list(callbacks) if callbacks else None

    timesteps = int(args.timesteps or cfg.get("train", {}).get("total_timesteps", 100000))
    model.learn(total_timesteps=timesteps, callback=callback, progress_bar=args.progress)

    model_path = paths["model_dir"] / f"{run_name}.zip"
    model.save(model_path)
    print(f"Saved model: {model_path}")


if __name__ == "__main__":
    main()
