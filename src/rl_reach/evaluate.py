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
from rl_reach.metrics import aggregate_summaries, summarize_episode


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate precision and stability metrics.")
    parser.add_argument("--config", default="configs/experiment.yaml")
    parser.add_argument("--algo", required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--episodes", type=int, default=50)
    parser.add_argument("--fixed-goal", action="store_true")
    parser.add_argument("--random-goal", action="store_true")
    parser.add_argument("--disturbance", choices=["off", "light", "medium", "strong"], default="medium")
    parser.add_argument("--output-name", default=None)
    parser.add_argument("--render", action="store_true")
    return parser.parse_args()


def evaluate_policy(
    *,
    cfg: dict[str, Any],
    algo: str,
    model_path: str | Path,
    episodes: int,
    render: bool = False,
) -> tuple[list[dict[str, Any]], dict[str, float]]:
    try:
        from stable_baselines3.common.monitor import Monitor
    except Exception as exc:
        raise RuntimeError("Install dependencies first: python -m pip install -r requirements.txt") from exc

    env_fn = make_env(cfg, render_mode=("human" if render else None), seed=int(cfg.get("seed", 0)) + 1000)
    env = Monitor(env_fn())
    model = load_model(algo, str(model_path), env, cfg)

    rows: list[dict[str, Any]] = []
    thresholds = list(cfg.get("env", {}).get("thresholds_m", [0.05, 0.03, 0.02, 0.01, 0.005]))
    wrapper = unwrap_high_precision(env)
    dt = wrapper.dt if wrapper is not None else 1.0 / 240.0

    for episode in range(episodes):
        obs, _info = env.reset(seed=int(cfg.get("seed", 0)) + 1000 + episode)
        done = False
        prev_action = None
        while not done:
            action, _state = model.predict(obs, deterministic=True)
            wrapper = unwrap_high_precision(env)
            action = precision_servo_if_needed(action, obs, prev_action, algo, cfg, wrapper)
            action = filter_action_if_needed(action, prev_action, algo, cfg)
            prev_action = np.asarray(action, dtype=float).copy()
            obs, _reward, terminated, truncated, _info = env.step(action)
            done = bool(terminated or truncated)
        wrapper = unwrap_high_precision(env)
        if wrapper is None:
            raise RuntimeError("HighPrecisionReachWrapper was not found during evaluation.")
        summary = summarize_episode(
            wrapper.get_episode_trace(),
            dt=dt,
            thresholds_m=thresholds,
            hold_threshold_m=0.005,
            hold_window=10,
        )
        summary["episode"] = episode
        rows.append(summary)
    env.close()
    return rows, aggregate_summaries(rows)


def precision_servo_if_needed(
    action: np.ndarray,
    obs: Any,
    prev_action: np.ndarray | None,
    algo: str,
    cfg: dict[str, Any],
    wrapper: Any | None = None,
) -> np.ndarray:
    eval_cfg = cfg.get("eval", {})
    servo_algos = {str(name).upper() for name in eval_cfg.get("precision_servo_algorithms", [])}
    if algo.upper() not in servo_algos or not isinstance(obs, dict):
        return action
    if wrapper is not None and hasattr(wrapper, "precision_servo_action"):
        return wrapper.precision_servo_action(action, obs, prev_action, eval_cfg)
    if "achieved_goal" not in obs or "desired_goal" not in obs:
        return action

    ee = np.asarray(obs["achieved_goal"], dtype=float).reshape(-1)
    goal = np.asarray(obs["desired_goal"], dtype=float).reshape(-1)
    if ee.size < 3 or goal.size < 3:
        return action

    gain = float(eval_cfg.get("precision_servo_gain", 10.0))
    beta = float(np.clip(float(eval_cfg.get("precision_servo_beta", 0.35)), 0.0, 1.0))
    max_action = float(eval_cfg.get("precision_servo_max_action", 0.4))
    servo = np.asarray(action, dtype=float).copy()
    servo[:3] = np.clip(gain * (goal[:3] - ee[:3]), -max_action, max_action)
    if prev_action is not None:
        servo = beta * servo + (1.0 - beta) * np.asarray(prev_action, dtype=float)
    return np.clip(servo, -1.0, 1.0)


def filter_action_if_needed(action: np.ndarray, prev_action: np.ndarray | None, algo: str, cfg: dict[str, Any]) -> np.ndarray:
    eval_cfg = cfg.get("eval", {})
    filtered_algos = {str(name).upper() for name in eval_cfg.get("stable_filter_algorithms", [])}
    servo_algos = {str(name).upper() for name in eval_cfg.get("precision_servo_algorithms", [])}
    if algo.upper() in servo_algos:
        return action
    if algo.upper() not in filtered_algos or prev_action is None:
        return action
    beta = float(np.clip(float(eval_cfg.get("action_filter_beta", 0.25)), 0.0, 1.0))
    smoothed = beta * np.asarray(action, dtype=float) + (1.0 - beta) * np.asarray(prev_action, dtype=float)
    return np.clip(smoothed, -1.0, 1.0)


def main() -> None:
    args = parse_args()
    cfg = load_config(args.config)
    if args.fixed_goal:
        cfg = deep_update(cfg, {"env": {"fixed_goal": True}})
    if args.random_goal:
        cfg = deep_update(cfg, {"env": {"fixed_goal": False}})
    cfg = deep_update(cfg, {"env": {"terminate_on_success": False}})
    if args.disturbance == "off":
        cfg = deep_update(cfg, {"disturbance": {"enabled": False}})
    else:
        strength = cfg.get("disturbance", {}).get("strengths", {}).get(args.disturbance, {})
        cfg = deep_update(cfg, {"disturbance": {"enabled": True, **strength}})

    paths = ensure_run_dirs(cfg)
    name = args.output_name or f"{args.algo}_{'fixed' if cfg['env'].get('fixed_goal') else 'random'}_{args.disturbance}"
    rows, aggregate = evaluate_policy(
        cfg=cfg,
        algo=args.algo,
        model_path=args.model,
        episodes=args.episodes,
        render=args.render,
    )

    csv_path = paths["result_dir"] / f"{name}_episodes.csv"
    json_path = paths["result_dir"] / f"{name}_summary.json"
    write_csv(csv_path, rows)
    json_path.write_text(json.dumps(aggregate, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"Saved episode metrics: {csv_path}")
    print(f"Saved summary metrics: {json_path}")
    print(json.dumps(aggregate, indent=2, ensure_ascii=False))


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
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
