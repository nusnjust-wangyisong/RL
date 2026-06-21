from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any

from rl_reach.config import deep_update, ensure_run_dirs, load_config
from rl_reach.evaluate import evaluate_policy


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Compare IK execution/training variants on fixed and random goals.")
    parser.add_argument("--config", default="configs/experiment_irb120.yaml")
    parser.add_argument("--model-dir", default=None)
    parser.add_argument("--episodes", type=int, default=30)
    parser.add_argument("--disturbance", choices=["off", "light", "medium", "strong"], default="medium")
    parser.add_argument("--output", default="ik_variant_summary.csv")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    base_cfg = load_config(args.config)
    paths = ensure_run_dirs(base_cfg)
    model_dir = Path(args.model_dir) if args.model_dir else paths["model_dir"]
    rows: list[dict[str, Any]] = []

    variants = [
        {
            "variant": "ik_servo",
            "model_prefix": "TD3_HER_CURRICULUM",
            "eval": {"precision_servo_mode": "ik_servo"},
        },
        {
            "variant": "ik_residual",
            "model_prefix": "TD3_HER_IK_RESIDUAL",
            "eval": {"precision_servo_mode": "ik_residual"},
        },
        {
            "variant": "her_ik_residual",
            "model_prefix": "TD3_HER_CURRICULUM",
            "eval": {"precision_servo_mode": "her_ik_residual"},
        },
        {
            "variant": "ik_observation",
            "model_prefix": "TD3_HER_IK_OBS",
            "env": {"include_ik_observation": True},
            "eval": {"precision_servo_algorithms": []},
        },
    ]

    for fixed_goal in (True, False):
        task = "fixed" if fixed_goal else "random"
        for variant in variants:
            model_path = model_dir / f"{variant['model_prefix']}_{task}.zip"
            if not model_path.exists():
                print(f"Skip missing model: {model_path}")
                continue
            cfg = deep_update(base_cfg, {"env": {"fixed_goal": fixed_goal, "terminate_on_success": False}})
            if args.disturbance == "off":
                cfg = deep_update(cfg, {"disturbance": {"enabled": False}})
            else:
                strength = cfg.get("disturbance", {}).get("strengths", {}).get(args.disturbance, {})
                cfg = deep_update(cfg, {"disturbance": {"enabled": True, **strength}})
            cfg = deep_update(cfg, {k: v for k, v in variant.items() if k in {"env", "eval"}})
            _episodes, summary = evaluate_policy(
                cfg=cfg,
                algo="TD3_HER_CURRICULUM",
                model_path=model_path,
                episodes=args.episodes,
                render=False,
            )
            row = {
                "variant": variant["variant"],
                "task": task,
                "disturbance": args.disturbance,
                **summary,
            }
            rows.append(row)
            print(json.dumps(row, ensure_ascii=False))

    out_path = paths["result_dir"] / args.output
    write_rows(out_path, rows)
    print(f"Saved IK variant summary: {out_path}")


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
