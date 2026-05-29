from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

from rl_reach.config import deep_update, ensure_run_dirs, load_config
from rl_reach.evaluate import evaluate_policy


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate a full algorithm suite from saved models.")
    parser.add_argument("--config", default="configs/experiment.yaml")
    parser.add_argument("--model-dir", default="runs/models")
    parser.add_argument("--episodes", type=int, default=30)
    parser.add_argument("--output", default="suite_summary.csv")
    parser.add_argument("--disturbance", choices=["off", "light", "medium", "strong"], default="medium")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    base_cfg = load_config(args.config)
    paths = ensure_run_dirs(base_cfg)
    rows: list[dict[str, object]] = []
    algorithms = base_cfg.get("experiments", {}).get("algorithms", [])
    model_dir = Path(args.model_dir)

    for fixed_goal in (True, False):
        for algo in algorithms:
            model_path = model_dir / f"{algo}_{'fixed' if fixed_goal else 'random'}.zip"
            if not model_path.exists():
                print(f"Skip missing model: {model_path}")
                continue
            cfg = deep_update(base_cfg, {"env": {"fixed_goal": fixed_goal, "terminate_on_success": False}})
            if args.disturbance == "off":
                cfg = deep_update(cfg, {"disturbance": {"enabled": False}})
            else:
                strength = cfg.get("disturbance", {}).get("strengths", {}).get(args.disturbance, {})
                cfg = deep_update(cfg, {"disturbance": {"enabled": True, **strength}})
            _episodes, summary = evaluate_policy(
                cfg=cfg,
                algo=algo,
                model_path=model_path,
                episodes=args.episodes,
                render=False,
            )
            row: dict[str, object] = {
                "algo": algo,
                "task": "fixed" if fixed_goal else "random",
                "disturbance": args.disturbance,
                **summary,
            }
            rows.append(row)
            print(json.dumps(row, ensure_ascii=False))

    out_path = paths["result_dir"] / args.output
    write_rows(out_path, rows)
    print(f"Saved suite summary: {out_path}")


def write_rows(path: Path, rows: list[dict[str, object]]) -> None:
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
