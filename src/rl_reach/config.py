from __future__ import annotations

import copy
from pathlib import Path
from typing import Any

import yaml


def load_config(path: str | Path) -> dict[str, Any]:
    with Path(path).open("r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def deep_update(base: dict[str, Any], overrides: dict[str, Any] | None) -> dict[str, Any]:
    result = copy.deepcopy(base)
    if not overrides:
        return result
    for key, value in overrides.items():
        if isinstance(value, dict) and isinstance(result.get(key), dict):
            result[key] = deep_update(result[key], value)
        else:
            result[key] = copy.deepcopy(value)
    return result


def ensure_run_dirs(cfg: dict[str, Any], root: str | Path = ".") -> dict[str, Path]:
    root = Path(root)
    paths = cfg.get("paths", {})
    run_dir = root / paths.get("run_dir", "runs")
    out = {
        "run_dir": run_dir,
        "model_dir": run_dir / paths.get("model_dir", "models"),
        "result_dir": run_dir / paths.get("result_dir", "results"),
        "figure_dir": run_dir / paths.get("figure_dir", "figures"),
    }
    for path in out.values():
        path.mkdir(parents=True, exist_ok=True)
    return out
