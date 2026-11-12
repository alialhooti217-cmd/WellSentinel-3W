"""Configuration loading and project paths."""
from __future__ import annotations

from pathlib import Path

import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG = PROJECT_ROOT / "configs" / "config.yaml"


def load_config(path: str | Path | None = None) -> dict:
    """Load the YAML config and resolve every entry under `paths` to an absolute Path."""
    path = Path(path) if path else DEFAULT_CONFIG
    with open(path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    cfg["paths"] = {k: (PROJECT_ROOT / v).resolve() for k, v in cfg["paths"].items()}
    for key in ("interim_dir", "processed_dir", "models_dir", "reports_dir", "figures_dir"):
        cfg["paths"][key].mkdir(parents=True, exist_ok=True)
    return cfg
