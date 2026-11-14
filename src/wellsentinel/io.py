"""Small I/O helpers."""
from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np


def to_jsonable(o):
    """Recursively convert numpy types and replace NaN/inf with None, so the output is strict JSON
    (Python's json would otherwise write bare NaN, which editors and other languages reject)."""
    if isinstance(o, dict):
        return {str(k): to_jsonable(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [to_jsonable(v) for v in o]
    if isinstance(o, np.ndarray):
        return [to_jsonable(v) for v in o.tolist()]
    if isinstance(o, (bool, np.bool_)):
        return bool(o)
    if isinstance(o, (int, np.integer)):
        return int(o)
    if isinstance(o, (float, np.floating)):
        o = float(o)
        return o if math.isfinite(o) else None
    return o


def write_json(obj, path: str | Path) -> None:
    with open(path, "w", encoding="utf-8") as f:
        json.dump(to_jsonable(obj), f, indent=2, allow_nan=False)
