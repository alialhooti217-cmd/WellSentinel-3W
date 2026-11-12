"""Model factory and persistence."""
from __future__ import annotations

from pathlib import Path

import joblib
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.pipeline import make_pipeline


def build_model(cfg: dict):
    """Create the classifier described in config.yaml -> model."""
    mtype = cfg["model"]["type"]
    params = dict(cfg["model"].get("params", {}))
    if mtype == "hist_gb":
        return HistGradientBoostingClassifier(**params)
    if mtype == "random_forest":
        params.pop("learning_rate", None)
        params.pop("max_iter", None)
        params.pop("max_leaf_nodes", None)
        params.pop("l2_regularization", None)
        return make_pipeline(
            SimpleImputer(strategy="median"),
            RandomForestClassifier(n_estimators=300, n_jobs=-1, **params),
        )
    raise ValueError(f"Unknown model type: {mtype}")


def save_bundle(model, feature_cols: list[str], cfg: dict, path: str | Path) -> None:
    """Save the model together with everything needed to use it later."""
    joblib.dump(
        {"model": model, "feature_cols": feature_cols, "windows": cfg["windows"],
         "stats": cfg["features"]["stats"], "alarm": cfg["alarm"]},
        path,
    )


def load_bundle(path: str | Path) -> dict:
    return joblib.load(path)
