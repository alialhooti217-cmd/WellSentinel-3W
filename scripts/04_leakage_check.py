"""Step 4 — the 'leakage gap': random window split vs. unseen-well split.

Same features, same model, same settings. Only the way we split changes.
Output: reports/leakage_check.json

SUPERSEDED by scripts/06_cross_validation.py (WORK_LOG E02). This script compares one random 20 % split
against ONE grouped fold with a different class mix, which is not like-for-like. Kept for history only.
"""
import _bootstrap  # noqa: F401

import json

import numpy as np
import pandas as pd
from sklearn.metrics import f1_score
from sklearn.model_selection import train_test_split

from wellsentinel.config import load_config
from wellsentinel.io import write_json
from wellsentinel.features import select_feature_columns
from wellsentinel.models import build_model


def main() -> None:
    cfg = load_config()
    P = cfg["paths"]
    table = pd.read_parquet(P["processed_dir"] / "features.parquet")
    feat_cols = select_feature_columns(table, cfg["features"]["drop_vars_missing_above"])

    real = table[table["instance_type"] == "real"]
    synth = table[table["instance_type"] != "real"]
    tr_idx, te_idx = train_test_split(real.index, test_size=0.2, random_state=cfg["split"]["seed"],
                                      stratify=real["target"])
    train = pd.concat([real.loc[tr_idx], synth])
    test = real.loc[te_idx]

    model = build_model(cfg).fit(train[feat_cols], train["target"])
    pred = model.predict(test[feat_cols])
    random_f1 = float(f1_score(test["target"], pred, average="macro"))

    with open(P["reports_dir"] / "metrics_baseline.json", encoding="utf-8") as f:
        well_f1 = json.load(f)["window"]["macro_f1"]

    out = {"random_split_macro_f1": random_f1, "unseen_well_macro_f1": well_f1,
           "inflation_factor": random_f1 / well_f1}
    write_json(out, P["reports_dir"] / "leakage_check.json")
    print(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
