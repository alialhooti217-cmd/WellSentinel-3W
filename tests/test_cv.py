"""CV split invariants (synthetic table, no dataset needed)."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import numpy as np
import pandas as pd

from wellsentinel.cv import cv_splits, window_scores


def _table():
    rng = np.random.default_rng(0)
    rows, inv = [], []
    for w in range(10):
        for i in range(2):
            iid = f"{w % 3}/WELL-{w:05d}_{i}"
            inv.append({"instance_id": iid, "fold": w % 5})
            rows += [{"instance_id": iid, "instance_type": "real", "well": f"WELL-{w:05d}",
                      "target": int(rng.integers(0, 3))} for _ in range(20)]
    for s in range(4):
        inv.append({"instance_id": f"1/SIM_{s}", "fold": -1})
        rows += [{"instance_id": f"1/SIM_{s}", "instance_type": "simulated" if s % 2 else "drawn",
                  "well": None, "target": 1} for _ in range(10)]
    return pd.DataFrame(rows), pd.DataFrame(inv)


def test_grouped_and_random_splits_keep_synthetic_out_of_test():
    t, inv = _table()
    synth = (t["instance_type"] != "real").to_numpy()
    for scheme in ("grouped", "random"):
        test_union = np.zeros(len(t), bool)
        for _, tr, te in cv_splits(t, inv, scheme, n_folds=5, seed=0):
            assert not (tr & te).any()
            assert not te[synth].any() and tr[synth].all()
            test_union |= te
        assert test_union[~synth].all()  # every real window tested exactly once overall


def test_grouped_split_never_shares_a_well():
    t, inv = _table()
    for _, tr, te in cv_splits(t, inv, "grouped", n_folds=5):
        assert not set(t.loc[tr, "well"].dropna()) & set(t.loc[te, "well"])


def test_macro_f1_uses_classes_present_in_truth():
    s = window_scores([0, 0, 2, 2], [0, 0, 2, 1])  # class 1 predicted but absent from truth
    assert np.isclose(s["macro_f1"], np.mean([1.0, 2 / 3]))
    assert s["macro_f1_union"] < s["macro_f1"]


def test_write_json_is_strict(tmp_path):
    import json

    from wellsentinel.io import write_json
    p = tmp_path / "x.json"
    write_json({"a": np.nan, "b": np.float64(1.5), "c": [np.inf, np.int64(2)], "d": np.bool_(True)}, p)
    txt = p.read_text()
    assert "NaN" not in txt and "Infinity" not in txt
    assert json.loads(txt) == {"a": None, "b": 1.5, "c": [None, 2], "d": True}
