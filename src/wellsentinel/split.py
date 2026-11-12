"""Leakage-safe splitting: whole wells go to either train or test, never both.

Random row-level splits on 3W give unrealistically high scores because
neighbouring windows from the same well/instance are nearly identical.
Grouping by well answers the honest question:
"Does the model work on a well it has never seen?"
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedGroupKFold


def assign_folds(inventory: pd.DataFrame, n_folds: int = 5, seed: int = 42) -> pd.DataFrame:
    """Add a `fold` column. Real instances get 0..n_folds-1 grouped by well;
    simulated / hand-drawn instances get fold = -1 (training-only augmentation)."""
    inv = inventory.copy()
    inv["fold"] = -1
    real = inv[inv["instance_type"] == "real"]
    sgkf = StratifiedGroupKFold(n_splits=n_folds, shuffle=True, random_state=seed)
    import warnings
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=UserWarning)  # rare classes have < n_folds wells
        for fold, (_, test_idx) in enumerate(sgkf.split(real, real["label"], groups=real["well"])):
            inv.loc[real.index[test_idx], "fold"] = fold
    return inv


def train_test_masks(
    table: pd.DataFrame,
    folds: pd.DataFrame,
    test_fold: int = 0,
    include_simulated: bool = True,
    include_drawn: bool = True,
) -> tuple[np.ndarray, np.ndarray]:
    """Boolean masks over the window table. Test = real instances from the held-out wells."""
    fold_of = folds.set_index("instance_id")["fold"]
    f = table["instance_id"].map(fold_of).to_numpy()
    itype = table["instance_type"].to_numpy()
    test = (itype == "real") & (f == test_fold)
    train = (itype == "real") & (f != test_fold)
    if include_simulated:
        train |= itype == "simulated"
    if include_drawn:
        train |= itype == "drawn"
    return train, test
