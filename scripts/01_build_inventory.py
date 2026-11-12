"""Step 1 — catalogue every instance and assign well-grouped CV folds.

Output: data/interim/inventory.csv
"""
import _bootstrap  # noqa: F401

import pandas as pd

from wellsentinel.config import load_config
from wellsentinel.data import build_inventory
from wellsentinel.split import assign_folds


def main() -> None:
    cfg = load_config()
    inv = build_inventory(cfg["paths"]["dataset_dir"])
    inv = assign_folds(inv, cfg["split"]["n_folds"], cfg["split"]["seed"])
    out = cfg["paths"]["interim_dir"] / "inventory.csv"
    inv.to_csv(out, index=False)

    print(f"Saved {len(inv)} instances -> {out}\n")
    summary = pd.crosstab([inv["label"], inv["event"]], inv["instance_type"], margins=True)
    print(summary.to_string())
    print(f"\nReal wells: {inv['well'].nunique()}  |  "
          f"hours of data: {inv['n_rows'].sum() / 3600:,.0f}")
    print("\nReal instances per fold:")
    print(inv[inv.instance_type == "real"].groupby("fold")
          .agg(instances=("instance_id", "size"), wells=("well", "nunique")).to_string())


if __name__ == "__main__":
    main()
