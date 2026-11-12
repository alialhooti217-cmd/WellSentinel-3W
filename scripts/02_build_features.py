"""Step 2 — turn raw 1 Hz streams into a window-level feature table.

Output: data/processed/features.parquet
Use --limit N for a quick test on N instances per event folder.
"""
import _bootstrap  # noqa: F401

import argparse
import time

import pandas as pd

from wellsentinel.config import load_config
from wellsentinel.features import build_feature_table


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=None, help="max instances per event folder")
    args = ap.parse_args()

    cfg = load_config()
    inv = pd.read_csv(cfg["paths"]["interim_dir"] / "inventory.csv")
    if args.limit:
        inv = inv.groupby(["label", "instance_type"], group_keys=False).head(args.limit)
    print(f"Extracting windows from {len(inv)} instances "
          f"(window {cfg['windows']['size_s']} s, step {cfg['windows']['step_s']} s)...")

    t0 = time.time()
    table = build_feature_table(inv, cfg)
    name = "features.parquet" if not args.limit else f"features_limit{args.limit}.parquet"
    out = cfg["paths"]["processed_dir"] / name
    table.to_parquet(out, index=False)

    print(f"\n{len(table):,} windows x {table.shape[1]} columns in {time.time() - t0:.0f} s -> {out}")
    print(table.groupby(["target", "is_transient"]).size().unstack(fill_value=0).to_string())


if __name__ == "__main__":
    main()
