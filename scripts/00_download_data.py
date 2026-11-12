"""Download the Petrobras 3W dataset (v2.0.0, ~1.9 GB, 2,228 Parquet files).

Resumable: files that already exist with the right size are skipped,
so you can stop and re-run at any time.

    python scripts/00_download_data.py            # everything
    python scripts/00_download_data.py --labels 0 8   # only some event folders

Data licence: CC BY 4.0 — (c) Petrobras, https://github.com/petrobras/3W
"""
from __future__ import annotations

import argparse
import json
import urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

REPO = "petrobras/3W"
BRANCH = "main"
ROOT = Path(__file__).resolve().parents[1] / "data" / "raw" / "3W"
EXTRA = {"dataset/dataset.ini", "dataset/LICENSE-CC-BY", "dataset/README.md"}


def list_files(labels: set[str] | None) -> list[tuple[str, int]]:
    url = f"https://api.github.com/repos/{REPO}/git/trees/{BRANCH}?recursive=1"
    with urllib.request.urlopen(url, timeout=60) as r:
        tree = json.load(r)["tree"]
    out = []
    for x in tree:
        p = x["path"]
        if x["type"] != "blob" or not p.startswith("dataset/"):
            continue
        parts = p.split("/")
        is_data = len(parts) == 3 and parts[1].isdigit() and p.endswith(".parquet")
        if p in EXTRA or (is_data and (labels is None or parts[1] in labels)):
            out.append((p, x["size"]))
    return out


def fetch(path: str, size: int) -> str:
    dest = ROOT / path
    if dest.exists() and dest.stat().st_size == size:
        return "skip"
    dest.parent.mkdir(parents=True, exist_ok=True)
    url = f"https://raw.githubusercontent.com/{REPO}/{BRANCH}/{path}"
    tmp = dest.with_suffix(dest.suffix + ".part")
    for attempt in range(3):
        try:
            with urllib.request.urlopen(url, timeout=120) as r, open(tmp, "wb") as f:
                f.write(r.read())
            tmp.replace(dest)
            return "ok"
        except Exception as e:  # noqa: BLE001
            if attempt == 2:
                return f"fail: {e}"
    return "fail"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--labels", nargs="*", help="event folders to download, e.g. 0 8 9")
    ap.add_argument("--workers", type=int, default=16)
    args = ap.parse_args()
    files = list_files(set(args.labels) if args.labels else None)
    print(f"{len(files)} files, {sum(s for _, s in files) / 1e9:.2f} GB -> {ROOT}")
    done = {"ok": 0, "skip": 0, "fail": 0}
    with ThreadPoolExecutor(args.workers) as ex:
        futs = {ex.submit(fetch, p, s): p for p, s in files}
        for i, fut in enumerate(as_completed(futs), 1):
            res = fut.result()
            done["fail" if res.startswith("fail") else res] += 1
            if res.startswith("fail"):
                print("  !", futs[fut], res)
            if i % 100 == 0 or i == len(files):
                print(f"  {i}/{len(files)}  {done}")
    print("Done." if done["fail"] == 0 else "Some files failed - just run the script again.")


if __name__ == "__main__":
    main()
