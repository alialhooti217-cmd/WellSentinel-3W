"""Loading the 3W dataset (v2.0.0, Parquet) and building an instance inventory."""
from __future__ import annotations

from pathlib import Path

import pandas as pd
import pyarrow.parquet as pq

# Event types defined in dataset/dataset.ini
EVENT_NAMES: dict[int, str] = {
    0: "Normal Operation",
    1: "Abrupt Increase of BSW",
    2: "Spurious Closure of DHSV",
    3: "Severe Slugging",
    4: "Flow Instability",
    5: "Rapid Productivity Loss",
    6: "Quick Restriction in PCK",
    7: "Scaling in PCK",
    8: "Hydrate in Production Line",
    9: "Hydrate in Service Line",
}
TRANSIENT_OFFSET = 100  # class 101..109 = transient phase of event 1..9

# The 27 process variables (sensor / valve-state tags)
VARIABLES: list[str] = [
    "ABER-CKGL", "ABER-CKP", "ESTADO-DHSV", "ESTADO-M1", "ESTADO-M2", "ESTADO-PXO",
    "ESTADO-SDV-GL", "ESTADO-SDV-P", "ESTADO-W1", "ESTADO-W2", "ESTADO-XO",
    "P-ANULAR", "P-JUS-BS", "P-JUS-CKGL", "P-JUS-CKP", "P-MON-CKGL", "P-MON-CKP",
    "P-MON-SDV-P", "P-PDG", "PT-P", "P-TPT", "QBS", "QGL",
    "T-JUS-CKP", "T-MON-CKP", "T-PDG", "T-TPT",
]

# Plain-English descriptions (from dataset.ini) used in plots and the dashboard
VARIABLE_DESCRIPTIONS: dict[str, str] = {
    "ABER-CKGL": "Gas-lift choke opening [%]",
    "ABER-CKP": "Production choke opening [%]",
    "P-ANULAR": "Annulus pressure [Pa]",
    "P-PDG": "Downhole pressure (PDG) [Pa]",
    "P-TPT": "Xmas-tree pressure (TPT) [Pa]",
    "P-MON-CKP": "Upstream production choke pressure [Pa]",
    "P-JUS-CKP": "Downstream production choke pressure [Pa]",
    "P-JUS-CKGL": "Downstream gas-lift choke pressure [Pa]",
    "QGL": "Gas-lift flow rate [m3/s]",
    "T-TPT": "Xmas-tree temperature (TPT) [degC]",
    "T-PDG": "Downhole temperature (PDG) [degC]",
    "T-JUS-CKP": "Downstream production choke temperature [degC]",
    "T-MON-CKP": "Upstream production choke temperature [degC]",
    "P-MON-CKGL": "Upstream gas-lift choke pressure [Pa]",
    "P-MON-SDV-P": "Upstream production SDV pressure [Pa]",
    "P-JUS-BS": "Downstream service-pump pressure [Pa]",
    "PT-P": "Xmas-tree pressure downstream of production wing valve [Pa]",
    "QBS": "Service-pump flow rate [m3/s]",
}


def instance_meta(path: str | Path) -> dict:
    """Parse metadata from a file path like dataset/8/WELL-00021_20170509013517.parquet."""
    path = Path(path)
    source = path.stem.split("_")[0]
    if source.startswith("WELL"):
        itype, well = "real", source
    elif source == "SIMULATED":
        itype, well = "simulated", None
    else:
        itype, well = "drawn", None
    return {
        "instance_id": f"{path.parent.name}/{path.stem}",
        "path": str(path),
        "label": int(path.parent.name),
        "event": EVENT_NAMES[int(path.parent.name)],
        "instance_type": itype,
        "well": well,
    }


def build_inventory(dataset_dir: str | Path, with_rows: bool = True) -> pd.DataFrame:
    """One row per instance file: label, real/simulated/drawn, well and (optionally) length."""
    dataset_dir = Path(dataset_dir)
    files = sorted(dataset_dir.glob("[0-9]/*.parquet"))
    if not files:
        raise FileNotFoundError(
            f"No Parquet files found in {dataset_dir}. Run `python scripts/00_download_data.py` first."
        )
    rows = []
    for f in files:
        meta = instance_meta(f)
        if with_rows:
            meta["n_rows"] = pq.ParquetFile(f).metadata.num_rows
        rows.append(meta)
    return pd.DataFrame(rows)


def load_instance(path: str | Path) -> pd.DataFrame:
    """Read one instance. Index = timestamp; columns = 27 variables + `class` + `state`."""
    df = pd.read_parquet(path)
    for col in VARIABLES:
        if col not in df.columns:
            df[col] = float("nan")
    return df


def to_event(cls) -> int | None:
    """Map a raw class (0, 1..9, 101..109) to the event id 0..9."""
    if pd.isna(cls):
        return None
    cls = int(cls)
    return cls - TRANSIENT_OFFSET if cls >= TRANSIENT_OFFSET else cls


def is_transient(cls) -> bool:
    return (not pd.isna(cls)) and int(cls) >= TRANSIENT_OFFSET
