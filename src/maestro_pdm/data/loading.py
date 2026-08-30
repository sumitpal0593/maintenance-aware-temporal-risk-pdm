from __future__ import annotations

from pathlib import Path
from typing import Dict, Iterator, Optional, Set, Tuple

import numpy as np
import pandas as pd


REQUIRED_FILES = (
    "train_operational_readouts.csv",
    "validation_operational_readouts.csv",
    "train_specifications.csv",
    "validation_specifications.csv",
    "train_tte.csv",
    "validation_labels.csv",
)


def validate_dataset_directory(data_dir: Path) -> None:
    """Raise a clear error if required public dataset files are missing."""
    missing = [name for name in REQUIRED_FILES if not (data_dir / name).exists()]
    if missing:
        formatted = "\n  - ".join(missing)
        raise FileNotFoundError(
            "The following required Scania Component X files were not found:\n"
            f"  - {formatted}\n"
            f"Checked directory: {data_dir}"
        )


def detect_id_time_columns(df: pd.DataFrame) -> Tuple[str, str]:
    """Detect vehicle identifier and relative time-step columns."""
    lower_to_original = {c.lower(): c for c in df.columns}

    id_col = next(
        (
            lower_to_original[x]
            for x in ("vehicle_id", "veh_id", "id")
            if x in lower_to_original
        ),
        None,
    )
    time_col = next(
        (
            lower_to_original[x]
            for x in ("time_step", "timestep", "time", "step")
            if x in lower_to_original
        ),
        None,
    )

    if id_col is None or time_col is None:
        raise ValueError(
            "Could not detect vehicle_id and/or time_step columns. "
            f"Columns found: {list(df.columns)}"
        )

    return id_col, time_col


def detect_id_column(df: pd.DataFrame) -> str:
    """Detect the vehicle identifier column in a non-temporal table."""
    lower_to_original = {c.lower(): c for c in df.columns}
    for name in ("vehicle_id", "veh_id", "id"):
        if name in lower_to_original:
            return lower_to_original[name]
    raise ValueError(f"Could not detect vehicle ID column. Columns: {list(df.columns)}")


def operational_columns(
    data_dir: Path,
    split: str,
) -> Tuple[str, str, list[str]]:
    """Return (id_col, time_col, operational_feature_cols)."""
    path = data_dir / f"{split}_operational_readouts.csv"
    header = pd.read_csv(path, nrows=5)
    id_col, time_col = detect_id_time_columns(header)
    feature_cols = [c for c in header.columns if c not in {id_col, time_col}]
    return id_col, time_col, feature_cols


def read_validation_labels(data_dir: Path) -> pd.DataFrame:
    return pd.read_csv(data_dir / "validation_labels.csv")


def read_train_tte(data_dir: Path) -> pd.DataFrame:
    return pd.read_csv(data_dir / "train_tte.csv")


def read_specifications(data_dir: Path, split: str) -> pd.DataFrame:
    if split not in {"train", "validation", "test"}:
        raise ValueError("split must be one of: train, validation, test")
    path = data_dir / f"{split}_specifications.csv"
    if not path.exists():
        raise FileNotFoundError(path)
    return pd.read_csv(path)


def iter_operational_chunks(
    data_dir: Path,
    split: str,
    chunk_rows: int = 400_000,
    usecols: Optional[list[str]] = None,
) -> Iterator[pd.DataFrame]:
    """Stream the large operational-readout CSV instead of loading it all at once."""
    if split not in {"train", "validation", "test"}:
        raise ValueError("split must be one of: train, validation, test")

    path = data_dir / f"{split}_operational_readouts.csv"
    if not path.exists():
        raise FileNotFoundError(path)

    yield from pd.read_csv(path, chunksize=chunk_rows, usecols=usecols)


def collect_vehicle_times(
    data_dir: Path,
    split: str,
    chunk_rows: int,
    allowed_vehicle_ids: Optional[Set[int]] = None,
) -> Tuple[Dict[int, np.ndarray], str, str]:
    """
    Collect only vehicle IDs and relative time-step values.

    This is deliberately lightweight: label auditing does not require loading
    the 105 operational feature columns into memory.
    """
    id_col, time_col, _ = operational_columns(data_dir, split)
    parts: Dict[int, list[np.ndarray]] = {}

    for chunk in iter_operational_chunks(
        data_dir,
        split,
        chunk_rows=chunk_rows,
        usecols=[id_col, time_col],
    ):
        chunk[id_col] = pd.to_numeric(chunk[id_col], errors="coerce")
        chunk[time_col] = pd.to_numeric(chunk[time_col], errors="coerce")
        chunk = chunk.dropna(subset=[id_col, time_col])

        if allowed_vehicle_ids is not None:
            chunk = chunk[chunk[id_col].astype(int).isin(allowed_vehicle_ids)]

        if chunk.empty:
            continue

        for vid, group in chunk.groupby(id_col, sort=False):
            arr = group[time_col].to_numpy(dtype=np.int64)
            parts.setdefault(int(vid), []).append(arr)

    times: Dict[int, np.ndarray] = {}
    for vid, arrays in parts.items():
        all_t = np.concatenate(arrays)
        times[vid] = np.sort(all_t)

    return times, id_col, time_col
