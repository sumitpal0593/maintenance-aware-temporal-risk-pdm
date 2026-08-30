from __future__ import annotations

from typing import Dict, Optional, Set, Tuple

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

from .labels import HorizonSample
from .loading import iter_operational_chunks, operational_columns
from .preprocessing import OperationalPreprocessor


def load_preprocessed_sequences(
    data_dir,
    split: str,
    preprocessor: OperationalPreprocessor,
    chunk_rows: int,
    allowed_vehicle_ids: Optional[Set[int]] = None,
    max_vehicles: Optional[int] = None,
) -> Tuple[Dict[int, Tuple[np.ndarray, np.ndarray]], str, str]:
    """
    Load and normalize variable-length operational sequences.

    This keeps the original chunked-loading idea while using a preprocessor
    fitted only on the model-training split.
    """
    id_col, time_col, feature_cols = operational_columns(data_dir, split)

    if feature_cols != preprocessor.feature_names:
        raise ValueError(
            "Operational feature columns do not match the fitted preprocessor."
        )

    parts: Dict[int, list[Tuple[np.ndarray, np.ndarray]]] = {}
    seen: Set[int] = set()

    for chunk in iter_operational_chunks(
        data_dir,
        split,
        chunk_rows=chunk_rows,
    ):
        ids_numeric = pd.to_numeric(chunk[id_col], errors="coerce")
        chunk = chunk.loc[ids_numeric.notna()].copy()
        chunk[id_col] = pd.to_numeric(chunk[id_col], errors="coerce").astype(int)

        if allowed_vehicle_ids is not None:
            chunk = chunk[chunk[id_col].isin(allowed_vehicle_ids)]

        if chunk.empty:
            continue

        chunk[time_col] = pd.to_numeric(chunk[time_col], errors="coerce")
        chunk = chunk.dropna(subset=[time_col])

        for vid, group in chunk.groupby(id_col, sort=False):
            vid = int(vid)
            if (
                max_vehicles is not None
                and vid not in seen
                and len(seen) >= max_vehicles
            ):
                continue

            seen.add(vid)
            t = group[time_col].to_numpy(dtype=np.int64)
            raw_X = (
                group[feature_cols]
                .apply(pd.to_numeric, errors="coerce")
                .to_numpy(dtype=np.float64)
            )
            X = preprocessor.transform(raw_X)
            parts.setdefault(vid, []).append((t, X))

    sequences: Dict[int, Tuple[np.ndarray, np.ndarray]] = {}
    for vid, vehicle_parts in parts.items():
        t_all = np.concatenate([x[0] for x in vehicle_parts])
        X_all = np.concatenate([x[1] for x in vehicle_parts], axis=0)
        order = np.argsort(t_all)
        sequences[vid] = (t_all[order], X_all[order])

    return sequences, id_col, time_col


class HorizonPrefixDataset(Dataset):
    """
    Prefix-level fixed-horizon dataset.

    Only samples with a known binary horizon outcome are accepted. Censored
    prefixes whose full tau-window is not observed are excluded upstream.
    """

    def __init__(
        self,
        sequences: Dict[int, Tuple[np.ndarray, np.ndarray]],
        static_vectors: Dict[int, np.ndarray],
        samples: list[HorizonSample],
        split: str,
    ):
        self.sequences = sequences
        self.static_vectors = static_vectors
        self.samples = [
            s
            for s in samples
            if s.split == split
            and s.label is not None
            and s.vehicle_id in sequences
            and s.vehicle_id in static_vectors
        ]

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        sample = self.samples[index]
        _, X = self.sequences[sample.vehicle_id]

        # prefix_index is inclusive
        X_prefix = X[: sample.prefix_index + 1]
        static = self.static_vectors[sample.vehicle_id]

        return (
            X_prefix.astype(np.float32),
            static.astype(np.float32),
            np.float32(sample.label),
            int(sample.vehicle_id),
            int(sample.prefix_time_step),
        )


def collate_variable_length(batch):
    X_list, static_list, y_list, vid_list, prefix_time_list = zip(*batch)

    lengths = torch.tensor(
        [x.shape[0] for x in X_list],
        dtype=torch.long,
    )
    batch_size = len(X_list)
    n_features = X_list[0].shape[1]
    max_length = int(lengths.max())

    X_pad = torch.zeros(
        (batch_size, max_length, n_features),
        dtype=torch.float32,
    )
    for i, x in enumerate(X_list):
        X_pad[i, : x.shape[0], :] = torch.from_numpy(x)

    static = torch.from_numpy(np.stack(static_list).astype(np.float32))
    y = torch.tensor(y_list, dtype=torch.float32)
    vids = torch.tensor(vid_list, dtype=torch.long)
    prefix_times = torch.tensor(prefix_time_list, dtype=torch.long)

    return X_pad, lengths, static, y, vids, prefix_times
