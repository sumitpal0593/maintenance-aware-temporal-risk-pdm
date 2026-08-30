from __future__ import annotations

from pathlib import Path
from typing import Dict, Iterable, Sequence

import joblib
import numpy as np
import pandas as pd
import torch
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader

from maestro_pdm.data.datasets import (
    HorizonPrefixDataset,
    collate_variable_length,
)
from maestro_pdm.evaluation.metrics import binary_ranking_metrics


def fit_logistic_control(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_tuning: np.ndarray,
    y_tuning: np.ndarray,
    C_grid: Sequence[float],
    seed: int,
) -> tuple[Pipeline, pd.DataFrame, float]:
    """
    Tune an L2 logistic-regression control on the vehicle-level tuning set.

    The public validation set is never consulted during hyperparameter
    selection. Class balancing is learned from the model-training labels only.
    """
    X_train = np.asarray(X_train, dtype=np.float32)
    X_tuning = np.asarray(X_tuning, dtype=np.float32)
    y_train = np.asarray(y_train, dtype=int).reshape(-1)
    y_tuning = np.asarray(y_tuning, dtype=int).reshape(-1)

    if len(X_train) != len(y_train):
        raise ValueError("Training features and labels are misaligned.")
    if len(X_tuning) != len(y_tuning):
        raise ValueError("Tuning features and labels are misaligned.")
    if set(np.unique(y_train)) != {0, 1}:
        raise ValueError("Training labels must contain both classes.")
    if set(np.unique(y_tuning)) != {0, 1}:
        raise ValueError("Tuning labels must contain both classes.")

    rows = []
    fitted: Dict[float, Pipeline] = {}

    for C in [float(x) for x in C_grid]:
        model = Pipeline(
            [
                ("scale", StandardScaler()),
                (
                    "classifier",
                    LogisticRegression(
                        C=C,
                        solver="lbfgs",
                        class_weight="balanced",
                        max_iter=3000,
                        random_state=int(seed),
                    ),
                ),
            ]
        )
        model.fit(X_train, y_train)
        p_tune = model.predict_proba(X_tuning)[:, 1]
        metrics = binary_ranking_metrics(y_tuning, p_tune)

        rows.append(
            {
                "C": C,
                "tuning_auroc": metrics["auroc"],
                "tuning_pr_auc": metrics["pr_auc"],
                "tuning_enrichment": metrics["enrichment"],
            }
        )
        fitted[C] = model

    tuning_table = pd.DataFrame(rows).sort_values(
        ["tuning_pr_auc", "tuning_auroc", "C"],
        ascending=[False, False, True],
        kind="mergesort",
    ).reset_index(drop=True)

    best_C = float(tuning_table.loc[0, "C"])
    return fitted[best_C], tuning_table, best_C


def extract_gru_prefix_embeddings(
    model,
    sequences,
    static_vectors,
    samples,
    split: str,
    device: torch.device,
    batch_size: int,
) -> tuple[pd.DataFrame, np.ndarray]:
    """
    Extract frozen h_i=f_thetaT(X_i) representations for prefix samples.
    """
    dataset = HorizonPrefixDataset(
        sequences=sequences,
        static_vectors=static_vectors,
        samples=samples,
        split=split,
    )

    if len(dataset) == 0:
        raise ValueError(
            f"No labelled samples available for GRU representation: {split}"
        )

    loader = DataLoader(
        dataset,
        batch_size=int(batch_size),
        shuffle=False,
        num_workers=0,
        collate_fn=collate_variable_length,
    )

    model.eval()
    rows = []
    embeddings = []

    with torch.no_grad():
        for X_pad, lengths, static, y, vids, prefix_times in loader:
            h = model.encode_temporal(
                X_pad.to(device),
                lengths.to(device),
            )
            h_np = h.cpu().numpy().astype(np.float32)
            embeddings.append(h_np)

            for i in range(len(y)):
                rows.append(
                    {
                        "vehicle_id": int(vids[i].item()),
                        "prefix_time_step": int(
                            prefix_times[i].item()
                        ),
                        "label": int(y[i].item()),
                    }
                )

    return (
        pd.DataFrame(rows),
        np.concatenate(embeddings, axis=0),
    )


def extract_gru_full_embeddings(
    model,
    sequences,
    vehicle_ids: Iterable[int],
    device: torch.device,
    batch_size: int,
) -> tuple[pd.DataFrame, np.ndarray]:
    """
    Extract one frozen temporal embedding from the full available history of
    each requested vehicle.
    """
    model.eval()

    vids = sorted(int(v) for v in vehicle_ids)
    metadata_rows = []
    embedding_rows = []

    with torch.no_grad():
        for start in range(0, len(vids), int(batch_size)):
            batch_vids = vids[start : start + int(batch_size)]
            X_list = []
            lengths = []

            for vid in batch_vids:
                if vid not in sequences:
                    raise KeyError(
                        f"Missing operational sequence for vehicle {vid}."
                    )
                times, X = sequences[vid]
                X_list.append(X.astype(np.float32))
                lengths.append(len(X))
                metadata_rows.append(
                    {
                        "vehicle_id": vid,
                        "prefix_time_step": int(times[-1]),
                        "prefix_length": int(len(X)),
                    }
                )

            lengths_tensor = torch.tensor(
                lengths,
                dtype=torch.long,
            )
            max_len = int(max(lengths))
            n_features = int(X_list[0].shape[1])

            X_pad = torch.zeros(
                (len(batch_vids), max_len, n_features),
                dtype=torch.float32,
            )
            for i, X in enumerate(X_list):
                X_pad[i, : len(X), :] = torch.from_numpy(X)

            h = model.encode_temporal(
                X_pad.to(device),
                lengths_tensor.to(device),
            )
            embedding_rows.append(
                h.cpu().numpy().astype(np.float32)
            )

    return (
        pd.DataFrame(metadata_rows),
        np.concatenate(embedding_rows, axis=0),
    )


def save_control_model(model: Pipeline, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, path)
