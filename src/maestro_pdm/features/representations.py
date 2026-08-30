from __future__ import annotations

from typing import Dict, Iterable, List, Sequence, Tuple

import numpy as np
import pandas as pd

from maestro_pdm.data.labels import HorizonSample


AGGREGATE_STATISTICS = (
    "mean",
    "std",
    "min",
    "max",
    "final",
    "trend",
)


def aggregate_sequence(X: np.ndarray) -> np.ndarray:
    """
    Convert a variable-length multivariate sequence into the conventional
    summary representation requested for the reviewer control.

    For each dynamic variable, the representation contains:
        mean, standard deviation, minimum, maximum, final value, linear trend.

    The linear trend is the ordinary least-squares slope against a normalized
    position index t in [0, 1]. Normalizing the index prevents slope magnitude
    from being a trivial function of sequence length.
    """
    X = np.asarray(X, dtype=np.float64)

    if X.ndim != 2:
        raise ValueError("X must have shape (time, features).")
    if X.shape[0] < 1:
        raise ValueError("X must contain at least one time point.")
    if not np.all(np.isfinite(X)):
        raise ValueError(
            "aggregate_sequence expects already-imputed finite values."
        )

    mean = X.mean(axis=0)
    std = X.std(axis=0, ddof=0)
    minimum = X.min(axis=0)
    maximum = X.max(axis=0)
    final = X[-1]

    if X.shape[0] == 1:
        trend = np.zeros(X.shape[1], dtype=np.float64)
    else:
        t = np.linspace(0.0, 1.0, X.shape[0], dtype=np.float64)
        t_centered = t - t.mean()
        denom = float(np.dot(t_centered, t_centered))
        trend = (t_centered[:, None] * X).sum(axis=0) / denom

    return np.concatenate(
        [mean, std, minimum, maximum, final, trend],
        axis=0,
    ).astype(np.float32)


def aggregate_feature_names(
    dynamic_feature_names: Sequence[str],
) -> List[str]:
    names: List[str] = []
    for statistic in AGGREGATE_STATISTICS:
        names.extend(
            f"{feature}__{statistic}"
            for feature in dynamic_feature_names
        )
    return names


def _static_vector(
    static_vectors: Dict[int, np.ndarray],
    vehicle_id: int,
) -> np.ndarray:
    if vehicle_id not in static_vectors:
        raise KeyError(
            f"Static vector is missing for vehicle {vehicle_id}."
        )
    return np.asarray(
        static_vectors[vehicle_id],
        dtype=np.float32,
    ).reshape(-1)


def build_prefix_representation(
    samples: Iterable[HorizonSample],
    sequences: Dict[int, Tuple[np.ndarray, np.ndarray]],
    static_vectors: Dict[int, np.ndarray],
) -> tuple[pd.DataFrame, np.ndarray, np.ndarray, np.ndarray]:
    """
    Build aggregate, static, and time-only representations for labelled
    prefix-level samples.

    Returns
    -------
    metadata:
        vehicle_id, prefix_time_step, prefix_length, label.
    aggregate_dynamic:
        shape (n, 6*p)
    static:
        shape (n, q_e)
    time_only:
        [prefix_time_step, prefix_length]
    """
    rows = []
    aggregate_rows = []
    static_rows = []
    time_rows = []

    for sample in samples:
        if sample.label is None:
            continue
        if sample.vehicle_id not in sequences:
            continue
        if sample.vehicle_id not in static_vectors:
            continue

        times, X = sequences[sample.vehicle_id]
        end = int(sample.prefix_index) + 1

        if end <= 0 or end > len(X):
            raise ValueError(
                f"Invalid prefix index {sample.prefix_index} for "
                f"vehicle {sample.vehicle_id} with {len(X)} readouts."
            )

        X_prefix = X[:end]

        aggregate_rows.append(aggregate_sequence(X_prefix))
        static_rows.append(
            _static_vector(static_vectors, sample.vehicle_id)
        )
        time_rows.append(
            np.asarray(
                [float(sample.prefix_time_step), float(end)],
                dtype=np.float32,
            )
        )

        rows.append(
            {
                "vehicle_id": int(sample.vehicle_id),
                "prefix_time_step": int(sample.prefix_time_step),
                "prefix_length": int(end),
                "label": int(sample.label),
            }
        )

    if not rows:
        raise ValueError("No labelled prefix samples were available.")

    return (
        pd.DataFrame(rows),
        np.stack(aggregate_rows).astype(np.float32),
        np.stack(static_rows).astype(np.float32),
        np.stack(time_rows).astype(np.float32),
    )


def build_full_sequence_representation(
    sequences: Dict[int, Tuple[np.ndarray, np.ndarray]],
    static_vectors: Dict[int, np.ndarray],
    vehicle_ids: Iterable[int],
) -> tuple[pd.DataFrame, np.ndarray, np.ndarray, np.ndarray]:
    """
    Build one representation per vehicle from the full available sequence.
    """
    rows = []
    aggregate_rows = []
    static_rows = []
    time_rows = []

    for vehicle_id in sorted(int(v) for v in vehicle_ids):
        if vehicle_id not in sequences:
            raise KeyError(
                f"Operational sequence is missing for validation vehicle "
                f"{vehicle_id}."
            )
        if vehicle_id not in static_vectors:
            raise KeyError(
                f"Static vector is missing for validation vehicle {vehicle_id}."
            )

        times, X = sequences[vehicle_id]
        if len(X) < 1:
            raise ValueError(
                f"Vehicle {vehicle_id} has an empty sequence."
            )

        aggregate_rows.append(aggregate_sequence(X))
        static_rows.append(_static_vector(static_vectors, vehicle_id))
        time_rows.append(
            np.asarray(
                [float(times[-1]), float(len(X))],
                dtype=np.float32,
            )
        )

        rows.append(
            {
                "vehicle_id": vehicle_id,
                "prefix_time_step": int(times[-1]),
                "prefix_length": int(len(X)),
            }
        )

    return (
        pd.DataFrame(rows),
        np.stack(aggregate_rows).astype(np.float32),
        np.stack(static_rows).astype(np.float32),
        np.stack(time_rows).astype(np.float32),
    )
