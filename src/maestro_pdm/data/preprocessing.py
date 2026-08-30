from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Optional, Set, Tuple

import joblib
import numpy as np
import pandas as pd
from sklearn.preprocessing import OneHotEncoder

from .loading import (
    detect_id_column,
    iter_operational_chunks,
    operational_columns,
)


@dataclass
class OperationalPreprocessor:
    feature_names: list[str]
    means: np.ndarray
    stds: np.ndarray
    observed_counts: np.ndarray
    total_rows: int
    epsilon: float = 1e-6

    def transform(self, X: np.ndarray) -> np.ndarray:
        """
        Mean-impute and standardize an operational feature matrix.

        Missing/non-finite values are replaced by training-only feature means.
        """
        X = np.asarray(X, dtype=np.float64)
        if X.ndim != 2 or X.shape[1] != len(self.feature_names):
            raise ValueError(
                f"Expected X shape (*, {len(self.feature_names)}), got {X.shape}"
            )

        finite = np.isfinite(X)
        X_imp = np.where(finite, X, self.means[None, :])
        X_scaled = (X_imp - self.means[None, :]) / self.stds[None, :]
        return X_scaled.astype(np.float32)

    @property
    def missing_counts(self) -> np.ndarray:
        return self.total_rows - self.observed_counts

    @property
    def missing_rates(self) -> np.ndarray:
        return self.missing_counts / float(self.total_rows)

    def summary_frame(self) -> pd.DataFrame:
        return pd.DataFrame(
            {
                "feature": self.feature_names,
                "training_rows": self.total_rows,
                "observed_count": self.observed_counts.astype(np.int64),
                "missing_count": self.missing_counts.astype(np.int64),
                "missing_rate": self.missing_rates,
                "imputation_mean": self.means,
                "post_imputation_std": self.stds,
            }
        )


def fit_operational_preprocessor(
    data_dir: Path,
    allowed_vehicle_ids: Set[int],
    chunk_rows: int,
    epsilon: float = 1e-6,
) -> OperationalPreprocessor:
    """
    Fit operational preprocessing on model-training vehicles only.

    The implementation streams the full training operational CSV. It computes
    training-only feature means and the standard deviation *after mean
    imputation*, matching the revised manuscript:

        x_imp = x if observed else mu_train
        x_tilde = (x_imp - mu_train) / (sigma_train + epsilon)

    Because mean-imputed entries equal the feature mean, they contribute zero
    squared deviation. This permits exact one-pass sufficient-statistic
    calculation without loading the >1 GB CSV into memory.
    """
    id_col, time_col, feature_cols = operational_columns(data_dir, "train")
    n_features = len(feature_cols)

    sums = np.zeros(n_features, dtype=np.float64)
    sums_sq = np.zeros(n_features, dtype=np.float64)
    observed_counts = np.zeros(n_features, dtype=np.int64)
    total_rows = 0

    usecols = [id_col, time_col] + feature_cols

    for chunk in iter_operational_chunks(
        data_dir,
        "train",
        chunk_rows=chunk_rows,
        usecols=usecols,
    ):
        ids = pd.to_numeric(chunk[id_col], errors="coerce")
        mask = ids.notna() & ids.astype("Int64").isin(allowed_vehicle_ids)
        chunk = chunk.loc[mask]

        if chunk.empty:
            continue

        numeric = chunk[feature_cols].apply(pd.to_numeric, errors="coerce")
        arr = numeric.to_numpy(dtype=np.float64)
        finite = np.isfinite(arr)

        total_rows += arr.shape[0]
        observed_counts += finite.sum(axis=0).astype(np.int64)

        safe = np.where(finite, arr, 0.0)
        sums += safe.sum(axis=0)
        sums_sq += np.square(safe).sum(axis=0)

    if total_rows == 0:
        raise RuntimeError(
            "No operational rows were found for model-training vehicle IDs."
        )
    if np.any(observed_counts == 0):
        bad = [
            feature_cols[i]
            for i, count in enumerate(observed_counts)
            if count == 0
        ]
        raise RuntimeError(
            "At least one operational feature has no observed values in the "
            f"model-training split: {bad}"
        )

    means = sums / observed_counts

    # Sum of squared deviations for observed values:
    # sum(x^2) - n_obs * mean^2.
    # Missing values are mean-imputed and therefore add zero deviation.
    ss_dev = sums_sq - observed_counts * np.square(means)
    ss_dev = np.maximum(ss_dev, 0.0)

    variance_after_mean_imputation = ss_dev / float(total_rows)
    stds = np.sqrt(variance_after_mean_imputation)

    # Numerical safeguard for constant/nearly constant features.
    stds = np.where(stds < epsilon, 1.0, stds)

    return OperationalPreprocessor(
        feature_names=feature_cols,
        means=means,
        stds=stds,
        observed_counts=observed_counts,
        total_rows=int(total_rows),
        epsilon=float(epsilon),
    )


def save_operational_preprocessor(
    preprocessor: OperationalPreprocessor,
    path: Path,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(preprocessor, path)


def load_operational_preprocessor(path: Path) -> OperationalPreprocessor:
    return joblib.load(path)


def _prepare_spec_matrix(
    specs_df: pd.DataFrame,
    id_col: str,
    feature_cols: list[str],
) -> pd.DataFrame:
    # Component X specification variables are categorical. Force a stable
    # string representation and reserve an explicit missing token.
    return (
        specs_df[feature_cols]
        .astype("string")
        .fillna("__MISSING__")
    )


def fit_static_encoder(
    train_specs: pd.DataFrame,
    allowed_vehicle_ids: Set[int],
) -> Tuple[OneHotEncoder, str, list[str]]:
    """
    Fit one-hot encoding on model-training vehicles only.
    """
    id_col = detect_id_column(train_specs)
    feature_cols = [c for c in train_specs.columns if c != id_col]

    subset = train_specs[
        train_specs[id_col].astype(int).isin(allowed_vehicle_ids)
    ].copy()

    if subset.empty:
        raise RuntimeError("No model-training specification rows were found.")

    encoder = OneHotEncoder(
        handle_unknown="ignore",
        sparse_output=False,
        dtype=np.float32,
    )
    encoder.fit(_prepare_spec_matrix(subset, id_col, feature_cols))
    return encoder, id_col, feature_cols


def transform_static_specs(
    encoder: OneHotEncoder,
    specs_df: pd.DataFrame,
    id_col: str,
    feature_cols: list[str],
) -> Dict[int, np.ndarray]:
    X = encoder.transform(_prepare_spec_matrix(specs_df, id_col, feature_cols))
    ids = specs_df[id_col].astype(int).to_numpy()

    return {
        int(vid): np.asarray(X[i], dtype=np.float32).reshape(-1)
        for i, vid in enumerate(ids)
    }


def static_encoding_summary(
    encoder: OneHotEncoder,
    feature_cols: list[str],
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    category_rows = []
    for feature, categories in zip(feature_cols, encoder.categories_):
        category_rows.append(
            {
                "feature": feature,
                "n_training_categories": int(len(categories)),
                "categories": "|".join(str(x) for x in categories),
            }
        )

    q_e = int(sum(len(x) for x in encoder.categories_))
    overview = pd.DataFrame(
        [
            {
                "static_encoding_method": "one_hot",
                "original_static_dimension_q": int(len(feature_cols)),
                "encoded_static_dimension_q_e": q_e,
                "unknown_category_policy": "ignore",
            }
        ]
    )
    return overview, pd.DataFrame(category_rows)


def save_static_encoder(encoder: OneHotEncoder, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(encoder, path)
