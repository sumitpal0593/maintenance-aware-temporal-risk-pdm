from __future__ import annotations

import numpy as np
import pandas as pd


def validate_survival_targets(durations, events):
    t = np.asarray(durations, dtype=float).reshape(-1)
    e = np.asarray(events, dtype=int).reshape(-1)

    if len(t) != len(e):
        raise ValueError("durations and events must have equal length.")
    if len(t) == 0:
        raise ValueError("Survival target is empty.")
    if np.any(~np.isfinite(t)):
        raise ValueError("Survival durations contain non-finite values.")
    if np.any(t <= 0):
        raise ValueError("All survival durations must be positive.")
    if not set(np.unique(e)).issubset({0, 1}):
        raise ValueError("events must be binary.")

    return t, e


def build_survival_target_frame(
    feature_vehicle_ids,
    feature_prefix_times,
    source_table: pd.DataFrame,
    source_split: str | None = None,
) -> pd.DataFrame:
    """
    Align saved Stage-4A feature rows with residual survival targets.

    Duplicate (vehicle_id, prefix_time_step) source rows are allowed only when
    their survival target information is identical. This handles repeated
    discrete time-step keys without silently accepting contradictory outcomes.
    """
    required = {
        "vehicle_id",
        "prefix_time_step",
        "remaining_time_steps",
        "event_observed",
        "label",
    }
    missing = required.difference(source_table.columns)
    if missing:
        raise ValueError(
            f"Survival source table is missing columns: {sorted(missing)}"
        )

    source = source_table.copy()

    if source_split is not None:
        if "split" not in source.columns:
            raise ValueError(
                "source_split was supplied but the source table has no "
                "'split' column."
            )
        source = source[source["split"] == source_split].copy()

    source = source[source["label"].notna()].copy()
    source["vehicle_id"] = source["vehicle_id"].astype(int)
    source["prefix_time_step"] = source["prefix_time_step"].astype(int)

    key_cols = ["vehicle_id", "prefix_time_step"]
    target_cols = [
        "remaining_time_steps",
        "event_observed",
        "label",
    ]
    if "status" in source.columns:
        target_cols.append("status")

    duplicate_mask = source.duplicated(key_cols, keep=False)

    if duplicate_mask.any():
        duplicate_source = source.loc[
            duplicate_mask,
            key_cols + target_cols,
        ].copy()

        conflicting_keys = []
        for key, group in duplicate_source.groupby(
            key_cols,
            sort=False,
            dropna=False,
        ):
            normalized = group[target_cols].copy()
            for col in target_cols:
                normalized[col] = normalized[col].astype(object).where(
                    normalized[col].notna(),
                    "__NA__",
                )
            if len(normalized.drop_duplicates()) > 1:
                conflicting_keys.append(key)

        if conflicting_keys:
            raise ValueError(
                "Duplicate vehicle/prefix keys have conflicting survival "
                f"targets. Examples: {conflicting_keys[:10]}"
            )

        source = source.drop_duplicates(
            subset=key_cols,
            keep="first",
        ).copy()

    keys = pd.DataFrame(
        {
            "vehicle_id": np.asarray(feature_vehicle_ids, dtype=np.int64),
            "prefix_time_step": np.asarray(
                feature_prefix_times,
                dtype=np.int64,
            ),
            "_feature_order": np.arange(
                len(feature_vehicle_ids),
                dtype=np.int64,
            ),
        }
    )

    keep_cols = key_cols + [
        "remaining_time_steps",
        "event_observed",
        "label",
    ]
    if "status" in source.columns:
        keep_cols.append("status")

    aligned = keys.merge(
        source[keep_cols],
        on=key_cols,
        how="left",
        validate="many_to_one",
    ).sort_values("_feature_order")

    if aligned["remaining_time_steps"].isna().any():
        examples = aligned.loc[
            aligned["remaining_time_steps"].isna(),
            key_cols,
        ].head(10)
        raise ValueError(
            "Could not recover survival targets for every feature row. "
            f"Examples:\n{examples}"
        )

    aligned["remaining_time_steps"] = aligned[
        "remaining_time_steps"
    ].astype(float)
    aligned["event_observed"] = aligned["event_observed"].astype(int)
    aligned["label"] = aligned["label"].astype(int)

    validate_survival_targets(
        aligned["remaining_time_steps"],
        aligned["event_observed"],
    )

    return (
        aligned
        .drop(columns=["_feature_order"])
        .reset_index(drop=True)
    )
