from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Dict, Iterable, List, Optional

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class HorizonLabelResult:
    label: Optional[int]
    status: str
    remaining_time_steps: int


@dataclass(frozen=True)
class HorizonSample:
    vehicle_id: int
    split: str
    prefix_index: int
    prefix_time_step: int
    study_end_time_step: int
    event_observed: int
    remaining_time_steps: int
    label: Optional[int]
    status: str


def build_horizon_label(
    prefix_time_step: int,
    study_end_time_step: int,
    event_observed: int,
    tau: int,
) -> HorizonLabelResult:
    """
    Construct a censoring-aware fixed-horizon target Y_i(tau).

    Positive:
        An observed repair/failure occurs strictly after the prefix and within
        tau relative time steps.

    Negative:
        We can verify that no repair/failure occurs within tau:
        - an observed event occurs later than tau, or
        - a censored vehicle remains under observation for at least tau after
          the prefix.

    Excluded:
        - censored follow-up ends before tau, so the horizon outcome is unknown;
        - the prefix is at/after the study-end/event boundary, so there is no
          forward prediction window.

    The exact event/censoring boundary is excluded to prevent using an
    observation taken at the outcome time as a forward-looking prediction
    example.
    """
    tau = int(tau)
    if tau <= 0:
        raise ValueError("tau must be positive.")

    event_observed = int(event_observed)
    if event_observed not in (0, 1):
        raise ValueError("event_observed must be 0 or 1.")

    remaining = int(study_end_time_step) - int(prefix_time_step)

    if remaining <= 0:
        return HorizonLabelResult(
            label=None,
            status="excluded_no_forward_window",
            remaining_time_steps=remaining,
        )

    if event_observed == 1:
        if remaining <= tau:
            return HorizonLabelResult(
                label=1,
                status="positive_event_within_horizon",
                remaining_time_steps=remaining,
            )
        return HorizonLabelResult(
            label=0,
            status="negative_event_after_horizon",
            remaining_time_steps=remaining,
        )

    # Right-censored vehicle: a negative label is valid only when at least tau
    # future time steps are observed without an event.
    if remaining >= tau:
        return HorizonLabelResult(
            label=0,
            status="negative_event_free_through_horizon",
            remaining_time_steps=remaining,
        )

    return HorizonLabelResult(
        label=None,
        status="excluded_censored_before_horizon",
        remaining_time_steps=remaining,
    )


def deterministic_prefix_indices(
    n_readouts: int,
    min_prefix_length: int,
    samples_per_vehicle: int,
) -> np.ndarray:
    """
    Reproduce the original evenly spaced prefix-sampling idea deterministically.
    """
    n_readouts = int(n_readouts)
    min_prefix_length = int(min_prefix_length)
    samples_per_vehicle = int(samples_per_vehicle)

    if n_readouts < min_prefix_length or samples_per_vehicle <= 0:
        return np.asarray([], dtype=int)

    n_available = n_readouts - min_prefix_length + 1
    n_samples = min(samples_per_vehicle, n_available)

    return np.linspace(
        min_prefix_length - 1,
        n_readouts - 1,
        num=n_samples,
        dtype=int,
    )


def build_horizon_samples(
    vehicle_times: Dict[int, np.ndarray],
    tte_df: pd.DataFrame,
    id_col: str,
    split_assignments: pd.DataFrame,
    tau: int,
    min_prefix_length: int,
    samples_per_vehicle: int,
) -> List[HorizonSample]:
    """Build deterministic prefix-level fixed-horizon samples for all splits."""
    required_tte = {"length_of_study_time_step", "in_study_repair"}
    missing = required_tte.difference(tte_df.columns)
    if missing:
        raise ValueError(f"train_tte.csv is missing columns: {sorted(missing)}")

    if "split" not in split_assignments.columns or "vehicle_id" not in split_assignments.columns:
        raise ValueError("split_assignments must contain vehicle_id and split.")

    T_map = dict(
        zip(
            tte_df[id_col].astype(int),
            tte_df["length_of_study_time_step"].astype(int),
        )
    )
    E_map = dict(
        zip(
            tte_df[id_col].astype(int),
            tte_df["in_study_repair"].astype(int),
        )
    )
    split_map = dict(
        zip(
            split_assignments["vehicle_id"].astype(int),
            split_assignments["split"].astype(str),
        )
    )

    samples: List[HorizonSample] = []

    for vid in sorted(vehicle_times):
        if vid not in T_map or vid not in E_map or vid not in split_map:
            continue

        times = np.asarray(vehicle_times[vid], dtype=np.int64)
        idxs = deterministic_prefix_indices(
            n_readouts=len(times),
            min_prefix_length=min_prefix_length,
            samples_per_vehicle=samples_per_vehicle,
        )

        for prefix_index in idxs:
            prefix_time = int(times[int(prefix_index)])
            result = build_horizon_label(
                prefix_time_step=prefix_time,
                study_end_time_step=T_map[vid],
                event_observed=E_map[vid],
                tau=tau,
            )
            samples.append(
                HorizonSample(
                    vehicle_id=int(vid),
                    split=split_map[vid],
                    prefix_index=int(prefix_index),
                    prefix_time_step=prefix_time,
                    study_end_time_step=int(T_map[vid]),
                    event_observed=int(E_map[vid]),
                    remaining_time_steps=int(result.remaining_time_steps),
                    label=result.label,
                    status=result.status,
                )
            )

    return samples


def samples_to_frame(samples: Iterable[HorizonSample]) -> pd.DataFrame:
    return pd.DataFrame([asdict(x) for x in samples])


def summarize_horizon_samples(samples_df: pd.DataFrame) -> pd.DataFrame:
    """
    Produce reviewer-facing counts for each development split.

    Class weights are reported only for model_train because they are used by the
    weighted BCE training objective.
    """
    rows = []

    for split_name, group in samples_df.groupby("split", sort=False):
        included = group[group["label"].notna()].copy()
        positives = int((included["label"] == 1).sum())
        negatives = int((included["label"] == 0).sum())
        total = positives + negatives

        row = {
            "split": split_name,
            "candidate_prefixes": int(len(group)),
            "included_labeled_samples": int(total),
            "positive_samples": positives,
            "negative_samples": negatives,
            "horizon_positive_prevalence": (
                positives / total if total else np.nan
            ),
            "excluded_censored_before_horizon": int(
                (group["status"] == "excluded_censored_before_horizon").sum()
            ),
            "excluded_no_forward_window": int(
                (group["status"] == "excluded_no_forward_window").sum()
            ),
            "omega_0": np.nan,
            "omega_1": np.nan,
        }

        if split_name == "model_train" and positives > 0 and negatives > 0:
            row["omega_0"] = total / (2.0 * negatives)
            row["omega_1"] = total / (2.0 * positives)

        rows.append(row)

    return pd.DataFrame(rows)


def frame_to_horizon_samples(df: pd.DataFrame) -> List[HorizonSample]:
    """Reconstruct HorizonSample objects from horizon_prefix_samples.csv."""
    required = {
        "vehicle_id",
        "split",
        "prefix_index",
        "prefix_time_step",
        "study_end_time_step",
        "event_observed",
        "remaining_time_steps",
        "label",
        "status",
    }
    missing = required.difference(df.columns)
    if missing:
        raise ValueError(
            f"Horizon-sample table is missing columns: {sorted(missing)}"
        )

    samples: List[HorizonSample] = []
    for row in df.to_dict(orient="records"):
        raw_label = row["label"]
        label = None if pd.isna(raw_label) else int(raw_label)
        samples.append(
            HorizonSample(
                vehicle_id=int(row["vehicle_id"]),
                split=str(row["split"]),
                prefix_index=int(row["prefix_index"]),
                prefix_time_step=int(row["prefix_time_step"]),
                study_end_time_step=int(row["study_end_time_step"]),
                event_observed=int(row["event_observed"]),
                remaining_time_steps=int(row["remaining_time_steps"]),
                label=label,
                status=str(row["status"]),
            )
        )
    return samples
