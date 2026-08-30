from __future__ import annotations

from typing import Dict, Iterable, List, Set, Tuple

import numpy as np
import pandas as pd

from maestro_pdm.data.labels import HorizonSample, build_horizon_label


def select_validation_like_landmarks(
    sequences: Dict[int, Tuple[np.ndarray, np.ndarray]],
    vehicle_ids: Set[int],
    tte_df: pd.DataFrame,
    tte_id_col: str,
    tau: int,
    min_prefix_length: int,
    seed: int,
    split_name: str = "tuning",
) -> Tuple[List[HorizonSample], pd.DataFrame]:
    """
    Select exactly one reproducible pseudo-validation landmark per vehicle.

    The public Scania validation set contains one randomly truncated operational
    history per vehicle. To make internal model selection more comparable, this
    function selects one readout position per tuning vehicle rather than using
    all of the multiple training-style prefixes.

    Important:
    - The landmark is selected BEFORE inspecting its horizon label.
    - Selection is uniform over available readout indices from
      min_prefix_length-1 through the final available readout.
    - A vehicle-specific RNG stream derived from (seed, vehicle_id) makes the
      selection deterministic and independent of iteration order.
    - If the selected landmark does not have an identifiable Y_i(tau) because
      a censored training trajectory ends before the full horizon, that vehicle
      is excluded from the labelled model-selection set. We do not resample a
      more convenient landmark, because doing so would condition selection on
      label observability.
    """
    required = {"length_of_study_time_step", "in_study_repair"}
    missing = required.difference(tte_df.columns)
    if missing:
        raise ValueError(f"train_tte.csv missing columns: {sorted(missing)}")

    T_map = dict(
        zip(
            tte_df[tte_id_col].astype(int),
            tte_df["length_of_study_time_step"].astype(int),
        )
    )
    E_map = dict(
        zip(
            tte_df[tte_id_col].astype(int),
            tte_df["in_study_repair"].astype(int),
        )
    )

    samples: List[HorizonSample] = []
    audit_rows = []

    for vid in sorted(int(v) for v in vehicle_ids):
        if vid not in sequences:
            audit_rows.append(
                {
                    "vehicle_id": vid,
                    "split": split_name,
                    "prefix_index": np.nan,
                    "prefix_time_step": np.nan,
                    "study_end_time_step": T_map.get(vid, np.nan),
                    "event_observed": E_map.get(vid, np.nan),
                    "remaining_time_steps": np.nan,
                    "label": np.nan,
                    "status": "excluded_missing_operational_sequence",
                }
            )
            continue

        if vid not in T_map or vid not in E_map:
            audit_rows.append(
                {
                    "vehicle_id": vid,
                    "split": split_name,
                    "prefix_index": np.nan,
                    "prefix_time_step": np.nan,
                    "study_end_time_step": np.nan,
                    "event_observed": np.nan,
                    "remaining_time_steps": np.nan,
                    "label": np.nan,
                    "status": "excluded_missing_tte",
                }
            )
            continue

        times = np.asarray(sequences[vid][0], dtype=np.int64)
        n_readouts = len(times)

        if n_readouts < int(min_prefix_length):
            audit_rows.append(
                {
                    "vehicle_id": vid,
                    "split": split_name,
                    "prefix_index": np.nan,
                    "prefix_time_step": np.nan,
                    "study_end_time_step": int(T_map[vid]),
                    "event_observed": int(E_map[vid]),
                    "remaining_time_steps": np.nan,
                    "label": np.nan,
                    "status": "excluded_short_history",
                }
            )
            continue

        # Stable vehicle-specific RNG: the selected landmark does not depend
        # on dictionary order or which other vehicles happen to be present.
        rng = np.random.default_rng(
            np.random.SeedSequence([int(seed), int(vid)])
        )
        prefix_index = int(
            rng.integers(
                low=int(min_prefix_length) - 1,
                high=n_readouts,
            )
        )
        prefix_time = int(times[prefix_index])

        result = build_horizon_label(
            prefix_time_step=prefix_time,
            study_end_time_step=int(T_map[vid]),
            event_observed=int(E_map[vid]),
            tau=int(tau),
        )

        label_value = np.nan if result.label is None else int(result.label)

        audit_rows.append(
            {
                "vehicle_id": vid,
                "split": split_name,
                "prefix_index": prefix_index,
                "prefix_time_step": prefix_time,
                "study_end_time_step": int(T_map[vid]),
                "event_observed": int(E_map[vid]),
                "remaining_time_steps": int(result.remaining_time_steps),
                "label": label_value,
                "status": result.status,
            }
        )

        if result.label is not None:
            samples.append(
                HorizonSample(
                    vehicle_id=vid,
                    split=split_name,
                    prefix_index=prefix_index,
                    prefix_time_step=prefix_time,
                    study_end_time_step=int(T_map[vid]),
                    event_observed=int(E_map[vid]),
                    remaining_time_steps=int(result.remaining_time_steps),
                    label=int(result.label),
                    status=result.status,
                )
            )

    audit = pd.DataFrame(audit_rows)

    if audit["vehicle_id"].duplicated().any():
        raise RuntimeError(
            "Landmark selection produced more than one row for a tuning vehicle."
        )

    return samples, audit


def summarize_landmarks(audit: pd.DataFrame) -> pd.DataFrame:
    """Create a one-row reviewer-facing audit of the landmark selection."""
    labelled = audit[audit["label"].notna()].copy()

    n_positive = int((labelled["label"] == 1).sum())
    n_negative = int((labelled["label"] == 0).sum())
    n_labelled = n_positive + n_negative

    return pd.DataFrame(
        [
            {
                "candidate_vehicles": int(len(audit)),
                "selected_landmarks": int(
                    audit["prefix_index"].notna().sum()
                ),
                "included_labelled_vehicles": n_labelled,
                "positive_vehicles": n_positive,
                "negative_vehicles": n_negative,
                "positive_prevalence": (
                    n_positive / n_labelled if n_labelled else np.nan
                ),
                "excluded_censored_before_horizon": int(
                    (
                        audit["status"]
                        == "excluded_censored_before_horizon"
                    ).sum()
                ),
                "excluded_no_forward_window": int(
                    (
                        audit["status"]
                        == "excluded_no_forward_window"
                    ).sum()
                ),
                "excluded_short_history": int(
                    (audit["status"] == "excluded_short_history").sum()
                ),
                "excluded_missing_sequence": int(
                    (
                        audit["status"]
                        == "excluded_missing_operational_sequence"
                    ).sum()
                ),
                "excluded_missing_tte": int(
                    (audit["status"] == "excluded_missing_tte").sum()
                ),
            }
        ]
    )
