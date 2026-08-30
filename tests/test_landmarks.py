import numpy as np
import pandas as pd

from maestro_pdm.evaluation.landmarks import (
    select_validation_like_landmarks,
    summarize_landmarks,
)


def _sequence(times):
    times = np.asarray(times, dtype=np.int64)
    X = np.zeros((len(times), 2), dtype=np.float32)
    return times, X


def test_landmark_selection_is_reproducible_and_one_per_vehicle():
    sequences = {
        1: _sequence(range(1, 101)),
        2: _sequence(range(1, 101)),
        3: _sequence(range(1, 101)),
    }
    tte = pd.DataFrame(
        {
            "vehicle_id": [1, 2, 3],
            "length_of_study_time_step": [120, 120, 120],
            "in_study_repair": [1, 0, 1],
        }
    )

    samples_a, audit_a = select_validation_like_landmarks(
        sequences=sequences,
        vehicle_ids={1, 2, 3},
        tte_df=tte,
        tte_id_col="vehicle_id",
        tau=48,
        min_prefix_length=20,
        seed=42,
    )
    samples_b, audit_b = select_validation_like_landmarks(
        sequences=sequences,
        vehicle_ids={1, 2, 3},
        tte_df=tte,
        tte_id_col="vehicle_id",
        tau=48,
        min_prefix_length=20,
        seed=42,
    )

    pd.testing.assert_frame_equal(audit_a, audit_b)
    assert len(audit_a) == 3
    assert not audit_a["vehicle_id"].duplicated().any()
    assert (audit_a["prefix_index"].dropna() >= 19).all()


def test_landmark_is_selected_before_label_observability():
    # Every selectable readout is too close to censoring to establish a
    # complete 48-step event-free horizon. The function must exclude the
    # selected landmark rather than resampling an earlier convenient point.
    sequences = {
        10: _sequence(range(70, 101)),
    }
    tte = pd.DataFrame(
        {
            "vehicle_id": [10],
            "length_of_study_time_step": [100],
            "in_study_repair": [0],
        }
    )

    samples, audit = select_validation_like_landmarks(
        sequences=sequences,
        vehicle_ids={10},
        tte_df=tte,
        tte_id_col="vehicle_id",
        tau=48,
        min_prefix_length=20,
        seed=42,
    )

    assert len(samples) == 0
    assert audit.loc[0, "status"] == "excluded_censored_before_horizon"


def test_landmark_summary_counts_labelled_vehicles():
    audit = pd.DataFrame(
        {
            "vehicle_id": [1, 2, 3, 4],
            "prefix_index": [20, 21, 22, 23],
            "label": [1.0, 0.0, np.nan, 0.0],
            "status": [
                "positive_event_within_horizon",
                "negative_event_after_horizon",
                "excluded_censored_before_horizon",
                "negative_event_free_through_horizon",
            ],
        }
    )

    summary = summarize_landmarks(audit).iloc[0]

    assert summary["candidate_vehicles"] == 4
    assert summary["included_labelled_vehicles"] == 3
    assert summary["positive_vehicles"] == 1
    assert summary["negative_vehicles"] == 2
    assert np.isclose(summary["positive_prevalence"], 1 / 3)
