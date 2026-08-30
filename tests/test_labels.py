import numpy as np
import pandas as pd

from maestro_pdm.data.labels import (
    build_horizon_label,
    build_horizon_samples,
    deterministic_prefix_indices,
    samples_to_frame,
    summarize_horizon_samples,
)


def test_event_within_horizon_is_positive():
    result = build_horizon_label(
        prefix_time_step=60,
        study_end_time_step=100,
        event_observed=1,
        tau=48,
    )
    assert result.label == 1
    assert result.remaining_time_steps == 40


def test_event_after_horizon_is_negative():
    result = build_horizon_label(
        prefix_time_step=40,
        study_end_time_step=100,
        event_observed=1,
        tau=48,
    )
    assert result.label == 0
    assert result.status == "negative_event_after_horizon"


def test_censored_with_full_horizon_is_negative():
    result = build_horizon_label(
        prefix_time_step=40,
        study_end_time_step=100,
        event_observed=0,
        tau=48,
    )
    assert result.label == 0
    assert result.status == "negative_event_free_through_horizon"


def test_censored_before_horizon_is_excluded():
    result = build_horizon_label(
        prefix_time_step=70,
        study_end_time_step=100,
        event_observed=0,
        tau=48,
    )
    assert result.label is None
    assert result.status == "excluded_censored_before_horizon"


def test_exact_boundary_without_forward_window_is_excluded():
    result = build_horizon_label(
        prefix_time_step=100,
        study_end_time_step=100,
        event_observed=1,
        tau=48,
    )
    assert result.label is None
    assert result.status == "excluded_no_forward_window"


def test_prefix_sampling_is_deterministic():
    idx = deterministic_prefix_indices(
        n_readouts=100,
        min_prefix_length=20,
        samples_per_vehicle=3,
    )
    np.testing.assert_array_equal(idx, np.array([19, 59, 99]))


def test_training_class_weights_match_definition():
    samples = pd.DataFrame(
        {
            "split": ["model_train"] * 4,
            "label": [0, 0, 0, 1],
            "status": [
                "negative_event_after_horizon",
                "negative_event_after_horizon",
                "negative_event_after_horizon",
                "positive_event_within_horizon",
            ],
        }
    )
    audit = summarize_horizon_samples(samples).iloc[0]

    assert audit["omega_0"] == 4 / (2 * 3)
    assert audit["omega_1"] == 4 / (2 * 1)
