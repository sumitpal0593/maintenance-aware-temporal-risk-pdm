import numpy as np
import pandas as pd

from maestro_pdm.evaluation.survival_utils import (
    build_survival_target_frame,
    choose_best_configuration,
    validate_survival_targets,
)


def test_validate_survival_targets():
    t, e = validate_survival_targets(
        [10, 20, 30],
        [1, 0, 1],
    )
    assert np.array_equal(
        t,
        np.array([10.0, 20.0, 30.0]),
    )
    assert np.array_equal(
        e,
        np.array([1, 0, 1]),
    )


def test_build_survival_target_frame_alignment():
    source = pd.DataFrame(
        {
            "vehicle_id": [2, 1],
            "split": ["model_train", "model_train"],
            "prefix_time_step": [50, 40],
            "remaining_time_steps": [20, 60],
            "event_observed": [1, 0],
            "label": [1, 0],
            "status": ["positive", "negative"],
        }
    )

    aligned = build_survival_target_frame(
        feature_vehicle_ids=[1, 2],
        feature_prefix_times=[40, 50],
        source_table=source,
        source_split="model_train",
    )

    assert aligned["vehicle_id"].tolist() == [1, 2]
    assert aligned["remaining_time_steps"].tolist() == [60.0, 20.0]


def test_choose_best_configuration_uses_pr_auc_first():
    table = pd.DataFrame(
        {
            "configuration": ["a", "b", "c"],
            "tuning_pr_auc": [0.10, 0.12, 0.12],
            "tuning_auroc": [0.70, 0.60, 0.65],
        }
    )

    best = choose_best_configuration(table)
    assert best["configuration"] == "c"



def test_duplicate_source_keys_with_identical_targets_are_allowed():
    source = pd.DataFrame(
        {
            "vehicle_id": [1, 1, 2],
            "split": ["model_train", "model_train", "model_train"],
            "prefix_time_step": [40, 40, 50],
            "remaining_time_steps": [60, 60, 20],
            "event_observed": [0, 0, 1],
            "label": [0, 0, 1],
            "status": ["negative", "negative", "positive"],
        }
    )

    aligned = build_survival_target_frame(
        feature_vehicle_ids=[1, 1, 2],
        feature_prefix_times=[40, 40, 50],
        source_table=source,
        source_split="model_train",
    )

    # Both feature rows are retained, but the redundant source target is
    # collapsed to a unique mapping.
    assert aligned["vehicle_id"].tolist() == [1, 1, 2]
    assert aligned["remaining_time_steps"].tolist() == [60.0, 60.0, 20.0]


def test_duplicate_source_keys_with_conflicting_targets_fail():
    source = pd.DataFrame(
        {
            "vehicle_id": [1, 1],
            "split": ["model_train", "model_train"],
            "prefix_time_step": [40, 40],
            "remaining_time_steps": [60, 55],
            "event_observed": [0, 0],
            "label": [0, 0],
            "status": ["negative", "negative"],
        }
    )

    try:
        build_survival_target_frame(
            feature_vehicle_ids=[1],
            feature_prefix_times=[40],
            source_table=source,
            source_split="model_train",
        )
    except ValueError as exc:
        assert "conflicting survival targets" in str(exc)
    else:
        raise AssertionError("Conflicting duplicate targets should fail.")
