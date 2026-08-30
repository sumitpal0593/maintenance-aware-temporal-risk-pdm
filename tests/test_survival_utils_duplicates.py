import pandas as pd

from maestro_pdm.evaluation.survival_utils import (
    build_survival_target_frame,
)


def test_identical_duplicate_targets_are_allowed():
    source = pd.DataFrame(
        {
            "vehicle_id": [1, 1, 2],
            "split": ["model_train"] * 3,
            "prefix_time_step": [40, 40, 50],
            "remaining_time_steps": [60, 60, 20],
            "event_observed": [0, 0, 1],
            "label": [0, 0, 1],
            "status": ["negative", "negative", "positive"],
        }
    )
    aligned = build_survival_target_frame(
        [1, 1, 2],
        [40, 40, 50],
        source,
        "model_train",
    )
    assert len(aligned) == 3
    assert aligned["remaining_time_steps"].tolist() == [60.0, 60.0, 20.0]


def test_conflicting_duplicate_targets_fail():
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
            [1], [40], source, "model_train"
        )
    except ValueError as exc:
        assert "conflicting survival targets" in str(exc)
    else:
        raise AssertionError("Conflicting duplicate targets should fail.")
