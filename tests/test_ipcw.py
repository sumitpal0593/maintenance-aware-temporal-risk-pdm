import numpy as np
import pandas as pd

from maestro_pdm.evaluation.ipcw import (
    build_censoring_km,
    censor_survival_before,
    compute_horizon_ipcw_weights,
    effective_sample_size,
    fit_ipcw_platt_scaler,
)


def test_km_censoring_survival_before_time():
    audit = pd.DataFrame(
        {
            "vehicle_id": [1, 2, 3, 4],
            "prefix_index": [10, 10, 10, 10],
            "remaining_time_steps": [10, 20, 30, 40],
            "event_observed": [0, 1, 0, 1],
        }
    )

    km = build_censoring_km(audit)

    # Before the first censoring time there has been no censoring.
    assert np.isclose(
        censor_survival_before(km, [10])[0],
        1.0,
    )

    # At time 15, the censoring event at 10 has occurred:
    # G = 1 - 1/4 = 0.75.
    assert np.isclose(
        censor_survival_before(km, [15])[0],
        0.75,
    )


def test_horizon_ipcw_weights_positive_and_negative():
    audit = pd.DataFrame(
        {
            "vehicle_id": [1, 2, 3, 4],
            "prefix_index": [10, 10, 10, 10],
            "remaining_time_steps": [10, 20, 60, 70],
            "event_observed": [0, 1, 0, 1],
        }
    )
    km = build_censoring_km(audit)

    labelled = pd.DataFrame(
        {
            "vehicle_id": [2, 3, 4],
            "remaining_time_steps": [20, 60, 70],
            "event_observed": [1, 0, 1],
            "label": [1, 0, 0],
            "status": [
                "positive_event_within_horizon",
                "negative_event_free_through_horizon",
                "negative_event_after_horizon",
            ],
        }
    )

    weighted = compute_horizon_ipcw_weights(
        labelled,
        km_table=km,
        tau=48,
    )

    assert np.all(weighted["ipcw_raw"] > 0)
    assert np.all(weighted["ipcw_normalized"] > 0)
    assert np.isclose(
        weighted["ipcw_normalized"].mean(),
        1.0,
    )


def test_effective_sample_size_equal_weights():
    w = np.ones(10)
    assert np.isclose(effective_sample_size(w), 10.0)


def test_ipcw_platt_scaler_accepts_sample_weights():
    logits = np.array([-2.0, -1.0, -0.5, 0.5, 1.0, 2.0])
    y = np.array([0, 0, 0, 1, 1, 1])
    weights = np.array([1.0, 1.0, 1.0, 2.0, 2.0, 2.0])

    model = fit_ipcw_platt_scaler(
        raw_logits=logits,
        y_true=y,
        sample_weight=weights,
        C=1e6,
        seed=42,
    )

    p = model.predict_proba(logits.reshape(-1, 1))[:, 1]
    assert np.all((p > 0) & (p < 1))
    assert model.coef_[0, 0] > 0
