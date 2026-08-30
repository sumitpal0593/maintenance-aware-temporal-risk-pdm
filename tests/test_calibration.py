import numpy as np

from maestro_pdm.evaluation.calibration import (
    analytical_weight_correction,
    analytical_weight_correction_from_logits,
    apply_platt_scaler,
    expected_calibration_error,
    fit_platt_scaler,
    probability_metrics,
    threshold_support_table,
)


def test_analytical_correction_matches_logit_shift():
    q = np.array([0.10, 0.25, 0.50, 0.75, 0.90])
    logits = np.log(q / (1.0 - q))

    omega_0 = 0.55
    omega_1 = 6.2

    a = analytical_weight_correction(
        q,
        omega_0=omega_0,
        omega_1=omega_1,
    )
    b = analytical_weight_correction_from_logits(
        logits,
        omega_0=omega_0,
        omega_1=omega_1,
    )

    np.testing.assert_allclose(a, b, rtol=1e-10, atol=1e-10)


def test_analytical_correction_is_monotonic():
    q = np.linspace(0.01, 0.99, 99)
    corrected = analytical_weight_correction(
        q,
        omega_0=0.5,
        omega_1=5.0,
    )
    assert np.all(np.diff(corrected) > 0)


def test_platt_scaler_returns_valid_probabilities():
    logits = np.array([-2.0, -1.0, -0.2, 0.2, 1.0, 2.0])
    y = np.array([0, 0, 0, 1, 1, 1])

    model = fit_platt_scaler(logits, y, C=1e6, seed=42)
    p = apply_platt_scaler(model, logits)

    assert np.all(p > 0)
    assert np.all(p < 1)
    assert model.coef_[0, 0] > 0


def test_ece_is_zero_for_perfect_group_probabilities():
    y = np.array([0, 0, 1, 1])
    p = np.array([0.0, 0.0, 1.0, 1.0])
    ece = expected_calibration_error(y, p, n_bins=2)
    assert ece < 1e-5


def test_probability_metrics_contains_required_fields():
    y = np.array([0, 0, 0, 1, 1, 1])
    p = np.array([0.05, 0.10, 0.20, 0.60, 0.80, 0.90])

    metrics = probability_metrics(y, p)

    for key in (
        "auroc",
        "pr_auc",
        "enrichment",
        "brier_score",
        "log_loss",
        "ece_equal_width",
        "diagnostic_calibration_intercept",
        "diagnostic_calibration_slope",
    ):
        assert key in metrics
        assert np.isfinite(metrics[key])


def test_threshold_support_finds_nearest_probability():
    p = np.array([0.1, 0.2, 0.31, 0.49, 0.72])
    table = threshold_support_table(
        p,
        method="test",
        target_levels=[0.30, 0.50, 0.70],
    )

    assert np.isclose(
        table.loc[
            table["target_probability"] == 0.30,
            "nearest_observed_probability",
        ].iloc[0],
        0.31,
    )
