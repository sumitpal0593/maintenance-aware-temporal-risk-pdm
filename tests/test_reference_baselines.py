import numpy as np
import pandas as pd

from maestro_pdm.evaluation.reference_baselines import (
    binary_ranking_metrics,
    select_best_by_tuning,
    step_survival_array_risk_at_tau,
    survival_df_risk_at_tau,
)


def test_binary_metrics():
    y = np.array([0, 0, 1, 1])
    s = np.array([0.1, 0.2, 0.8, 0.9])
    m = binary_ranking_metrics(y, s)
    assert np.isclose(m["auroc"], 1.0)
    assert np.isclose(m["pr_auc"], 1.0)


def test_select_best_uses_tuning_pr_auc():
    df = pd.DataFrame(
        {
            "configuration": ["a", "b"],
            "status": ["ok", "ok"],
            "tuning_pr_auc": [0.10, 0.12],
            "tuning_auroc": [0.80, 0.70],
        }
    )
    best = select_best_by_tuning(df)
    assert best["configuration"] == "b"


def test_survival_dataframe_uses_last_time_not_after_tau():
    df = pd.DataFrame(
        {
            0: [1.0, 0.8, 0.5],
            1: [1.0, 0.9, 0.7],
        },
        index=[10.0, 40.0, 60.0],
    )
    risk = survival_df_risk_at_tau(df, tau=48)
    np.testing.assert_allclose(risk, [0.2, 0.1])


def test_survival_array_at_tau():
    surv = np.array(
        [
            [1.0, 0.8, 0.5],
            [1.0, 0.9, 0.7],
        ]
    )
    risk = step_survival_array_risk_at_tau(
        surv,
        [10, 40, 60],
        tau=48,
    )
    np.testing.assert_allclose(risk, [0.2, 0.1])
