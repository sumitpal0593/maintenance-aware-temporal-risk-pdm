import numpy as np

from maestro_pdm.evaluation.metrics import binary_ranking_metrics


def test_binary_ranking_metrics_perfect_ranking():
    y = np.array([0, 0, 1, 1])
    s = np.array([0.1, 0.2, 0.8, 0.9])

    metrics = binary_ranking_metrics(y, s)

    assert np.isclose(metrics["auroc"], 1.0)
    assert np.isclose(metrics["pr_auc"], 1.0)
    assert np.isclose(metrics["positive_prevalence"], 0.5)
    assert np.isclose(metrics["enrichment"], 2.0)
