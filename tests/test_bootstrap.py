import numpy as np

from maestro_pdm.evaluation.bootstrap import (
    BootstrapConfig,
    paired_stratified_bootstrap,
    percentile_interval,
    stratified_bootstrap_indices,
)


def test_stratified_bootstrap_preserves_class_counts():
    y = np.array([0, 0, 0, 1, 1])
    idx = stratified_bootstrap_indices(y, n_resamples=25, seed=42)
    assert idx.shape == (25, 5)
    for row in idx:
        yr = y[row]
        assert int((yr == 1).sum()) == 2
        assert int((yr == 0).sum()) == 3


def test_paired_bootstrap_identical_models_have_zero_difference():
    y = np.array([0, 0, 0, 0, 1, 1, 1, 1])
    s = np.array([0.1, 0.2, 0.3, 0.4, 0.6, 0.7, 0.8, 0.9])
    cfg = BootstrapConfig(n_resamples=50, confidence_level=0.95, seed=1)
    _, dist = paired_stratified_bootstrap(y, {"a": s, "b": s.copy()}, cfg)
    for metric in ("auroc", "pr_auc", "enrichment"):
        d = dist["a"][metric] - dist["b"][metric]
        assert np.allclose(d, 0.0)
        lo, hi = percentile_interval(d, 0.95)
        assert lo == 0.0
        assert hi == 0.0
