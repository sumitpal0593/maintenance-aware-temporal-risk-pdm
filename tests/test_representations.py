import numpy as np

from maestro_pdm.features.representations import (
    aggregate_feature_names,
    aggregate_sequence,
)


def test_aggregate_sequence_dimension():
    X = np.arange(30, dtype=float).reshape(10, 3)
    features = aggregate_sequence(X)
    assert features.shape == (18,)


def test_constant_sequence_has_zero_trend_and_std():
    X = np.ones((8, 4), dtype=float)
    f = aggregate_sequence(X)

    p = X.shape[1]
    std = f[p : 2 * p]
    trend = f[5 * p : 6 * p]

    assert np.allclose(std, 0.0)
    assert np.allclose(trend, 0.0)


def test_linear_sequence_trend_positive():
    t = np.linspace(0.0, 1.0, 20)
    X = np.column_stack([t, 2.0 * t, -t])
    f = aggregate_sequence(X)

    p = X.shape[1]
    trend = f[5 * p : 6 * p]

    assert trend[0] > 0
    assert trend[1] > trend[0]
    assert trend[2] < 0


def test_aggregate_feature_names_match_dimension():
    names = aggregate_feature_names(["a", "b", "c"])
    assert len(names) == 18
    assert names[0] == "a__mean"
    assert names[-1] == "c__trend"
