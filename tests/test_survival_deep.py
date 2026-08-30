import numpy as np
import torch

from maestro_pdm.models.survival_deep import (
    DeepCoxPHNet,
    DeepSurvivalMachine,
    breslow_baseline_cumulative_hazard_at_tau,
    cox_breslow_negative_log_likelihood,
    cox_risk_at_tau,
)


def test_cox_loss_is_finite():
    scores = torch.tensor(
        [0.1, -0.2, 0.4, 0.0],
        dtype=torch.float32,
        requires_grad=True,
    )
    durations = torch.tensor(
        [10.0, 20.0, 20.0, 30.0]
    )
    events = torch.tensor(
        [1.0, 1.0, 0.0, 1.0]
    )

    loss = cox_breslow_negative_log_likelihood(
        scores,
        durations,
        events,
    )
    assert torch.isfinite(loss)
    loss.backward()
    assert scores.grad is not None


def test_deep_cox_output_shape():
    model = DeepCoxPHNet(
        input_dim=7,
        hidden_dims=(8, 4),
    )
    X = torch.randn(5, 7)
    scores = model(X)
    assert scores.shape == (5,)


def test_breslow_risk_is_monotonic_in_score():
    scores = np.array([-1.0, 0.0, 1.0])
    durations = np.array([20, 30, 40])
    events = np.array([1, 1, 1])

    H0, offset = breslow_baseline_cumulative_hazard_at_tau(
        scores=scores,
        durations=durations,
        events=events,
        tau=48,
    )
    risk = cox_risk_at_tau(
        scores,
        H0_tau=H0,
        score_offset=offset,
    )

    assert np.all(np.diff(risk) > 0)


def test_dsm_nll_and_risk_are_finite():
    model = DeepSurvivalMachine(
        input_dim=6,
        components=2,
        hidden_dims=(8, 4),
    )
    X = torch.randn(12, 6)
    durations = torch.linspace(
        5.0,
        60.0,
        steps=12,
    )
    events = torch.tensor(
        [0, 1] * 6,
        dtype=torch.float32,
    )

    loss = model.negative_log_likelihood(
        X,
        durations,
        events,
    )
    risk = model.risk_at_tau(
        X,
        tau=48.0,
    )

    assert torch.isfinite(loss)
    assert torch.all(torch.isfinite(risk))
    assert torch.all(risk >= 0)
    assert torch.all(risk <= 1)
