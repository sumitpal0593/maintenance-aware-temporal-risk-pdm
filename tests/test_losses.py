import numpy as np
import torch

from maestro_pdm.training.losses import (
    compute_balanced_class_weights,
    weighted_bce_with_logits,
)


def test_balanced_class_weights():
    weights = compute_balanced_class_weights([0, 0, 0, 1])
    assert np.isclose(weights.omega_0, 4 / 6)
    assert np.isclose(weights.omega_1, 2.0)


def test_weighted_bce_is_finite():
    logits = torch.tensor([0.0, 0.5, -0.2, 1.0])
    targets = torch.tensor([0.0, 1.0, 0.0, 1.0])

    loss = weighted_bce_with_logits(
        logits,
        targets,
        omega_0=1.0,
        omega_1=2.0,
    )
    assert torch.isfinite(loss)
    assert loss.item() > 0
