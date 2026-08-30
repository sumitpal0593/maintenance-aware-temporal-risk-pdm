from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn.functional as F


@dataclass(frozen=True)
class ClassWeights:
    omega_0: float
    omega_1: float
    n_negative: int
    n_positive: int


def compute_balanced_class_weights(labels) -> ClassWeights:
    """
    Compute the manuscript's balanced class weights:

        omega_1 = N / (2 N_1)
        omega_0 = N / (2 N_0)
    """
    y = torch.as_tensor(labels, dtype=torch.float32).reshape(-1)
    n_positive = int((y == 1).sum().item())
    n_negative = int((y == 0).sum().item())
    n_total = n_positive + n_negative

    if n_positive == 0 or n_negative == 0:
        raise ValueError(
            "Both positive and negative training samples are required."
        )

    return ClassWeights(
        omega_0=n_total / (2.0 * n_negative),
        omega_1=n_total / (2.0 * n_positive),
        n_negative=n_negative,
        n_positive=n_positive,
    )


def weighted_bce_with_logits(
    logits: torch.Tensor,
    targets: torch.Tensor,
    omega_0: float,
    omega_1: float,
) -> torch.Tensor:
    """
    Class-weighted binary cross entropy matching the manuscript exactly.
    """
    logits = logits.reshape(-1)
    targets = targets.float().reshape(-1)

    element_loss = F.binary_cross_entropy_with_logits(
        logits,
        targets,
        reduction="none",
    )

    weights = torch.where(
        targets > 0.5,
        torch.full_like(targets, float(omega_1)),
        torch.full_like(targets, float(omega_0)),
    )

    return (weights * element_loss).mean()
