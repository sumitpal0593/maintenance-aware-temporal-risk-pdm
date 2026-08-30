from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass
from typing import Dict, Iterable, List, Sequence

import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from maestro_pdm.evaluation.metrics import binary_ranking_metrics
from maestro_pdm.evaluation.survival_utils import validate_survival_targets


def cox_breslow_negative_log_likelihood(
    scores: torch.Tensor,
    durations: torch.Tensor,
    events: torch.Tensor,
) -> torch.Tensor:
    """
    Breslow partial negative log likelihood with tied event times.

    Durations are sorted in descending order. For every tied duration, all
    events at that duration share the same risk-set denominator.
    """
    scores = scores.reshape(-1)
    durations = durations.reshape(-1)
    events = events.reshape(-1).float()

    order = torch.argsort(
        durations,
        descending=True,
        stable=True,
    )
    s = scores[order]
    t = durations[order]
    e = events[order]

    if torch.sum(e) <= 0:
        raise ValueError(
            "Deep CoxPH requires at least one observed event."
        )

    _, counts = torch.unique_consecutive(
        t,
        return_counts=True,
    )
    group_id = torch.repeat_interleave(
        torch.arange(
            len(counts),
            device=s.device,
        ),
        counts,
    )
    group_end = torch.cumsum(counts, dim=0) - 1

    log_risk_sum = torch.logcumsumexp(
        s,
        dim=0,
    )[group_end]

    event_count = torch.zeros(
        len(counts),
        dtype=s.dtype,
        device=s.device,
    ).scatter_add_(
        0,
        group_id,
        e,
    )
    event_score_sum = torch.zeros(
        len(counts),
        dtype=s.dtype,
        device=s.device,
    ).scatter_add_(
        0,
        group_id,
        s * e,
    )

    partial_log_likelihood = (
        event_score_sum
        - event_count * log_risk_sum
    ).sum()

    return -partial_log_likelihood / torch.sum(e)


def breslow_baseline_cumulative_hazard_at_tau(
    scores,
    durations,
    events,
    tau: float,
) -> tuple[float, float]:
    """
    Estimate H0(tau) from training data.

    Returns
    -------
    H0_tau
    score_offset

    Scores are centered before exponentiation for numerical stability. The same
    score_offset must be subtracted from new scores at prediction time.
    """
    scores = np.asarray(scores, dtype=float).reshape(-1)
    durations, events = validate_survival_targets(
        durations,
        events,
    )
    tau = float(tau)

    if len(scores) != len(durations):
        raise ValueError("scores and survival targets are misaligned.")

    offset = float(np.mean(scores))
    relative_risk = np.exp(
        np.clip(scores - offset, -40.0, 40.0)
    )

    h0 = 0.0
    event_times = np.sort(
        np.unique(durations[events == 1])
    )

    for time in event_times:
        if time > tau:
            break
        d = int(
            np.sum(
                (durations == time)
                & (events == 1)
            )
        )
        denominator = float(
            np.sum(
                relative_risk[
                    durations >= time
                ]
            )
        )
        if denominator > 0:
            h0 += d / denominator

    return float(h0), offset


def cox_risk_at_tau(
    scores,
    H0_tau: float,
    score_offset: float,
) -> np.ndarray:
    scores = np.asarray(scores, dtype=float).reshape(-1)
    rr = np.exp(
        np.clip(
            scores - float(score_offset),
            -40.0,
            40.0,
        )
    )
    survival = np.exp(
        -float(H0_tau) * rr
    )
    return np.clip(
        1.0 - survival,
        0.0,
        1.0,
    )


class DeepCoxPHNet(nn.Module):
    def __init__(
        self,
        input_dim: int,
        hidden_dims: Sequence[int] = (128, 64),
        dropout: float = 0.10,
    ):
        super().__init__()
        layers: List[nn.Module] = []
        dim = int(input_dim)

        for hidden in hidden_dims:
            layers.extend(
                [
                    nn.Linear(dim, int(hidden)),
                    nn.ReLU(),
                    nn.Dropout(float(dropout)),
                ]
            )
            dim = int(hidden)

        layers.append(nn.Linear(dim, 1))
        self.network = nn.Sequential(*layers)

    def forward(self, X: torch.Tensor) -> torch.Tensor:
        return self.network(X).squeeze(1)


@dataclass
class DeepCoxFitResult:
    state_dict: dict
    best_epoch: int
    best_tuning_pr_auc: float
    training_history: pd.DataFrame
    H0_tau: float
    score_offset: float


def fit_deep_coxph(
    X_train: np.ndarray,
    durations_train,
    events_train,
    X_tuning: np.ndarray,
    y_tuning,
    tau: float,
    hidden_dims: Sequence[int],
    dropout: float,
    learning_rate: float,
    weight_decay: float,
    max_epochs: int,
    patience: int,
    seed: int,
    device: torch.device,
) -> tuple[DeepCoxPHNet, DeepCoxFitResult]:
    torch.manual_seed(int(seed))

    X_train = np.asarray(
        X_train,
        dtype=np.float32,
    )
    X_tuning = np.asarray(
        X_tuning,
        dtype=np.float32,
    )
    durations_train, events_train = validate_survival_targets(
        durations_train,
        events_train,
    )
    y_tuning = np.asarray(
        y_tuning,
        dtype=int,
    ).reshape(-1)

    Xtr = torch.from_numpy(X_train).to(device)
    ttr = torch.from_numpy(
        durations_train.astype(np.float32)
    ).to(device)
    etr = torch.from_numpy(
        events_train.astype(np.float32)
    ).to(device)
    Xtu = torch.from_numpy(X_tuning).to(device)

    model = DeepCoxPHNet(
        input_dim=X_train.shape[1],
        hidden_dims=hidden_dims,
        dropout=dropout,
    ).to(device)

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(learning_rate),
        weight_decay=float(weight_decay),
    )

    best_state = None
    best_pr_auc = -np.inf
    best_epoch = 0
    epochs_without_improvement = 0
    history = []

    for epoch in range(1, int(max_epochs) + 1):
        model.train()
        optimizer.zero_grad(set_to_none=True)

        scores = model(Xtr)
        loss = cox_breslow_negative_log_likelihood(
            scores,
            ttr,
            etr,
        )
        loss.backward()
        torch.nn.utils.clip_grad_norm_(
            model.parameters(),
            2.0,
        )
        optimizer.step()

        model.eval()
        with torch.no_grad():
            tuning_scores = model(Xtu).cpu().numpy()

        tuning_metrics = binary_ranking_metrics(
            y_tuning,
            tuning_scores,
        )

        history.append(
            {
                "epoch": epoch,
                "cox_negative_partial_log_likelihood": float(
                    loss.detach().cpu().item()
                ),
                "tuning_auroc": tuning_metrics["auroc"],
                "tuning_pr_auc": tuning_metrics["pr_auc"],
                "tuning_enrichment": tuning_metrics["enrichment"],
            }
        )

        if tuning_metrics["pr_auc"] > best_pr_auc + 1e-12:
            best_pr_auc = float(
                tuning_metrics["pr_auc"]
            )
            best_epoch = int(epoch)
            best_state = deepcopy(
                model.state_dict()
            )
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1

        if epochs_without_improvement >= int(patience):
            break

    if best_state is None:
        raise RuntimeError(
            "Deep CoxPH training did not produce a valid checkpoint."
        )

    model.load_state_dict(best_state)
    model.eval()

    with torch.no_grad():
        train_scores = model(Xtr).cpu().numpy()

    H0_tau, offset = (
        breslow_baseline_cumulative_hazard_at_tau(
            scores=train_scores,
            durations=durations_train,
            events=events_train,
            tau=tau,
        )
    )

    result = DeepCoxFitResult(
        state_dict=best_state,
        best_epoch=best_epoch,
        best_tuning_pr_auc=best_pr_auc,
        training_history=pd.DataFrame(history),
        H0_tau=H0_tau,
        score_offset=offset,
    )
    return model, result


class DeepSurvivalMachine(nn.Module):
    """
    Mixture-of-Weibulls Deep Survival Machine.

    The network predicts mixture weights, Weibull scales, and Weibull shapes.
    Training uses the exact right-censored mixture likelihood.
    """

    def __init__(
        self,
        input_dim: int,
        components: int = 3,
        hidden_dims: Sequence[int] = (128, 64),
        dropout: float = 0.10,
    ):
        super().__init__()

        self.components = int(components)
        if self.components < 1:
            raise ValueError(
                "components must be at least one."
            )

        layers: List[nn.Module] = []
        dim = int(input_dim)
        for hidden in hidden_dims:
            layers.extend(
                [
                    nn.Linear(dim, int(hidden)),
                    nn.ReLU(),
                    nn.Dropout(float(dropout)),
                ]
            )
            dim = int(hidden)

        self.backbone = nn.Sequential(*layers)
        self.mixture_head = nn.Linear(
            dim,
            self.components,
        )
        self.log_scale_head = nn.Linear(
            dim,
            self.components,
        )
        self.raw_shape_head = nn.Linear(
            dim,
            self.components,
        )

    def parameters_from_X(
        self,
        X: torch.Tensor,
    ):
        h = self.backbone(X)

        log_pi = torch.log_softmax(
            self.mixture_head(h),
            dim=1,
        )
        log_scale = torch.clamp(
            self.log_scale_head(h),
            min=-4.0,
            max=8.0,
        )
        shape = torch.nn.functional.softplus(
            self.raw_shape_head(h)
        ) + 0.10
        shape = torch.clamp(
            shape,
            min=0.10,
            max=10.0,
        )

        return log_pi, log_scale, shape

    def negative_log_likelihood(
        self,
        X: torch.Tensor,
        durations: torch.Tensor,
        events: torch.Tensor,
    ) -> torch.Tensor:
        log_pi, log_scale, shape = (
            self.parameters_from_X(X)
        )

        t = torch.clamp(
            durations.reshape(-1, 1),
            min=1e-6,
        )
        e = events.reshape(-1)

        log_t = torch.log(t)
        z = shape * (log_t - log_scale)
        cumulative_hazard = torch.exp(
            torch.clamp(
                z,
                min=-30.0,
                max=30.0,
            )
        )

        log_survival_component = (
            -cumulative_hazard
        )
        log_density_component = (
            torch.log(shape)
            - log_scale
            + (shape - 1.0)
            * (log_t - log_scale)
            - cumulative_hazard
        )

        log_survival = torch.logsumexp(
            log_pi + log_survival_component,
            dim=1,
        )
        log_density = torch.logsumexp(
            log_pi + log_density_component,
            dim=1,
        )

        log_likelihood = (
            e * log_density
            + (1.0 - e) * log_survival
        )

        return -torch.mean(log_likelihood)

    def risk_at_tau(
        self,
        X: torch.Tensor,
        tau: float,
    ) -> torch.Tensor:
        log_pi, log_scale, shape = (
            self.parameters_from_X(X)
        )

        log_t = torch.log(
            torch.full(
                (X.shape[0], 1),
                float(tau),
                dtype=X.dtype,
                device=X.device,
            )
        )

        z = shape * (log_t - log_scale)
        cumulative_hazard = torch.exp(
            torch.clamp(
                z,
                min=-30.0,
                max=30.0,
            )
        )
        log_survival_component = (
            -cumulative_hazard
        )

        log_survival = torch.logsumexp(
            log_pi + log_survival_component,
            dim=1,
        )
        survival = torch.exp(
            torch.clamp(
                log_survival,
                min=-50.0,
                max=0.0,
            )
        )

        return torch.clamp(
            1.0 - survival,
            0.0,
            1.0,
        )


@dataclass
class DSMFitResult:
    state_dict: dict
    best_epoch: int
    best_tuning_pr_auc: float
    training_history: pd.DataFrame


def fit_deep_survival_machine(
    X_train: np.ndarray,
    durations_train,
    events_train,
    X_tuning: np.ndarray,
    y_tuning,
    tau: float,
    components: int,
    hidden_dims: Sequence[int],
    dropout: float,
    learning_rate: float,
    weight_decay: float,
    batch_size: int,
    max_epochs: int,
    patience: int,
    seed: int,
    device: torch.device,
) -> tuple[DeepSurvivalMachine, DSMFitResult]:
    torch.manual_seed(int(seed))

    X_train = np.asarray(
        X_train,
        dtype=np.float32,
    )
    X_tuning = np.asarray(
        X_tuning,
        dtype=np.float32,
    )
    durations_train, events_train = validate_survival_targets(
        durations_train,
        events_train,
    )
    y_tuning = np.asarray(
        y_tuning,
        dtype=int,
    ).reshape(-1)

    dataset = TensorDataset(
        torch.from_numpy(X_train),
        torch.from_numpy(
            durations_train.astype(np.float32)
        ),
        torch.from_numpy(
            events_train.astype(np.float32)
        ),
    )

    generator = torch.Generator()
    generator.manual_seed(int(seed))

    loader = DataLoader(
        dataset,
        batch_size=int(batch_size),
        shuffle=True,
        num_workers=0,
        generator=generator,
    )

    Xtu = torch.from_numpy(
        X_tuning
    ).to(device)

    model = DeepSurvivalMachine(
        input_dim=X_train.shape[1],
        components=int(components),
        hidden_dims=hidden_dims,
        dropout=dropout,
    ).to(device)

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(learning_rate),
        weight_decay=float(weight_decay),
    )

    best_state = None
    best_pr_auc = -np.inf
    best_epoch = 0
    epochs_without_improvement = 0
    history = []

    for epoch in range(1, int(max_epochs) + 1):
        model.train()
        batch_losses = []

        for Xb, tb, eb in loader:
            Xb = Xb.to(device)
            tb = tb.to(device)
            eb = eb.to(device)

            optimizer.zero_grad(set_to_none=True)
            loss = model.negative_log_likelihood(
                Xb,
                tb,
                eb,
            )
            loss.backward()
            torch.nn.utils.clip_grad_norm_(
                model.parameters(),
                2.0,
            )
            optimizer.step()

            batch_losses.append(
                float(
                    loss.detach().cpu().item()
                )
            )

        model.eval()
        with torch.no_grad():
            tune_risk = (
                model.risk_at_tau(
                    Xtu,
                    tau=float(tau),
                )
                .cpu()
                .numpy()
            )

        tuning_metrics = binary_ranking_metrics(
            y_tuning,
            tune_risk,
        )

        history.append(
            {
                "epoch": epoch,
                "training_negative_log_likelihood": float(
                    np.mean(batch_losses)
                ),
                "tuning_auroc": tuning_metrics["auroc"],
                "tuning_pr_auc": tuning_metrics["pr_auc"],
                "tuning_enrichment": tuning_metrics["enrichment"],
            }
        )

        if tuning_metrics["pr_auc"] > best_pr_auc + 1e-12:
            best_pr_auc = float(
                tuning_metrics["pr_auc"]
            )
            best_epoch = int(epoch)
            best_state = deepcopy(
                model.state_dict()
            )
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1

        if epochs_without_improvement >= int(patience):
            break

    if best_state is None:
        raise RuntimeError(
            "DSM training did not produce a valid checkpoint."
        )

    model.load_state_dict(best_state)
    model.eval()

    return (
        model,
        DSMFitResult(
            state_dict=best_state,
            best_epoch=best_epoch,
            best_tuning_pr_auc=best_pr_auc,
            training_history=pd.DataFrame(history),
        ),
    )
