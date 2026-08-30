from __future__ import annotations

from copy import deepcopy
from pathlib import Path
from typing import Dict, Iterable, Optional

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

from maestro_pdm.evaluation.metrics import binary_ranking_metrics
from maestro_pdm.training.losses import weighted_bce_with_logits


def make_optimizer(model, config: dict):
    training_cfg = config["training"]
    name = str(training_cfg["optimizer"]).lower()

    if name == "adamw":
        return torch.optim.AdamW(
            model.parameters(),
            lr=float(training_cfg["learning_rate"]),
            weight_decay=float(training_cfg["weight_decay"]),
        )

    if name == "adam":
        return torch.optim.Adam(
            model.parameters(),
            lr=float(training_cfg["learning_rate"]),
            weight_decay=float(training_cfg["weight_decay"]),
        )

    raise ValueError(f"Unsupported optimizer: {name}")


def train_one_epoch(
    model,
    loader: DataLoader,
    optimizer,
    device: torch.device,
    omega_0: float,
    omega_1: float,
    gradient_clip_norm: Optional[float],
) -> float:
    model.train()
    losses = []

    for X_pad, lengths, static, y, _, _ in loader:
        X_pad = X_pad.to(device)
        lengths = lengths.to(device)
        static = static.to(device)
        y = y.to(device)

        optimizer.zero_grad(set_to_none=True)

        logits, _, _ = model(X_pad, lengths, static)
        loss = weighted_bce_with_logits(
            logits,
            y,
            omega_0=omega_0,
            omega_1=omega_1,
        )

        loss.backward()

        if gradient_clip_norm is not None:
            torch.nn.utils.clip_grad_norm_(
                model.parameters(),
                float(gradient_clip_norm),
            )

        optimizer.step()
        losses.append(float(loss.detach().cpu().item()))

    return float(np.mean(losses))


@torch.no_grad()
def predict_loader(
    model,
    loader: DataLoader,
    device: torch.device,
) -> pd.DataFrame:
    model.eval()
    rows = []

    for X_pad, lengths, static, y, vids, prefix_times in loader:
        X_pad = X_pad.to(device)
        lengths_device = lengths.to(device)
        static = static.to(device)

        logits, temporal, fused = model(
            X_pad,
            lengths_device,
            static,
        )
        probs = torch.sigmoid(logits)

        logits_np = logits.cpu().numpy()
        probs_np = probs.cpu().numpy()
        y_np = y.numpy()
        vids_np = vids.numpy()
        prefix_np = prefix_times.numpy()

        for i in range(len(y_np)):
            rows.append(
                {
                    "vehicle_id": int(vids_np[i]),
                    "prefix_time_step": int(prefix_np[i]),
                    "label": int(y_np[i]),
                    "raw_logit": float(logits_np[i]),
                    "raw_probability": float(probs_np[i]),
                }
            )

    return pd.DataFrame(rows)


def fit_with_early_stopping(
    model,
    train_loader: DataLoader,
    tuning_loader: DataLoader,
    optimizer,
    device: torch.device,
    omega_0: float,
    omega_1: float,
    epochs: int,
    patience: int,
    gradient_clip_norm: Optional[float],
    checkpoint_path: Path,
):
    """
    Train on model_train and select the epoch by tuning-set PR-AUC.

    The public validation set is never used for early stopping.
    """
    best_state = None
    best_pr_auc = -np.inf
    best_epoch = None
    epochs_without_improvement = 0
    history = []

    for epoch in range(1, int(epochs) + 1):
        train_loss = train_one_epoch(
            model=model,
            loader=train_loader,
            optimizer=optimizer,
            device=device,
            omega_0=omega_0,
            omega_1=omega_1,
            gradient_clip_norm=gradient_clip_norm,
        )

        tuning_pred = predict_loader(
            model=model,
            loader=tuning_loader,
            device=device,
        )
        tuning_metrics = binary_ranking_metrics(
            tuning_pred["label"],
            tuning_pred["raw_probability"],
        )

        history.append(
            {
                "epoch": epoch,
                "train_weighted_bce": train_loss,
                "tuning_auroc": tuning_metrics["auroc"],
                "tuning_pr_auc": tuning_metrics["pr_auc"],
                "tuning_enrichment": tuning_metrics["enrichment"],
            }
        )

        print(
            f"Epoch {epoch:02d} | "
            f"train WBCE={train_loss:.6f} | "
            f"tune AUROC={tuning_metrics['auroc']:.6f} | "
            f"tune PR-AUC={tuning_metrics['pr_auc']:.6f}"
        )

        if tuning_metrics["pr_auc"] > best_pr_auc + 1e-12:
            best_pr_auc = tuning_metrics["pr_auc"]
            best_epoch = epoch
            best_state = deepcopy(model.state_dict())
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1

        if epochs_without_improvement >= int(patience):
            print(
                f"Early stopping after epoch {epoch}; "
                f"best epoch was {best_epoch}."
            )
            break

    if best_state is None:
        raise RuntimeError("Training did not produce a valid best model.")

    model.load_state_dict(best_state)

    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "state_dict": best_state,
            "best_epoch": int(best_epoch),
            "best_tuning_pr_auc": float(best_pr_auc),
        },
        checkpoint_path,
    )

    return pd.DataFrame(history), int(best_epoch), float(best_pr_auc)


@torch.no_grad()
def predict_full_sequences(
    model,
    sequences: Dict[int, tuple],
    static_vectors: Dict[int, np.ndarray],
    device: torch.device,
    batch_size: int = 256,
) -> pd.DataFrame:
    """
    Predict one raw fixed-horizon risk score per vehicle using the full
    available sequence.

    Full-sequence validation prediction corresponds to the public dataset's
    final available readout for each vehicle.
    """
    model.eval()

    vids = sorted(set(sequences).intersection(static_vectors))
    rows = []

    for start in range(0, len(vids), int(batch_size)):
        batch_vids = vids[start : start + int(batch_size)]

        X_list = [sequences[v][1] for v in batch_vids]
        static_list = [static_vectors[v] for v in batch_vids]
        final_times = [int(sequences[v][0][-1]) for v in batch_vids]

        lengths = torch.tensor(
            [len(x) for x in X_list],
            dtype=torch.long,
        )
        max_len = int(lengths.max())
        n_features = int(X_list[0].shape[1])

        X_pad = torch.zeros(
            (len(batch_vids), max_len, n_features),
            dtype=torch.float32,
        )
        for i, x in enumerate(X_list):
            X_pad[i, : len(x), :] = torch.from_numpy(x)

        static = torch.from_numpy(
            np.stack(static_list).astype(np.float32)
        )

        logits, temporal, fused = model(
            X_pad.to(device),
            lengths.to(device),
            static.to(device),
        )
        probs = torch.sigmoid(logits)

        logits = logits.cpu().numpy()
        probs = probs.cpu().numpy()

        for i, vid in enumerate(batch_vids):
            rows.append(
                {
                    "vehicle_id": int(vid),
                    "final_time_step": int(final_times[i]),
                    "raw_logit": float(logits[i]),
                    "raw_probability": float(probs[i]),
                }
            )

    return pd.DataFrame(rows)
