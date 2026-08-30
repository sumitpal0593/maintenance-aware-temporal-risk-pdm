from __future__ import annotations

from typing import Dict

import numpy as np
from sklearn.metrics import average_precision_score, roc_auc_score


def binary_ranking_metrics(y_true, y_score) -> Dict[str, float]:
    """
    AUROC, PR-AUC, prevalence, and enrichment over random ranking.
    """
    y_true = np.asarray(y_true, dtype=int).reshape(-1)
    y_score = np.asarray(y_score, dtype=float).reshape(-1)

    if len(y_true) != len(y_score):
        raise ValueError("y_true and y_score must have equal length.")
    if len(np.unique(y_true)) != 2:
        raise ValueError(
            "Both binary classes are required for ranking metrics."
        )

    prevalence = float(np.mean(y_true))
    auroc = float(roc_auc_score(y_true, y_score))
    pr_auc = float(average_precision_score(y_true, y_score))
    enrichment = (
        float(pr_auc / prevalence)
        if prevalence > 0
        else np.nan
    )

    return {
        "n": int(len(y_true)),
        "n_positive": int(y_true.sum()),
        "n_negative": int(
            len(y_true) - y_true.sum()
        ),
        "positive_prevalence": prevalence,
        "auroc": auroc,
        "pr_auc": pr_auc,
        "enrichment": enrichment,
    }
