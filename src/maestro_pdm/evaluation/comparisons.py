from __future__ import annotations

import numpy as np
import pandas as pd


def representation_comparison_summary(
    metrics: pd.DataFrame,
) -> pd.DataFrame:
    """
    Reviewer-facing pairwise comparisons that isolate temporal encoding.
    """
    table = metrics.set_index("model")

    pairs = [
        (
            "gru_temporal_logistic",
            "aggregate_dynamic_logistic",
            "learned temporal representation vs hand-engineered dynamic summary",
        ),
        (
            "gru_temporal_plus_static_logistic",
            "aggregate_dynamic_plus_static_logistic",
            "learned temporal representation plus static vs aggregate plus static",
        ),
        (
            "frozen_proposed_gru",
            "aggregate_dynamic_plus_static_logistic",
            "full proposed GRU vs conventional aggregate plus static logistic",
        ),
        (
            "frozen_proposed_gru",
            "time_only_logistic",
            "full proposed GRU vs age/sequence-length control",
        ),
    ]

    rows = []
    for model_a, model_b, interpretation in pairs:
        if model_a not in table.index or model_b not in table.index:
            continue

        a = table.loc[model_a]
        b = table.loc[model_b]

        rows.append(
            {
                "model_a": model_a,
                "model_b": model_b,
                "comparison": interpretation,
                "auroc_a": float(a["auroc"]),
                "auroc_b": float(b["auroc"]),
                "auroc_difference_a_minus_b": float(
                    a["auroc"] - b["auroc"]
                ),
                "pr_auc_a": float(a["pr_auc"]),
                "pr_auc_b": float(b["pr_auc"]),
                "pr_auc_difference_a_minus_b": float(
                    a["pr_auc"] - b["pr_auc"]
                ),
                "enrichment_a": float(a["enrichment"]),
                "enrichment_b": float(b["enrichment"]),
                "enrichment_difference_a_minus_b": float(
                    a["enrichment"] - b["enrichment"]
                ),
            }
        )

    return pd.DataFrame(rows)
