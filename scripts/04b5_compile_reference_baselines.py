from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from maestro_pdm.config import get_output_dir, load_config
from maestro_pdm.evaluation.reference_baselines import (
    binary_ranking_metrics,
)


MODEL_FILES = {
    "CoxPH": "coxph_reference",
    "RSF": "rsf_reference",
    "Deep CoxPH": "deep_coxph_reference",
    "DSM": "dsm_reference",
}


def args():
    p = argparse.ArgumentParser()
    p.add_argument("--config", default="configs/paper_experiment.yaml")
    return p.parse_args()


def main():
    a = args()
    cfg = load_config(a.config)
    output_root = get_output_dir(cfg)
    tables = output_root / "tables"
    predictions = output_root / "predictions"
    logs = output_root / "logs"

    metric_frames = []
    pred_frames = []

    for model, slug in MODEL_FILES.items():
        metric_path = tables / f"{slug}_validation_metrics.csv"
        pred_path = predictions / f"{slug}_validation_predictions.csv"

        if not metric_path.exists() or not pred_path.exists():
            raise FileNotFoundError(
                f"Missing completed {model} output. Expected:\n"
                f"  {metric_path}\n  {pred_path}"
            )

        metric_frames.append(pd.read_csv(metric_path))
        p = pd.read_csv(pred_path)[
            ["vehicle_id", "binary_label", "risk_tau"]
        ].rename(columns={"risk_tau": model})
        pred_frames.append(p)

    metrics = pd.concat(metric_frames, ignore_index=True)

    merged = pred_frames[0]
    for frame in pred_frames[1:]:
        merged = merged.merge(
            frame,
            on=["vehicle_id", "binary_label"],
            how="inner",
            validate="one_to_one",
        )

    proposed_path = (
        predictions / "validation_raw_predictions_vehicle_tuned.csv"
    )
    if not proposed_path.exists():
        raise FileNotFoundError(proposed_path)

    proposed = pd.read_csv(proposed_path)[
        ["vehicle_id", "binary_label", "raw_probability"]
    ].rename(columns={"raw_probability": "Frozen proposed GRU"})

    merged = merged.merge(
        proposed,
        on=["vehicle_id", "binary_label"],
        how="inner",
        validate="one_to_one",
    )

    proposed_metrics = binary_ranking_metrics(
        merged["binary_label"],
        merged["Frozen proposed GRU"],
    )

    proposed_row = {
        "model": "Frozen proposed GRU",
        "implementation": "Stage 2E.1 frozen temporal-risk model",
        "configuration": "best epoch selected on vehicle-level tuning",
        **proposed_metrics,
    }
    metrics = pd.concat(
        [metrics, pd.DataFrame([proposed_row])],
        ignore_index=True,
    )

    comparison_rows = []
    for _, row in metrics[
        metrics["model"] != "Frozen proposed GRU"
    ].iterrows():
        comparison_rows.append(
            {
                "baseline": row["model"],
                "baseline_auroc": row["auroc"],
                "proposed_auroc": proposed_metrics["auroc"],
                "auroc_difference_proposed_minus_baseline": (
                    proposed_metrics["auroc"] - row["auroc"]
                ),
                "relative_auroc_improvement_percent": (
                    (
                        proposed_metrics["auroc"] - row["auroc"]
                    )
                    / row["auroc"]
                    * 100.0
                ),
                "baseline_pr_auc": row["pr_auc"],
                "proposed_pr_auc": proposed_metrics["pr_auc"],
                "pr_auc_difference_proposed_minus_baseline": (
                    proposed_metrics["pr_auc"] - row["pr_auc"]
                ),
                "relative_pr_auc_improvement_percent": (
                    (
                        proposed_metrics["pr_auc"] - row["pr_auc"]
                    )
                    / row["pr_auc"]
                    * 100.0
                ),
            }
        )

    comparison = pd.DataFrame(comparison_rows)

    metrics.to_csv(
        tables / "survival_baseline_validation_metrics_reference.csv",
        index=False,
    )
    comparison.to_csv(
        tables / "survival_baseline_comparison_with_proposed_reference.csv",
        index=False,
    )
    merged.to_csv(
        predictions / "survival_baseline_validation_predictions_reference.csv",
        index=False,
    )

    summary = {
        "stage": "4B-reference",
        "tau": float(cfg["experiment"]["tau"]),
        "validation_used_for_model_selection": False,
        "baseline_metrics": metrics.to_dict(orient="records"),
        "comparisons": comparison.to_dict(orient="records"),
    }
    logs.mkdir(parents=True, exist_ok=True)
    with (
        logs / "stage4b_reference_baselines_summary.json"
    ).open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    print()
    print("=" * 78)
    print("STAGE 4B REFERENCE-IMPLEMENTATION BASELINE RESULTS")
    print("=" * 78)
    print(
        metrics[
            [
                "model",
                "implementation",
                "auroc",
                "pr_auc",
                "enrichment",
                "configuration",
            ]
        ].to_string(
            index=False,
            float_format=lambda x: f"{x:.6f}",
        )
    )
    print()
    print("Proposed-model differences:")
    print(
        comparison[
            [
                "baseline",
                "auroc_difference_proposed_minus_baseline",
                "relative_auroc_improvement_percent",
                "pr_auc_difference_proposed_minus_baseline",
                "relative_pr_auc_improvement_percent",
            ]
        ].to_string(
            index=False,
            float_format=lambda x: f"{x:.6f}",
        )
    )


if __name__ == "__main__":
    main()
