from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from maestro_pdm.config import get_output_dir, load_config
from maestro_pdm.evaluation.bootstrap import (
    BootstrapConfig,
    paired_stratified_bootstrap,
    percentile_interval,
)


PROPOSED = "Frozen proposed GRU"
METRICS = ("auroc", "pr_auc", "enrichment")


def args():
    p = argparse.ArgumentParser()
    p.add_argument("--config", default="configs/paper_experiment.yaml")
    p.add_argument("--n-resamples", type=int, default=None)
    p.add_argument("--seed", type=int, default=None)
    return p.parse_args()


def main():
    a = args()
    cfg = load_config(a.config)
    output_root = get_output_dir(cfg)
    pred_path = (
        output_root
        / "predictions"
        / "survival_baseline_validation_predictions_reference.csv"
    )
    if not pred_path.exists():
        raise FileNotFoundError(
            f"Missing Stage 4B compiled predictions: {pred_path}\n"
            "Run scripts/04b5_compile_reference_baselines.py first."
        )

    df = pd.read_csv(pred_path)
    required = {
        "vehicle_id",
        "binary_label",
        "CoxPH",
        "RSF",
        "Deep CoxPH",
        "DSM",
        PROPOSED,
    }
    missing = sorted(required - set(df.columns))
    if missing:
        raise ValueError(f"Compiled prediction file is missing columns: {missing}")

    if df["vehicle_id"].duplicated().any():
        raise ValueError("Validation prediction file contains duplicate vehicle_id values.")

    y = df["binary_label"].to_numpy(dtype=int)
    unique = set(np.unique(y))
    if unique != {0, 1}:
        raise ValueError(f"Expected binary labels {{0,1}}, found {sorted(unique)}")

    model_names = ["CoxPH", "RSF", "Deep CoxPH", "DSM", PROPOSED]
    scores = {name: df[name].to_numpy(dtype=float) for name in model_names}

    boot_cfg = cfg.get("bootstrap", {})
    n_resamples = int(
        a.n_resamples
        if a.n_resamples is not None
        else boot_cfg.get("n_resamples", 5000)
    )
    confidence_level = float(boot_cfg.get("confidence_level", 0.95))
    seed = int(
        a.seed
        if a.seed is not None
        else boot_cfg.get("seed", cfg.get("experiment", {}).get("seed", 42))
    )

    bc = BootstrapConfig(
        n_resamples=n_resamples,
        confidence_level=confidence_level,
        seed=seed,
    )

    n = int(len(y))
    n_pos = int(y.sum())
    n_neg = int(n - n_pos)
    prevalence = float(y.mean())

    print("=" * 78)
    print("STAGE 4C PAIRED STRATIFIED BOOTSTRAP")
    print("=" * 78)
    print(f"Validation vehicles : {n}")
    print(f"Positive vehicles   : {n_pos}")
    print(f"Negative vehicles   : {n_neg}")
    print(f"Prevalence          : {prevalence:.6f}")
    print(f"Bootstrap resamples : {n_resamples}")
    print(f"Confidence level    : {confidence_level:.3f}")
    print(f"Seed                : {seed}")
    print("Resampling scheme   : positive and negative vehicles sampled separately")
    print("Pairing              : identical bootstrap indices used for every model")
    print()

    point, dist = paired_stratified_bootstrap(y, scores, bc)

    ci_rows = []
    for model in model_names:
        for metric in METRICS:
            lo, hi = percentile_interval(dist[model][metric], confidence_level)
            ci_rows.append(
                {
                    "model": model,
                    "metric": metric,
                    "point_estimate": point[model][metric],
                    "bootstrap_mean": float(np.mean(dist[model][metric])),
                    "ci_lower": lo,
                    "ci_upper": hi,
                    "confidence_level": confidence_level,
                    "n_resamples": n_resamples,
                }
            )
    ci_df = pd.DataFrame(ci_rows)

    diff_rows = []
    for baseline in ["CoxPH", "RSF", "Deep CoxPH", "DSM"]:
        for metric in METRICS:
            diff = dist[PROPOSED][metric] - dist[baseline][metric]
            lo, hi = percentile_interval(diff, confidence_level)
            point_diff = point[PROPOSED][metric] - point[baseline][metric]
            diff_rows.append(
                {
                    "baseline": baseline,
                    "metric": metric,
                    "proposed_point_estimate": point[PROPOSED][metric],
                    "baseline_point_estimate": point[baseline][metric],
                    "difference_proposed_minus_baseline": point_diff,
                    "bootstrap_difference_mean": float(np.mean(diff)),
                    "ci_lower": lo,
                    "ci_upper": hi,
                    "ci_excludes_zero": bool(lo > 0.0 or hi < 0.0),
                    "direction": (
                        "proposed_higher" if point_diff > 0
                        else "proposed_lower" if point_diff < 0
                        else "equal"
                    ),
                    "confidence_level": confidence_level,
                    "n_resamples": n_resamples,
                }
            )
    diff_df = pd.DataFrame(diff_rows)

    tables_dir = output_root / "tables"
    logs_dir = output_root / "logs"
    bootstrap_dir = output_root / "bootstrap"
    tables_dir.mkdir(parents=True, exist_ok=True)
    logs_dir.mkdir(parents=True, exist_ok=True)
    bootstrap_dir.mkdir(parents=True, exist_ok=True)

    ci_path = tables_dir / "stage4c_bootstrap_model_metric_cis.csv"
    diff_path = tables_dir / "stage4c_bootstrap_paired_differences_vs_proposed.csv"
    audit_path = tables_dir / "stage4c_bootstrap_validation_audit.csv"
    npz_path = bootstrap_dir / "stage4c_bootstrap_distributions.npz"
    json_path = logs_dir / "stage4c_bootstrap_summary.json"

    ci_df.to_csv(ci_path, index=False)
    diff_df.to_csv(diff_path, index=False)
    pd.DataFrame(
        [{
            "validation_vehicles": n,
            "positive_vehicles": n_pos,
            "negative_vehicles": n_neg,
            "prevalence": prevalence,
            "n_resamples": n_resamples,
            "confidence_level": confidence_level,
            "seed": seed,
            "stratified_by_binary_label": True,
            "paired_across_models": True,
        }]
    ).to_csv(audit_path, index=False)

    npz_payload = {}
    for model in model_names:
        slug = model.lower().replace(" ", "_")
        for metric in METRICS:
            npz_payload[f"{slug}__{metric}"] = dist[model][metric]
    np.savez_compressed(npz_path, **npz_payload)

    summary = {
        "stage": "4C",
        "method": "paired stratified nonparametric bootstrap",
        "resampling_unit": "validation vehicle",
        "n_validation": n,
        "n_positive": n_pos,
        "n_negative": n_neg,
        "prevalence": prevalence,
        "n_resamples": n_resamples,
        "confidence_level": confidence_level,
        "seed": seed,
        "same_indices_for_all_models": True,
        "model_metric_intervals": ci_df.to_dict(orient="records"),
        "paired_differences_vs_proposed": diff_df.to_dict(orient="records"),
    }
    with json_path.open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    print("MODEL METRIC 95% BOOTSTRAP INTERVALS")
    print("-" * 78)
    wide = ci_df.copy()
    wide["estimate_ci"] = wide.apply(
        lambda r: f"{r['point_estimate']:.6f} [{r['ci_lower']:.6f}, {r['ci_upper']:.6f}]",
        axis=1,
    )
    print(
        wide.pivot(index="model", columns="metric", values="estimate_ci")
        .reindex(model_names)
        .to_string()
    )

    print("\nPAIRED DIFFERENCES: PROPOSED - BASELINE")
    print("-" * 78)
    show = diff_df.copy()
    show["difference_ci"] = show.apply(
        lambda r: f"{r['difference_proposed_minus_baseline']:.6f} "
                  f"[{r['ci_lower']:.6f}, {r['ci_upper']:.6f}]",
        axis=1,
    )
    print(
        show[["baseline", "metric", "difference_ci", "ci_excludes_zero"]]
        .to_string(index=False)
    )

    print("\nSaved:")
    for p in [ci_path, diff_path, audit_path, npz_path, json_path]:
        print(f"  {p}")


if __name__ == "__main__":
    main()
