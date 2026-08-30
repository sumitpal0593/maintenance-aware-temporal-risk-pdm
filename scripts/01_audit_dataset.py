from __future__ import annotations

import argparse
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

from maestro_pdm.config import get_data_dir, get_output_dir, load_config
from maestro_pdm.data.labels import (
    build_horizon_samples,
    samples_to_frame,
    summarize_horizon_samples,
)
from maestro_pdm.data.loading import (
    collect_vehicle_times,
    detect_id_column,
    operational_columns,
    read_specifications,
    read_train_tte,
    read_validation_labels,
    validate_dataset_directory,
)
from maestro_pdm.data.preprocessing import (
    fit_operational_preprocessor,
    fit_static_encoder,
    save_operational_preprocessor,
    save_static_encoder,
    static_encoding_summary,
)
from maestro_pdm.reproducibility import set_global_seed


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Stage 2A+2B: audit the Scania dataset, create vehicle-level "
            "development splits, fit training-only preprocessing, and "
            "construct censoring-aware Y_i(tau) labels."
        )
    )
    parser.add_argument(
        "--config",
        default="configs/paper_experiment.yaml",
        help="Path to YAML experiment configuration.",
    )
    parser.add_argument(
        "--skip-operational-preprocessor",
        action="store_true",
        help=(
            "Skip the full streaming operational-statistics pass. Useful only "
            "for a quick label-audit smoke run; do not use for final paper runs."
        ),
    )
    return parser.parse_args()


def make_vehicle_splits(
    tte: pd.DataFrame,
    id_col: str,
    config: dict,
) -> pd.DataFrame:
    split_cfg = config["split"]
    train_fraction = float(split_cfg["training_fraction"])
    tune_fraction = float(split_cfg["tuning_fraction"])
    calibration_fraction = float(split_cfg["calibration_fraction"])

    total = train_fraction + tune_fraction + calibration_fraction
    if not np.isclose(total, 1.0):
        raise ValueError(
            "training_fraction + tuning_fraction + calibration_fraction "
            f"must equal 1.0; got {total}"
        )

    ids = tte[id_col].astype(int).to_numpy()
    events = tte["in_study_repair"].astype(int).to_numpy()

    seed = int(config["experiment"]["seed"])
    stratify = events if bool(split_cfg.get("stratify", True)) else None

    fit_ids, remainder_ids, fit_events, remainder_events = train_test_split(
        ids,
        events,
        test_size=(1.0 - train_fraction),
        random_state=seed,
        stratify=stratify,
    )

    remainder_fraction = tune_fraction + calibration_fraction
    calibration_share_of_remainder = calibration_fraction / remainder_fraction
    remainder_stratify = (
        remainder_events if bool(split_cfg.get("stratify", True)) else None
    )

    tune_ids, calibration_ids, _, _ = train_test_split(
        remainder_ids,
        remainder_events,
        test_size=calibration_share_of_remainder,
        random_state=seed,
        stratify=remainder_stratify,
    )

    event_map = dict(zip(ids, events))
    rows = []
    for split_name, split_ids in (
        ("model_train", fit_ids),
        ("tuning", tune_ids),
        ("calibration", calibration_ids),
    ):
        for vid in split_ids:
            rows.append(
                {
                    "vehicle_id": int(vid),
                    "split": split_name,
                    "in_study_repair": int(event_map[int(vid)]),
                }
            )

    split_df = pd.DataFrame(rows).sort_values(
        ["split", "vehicle_id"]
    ).reset_index(drop=True)

    if split_df["vehicle_id"].duplicated().any():
        raise RuntimeError("A vehicle appears in more than one development split.")
    if len(split_df) != len(ids):
        raise RuntimeError(
            f"Expected {len(ids)} split assignments, got {len(split_df)}."
        )

    return split_df


def main():
    args = parse_args()
    cfg = load_config(args.config)
    set_global_seed(int(cfg["experiment"]["seed"]))

    data_dir = get_data_dir()
    validate_dataset_directory(data_dir)
    output_root = get_output_dir(cfg)

    tables_dir = output_root / "tables"
    checkpoints_dir = output_root / "checkpoints"
    logs_dir = output_root / "logs"
    for p in (tables_dir, checkpoints_dir, logs_dir):
        p.mkdir(parents=True, exist_ok=True)

    tau = int(cfg["experiment"]["tau"])
    chunk_rows = int(cfg["data"]["ops_chunk_rows"])
    min_prefix_length = int(cfg["data"]["min_prefix_length"])
    samples_per_vehicle = int(cfg["data"]["samples_per_vehicle"])
    epsilon = float(cfg["preprocessing"]["epsilon"])

    print("=" * 78)
    print("STAGE 2A + 2B")
    print("Scania preprocessing and fixed-horizon label audit")
    print("=" * 78)
    print(f"Data directory       : {data_dir}")
    print(f"Prediction horizon   : tau = {tau} relative time steps")
    print(f"Chunk rows           : {chunk_rows:,}")
    print()

    # --------------------------------------------------------------
    # Core public tables
    # --------------------------------------------------------------
    tte = read_train_tte(data_dir)
    tte_id_col = detect_id_column(tte)

    required_tte = {"length_of_study_time_step", "in_study_repair"}
    missing = required_tte.difference(tte.columns)
    if missing:
        raise ValueError(f"train_tte.csv missing columns: {sorted(missing)}")

    validation_labels = read_validation_labels(data_dir)
    val_id_col = detect_id_column(validation_labels)
    label_cols = [c for c in validation_labels.columns if c != val_id_col]
    if "class_label" in validation_labels.columns:
        class_col = "class_label"
    elif len(label_cols) == 1:
        class_col = label_cols[0]
    else:
        raise ValueError(
            "Could not unambiguously identify the validation class-label column."
        )

    train_specs = read_specifications(data_dir, "train")
    val_specs = read_specifications(data_dir, "validation")

    op_id_col, op_time_col, operational_feature_cols = operational_columns(
        data_dir, "train"
    )

    # --------------------------------------------------------------
    # Vehicle-level model-training / tuning / calibration split
    # --------------------------------------------------------------
    split_df = make_vehicle_splits(tte, tte_id_col, cfg)
    split_df.to_csv(tables_dir / "development_split.csv", index=False)

    split_summary = (
        split_df.groupby("split", sort=False)["in_study_repair"]
        .agg(["count", "sum"])
        .reset_index()
        .rename(columns={"count": "vehicles", "sum": "event_vehicles"})
    )
    split_summary["event_rate"] = (
        split_summary["event_vehicles"] / split_summary["vehicles"]
    )
    split_summary.to_csv(
        tables_dir / "development_split_summary.csv",
        index=False,
    )

    model_train_ids = set(
        split_df.loc[
            split_df["split"] == "model_train",
            "vehicle_id",
        ].astype(int)
    )

    # --------------------------------------------------------------
    # Static specification encoder: fit on model-training vehicles only
    # --------------------------------------------------------------
    encoder, spec_id_col, spec_feature_cols = fit_static_encoder(
        train_specs,
        model_train_ids,
    )
    save_static_encoder(
        encoder,
        checkpoints_dir / "static_onehot_encoder.joblib",
    )
    static_overview, static_categories = static_encoding_summary(
        encoder,
        spec_feature_cols,
    )
    static_overview.to_csv(
        tables_dir / "static_encoding_summary.csv",
        index=False,
    )
    static_categories.to_csv(
        tables_dir / "static_encoding_categories.csv",
        index=False,
    )

    # --------------------------------------------------------------
    # Horizon labels: load only vehicle_id/time_step, not 105 features
    # --------------------------------------------------------------
    print("Collecting training vehicle time-step histories for label audit...")
    vehicle_times, _, _ = collect_vehicle_times(
        data_dir=data_dir,
        split="train",
        chunk_rows=chunk_rows,
        allowed_vehicle_ids=set(split_df["vehicle_id"].astype(int)),
    )

    horizon_samples = build_horizon_samples(
        vehicle_times=vehicle_times,
        tte_df=tte,
        id_col=tte_id_col,
        split_assignments=split_df,
        tau=tau,
        min_prefix_length=min_prefix_length,
        samples_per_vehicle=samples_per_vehicle,
    )
    horizon_samples_df = samples_to_frame(horizon_samples)
    horizon_samples_df.to_csv(
        tables_dir / "horizon_prefix_samples.csv",
        index=False,
    )

    label_audit = summarize_horizon_samples(horizon_samples_df)

    vehicle_counts = (
        split_df.groupby("split")["vehicle_id"]
        .nunique()
        .rename("vehicles")
        .reset_index()
    )
    label_audit = vehicle_counts.merge(
        label_audit,
        on="split",
        how="left",
    )
    label_audit.to_csv(
        tables_dir / "training_label_audit.csv",
        index=False,
    )

    # --------------------------------------------------------------
    # Dataset-level audit
    # --------------------------------------------------------------
    train_event_count = int(tte["in_study_repair"].astype(int).sum())
    train_vehicle_count = int(len(tte))
    train_censored_count = train_vehicle_count - train_event_count

    validation_binary = (
        validation_labels[class_col].astype(int) > 0
    ).astype(int)
    validation_positive = int(validation_binary.sum())
    validation_negative = int(len(validation_binary) - validation_positive)

    validation_class_distribution = (
        validation_labels[class_col]
        .value_counts()
        .sort_index()
        .rename_axis("class_label")
        .reset_index(name="count")
    )
    validation_class_distribution["percentage"] = (
        validation_class_distribution["count"]
        / len(validation_labels)
        * 100.0
    )
    validation_class_distribution.to_csv(
        tables_dir / "validation_class_distribution.csv",
        index=False,
    )

    audit_rows = [
        ("prediction_horizon_tau", tau),
        ("train_tte_vehicles", train_vehicle_count),
        ("train_event_vehicles", train_event_count),
        ("train_censored_vehicles", train_censored_count),
        ("train_vehicle_event_rate", train_event_count / train_vehicle_count),
        ("validation_vehicles", int(len(validation_labels))),
        ("validation_binary_positive", validation_positive),
        ("validation_binary_negative", validation_negative),
        (
            "validation_binary_positive_rate",
            validation_positive / len(validation_labels),
        ),
        ("operational_feature_dimension_p", len(operational_feature_cols)),
        ("original_static_dimension_q", len(spec_feature_cols)),
        (
            "encoded_static_dimension_q_e",
            int(static_overview.loc[0, "encoded_static_dimension_q_e"]),
        ),
        ("model_train_vehicles", len(model_train_ids)),
        (
            "tuning_vehicles",
            int((split_df["split"] == "tuning").sum()),
        ),
        (
            "calibration_vehicles",
            int((split_df["split"] == "calibration").sum()),
        ),
    ]
    dataset_audit = pd.DataFrame(audit_rows, columns=["quantity", "value"])
    dataset_audit.to_csv(
        tables_dir / "dataset_audit.csv",
        index=False,
    )

    # --------------------------------------------------------------
    # Full training-only operational preprocessing statistics
    # --------------------------------------------------------------
    if args.skip_operational_preprocessor:
        print()
        print(
            "[WARNING] Operational preprocessor was skipped. "
            "This is not a final-paper run."
        )
    else:
        print()
        print(
            "Fitting operational mean-imputation/standardization statistics "
            "on model-training vehicles only..."
        )
        preprocessor = fit_operational_preprocessor(
            data_dir=data_dir,
            allowed_vehicle_ids=model_train_ids,
            chunk_rows=chunk_rows,
            epsilon=epsilon,
        )
        save_operational_preprocessor(
            preprocessor,
            checkpoints_dir / "operational_preprocessor.joblib",
        )
        preprocessing_summary = preprocessor.summary_frame()
        preprocessing_summary.to_csv(
            tables_dir / "operational_preprocessing_summary.csv",
            index=False,
        )

    # --------------------------------------------------------------
    # Machine-readable stage summary
    # --------------------------------------------------------------
    train_audit_row = label_audit[
        label_audit["split"] == "model_train"
    ].iloc[0]

    run_summary = {
        "tau": tau,
        "train_tte_vehicle_event_rate": train_event_count / train_vehicle_count,
        "validation_binary_positive_rate": (
            validation_positive / len(validation_labels)
        ),
        "training_horizon_positive_rate": (
            None
            if pd.isna(train_audit_row["horizon_positive_prevalence"])
            else float(train_audit_row["horizon_positive_prevalence"])
        ),
        "omega_0": (
            None
            if pd.isna(train_audit_row["omega_0"])
            else float(train_audit_row["omega_0"])
        ),
        "omega_1": (
            None
            if pd.isna(train_audit_row["omega_1"])
            else float(train_audit_row["omega_1"])
        ),
        "q_e": int(static_overview.loc[0, "encoded_static_dimension_q_e"]),
        "operational_feature_dimension_p": len(operational_feature_cols),
        "validation_vehicle_count": int(len(validation_labels)),
        "operational_preprocessor_fitted": (
            not args.skip_operational_preprocessor
        ),
    }
    with (logs_dir / "stage2a2b_summary.json").open(
        "w",
        encoding="utf-8",
    ) as f:
        json.dump(run_summary, f, indent=2)

    print()
    print("-" * 78)
    print("AUDIT SUMMARY")
    print("-" * 78)
    print(
        f"Raw train repair-event rate        : "
        f"{train_event_count}/{train_vehicle_count} "
        f"= {train_event_count / train_vehicle_count:.6f}"
    )
    print(
        f"Validation horizon-positive rate   : "
        f"{validation_positive}/{len(validation_labels)} "
        f"= {validation_positive / len(validation_labels):.6f}"
    )
    print(
        f"Model-train horizon-positive rate  : "
        f"{train_audit_row['horizon_positive_prevalence']:.6f}"
    )
    print(
        f"Weighted BCE class weights         : "
        f"omega_0={train_audit_row['omega_0']:.6f}, "
        f"omega_1={train_audit_row['omega_1']:.6f}"
    )
    print(
        f"Operational dimension p            : "
        f"{len(operational_feature_cols)}"
    )
    print(
        f"Static dimensions q -> q_e         : "
        f"{len(spec_feature_cols)} -> "
        f"{int(static_overview.loc[0, 'encoded_static_dimension_q_e'])}"
    )
    print(f"Validation vehicles                : {len(validation_labels)}")
    print()
    print("Saved:")
    for name in [
        "dataset_audit.csv",
        "development_split.csv",
        "development_split_summary.csv",
        "training_label_audit.csv",
        "horizon_prefix_samples.csv",
        "validation_class_distribution.csv",
        "static_encoding_summary.csv",
        "static_encoding_categories.csv",
    ]:
        print(f"  - {tables_dir / name}")

    if not args.skip_operational_preprocessor:
        print(
            f"  - {tables_dir / 'operational_preprocessing_summary.csv'}"
        )
        print(
            f"  - {checkpoints_dir / 'operational_preprocessor.joblib'}"
        )

    print(f"  - {checkpoints_dir / 'static_onehot_encoder.joblib'}")
    print(f"  - {logs_dir / 'stage2a2b_summary.json'}")
    print()
    print("Stage 2A + 2B completed successfully.")


if __name__ == "__main__":
    main()
