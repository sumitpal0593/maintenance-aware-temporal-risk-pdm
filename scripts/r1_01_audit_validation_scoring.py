"""R1 reviewer audit of Scania validation scoring, without model retraining.

Example:
    python scripts/r1_01_audit_validation_scoring.py \
      --controls outputs/predictions/temporal_control_validation_predictions.csv \
      --raw outputs/predictions/validation_raw_predictions_vehicle_tuned.csv \
      --calibrated outputs/predictions/validation_calibrated_predictions_stage3_1.csv

Optional independent readout verification:
    --validation-readouts path/to/validation_operational_readouts.csv
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import numpy as np
import pandas as pd


def check_unique(df: pd.DataFrame, name: str, required: list[str]) -> None:
    missing = sorted(set(required) - set(df.columns))
    if missing:
        raise ValueError(f"{name} missing columns: {missing}")
    if df.vehicle_id.isna().any() or df.vehicle_id.duplicated().any():
        raise ValueError(f"{name}: null or duplicate vehicle_id")
    if df[required].isna().any().any():
        raise ValueError(f"{name}: null required values")


def equal_counts(a, b, title: str) -> None:
    if not np.array_equal(a, b):
        mismatches = int(np.count_nonzero(a != b))
        raise ValueError(f"{title}: {mismatches} inconsistent values")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--controls',type=Path,default=Path('outputs/predictions/temporal_control_validation_predictions.csv'))
    parser.add_argument('--raw',type=Path,default=Path('outputs/predictions/validation_raw_predictions_vehicle_tuned.csv'))
    parser.add_argument('--calibrated',type=Path,default=Path('outputs/predictions/validation_calibrated_predictions_stage3_1.csv'))
    parser.add_argument('--validation-readouts',type=Path,default=None,help='Optional raw Scania CSV for independent final-time and readout-count verification')
    parser.add_argument('--out-dir',type=Path,default=Path('outputs/r1_revision/scoring_audit'))
    args=parser.parse_args()

    controls=pd.read_csv(args.controls)
    raw=pd.read_csv(args.raw)
    calibrated=pd.read_csv(args.calibrated)
    check_unique(controls,'controls',['vehicle_id','binary_label','prefix_time_step','prefix_length','frozen_proposed_gru'])
    check_unique(raw,'raw',['vehicle_id','binary_label','class_label','final_time_step','raw_probability','raw_logit'])
    check_unique(calibrated,'calibrated',['vehicle_id','binary_label','class_label','final_time_step','raw_probability','raw_logit','ipcw_platt_probability'])
    if set(controls.vehicle_id)!=set(raw.vehicle_id) or set(controls.vehicle_id)!=set(calibrated.vehicle_id):
        raise ValueError('Vehicle ID sets differ across exported files')

    joined=(controls[['vehicle_id','binary_label','prefix_time_step','prefix_length','frozen_proposed_gru']]
      .merge(raw[['vehicle_id','binary_label','class_label','final_time_step','raw_probability','raw_logit']],on='vehicle_id',validate='one_to_one',suffixes=('_ctrl','_raw'))
      .merge(calibrated[['vehicle_id','binary_label','class_label','final_time_step','raw_probability','raw_logit','ipcw_platt_probability']],on='vehicle_id',validate='one_to_one',suffixes=('_raw','_cal')))
    equal_counts(joined.binary_label_ctrl.to_numpy(),joined.binary_label_raw.to_numpy(),'control/raw labels')
    equal_counts(joined.binary_label_ctrl.to_numpy(),joined.binary_label.to_numpy(),'control/calibrated labels')
    equal_counts(joined.class_label_raw.to_numpy(),joined.class_label_cal.to_numpy(),'class labels')
    equal_counts(joined.binary_label.to_numpy(),(joined.class_label_cal>0).astype(int).to_numpy(),'binary class derivation')
    equal_counts(joined.prefix_time_step.to_numpy(),joined.final_time_step_raw.to_numpy(),'control/raw scoring time')
    equal_counts(joined.prefix_time_step.to_numpy(),joined.final_time_step_cal.to_numpy(),'control/calibrated scoring time')
    if not np.allclose(joined.frozen_proposed_gru,joined.raw_probability_raw,rtol=0,atol=1e-8):
        raise ValueError('Frozen GRU control prediction does not match proposed raw prediction')
    if not np.allclose(joined.raw_probability_raw,joined.raw_probability_cal,rtol=0,atol=1e-8):
        raise ValueError('Raw score does not match between raw and calibration exports')
    if not np.allclose(joined.raw_logit_raw,joined.raw_logit_cal,rtol=0,atol=1e-8):
        raise ValueError('Raw logit does not match between raw and calibration exports')
    if not (joined.prefix_length >= 1).all():
        raise ValueError('Invalid empty validation history')
    if len(joined)!=5046 or int(joined.binary_label.sum())!=136:
        raise ValueError('Validation cohort is not the expected 5046/136')

    summary={
        'status':'PASS_EXPORTED_FILE_ALIGNMENT',
        'n_vehicles':len(joined),
        'n_positive':int(joined.binary_label.sum()),
        'n_negative':int((joined.binary_label==0).sum()),
        'unique_vehicle_ids':int(joined.vehicle_id.nunique()),
        'control_scoring_time_equals_raw_and_calibrated':True,
        'frozen_gru_scores_equal_raw_proposed_scores':True,
        'class_labels_and_binary_labels_aligned':True,
        'prefix_length_min':int(joined.prefix_length.min()),
        'prefix_length_median':float(joined.prefix_length.median()),
        'prefix_length_max':int(joined.prefix_length.max()),
        'final_time_step_min':int(joined.prefix_time_step.min()),
        'final_time_step_max':int(joined.prefix_time_step.max()),
        'raw_operational_csv_checked':False,
        'limitations':'Without --validation-readouts, verifies exported scoring metadata and code contract, not independent raw-source last-readout timestamps.'
    }

    if args.validation_readouts is not None:
        path=args.validation_readouts
        if not path.is_file():
            raise FileNotFoundError(path)
        header=pd.read_csv(path,nrows=0)
        columns={c.lower():c for c in header.columns}
        vehicle_col=next((columns[k] for k in ('vehicle_id','veh_id','id') if k in columns),None)
        time_col=next((columns[k] for k in ('time_step','timestep','time','step') if k in columns),None)
        if not vehicle_col or not time_col:
            raise ValueError('Could not detect ID/time columns in validation readouts')
        max_times={}; counts={}
        for chunk in pd.read_csv(path,usecols=[vehicle_col,time_col],chunksize=400000):
            chunk[vehicle_col]=pd.to_numeric(chunk[vehicle_col],errors='coerce')
            chunk[time_col]=pd.to_numeric(chunk[time_col],errors='coerce')
            chunk=chunk.dropna(subset=[vehicle_col,time_col])
            for vid, sub in chunk.groupby(vehicle_col,sort=False):
                k=int(vid)
                counts[k]=counts.get(k,0)+len(sub)
                max_times[k]=max(max_times.get(k,float('-inf')),float(sub[time_col].max()))
        all_ids=set(joined.vehicle_id.astype(int).tolist())
        if set(counts)!=all_ids:
            raise ValueError('Raw operational readouts vehicle set differs from predictions')
        raw_last = np.array([max_times[int(v)] for v in joined.vehicle_id], dtype=np.float64)
        lengths = np.array([counts[int(v)] for v in joined.vehicle_id], dtype=np.int64)
        exported = joined.prefix_time_step.to_numpy(dtype=np.int64)
        exported_lengths = joined.prefix_length.to_numpy(dtype=np.int64)

        # The actual project loader converts each raw time_step to np.int64
        # before sorting the vehicle history and exporting times[-1].
        # For finite values NumPy truncates toward zero; use that exact
        # convention here rather than comparing integer exports to raw floats.
        pipeline_last = raw_last.astype(np.int64)
        exact_last_matches = np.isclose(exported, raw_last, rtol=0, atol=1e-9)
        pipeline_last_matches = exported == pipeline_last
        length_matches = exported_lengths == lengths
        args.out_dir.mkdir(parents=True, exist_ok=True)
        diagnostics = pd.DataFrame({
            'vehicle_id': joined.vehicle_id.astype(int),
            'exported_final_time_step': exported,
            'raw_max_time_step': raw_last,
            'raw_max_as_int64': pipeline_last,
            'export_equals_raw_exactly': exact_last_matches,
            'export_equals_pipeline_int_time': pipeline_last_matches,
            'exported_prefix_length': exported_lengths,
            'raw_readout_count': lengths,
            'readout_count_matches': length_matches,
        })
        diagnostics.to_csv(args.out_dir / 'r1_raw_timestamp_diagnostics.csv', index=False)
        n_raw_float_mismatch = int((~exact_last_matches).sum())
        n_pipeline_mismatch = int((~pipeline_last_matches).sum())
        n_length_mismatch = int((~length_matches).sum())
        print(f'Raw floating-point time-step mismatches: {n_raw_float_mismatch}')
        print(f'Mismatches after matching loader int64 convention: {n_pipeline_mismatch}')
        print(f'Readout-count mismatches: {n_length_mismatch}')
        if n_pipeline_mismatch or n_length_mismatch:
            bad = diagnostics.loc[~pipeline_last_matches | ~length_matches].head(10)
            print('Example remaining mismatches:\n' + bad.to_string(index=False))
            raise ValueError(
                f'Raw source audit FAILED: {n_pipeline_mismatch} time-step '
                f'mismatches after int64 conversion; {n_length_mismatch} readout-count mismatches. '
                'See ' + str(args.out_dir / 'r1_raw_timestamp_diagnostics.csv') + '.'
            )
        summary['raw_operational_csv_checked'] = True
        summary['raw_timestamp_mismatches_before_int64_normalization'] = n_raw_float_mismatch
        summary['raw_timestamp_mismatches_after_int64_normalization'] = n_pipeline_mismatch
        summary['raw_readout_count_mismatches'] = n_length_mismatch
        summary['time_step_export_convention'] = 'int64(raw time_step), matching load_preprocessed_sequences()'
        if n_raw_float_mismatch:
            summary['status'] = 'PASS_RAW_SOURCE_VALIDATED_WITH_TIME_TRUNCATION'
            summary['limitations'] = (
                'Final available observations and readout counts match the raw source. '
                'Exported time-step fields are int64-truncated representations of raw time values, '
                'not exact original fractional time-step values.'
            )
        else:
            summary['status'] = 'PASS_RAW_SOURCE_VALIDATED'
            summary['limitations'] = (
                'Last-readout and sequence-length metadata independently verified against raw validation operational file.'
            )

    args.out_dir.mkdir(parents=True,exist_ok=True)
    joined[['vehicle_id','binary_label_ctrl','prefix_time_step','prefix_length','final_time_step_raw','final_time_step_cal']].rename(columns={'binary_label_ctrl':'binary_label'}).to_csv(args.out_dir/'r1_validation_scoring_vehicle_audit.csv',index=False)
    (args.out_dir/'r1_validation_scoring_summary.json').write_text(json.dumps(summary,indent=2)+'\n',encoding='utf-8')
    print(json.dumps(summary,indent=2))

if __name__=='__main__':
    main()
