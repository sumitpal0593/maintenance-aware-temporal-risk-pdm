#!/usr/bin/env python3
"""R1-03: retrospective probability and cost-decision sensitivity analysis.

Standalone; uses existing Stage 3.1 IPCW-Platt validation prediction exports.
This script NEVER refits the model and NEVER overwrites Stage 3.1 results.

IMPORTANT: The target prevalence is computed from the public validation labels.
All prevalence-based adjustments are RETROSPECTIVE sensitivities, not a valid
out-of-sample recalibration or deployable threshold policy.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import yaml
from sklearn.metrics import average_precision_score, roc_auc_score

DEFAULT_WEIGHTED_PREVALENCE = 0.06361428129123482
DEFAULT_UNWEIGHTED_PREVALENCE = 0.08112493239588967
METHODS = (
    'ipcw_platt_original',
    'uniform_20pct_risk_reduction',
    'odds_shift_weighted_prevalence',
    'odds_shift_unweighted_prevalence',
    'proportional_mean_matching_stress',
)


def odds_shift(p: np.ndarray, pi_reference: float, pi_target: float) -> np.ndarray:
    """Positive monotone prior-odds shift, assuming invariant likelihood ratios.

    Using prevalence drawn from the held-out validation labels to choose
    pi_target makes this an in-sample, retrospective sensitivity only.
    """
    if not 0 < pi_reference < 1 or not 0 < pi_target < 1:
        raise ValueError('Reference and target prevalence must lie strictly between 0 and 1.')
    p = np.asarray(p, dtype=np.float64)
    if not np.isfinite(p).all() or np.any((p < 0) | (p > 1)):
        raise ValueError('Probabilities must be finite and within [0,1].')
    odds_multiplier = (pi_target / (1.0 - pi_target)) / (pi_reference / (1.0 - pi_reference))
    return odds_multiplier * p / (1.0 - p + odds_multiplier * p)


def load_predictions(path: Path) -> pd.DataFrame:
    if not path.is_file():
        raise FileNotFoundError(f'Prediction input not found: {path}')
    frame = pd.read_csv(path)
    required = {'vehicle_id', 'binary_label', 'ipcw_platt_probability'}
    missing = required.difference(frame.columns)
    if missing:
        raise ValueError(f'Prediction file is missing columns: {sorted(missing)}')
    if len(frame) == 0 or frame['vehicle_id'].isna().any():
        raise ValueError('Vehicle IDs must be nonmissing.')
    if frame['vehicle_id'].duplicated().any():
        raise ValueError('Expected exactly one prediction per vehicle.')
    if frame[list(required)].isna().any().any():
        raise ValueError('Input has missing labels or probability values.')
    y = pd.to_numeric(frame['binary_label'], errors='raise').to_numpy()
    p = pd.to_numeric(frame['ipcw_platt_probability'], errors='raise').to_numpy(dtype=float)
    if not set(np.unique(y)).issubset({0, 1}) or len(np.unique(y)) != 2:
        raise ValueError('Binary validation labels must contain both classes 0 and 1.')
    if not np.isfinite(p).all() or np.any((p <= 0) | (p >= 1)):
        raise ValueError('Expected finite strict probabilities in (0,1).')
    return frame.sort_values('vehicle_id').reset_index(drop=True)


def load_cost_scenarios(path: Path) -> dict[str, dict[str, float]]:
    if not path.is_file():
        raise FileNotFoundError(f'Cost-scenario configuration not found: {path}')
    data = yaml.safe_load(path.read_text(encoding='utf-8'))
    if not isinstance(data, dict) or not isinstance(data.get('scenarios'), dict):
        raise ValueError('Expected YAML mapping under key scenarios.')
    out = {}
    for name, raw in data['scenarios'].items():
        ca, cf, rho = (float(raw[k]) for k in ('C_A', 'C_F', 'rho'))
        if ca <= 0 or cf <= 0 or not 0 <= rho < 1:
            raise ValueError(f'Invalid cost parameters in {name}: {raw}')
        out[str(name)] = {'C_A': ca, 'C_F': cf, 'rho': rho}
    if not out:
        raise ValueError('No cost scenarios present.')
    return out


def calibration_diagnostics(y: np.ndarray, p: np.ndarray) -> dict[str, float]:
    y = np.asarray(y, dtype=int)
    p = np.asarray(p, dtype=float)
    # Equal-width ECE reproduces the Stage 3.1 implementation.
    bins = np.digitize(p, np.linspace(0.0, 1.0, 11)[1:-1], right=False)
    ece = 0.0
    for idx in range(10):
        mask = bins == idx
        if np.any(mask):
            ece += (mask.mean()) * abs(p[mask].mean() - y[mask].mean())
    safe = np.clip(p, 1e-12, 1 - 1e-12)
    return {
        'mean_probability': float(p.mean()),
        'p25': float(np.quantile(p, 0.25)),
        'p50': float(np.quantile(p, 0.5)),
        'p90': float(np.quantile(p, 0.9)),
        'max_probability': float(p.max()),
        'brier_score_diagnostic_only': float(np.mean((p - y) ** 2)),
        'log_loss_diagnostic_only': float(np.mean(-y * np.log(safe) - (1 - y) * np.log1p(-safe))),
        'ece_10_equal_width_diagnostic_only': float(ece),
        'auroc_unchanged_by_monotone_adjustment': float(roc_auc_score(y, p)),
        'pr_auc_unchanged_by_monotone_adjustment': float(average_precision_score(y, p)),
    }


def benefit_and_action(p: np.ndarray, ca: float, cf: float, rho: float) -> tuple[np.ndarray, np.ndarray]:
    p = np.asarray(p, dtype=np.float64)
    benefit = (1 - rho) * cf * p - ca
    return benefit, benefit > 0.0


def get_calibration_prevalences(summary_path: Path | None) -> tuple[float, float, str]:
    if summary_path is not None and summary_path.is_file():
        obj = json.loads(summary_path.read_text(encoding='utf-8'))
        d = obj['ipcw_diagnostics']
        return float(d['ipcw_weighted_positive_prevalence']), float(d['unweighted_positive_prevalence']), summary_path.name
    return DEFAULT_WEIGHTED_PREVALENCE, DEFAULT_UNWEIGHTED_PREVALENCE, 'Stage 3.1 reported values (built-in fallbacks)'


def run_analysis(input_path: Path, scenario_path: Path, outdir: Path, summary_path: Path | None = None) -> dict[str, Any]:
    frame = load_predictions(input_path)
    scenarios = load_cost_scenarios(scenario_path)
    y = frame['binary_label'].to_numpy(dtype=int)
    original = frame['ipcw_platt_probability'].to_numpy(dtype=float)
    pi_val = float(y.mean())
    pi_weighted, pi_unweighted, pi_source = get_calibration_prevalences(summary_path)
    if not (0 < pi_weighted < 1 and 0 < pi_unweighted < 1):
        raise ValueError('Invalid calibration prevalences.')
    factor = float(original.mean() / pi_val)
    probs = {
        'ipcw_platt_original': original,
        'uniform_20pct_risk_reduction': original * 0.8,
        'odds_shift_weighted_prevalence': odds_shift(original, pi_weighted, pi_val),
        'odds_shift_unweighted_prevalence': odds_shift(original, pi_unweighted, pi_val),
        'proportional_mean_matching_stress': original / factor,
    }
    assert list(probs) == list(METHODS)
    for name, p in probs.items():
        if not np.isfinite(p).all() or np.any((p <= 0) | (p >= 1)):
            raise AssertionError(f'Invalid adjusted risks for {name}')
        if name != 'ipcw_platt_original' and np.any(np.diff(p[np.argsort(original, kind='stable')]) < -1e-12):
            raise AssertionError(f'Monotonicity failure in {name}')

    outdir.mkdir(parents=True, exist_ok=True)
    data = frame[['vehicle_id', 'binary_label']].copy()
    if 'final_time_step' in frame.columns:
        data['final_time_step'] = frame['final_time_step'].to_numpy()
    for method, arr in probs.items():
        data[f'risk__{method}'] = arr

    metrics_rows = []
    policy_rows = []
    quantile_rows = []
    decision_rows = []
    for method, p in probs.items():
        metrics_rows.append({
            'method': method,
            'validation_n': int(len(y)),
            'validation_positives': int(y.sum()),
            'validation_prevalence': pi_val,
            'mean_risk_to_prevalence_ratio': float(p.mean() / pi_val),
            **calibration_diagnostics(y, p),
        })
        for scenario_name, c in scenarios.items():
            threshold = c['C_A'] / ((1 - c['rho']) * c['C_F'])
            b, act = benefit_and_action(p, **{'ca': c['C_A'], 'cf': c['C_F'], 'rho': c['rho']})
            original_b, original_act = benefit_and_action(original, c['C_A'], c['C_F'], c['rho'])
            key = f'{scenario_name}__{method}'
            data[f'action__{key}'] = act.astype(int)
            data[f'benefit__{key}'] = b
            policy_rows.append({
                'method': method,
                'scenario': scenario_name,
                'C_A': c['C_A'], 'C_F': c['C_F'], 'rho': c['rho'],
                'threshold': threshold,
                'n_maintain': int(act.sum()),
                'pct_maintain': float(100 * act.mean()),
                'n_original_maintain': int(original_act.sum()),
                'n_maintain_to_no_maintain': int(np.sum(original_act & ~act)),
                'n_no_maintain_to_maintain': int(np.sum(~original_act & act)),
                'pct_action_changed': float(100 * np.mean(act != original_act)),
                'n_observed_positives_among_selected_diagnostic_only': int(y[act].sum()),
                'sum_model_based_net_benefit_of_selected': float(np.clip(b, 0, None).sum()),
                'mean_model_based_net_benefit_per_vehicle': float(np.clip(b, 0, None).mean()),
                'mean_model_based_net_benefit_among_selected': float(b[act].mean()) if np.any(act) else 0.0,
            })
            decision_rows.append({
                'method': method, 'scenario': scenario_name,
                'n_changed_relative_to_original': int(np.sum(act != original_act)),
                'n_abandoned_interventions': int(np.sum(original_act & ~act)),
                'n_new_interventions': int(np.sum(~original_act & act)),
            })
            for label, q in [('P25', 0.25), ('P50', 0.5), ('P90', 0.9)]:
                risk = float(np.quantile(p, q))
                val = (1 - c['rho']) * c['C_F'] * risk - c['C_A']
                quantile_rows.append({
                    'method': method, 'scenario': scenario_name,
                    'quantile': label, 'risk': risk,
                    'threshold': threshold,
                    'expected_net_benefit': val,
                    'intervene': bool(val > 0),
                })

    outputs = {
        'r1_03_probability_sensitivity.csv': pd.DataFrame(metrics_rows),
        'r1_03_scenario_policy_sensitivity.csv': pd.DataFrame(policy_rows),
        'r1_03_quantile_net_benefit.csv': pd.DataFrame(quantile_rows),
        'r1_03_decision_changes.csv': pd.DataFrame(decision_rows),
        'r1_03_vehicle_risks_actions.csv': data,
    }
    for filename, table in outputs.items():
        table.to_csv(outdir / filename, index=False, float_format='%.15g')

    note = (
        'RETROSPECTIVE/VALIDATION-INFORMED SENSITIVITY ONLY. Target prevalence is derived '
        'from the same validation labels used for diagnostics; adjusted calibration scores '
        'are NOT a clean held-out calibration assessment and must NOT replace the '
        'original IPCW-Platt scores. Odds adjustment assumes a label-prior-shift model '
        'which has not been validated for this dataset. All net benefits are conditional '
        'model-based expectations under illustrative cost and intervention-effect assumptions, '
        'not observed causal treatment effects or actual monetary savings.'
    )
    summary = {
        'status': 'PASS_RETROSPECTIVE_SENSITIVITY_ONLY',
        'n_validation': int(len(y)),
        'n_positive': int(y.sum()), 'n_negative': int((1-y).sum()),
        'validation_prevalence': pi_val,
        'calibration_prevalence_weighted': pi_weighted,
        'calibration_prevalence_unweighted': pi_unweighted,
        'calibration_prevalence_source': pi_source,
        'ipcw_mean_predicted_probability': float(original.mean()),
        'ipcw_mean_to_validation_prevalence_ratio': factor,
        'method_names': list(METHODS),
        'scenario_names': list(scenarios.keys()),
        'method_descriptions': {
            'ipcw_platt_original': 'Frozen Stage 3.1 IPCW-Platt probability; no changes',
            'uniform_20pct_risk_reduction': 'Label-independent illustrative 20% downscale of all risks',
            'odds_shift_weighted_prevalence': 'Retrospective odds shift using IPCW-weighted calibration prevalence and validation prevalence',
            'odds_shift_unweighted_prevalence': 'Retrospective odds shift using raw labeled calibration prevalence and validation prevalence',
            'proportional_mean_matching_stress': 'Retrospective proportional scaling to force adjusted mean probability equal to observed validation prevalence',
        },
        'interpretation_warning': note,
        'outputs': list(outputs.keys()) + ['r1_03_sensitivity_summary.json','r1_03_high_consequence_p90_benefit.png'],
    }
    (outdir / 'r1_03_sensitivity_summary.json').write_text(json.dumps(summary, indent=2), encoding='utf-8')
    create_figure(pd.DataFrame(quantile_rows), outdir)
    return summary


def create_figure(quantile_table: pd.DataFrame, outdir: Path) -> None:
    """Publication-style diagnostic plot: no grid, white background."""
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    selected = quantile_table[(quantile_table['scenario'] == 'high_consequence') & (quantile_table['quantile'] == 'P90')].copy()
    labels = {
        'ipcw_platt_original': 'Original IPCW-Platt',
        'uniform_20pct_risk_reduction': '20% lower risk',
        'odds_shift_weighted_prevalence': 'Weighted-prevalence odds shift',
        'odds_shift_unweighted_prevalence': 'Unweighted-prevalence odds shift',
        'proportional_mean_matching_stress': 'Mean-matching stress',
    }
    selected = selected.set_index('method').loc[list(METHODS)].reset_index()
    values = selected['expected_net_benefit'].to_numpy()
    fig, ax = plt.subplots(figsize=(9.3, 4.7), dpi=180)
    fig.patch.set_facecolor('white')
    ax.set_facecolor('white')
    x = np.arange(len(selected))
    ax.bar(x, values, width=0.62, edgecolor='black', linewidth=0.65, color=['#2f6c8d' if val>=0 else '#b87365' for val in values])
    ax.axhline(0, color='black', linewidth=0.9)
    ax.set_xticks(x, [labels[m] for m in selected['method']], rotation=18, ha='right')
    ax.set_ylabel('Expected net benefit at P90 (normalized cost units)')
    ax.set_title('High-consequence scenario: sensitivity to risk adjustment', loc='left', fontsize=11)
    ax.grid(False)
    ax.spines[['top','right']].set_visible(False)
    for xi, yi in zip(x, values):
        ax.text(xi, yi + (2 if yi >=0 else -3), f'{yi:+.2f}', ha='center', va='bottom' if yi >=0 else 'top', fontsize=9)
    ax.set_ylim(min(-115,float(min(values))-12),max(42,float(max(values))+12))
    fig.tight_layout()
    fig.savefig(outdir/'r1_03_high_consequence_p90_benefit.png', bbox_inches='tight', facecolor='white')
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--predictions', type=Path, default=Path('outputs/predictions/validation_calibrated_predictions_stage3_1.csv'))
    parser.add_argument('--cost-config', type=Path, default=Path('configs/cost_scenarios.yaml'))
    parser.add_argument('--ipcw-summary', type=Path, default=Path('outputs/logs/stage3_1_ipcw_summary.json'))
    parser.add_argument('--output-dir', type=Path, default=Path('outputs/r1_revision/calibration_sensitivity'))
    args = parser.parse_args()
    result = run_analysis(args.predictions, args.cost_config, args.output_dir, args.ipcw_summary)
    print(json.dumps(result, indent=2))
    print('\nSummary written to:', args.output_dir/'r1_03_sensitivity_summary.json')


if __name__ == '__main__':
    main()
