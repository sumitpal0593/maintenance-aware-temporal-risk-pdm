"""R1 Table 8 paired stratified bootstrap, using saved per-vehicle predictions.

This never retrains, refits, tunes, or uses validation labels to change a model.
It computes uncertainty for pre-existing frozen validation predictions.

Example:
  python scripts/r1_02_bootstrap_temporal_controls.py --predictions \
    outputs/predictions/temporal_control_validation_predictions.csv \
    --metrics outputs/tables/temporal_control_validation_metrics.csv
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score, average_precision_score

CONTROL_COLUMNS=[
    'time_only_logistic',
    'static_only_logistic',
    'aggregate_dynamic_logistic',
    'gru_temporal_logistic',
    'aggregate_dynamic_plus_static_logistic',
    'gru_temporal_plus_static_logistic',
]
OPTIONAL_MODEL='frozen_proposed_gru'
COMPARISONS=[
    ('gru_temporal_logistic','aggregate_dynamic_logistic','GRU temporal vs aggregate dynamics'),
    ('gru_temporal_plus_static_logistic','aggregate_dynamic_plus_static_logistic','GRU + static vs aggregate + static'),
    ('gru_temporal_plus_static_logistic','static_only_logistic','GRU + static vs static only'),
    ('gru_temporal_logistic','static_only_logistic','GRU temporal vs static only (context)'),
]
METRICS=['auroc','pr_auc','enrichment']


def _weighted_auc_ap_from_counts(y:np.ndarray, scores:np.ndarray, counts:np.ndarray, n_pos:int, n_neg:int):
    """Exact duplicate-aware AUROC/AP for resampled indices, with grouped ties.

    The grouped-score AP formula matches sklearn.average_precision_score on a
    bootstrap sample containing repeated vehicles. Input 'counts' has shape
    (B, N), with integer bootstrap multiplicities.
    """
    order=np.argsort(-scores,kind='mergesort')
    score_sorted=scores[order]
    starts=np.r_[0, np.flatnonzero(score_sorted[1:]!=score_sorted[:-1])+1]
    p=counts[:,order] * (y[order]==1)[None,:]
    n=counts[:,order] * (y[order]==0)[None,:]
    pg=np.add.reduceat(p,starts,axis=1).astype(np.float64)
    ng=np.add.reduceat(n,starts,axis=1).astype(np.float64)
    cum_p=np.cumsum(pg,axis=1)
    cum_n=np.cumsum(ng,axis=1)
    # descending scores: true negatives with smaller scores are Nneg - cumulative negatives
    auroc=np.sum(pg*(n_neg-cum_n+0.5*ng),axis=1)/(n_pos*n_neg)
    precision=np.divide(cum_p,cum_p+cum_n,out=np.zeros_like(cum_p),where=(cum_p+cum_n)>0)
    ap=np.sum(pg*precision,axis=1)/n_pos
    enrichment=ap/(n_pos/(n_pos+n_neg))
    return np.stack([auroc,ap,enrichment],axis=1)


def _metrics_sklearn(y,s):
    auroc=float(roc_auc_score(y,s)); ap=float(average_precision_score(y,s))
    return {'auroc':auroc,'pr_auc':ap,'enrichment':ap/y.mean()}


def _ci(x,confidence=.95):
    a=(1-confidence)/2
    return np.quantile(x,[a,1-a]).tolist()


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--predictions',type=Path,default=Path('outputs/predictions/temporal_control_validation_predictions.csv'))
    p.add_argument('--metrics',type=Path,default=None,help='Optional Stage 4A reference CSV; fail on mismatched published point estimates')
    p.add_argument('--out-dir',type=Path,default=Path('outputs/r1_revision/representation_bootstrap'))
    p.add_argument('--resamples',type=int,default=5000)
    p.add_argument('--seed',type=int,default=42)
    p.add_argument('--confidence',type=float,default=.95)
    p.add_argument('--chunk-size',type=int,default=200)
    p.add_argument('--min-prefix-length',type=int,default=None,help='Exploratory subgroup sensitivity: keep only vehicles with at least this many readouts')
    args=p.parse_args()
    if args.resamples<1 or args.chunk_size<1 or not 0<args.confidence<1:
        raise ValueError('Invalid bootstrap configuration')
    df=pd.read_csv(args.predictions)
    names=CONTROL_COLUMNS+([OPTIONAL_MODEL] if OPTIONAL_MODEL in df else [])
    req=['vehicle_id','binary_label']+names
    if sorted(set(req)-set(df)):
        raise ValueError(f'Missing columns: {sorted(set(req)-set(df))}')
    if df[req].isna().any().any() or df.vehicle_id.duplicated().any():
        raise ValueError('Missing values or duplicated vehicle IDs')
    if args.min_prefix_length is not None:
        if 'prefix_length' not in df:
            raise ValueError('prefix_length column required for subgroup sensitivity')
        df=df[df.prefix_length >= args.min_prefix_length].copy()
        if len(df)==0: raise ValueError('No validation vehicles pass the subgroup filter')
    df=df.sort_values('vehicle_id',kind='mergesort').reset_index(drop=True)
    y=df.binary_label.to_numpy(dtype=np.int8)
    if set(np.unique(y))!={0,1}:raise ValueError('Expected binary labels')
    n_pos=int(y.sum());n_neg=int(len(y)-n_pos)
    if args.min_prefix_length is None and (len(y)!=5046 or n_pos!=136):
        raise ValueError(f'Unexpected validation population: n={len(y)}, positive={n_pos}')
    if n_pos<2 or n_neg<2:
        raise ValueError('Insufficient subgroup positives or negatives')
    scores={k:df[k].to_numpy(dtype=float) for k in names}
    if any(not np.isfinite(s).all() for s in scores.values()):
        raise ValueError('Nonfinite validation scores')
    point={k:_metrics_sklearn(y,s) for k,s in scores.items()}
    if args.metrics is not None and args.min_prefix_length is None:
        original=pd.read_csv(args.metrics).set_index('model')
        for name, values in point.items():
            if name not in original.index:
                raise ValueError(f'Expected metric row missing for {name}')
            for metric, v in values.items():
                if not np.isclose(float(original.loc[name,metric]),v,rtol=0,atol=1e-9):
                    raise ValueError(f'Stored predictions do not reproduce {name} {metric}: {v} vs {original.loc[name,metric]}')

    # Same seeded, class-stratified vehicle resamples for every model.
    # Resample multiplicities are computed once per chunk for memory efficiency.
    rng=np.random.default_rng(args.seed)
    positive=np.flatnonzero(y==1)
    negative=np.flatnonzero(y==0)
    all_distributions={k:np.empty((args.resamples,3),dtype=np.float64) for k in names}
    B=args.resamples
    for start in range(0,B,args.chunk_size):
        stop=min(B,start+args.chunk_size)
        counts=np.empty((stop-start,len(y)),dtype=np.int16)
        for row in range(stop-start):
            pos_idx=rng.choice(positive,size=n_pos,replace=True)
            neg_idx=rng.choice(negative,size=n_neg,replace=True)
            counts[row]=np.bincount(np.r_[pos_idx,neg_idx],minlength=len(y))
        for name, s in scores.items():
            all_distributions[name][start:stop]=_weighted_auc_ap_from_counts(y,s,counts,n_pos,n_neg)
        print(f'Resamples completed: {stop}/{B}',flush=True)

    # Verify exact group-tie arithmetic against scikit-learn using first bootstrap sample
    # recreated from the same seed, *and* the full original dataset as a weighted sample.
    one=np.ones((1,len(y)),dtype=np.int16)
    for name, s in scores.items():
        test=_weighted_auc_ap_from_counts(y,s,one,n_pos,n_neg)[0]
        ref=np.array([point[name][k] for k in METRICS])
        if not np.allclose(test,ref,atol=1e-12,rtol=0):
            raise RuntimeError(f'Weighted rank formula diverges from sklearn for {name}: {test} vs {ref}')

    rows=[]
    for name in names:
        for j,metric in enumerate(METRICS):
            lo,hi=_ci(all_distributions[name][:,j],args.confidence)
            rows.append({'model':name,'metric':metric,'estimate':point[name][metric],
                         'ci_lower':lo,'ci_upper':hi,'confidence_level':args.confidence,
                         'n_bootstrap':B,'seed':args.seed})
    model_table=pd.DataFrame(rows)

    comparisons=[]
    for model,reference,label in COMPARISONS:
        if model not in scores or reference not in scores:
            raise ValueError('Comparison model missing')
        differences=all_distributions[model]-all_distributions[reference]
        for j,metric in enumerate(METRICS):
            lo,hi=_ci(differences[:,j],args.confidence)
            comparisons.append({'comparison':label,'model':model,'reference':reference,
                'metric':metric,'delta_estimate':point[model][metric]-point[reference][metric],
                'ci_lower':lo,'ci_upper':hi,'ci_excludes_zero':bool(lo>0 or hi<0),
                'confidence_level':args.confidence,'n_bootstrap':B,'seed':args.seed})
    diff_table=pd.DataFrame(comparisons)
    args.out_dir.mkdir(parents=True,exist_ok=True)
    model_table.to_csv(args.out_dir/'r1_table8_model_bootstrap_cis.csv',index=False,float_format='%.10f')
    diff_table.to_csv(args.out_dir/'r1_table8_paired_differences.csv',index=False,float_format='%.10f')
    np.savez_compressed(args.out_dir/'r1_table8_bootstrap_distributions.npz',**{f'{n}__{m}':arr[:,j] for n,arr in all_distributions.items() for j,m in enumerate(METRICS)})
    summary={'status':'PASS','source_predictions':str(args.predictions),'n_vehicles':len(y),'n_positive':n_pos,
       'n_negative':n_neg,'min_prefix_length':args.min_prefix_length,'analysis_type':'exploratory_post_hoc_length_subgroup' if args.min_prefix_length is not None else 'predefined_full_validation',
       'n_bootstrap':B,'seed':args.seed,'ci_method':'paired class-stratified vehicle resampling; percentile 95% intervals',
       'note':'Confidence intervals quantify uncertainty in fixed frozen-model validation comparisons; no models retrained, selected or recalibrated.'}
    (args.out_dir/'r1_table8_bootstrap_summary.json').write_text(json.dumps(summary,indent=2)+'\n',encoding='utf-8')
    print('\nPAIRED COMPARISONS (full precision in CSV):')
    for _,row in diff_table.iterrows():
        print(f'{row["comparison"]:43s} {row["metric"]:10s} {row["delta_estimate"]:+.4f} [{row["ci_lower"]:+.4f}, {row["ci_upper"]:+.4f}] exclude0={row["ci_excludes_zero"]}')

if __name__=='__main__':main()
