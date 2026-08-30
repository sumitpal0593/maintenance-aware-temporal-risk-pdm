# Maintenance-Aware Horizon-Based Temporal Risk Modeling for Decision-Oriented Predictive Maintenance

## Overview

This repository contains the reproducible research implementation associated with the manuscript:

> **Maintenance-Aware Horizon-Based Temporal Risk Modeling for Decision-Oriented Predictive Maintenance**

The study develops a decision-oriented predictive-maintenance framework for rare near-horizon failure events using the **Scania Component X** dataset.

The complete pipeline integrates:

- censoring-aware fixed-horizon target construction;
- leakage-controlled vehicle-level data partitioning;
- multivariate temporal representation learning using a GRU;
- static vehicle-specification encoding;
- class-imbalance-aware model training;
- temporal representation-control experiments;
- standardized classical and deep survival baselines;
- censoring-aware IPCW-Platt probability calibration;
- stratified paired-bootstrap uncertainty analysis; and
- scenario-based cost-aware maintenance decision analysis.

The primary prediction target is the probability of a failure-related event occurring within

```text
tau = 48
```

**relative dataset time steps**.

The Scania Component X temporal index is anonymized and does not correspond to a disclosed universal physical unit. Therefore, `tau = 48` should **not** be interpreted as 48 hours, days, kilometres, cycles, or another physical time unit.

---

## Repository status

> **Status: Complete journal-revision reproducibility pipeline**

The repository contains the computational workflow used for the revised manuscript, including:

1. dataset auditing;
2. vehicle-level model-development partitioning;
3. censoring-aware fixed-horizon label construction;
4. operational-variable preprocessing;
5. static categorical encoding;
6. proposed GRU temporal-risk model training;
7. tuning-based model selection;
8. temporal representation-control experiments;
9. CoxPH, Random Survival Forest, Deep CoxPH, and Deep Survival Machine baselines;
10. IPCW-Platt probability calibration;
11. model-specific bootstrap confidence intervals;
12. paired bootstrap comparisons;
13. cost-aware maintenance scenario analysis; and
14. generation of manuscript-facing experimental outputs.

The public validation cohort is reserved for final evaluation and is not used for preprocessing estimation, model fitting, hyperparameter selection, or calibration.

---

# 1. Reproducibility design

The repository was structured around the following principles.

### No machine-specific data paths

Dataset paths are supplied through environment/configuration files rather than hard-coded local paths.

### Vehicle-level partitioning

All landmarks generated from a vehicle remain in the same development partition. This prevents information from the same vehicle appearing in both model-training and model-selection/calibration data.

### Training-only preprocessing

Imputation, standardization, and categorical encoding are fitted exclusively using the model-training partition.

The fitted transformations are then applied unchanged to:

- tuning data;
- calibration data; and
- public validation data.

### Dedicated tuning and calibration partitions

The development data are separated into:

- **70% model-training vehicles**;
- **15% tuning vehicles**; and
- **15% calibration vehicles**.

The tuning partition is used for model selection and early stopping.

The calibration partition is used for post-training probability calibration.

### No public-validation leakage

The public validation cohort is not used for:

- preprocessing estimation;
- model fitting;
- early stopping;
- hyperparameter selection;
- calibration fitting; or
- baseline configuration selection.

### Executable evaluation

Reported performance metrics are generated from saved model predictions rather than being entered manually into plotting scripts.

### Reproducible uncertainty analysis

Model-specific and paired confidence intervals are obtained using a fixed, class-stratified nonparametric bootstrap procedure.

---

# 2. Dataset

The experiments use the **Scania Component X dataset**, introduced by Kharazian et al. (2025).

The dataset contains:

- multivariate operational readouts;
- vehicle-level static specifications;
- time-to-event/event-status information for model development; and
- failure-proximity labels for public validation.

The operational variables and temporal scale are anonymized.

## Key dimensions used in this study

| Quantity | Value |
|---|---:|
| Dynamic operational variables | 105 |
| Original static categorical variables | 8 |
| One-hot encoded static dimensions | 89 |
| Public training vehicles | 23,550 |
| Public validation vehicles | 5,046 |
| Positive validation vehicles | 136 |
| Validation positive prevalence | 2.6952% |
| Prediction horizon | 48 relative dataset time steps |

The raw Scania Component X files are **not redistributed in this repository**.

Users should obtain the dataset from its official public source and configure its local location as described below.

See:

```text
data/README.md
```

for the expected local dataset structure and filenames.

---

# 3. Repository structure

A typical repository layout is:

```text
maintenance-aware-pdm/
│
├── README.md
├── requirements.txt
├── pyproject.toml
├── .gitignore
├── .env.example
│
├── configs/
│   ├── paper_experiment.yaml
│   └── cost_scenarios.yaml
│
├── data/
│   └── README.md
│
├── src/
│   └── maestro_pdm/
│       ├── config.py
│       ├── reproducibility.py
│       │
│       ├── data/
│       │   ├── loading.py
│       │   ├── preprocessing.py
│       │   ├── labels.py
│       │   └── datasets.py
│       │
│       ├── models/
│       ├── training/
│       ├── evaluation/
│       ├── decision/
│       └── visualization/
│
├── scripts/
│   ├── 00_check_setup.py
│   ├── 01_audit_dataset.py
│   ├── 02_train_proposed.py
│   ├── 03_train_baselines.py
│   ├── 04_run_temporal_controls.py
│   ├── 04c_run_paired_bootstrap.py
│   ├── 05_calibrate_predictions.py
│   ├── 06_bootstrap_evaluation.py
│   ├── 07_cost_analysis.py
│   ├── 08_generate_figures.py
│   └── run_all.py
│
├── notebooks/
├── tests/
│
└── outputs/
    ├── tables/
    ├── figures/
    ├── checkpoints/
    ├── bootstrap/
    └── logs/
```

Generated artifacts and large intermediate files may be excluded from version control where appropriate.

---

# 4. Environment setup

Python should be run inside a dedicated virtual environment.

## Windows PowerShell

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1

python -m pip install --upgrade pip
pip install -r requirements.txt
pip install -e .
```

## Windows Command Prompt

```bat
python -m venv .venv
.venv\Scripts\activate

python -m pip install --upgrade pip
pip install -r requirements.txt
pip install -e .
```

## Linux/macOS

```bash
python3 -m venv .venv
source .venv/bin/activate

python -m pip install --upgrade pip
pip install -r requirements.txt
pip install -e .
```

---

# 5. Configure the dataset

Copy:

```text
.env.example
```

to:

```text
.env
```

and specify the local Scania Component X dataset directory.

For example:

```text
SCANIA_DATA_DIR=C:/your/local/path/to/scania_component_x/data
```

The `.env` file and raw dataset should **not** be committed to GitHub.

The repository `.gitignore` should exclude at least:

```text
.env
.venv/
__pycache__/
*.pyc
```

together with any confidential credentials or unnecessary large generated artifacts.

---

# 6. Verify the installation

Run:

```bash
python scripts/00_check_setup.py
```

A successful configuration should identify the fixed prediction horizon:

```text
tau = 48
```

and the public validation cohort should contain:

```text
5,046 vehicles
```

---

# 7. Fixed-horizon target construction

The primary prediction endpoint is a censoring-aware binary horizon label:

```text
Y_i(48)
```

For a prediction landmark at relative time `t`, let:

```text
Delta = T_observed - t
```

where `T_observed` is the recorded repair/event or censoring endpoint.

## Observed-event vehicles

For a vehicle with an observed event:

```text
Y_i(48) = 1
```

when:

```text
1 <= Delta <= 48
```

and:

```text
Y_i(48) = 0
```

when:

```text
Delta > 48
```

Thus, an event is positive only when it occurs after the prediction landmark and within the next 48 relative time steps.

## Right-censored vehicles

For a censored vehicle:

```text
Y_i(48) = 0
```

only when:

```text
Delta >= 48
```

because at least one complete event-free prediction horizon is observed.

When:

```text
0 < Delta < 48
```

the fixed-horizon outcome is unknown because follow-up ends before the full horizon is observed.

Such landmarks are excluded from supervised model training.

Landmarks satisfying:

```text
Delta <= 0
```

are also excluded because they occur at or beyond the recorded endpoint.

---

# 8. Landmark construction and development split

Landmarks are generated from observed vehicle-history prefixes.

Only information observed up to the corresponding prediction landmark is supplied to the model.

No future operational readouts are included.

The model-development procedure uses:

```text
maximum landmarks per vehicle = 3
minimum observed prefix length = 20
```

All landmarks from the same vehicle remain within one development partition.

The 23,550 public training vehicles are partitioned into:

```text
70% model training
15% tuning
15% calibration
```

using a fixed random seed of:

```text
42
```

---

# 9. Training-label audit

Run:

```bash
python scripts/01_audit_dataset.py --config configs/paper_experiment.yaml
```

The audit verifies:

- development-partition integrity;
- censoring-aware horizon labels;
- positive and negative landmark counts;
- class weights;
- validation label distribution;
- static categorical encoding; and
- training-only operational preprocessing.

For a quicker label-only diagnostic:

```bash
python scripts/01_audit_dataset.py \
    --config configs/paper_experiment.yaml \
    --skip-operational-preprocessor
```

The `--skip-operational-preprocessor` option is intended for debugging only and should not be used for the final experiment.

---

# 10. Model-training target statistics

Under the final censoring-aware horizon construction, the model-training partition contains:

```text
Total labelled landmarks : 23,583
Positive landmarks       : 1,896
Positive prevalence      : approximately 8.04%
```

The class weights used for weighted binary cross-entropy are:

```text
omega_0 = 0.543713
omega_1 = 6.219146
```

where:

```text
omega_0 = N / (2 N_0)
omega_1 = N / (2 N_1)
```

The model-training landmark prevalence should not be confused with either:

- the raw vehicle-level observed-event proportion in the training time-to-event file; or
- the 2.6952% positive prevalence in the complete public validation cohort.

The difference arises from landmark construction, horizon identifiability, censoring, and cohort composition.

---

# 11. Operational preprocessing

The 105 dynamic operational variables are preprocessed using statistics estimated exclusively from model-training vehicles.

For each operational variable:

1. the mean is estimated from the model-training partition;
2. missing values are imputed using that training-derived mean;
3. scale parameters are estimated from the model-training partition; and
4. the same fitted transformation is applied unchanged to tuning, calibration, and validation data.

The validation cohort is never used to estimate preprocessing parameters.

---

# 12. Static specification encoding

The dataset contains eight original categorical vehicle-specification variables.

These variables are transformed using one-hot encoding fitted exclusively on model-training vehicles.

Previously unseen categories are ignored during transformation.

For the final reported experiment:

```text
Original static variables : 8
Encoded static dimension  : 89
```

The fitted encoder is applied unchanged to the tuning, calibration, and validation partitions.

---

# 13. Proposed temporal-risk model

The proposed model combines a GRU-based temporal encoder with encoded static vehicle information.

## Architecture

| Component | Configuration |
|---|---|
| Dynamic input dimension | 105 |
| Static encoded dimension | 89 |
| GRU layers | 1 |
| GRU hidden dimension | 128 |
| Static branch hidden dimension | 128 |
| Fusion hidden dimension | 128 |
| Activation | ReLU |
| Dropout | 0.10 |
| Output | Fixed-horizon raw risk score |

The trained class-weighted model produces the raw score:

```text
r_hat_omega,i(48)
```

which is subsequently calibrated before probability-scale decision analysis.

---

# 14. Proposed-model training

The final training configuration is:

| Setting | Value |
|---|---:|
| Optimizer | AdamW |
| Learning rate | 3 × 10^-4 |
| Weight decay | 1 × 10^-4 |
| Batch size | 256 |
| Maximum epochs | 30 |
| Early-stopping patience | 5 |
| Gradient clipping | 2.0 |
| Random seed | 42 |
| Model-selection criterion | tuning PR-AUC |

Training uses class-weighted binary cross-entropy.

Regularization is applied through **decoupled AdamW weight decay**, rather than by explicitly adding an L2 penalty term to the empirical weighted binary cross-entropy objective.

The final selected checkpoint corresponds to:

```text
epoch 12
```

under the reported experiment.

---

# 15. Temporal representation-control experiments

A dedicated representation-control experiment isolates the effect of temporal encoding from the choice of downstream prediction architecture.

The controls compare:

1. time information only;
2. static specifications only;
3. aggregated dynamic summaries;
4. learned GRU temporal representation;
5. aggregated dynamics + static specifications; and
6. GRU temporal representation + static specifications.

The downstream classifier is held fixed as L2-regularized logistic regression.

For each operational variable, the aggregate-dynamic representation uses six summary statistics:

- mean;
- standard deviation;
- minimum;
- maximum;
- final observed value; and
- linear trend.

With 105 operational variables:

```text
105 × 6 = 630 aggregate-dynamic features
```

The aggregate-dynamic-plus-static representation therefore contains:

```text
630 + 89 = 719 features
```

The learned GRU temporal representation contains:

```text
128 features
```

and the GRU-plus-static control contains:

```text
128 + 89 = 217 features
```

## Selected logistic-control regularization values

| Representation | Selected C |
|---|---:|
| Time only | 1 |
| Static only | 1 |
| Aggregated dynamics | 1 |
| GRU temporal | 0.01 |
| Aggregated dynamics + static | 0.1 |
| GRU temporal + static | 10 |

All values are selected using the tuning partition only.

---

# 16. Standardized survival baselines

The complete proposed framework is compared against four survival-oriented reference models:

- Cox Proportional Hazards (CoxPH);
- Random Survival Forest (RSF);
- Deep CoxPH; and
- Deep Survival Machine (DSM).

All baselines begin from the same 719-dimensional aggregate-dynamic-plus-static representation.

For CoxPH, one training-derived zero/near-zero-variance feature is removed before fitting to avoid numerical degeneracy. The same feature mask is subsequently applied to tuning and validation data.

## CoxPH

Implementation:

```text
lifelines.CoxPHFitter
```

Candidate penalizers:

```text
{0.001, 0.01, 0.1, 1.0}
```

Selected configuration:

```text
penalizer = 0.1
l1_ratio  = 0
```

## Random Survival Forest

Implementation:

```text
sksurv.ensemble.RandomSurvivalForest
```

Selected configuration:

```text
trees              = 300
minimum split size = 20
minimum leaf size  = 10
max_features       = sqrt
random seed        = 42
```

## Deep CoxPH

Implementation:

```text
pycox.models.CoxPH
```

Selected configuration:

```text
hidden layers       = [128, 128]
batch normalization = enabled
dropout             = 0.10
optimizer           = AdamW
learning rate       = 3e-4
batch size          = 1024
maximum epochs      = 30
patience            = 5
```

## Deep Survival Machine

Implementation:

```text
auton_survival.models.dsm.DeepSurvivalMachines
```

Selected configuration:

```text
mixture components = 3
hidden layers      = [64, 32]
learning rate      = 1e-3
batch size         = 128
```

Baseline hyperparameters are selected using model-development data only.

---

# 17. Probability calibration

The maintenance decision layer requires probability-scale risk estimates rather than ranking scores alone.

The selected GRU checkpoint is therefore **frozen before calibration**.

No GRU parameter is re-estimated during calibration.

The final calibration procedure uses **inverse-probability-of-censoring-weighted Platt scaling (IPCW-Platt)** on the dedicated calibration partition.

The probability mapping is:

```text
r_hat_cal,i(tau) = sigmoid(a + b * logit_i)
```

where the fitted calibration parameters for the reported experiment are:

```text
intercept a = -2.605532
slope     b =  0.456284
```

## IPCW-Platt implementation

```text
inverse-regularization C = 1e6
solver                   = lbfgs
maximum iterations       = 2000
random seed              = 42
```

The censoring survival function is estimated from calibration landmarks using Kaplan-Meier estimation.

The normalized IPCW weights have mean one before the logistic calibration model is fitted.

Selected calibration diagnostics include:

```text
Calibration landmarks used for censoring model = 2,978
Censoring endpoints                           = 2,676
Observed event endpoints                      = 302
G_hat(48-)                                    = 0.609705
Effective sample size                         = 1,839.51
```

---

# 18. Validation cohort

The complete public validation cohort contains:

```text
Total vehicles    = 5,046
Positive vehicles = 136
Negative vehicles = 4,910
Positive rate     = 0.026952
```

All 5,046 validation vehicles are retained for final evaluation.

---

# 19. Evaluation metrics

Because the validation endpoint is highly imbalanced, evaluation emphasizes ranking-oriented metrics.

The primary metrics are:

### AUROC

Measures discrimination between positive and negative vehicles over all score thresholds.

### PR-AUC

Measures the concentration of positive cases among high-risk predictions and is especially informative under severe class imbalance.

### Enrichment

Defined as:

```text
Enrichment = PR-AUC / positive prevalence
```

For the validation cohort:

```text
random-ranking PR-AUC ≈ 0.026952
```

An enrichment greater than one indicates prioritization above the random-ranking reference.

---

# 20. Reference validation results

The following values correspond to the final manuscript experiment and are provided as reproducibility checks.

They must **not** be manually inserted into metric-generation or plotting code.

## Proposed model

| Metric | Estimate |
|---|---:|
| AUROC | 0.635555 |
| PR-AUC | 0.043084 |
| Enrichment | 1.598560 |

The associated 95% bootstrap intervals are:

| Metric | 95% CI |
|---|---|
| AUROC | [0.589076, 0.679347] |
| PR-AUC | [0.035931, 0.056556] |
| Enrichment | [1.333163, 2.098387] |

---

# 21. Representation-control results

| Input representation | Dimension | AUROC | PR-AUC | Enrichment |
|---|---:|---:|---:|---:|
| Time only | 2 | 0.475571 | 0.024302 | 0.901668 |
| Static specifications | 89 | 0.613402 | 0.039338 | 1.459542 |
| Aggregated dynamics | 630 | 0.510889 | 0.025849 | 0.959063 |
| GRU temporal representation | 128 | 0.637145 | 0.041915 | 1.555182 |
| Aggregated dynamics + static | 719 | 0.557593 | 0.032126 | 1.191951 |
| GRU representation + static | 217 | 0.609316 | 0.040071 | 1.486759 |

The direct learned-temporal-versus-aggregated-dynamic comparison changes:

```text
AUROC : 0.510889 -> 0.637145
PR-AUC: 0.025849 -> 0.041915
```

This experiment is intended to isolate temporal representation quality while holding the downstream classifier family fixed.

---

# 22. Standardized survival-baseline results

| Method | AUROC | PR-AUC | Enrichment |
|---|---:|---:|---:|
| Random Survival Forest | 0.592607 | 0.031792 | 1.179594 |
| Deep CoxPH | 0.617310 | 0.037833 | 1.403730 |
| Deep Survival Machine | 0.626559 | 0.048596 | 1.803068 |
| Cox Proportional Hazards | 0.640408 | 0.040306 | 1.495457 |
| Proposed temporal-risk model | 0.635555 | 0.043084 | 1.598560 |

These results should not be interpreted as showing universal superiority of the proposed model.

CoxPH has the largest point-estimate AUROC, while DSM has the largest point-estimate PR-AUC and enrichment.

Paired statistical uncertainty is evaluated separately.

---

# 23. Bootstrap uncertainty analysis

The manuscript uses:

```text
5,000 bootstrap resamples
95% percentile confidence intervals
random seed = 42
```

Bootstrap resampling is performed at the **vehicle level**, not over individual temporal readouts.

---

# 24. Paired bootstrap model comparisons

The paired-bootstrap procedure compares the proposed model with each standardized survival baseline.

Within each bootstrap replicate:

1. the 136 positive validation vehicles are sampled with replacement;
2. the 4,910 negative validation vehicles are sampled with replacement;
3. the same sampled vehicle indices are applied to predictions from every model;
4. AUROC, PR-AUC, and enrichment are recomputed; and
5. model differences are calculated as:

```text
Proposed - Reference
```

The same resample is therefore used for both models in each comparison.

This pairing preserves the correspondence between predictions made for the same validation vehicles.

## Run the Stage 4C paired-bootstrap analysis

```powershell
python scripts/04c_run_paired_bootstrap.py --config configs/paper_experiment.yaml
```

The analysis produces outputs including:

```text
outputs/tables/stage4c_bootstrap_model_metric_cis.csv
outputs/tables/stage4c_bootstrap_paired_differences_vs_proposed.csv
outputs/tables/stage4c_bootstrap_validation_audit.csv
outputs/bootstrap/stage4c_bootstrap_distributions.npz
outputs/logs/stage4c_bootstrap_summary.json
```

## Interpretation

A paired 95% interval excluding zero is interpreted as evidence that the corresponding model-performance difference is distinguishable from zero at the 95% bootstrap level.

It is **not automatically interpreted as a formal parametric hypothesis-test p-value**.

For the final experiment, the proposed model shows paired intervals excluding zero relative to RSF for AUROC, PR-AUC, and enrichment.

The paired intervals relative to CoxPH, Deep CoxPH, and DSM include zero.

---

# 25. Probability-calibration results

The final IPCW-Platt calibration achieves:

| Metric | IPCW-Platt |
|---|---:|
| Brier score | 0.033800 |
| Log loss | 0.163492 |
| Expected calibration error | 0.073794 |
| Mean predicted risk | 0.100746 |
| P25 | 0.055645 |
| P50 | 0.095738 |
| P90 | 0.180457 |
| Maximum | 0.353274 |

The calibrated probabilities should not be interpreted as perfectly calibrated.

Residual overprediction remains relative to the validation positive prevalence of approximately 0.02695.

---

# 26. Cost-aware maintenance decision analysis

The final calibrated probability is translated into a preventive-maintenance decision using an expected-cost rule.

For failure cost `C_F`, preventive-action cost `C_A`, and residual-risk fraction `rho_i`, preventive maintenance is selected when:

```text
r_hat_cal,i(tau) > C_A / ((1 - rho_i) * C_F)
```

The manuscript uses illustrative normalized cost scenarios rather than claiming verified industrial monetary savings.

The residual-risk factor used in the reported scenarios is:

```text
rho = 0.20
```

The scenario analysis uses empirical calibrated-risk quantiles:

```text
P25 = 0.055645
P50 = 0.095738
P90 = 0.180457
```

rather than arbitrary probability values outside the observed calibrated-score distribution.

---

# 27. Generated outputs

The pipeline writes reproducibility artifacts under:

```text
outputs/
```

Typical output groups include:

```text
outputs/tables/
outputs/figures/
outputs/checkpoints/
outputs/bootstrap/
outputs/logs/
```

Dataset-audit outputs include files such as:

```text
dataset_audit.csv
development_split.csv
development_split_summary.csv
training_label_audit.csv
horizon_prefix_samples.csv
validation_class_distribution.csv
static_encoding_summary.csv
static_encoding_categories.csv
operational_preprocessing_summary.csv
```

Serialized preprocessing objects may include:

```text
static_onehot_encoder.joblib
operational_preprocessor.joblib
```

Exact generated artifacts depend on the stage being executed.

---

# 28. Important reproducibility rule

**Do not manually enter manuscript performance values into experiment, metric, or plotting code.**

The manuscript tables and figures should be generated from saved model predictions and experiment outputs.

The numerical values listed in this README are provided only as reference checks for the final reported experiment.

---

# 29. Run tests

Run:

```bash
pytest
```

The test suite checks key reproducibility conditions, including:

- `tau = 48`;
- complete validation-cohort handling;
- vehicle-level development separation;
- positive event-within-horizon labeling;
- negative event-beyond-horizon labeling;
- exclusion of censored prefixes with insufficient follow-up;
- exclusion of prediction landmarks at or beyond the event/censoring boundary;
- weighted-BCE class-weight calculation;
- training-only preprocessing;
- and one-hot encoding behavior for unseen categories.

---

# 30. Reproducing the manuscript workflow

The complete manuscript experiment conceptually follows this order:

```text
Raw Scania Component X data
        |
        v
Dataset audit
        |
        v
Vehicle-level model-training / tuning / calibration split
        |
        v
Censoring-aware Y_i(48) construction
        |
        v
Training-only preprocessing
        |
        v
Proposed GRU temporal-risk model
        |
        v
Tuning-based checkpoint selection
        |
        +-----------------------------+
        |                             |
        v                             v
Temporal controls             Survival baselines
        |                             |
        +-------------+---------------+
                      |
                      v
              Frozen model scores
                      |
                      v
             IPCW-Platt calibration
                      |
                      v
             Public validation cohort
                      |
             +--------+---------+
             |                  |
             v                  v
      Ranking metrics      Bootstrap analysis
             |                  |
             +--------+---------+
                      |
                      v
             Cost-aware decisions
                      |
                      v
             Manuscript outputs
```

---

# 31. Data availability

The raw Scania Component X dataset is not redistributed with this repository.

Users should obtain the dataset from the original public source:

> Kharazian, Z., Lindgren, T., Magnússon, S., Steinert, O., and Andersson Reyna, O. (2025).  
> **SCANIA Component X dataset: A real-world multivariate time series dataset for predictive maintenance.**  
> *Scientific Data*, 12, 493.

See:

```text
data/README.md
```

for local dataset configuration instructions.

---

# 32. Reproducibility notes

Several aspects of the study should be interpreted carefully.

### Prediction horizon

`tau = 48` represents relative dataset time steps and is not converted into physical clock time, distance, or operating cycles.

### Fixed-horizon formulation

The proposed model is a fixed-horizon temporal-risk model rather than a complete continuous-time survival model.

### Calibration

IPCW addresses censoring-related selective observability under the estimated censoring model, but residual calibration error remains.

### Cost analysis

Cost values are illustrative normalized scenarios and are not Scania-specific industrial cost estimates.

### Intervention effectiveness

The public dataset does not provide sufficient intervention-history information to estimate the causal effect of preventive maintenance.

### Model comparisons

The survival-baseline comparison evaluates complete predictive frameworks.

The temporal representation-control experiment answers a different question: whether learned temporal encoding contains predictive information beyond hand-engineered summaries when the downstream classifier family is held fixed.

---

# 33. Citation

If this repository is used in academic work, please cite the associated manuscript and the Scania Component X dataset.

A `CITATION.cff` file may be added once the final manuscript bibliographic metadata are available.

---

# 34. Code availability statement

This repository is intended to provide the source code, preprocessing logic, training configurations, calibration procedure, comparative baselines, statistical uncertainty analysis, and decision-analysis workflow required to reproduce the experiments reported in the associated manuscript.

The raw Scania Component X data are obtained separately from the original public data source and are not redistributed here.

---

# 35. Contact

For questions regarding the implementation or reproducibility of the study, please use the contact information provided in the associated manuscript.