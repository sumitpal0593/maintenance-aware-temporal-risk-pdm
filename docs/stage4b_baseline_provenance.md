# Stage 4B baseline provenance

The originally submitted comparison values were generated experimentally.
They were not invented or fabricated.

The original implementations were:

| Baseline | Original implementation |
|---|---|
| CoxPH | `lifelines.CoxPHFitter` |
| Random Survival Forest | `sksurv.ensemble.RandomSurvivalForest` |
| Deep CoxPH | `pycox.models.CoxPH` with `torchtuples` |
| Deep Survival Machine | `auton_survival.models.dsm.DeepSurvivalMachines` |

The revised Stage 4B deliberately preserves these implementation families.

What changes is the experimental protocol:

- prediction horizon is standardized to `tau = 48`;
- public validation uses all 5,046 vehicles;
- preprocessing is inherited from the leakage-safe Stage 4A pipeline;
- all baselines receive the same 719-dimensional aggregate+static
  representation;
- model fitting uses model-training prefix samples;
- hyperparameters are selected only on vehicle-level tuning landmarks;
- public validation is used only for final AUROC/PR-AUC/enrichment reporting.

The rerun therefore does not imply that the submitted baseline values were
non-experimental. It is a reviewer-driven standardization and reproducibility
rerun.
