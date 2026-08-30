# Stage 2A + 2B methodology note

## Purpose

This stage resolves the code-side foundation needed for the editorial concern
about the mismatch between the raw training repair-event proportion and the
validation fixed-horizon positive prevalence.

## Key distinction

`train_tte.csv` contains one vehicle-level event indicator:

- `in_study_repair = 1`: a repair occurs at `length_of_study_time_step`;
- `in_study_repair = 0`: no in-study repair is observed before censoring.

That raw event indicator is **not** used directly as the proposed model's
binary classification target.

Instead, each sampled operational-history prefix is assigned a fixed-horizon
target `Y_i(48)` according to whether an observed repair occurs within the next
48 relative dataset time steps.

Right-censored prefixes without a complete 48-step follow-up window are
excluded rather than incorrectly labelled negative.

## Why this matters

The raw vehicle-level event proportion and the prefix-level fixed-horizon
training prevalence are different statistical quantities. The code now saves
both so that the revised manuscript and response letter can report the
distinction transparently.
