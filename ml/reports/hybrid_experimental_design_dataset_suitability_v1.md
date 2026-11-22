# Hybrid Experimental Design and Dataset Suitability v1

## Scope

This document is a design and dataset-suitability assessment only. It does not train a model, create a checkpoint, modify a dataset, change a split, alter production inference, or modify API, frontend, SHAP, RAG, BioMistral, database, dependency, or `REST/` content.

The purpose is to determine whether a future Hybrid MLP + 1D-CNN experiment can test incremental temporal value beyond lifestyle and engineered tabular features without overstating the evidence.

## Dataset Suitability Findings

### Active deployed-model dataset

| Asset | Rows | Participant identifier | Repeated observations | Suitability for grouped validation |
|---|---:|---|---|---|
| `ml/data/raw/lifestyle_sleep.csv` | 374 | None retained | Cannot be determined from active file | Not possible |
| `ml/data/processed/dataset_engineered.csv` | 374 | None retained | Cannot be determined from active file | Not possible |
| `ml/data/processed/sequences.npy` | 374 x 288 | No embedded identifier | Aligned by row position only | Not possible |

The active engineered dataset has no participant ID, observation date, or source-row lineage. It has no exact duplicate rows and no exact duplicates across its non-target features, but this does not establish that each row is a unique person. Participant-aware separation therefore cannot be performed from the active artifacts alone.

### Retained source provenance

`ml/scripts/prepare_real_sleep_health_data.py` reads `ml/data/raw/source_notebooks/Sleep_health_and_lifestyle_dataset.csv`. That source has 374 rows and a `Person ID` column with 374 unique values, with no repeated participant IDs. The preparation script iterates through source rows and writes a new 374-row `lifestyle_sleep.csv` plus an aligned `screen_activity_sequences.npy`, but it does not persist `Person ID` into either output.

This supports the limited observation that the source appears to have one record per person. It does **not** create a current participant-level split key. Row-order reconstruction should not be assumed or used for a future grouped-validation claim unless source-to-output lineage is explicitly verified and persisted in a separately approved experiment. Even if verified, grouping would be equivalent to row-level separation because each source participant has one observation; it would not test longitudinal generalization.

### 288-bin sequence provenance

The 288-bin sequences are synthetic daily activity/inactivity series, generated deterministically with `numpy.random.default_rng(42)` by `prepare_real_sleep_health_data.py`:

1. A synthetic expected sleep-start hour is calculated from synthetic alcohol, caffeine, stress, activity, and bedtime-consistency values.
2. A background 24-hour binary activity pattern is sampled.
3. A zero-valued inactivity window is imposed from expected sleep start to wake time, where wake time uses source sleep duration.
4. In-window activity bursts are added according to a fragmentation rate derived from stress and synthetic blue-light exposure.

The sleep-efficiency target is also generated in the same script from source sleep quality, sleep duration, stress, sleep disorder, and noise. Thus the sequence is not an observed device trace and shares generating variables with the target. It can support a controlled proof-of-concept experiment, but not a claim that real screen-inactivity behavior has improved prediction.

### Auxiliary datasets and external-validation feasibility

| Dataset | Identifier status | Outcome / features | Can validate the active 20-feature model? |
|---|---|---|---|
| `sleep_efficiency_nightly.csv` | 452 unique source IDs, one target row per ID | Sleep-efficiency target, but none of the active 20 inputs | No |
| `student_screen_features_daily.csv` | 49 repeated participant IDs, 3,083 dated rows | Screen-event features, but no paired sleep-efficiency target | No |
| `student_perceived_stress.csv` / `student_psqi.csv` | 46 participant IDs | Survey-only data without observation-date alignment | No |
| `fitbit_sleep_score.csv` | One placeholder participant ID for all 291 rows | Wearable benchmark data | No participant separation; not appropriate for this non-wearable project |
| `mmash_wearable_features_final.csv` | 22 episodes from 21 participants | Wearable sleep features and target | Separate wearable study only; too small and outside core non-wearable scope |

The existing integration quality report confirms that no external row has all 20 active-model inputs. It also documents that no cross-cohort participant joins were performed because identifiers and collection periods are unrelated. Consequently, no genuine independent external holdout for the current model can be formed from the available project assets without fabricating variables or unsafe joins.

## Validation Decision

The active 374-row dataset can support only **row-level exploratory development validation**. Its existing 75-row test split has already been examined and must not be described as a fresh final confirmation set. The 299-row train-plus-validation development partition may be used for approved repeated out-of-fold comparisons only.

A scientifically defensible independent evaluation requires one of the following before a final performance claim:

1. A newly collected non-wearable cohort with participant IDs, nightly sleep-efficiency labels, lifestyle inputs, and timestamped screen-activity observations.
2. A separate external cohort with the same predefined feature contract and outcome definition.
3. For repeated-night data, a participant-disjoint train/development/test split, with all nights from a participant assigned to exactly one partition.

The MMASH wearable file must not be folded into the core project experiment because the project scope excludes wearable sensor data and its 22 episodes are insufficient for stable model selection.

## Controlled Ablation Design

The future experiment retains RF, XGBoost, and Hybrid MLP + 1D-CNN as mandatory research models. All neural models are experimental copies stored separately from production artifacts.

| ID | Model | Inputs | Purpose |
|---|---|---|---|
| B1 | Existing RF | Existing 20 tabular features | Structured-data baseline; unchanged feature contract |
| B2 | Existing XGBoost | Existing 20 tabular features | Structured-data baseline; unchanged feature contract |
| N1 | Tabular-only MLP | Existing 20 tabular features | Determines whether neural tabular learning alone explains Hybrid behavior |
| N2 | Sequence-only 1D-CNN | Activity sequence plus explicit time-of-day channels | Measures temporal signal without lifestyle/tabular information |
| N3 | Existing Hybrid reproduction | Existing 20 tabular features plus single-channel raw sequence | Reproduces the current Hybrid design under controlled seeds and folds |
| N4 | Revised non-redundant Hybrid | Lifestyle-only tabular features plus activity, sine-time, and cosine-time sequence channels | Tests temporal information without duplicating sequence-derived summaries in the tabular branch |
| N5 | Incremental-temporal Hybrid | Existing 20 tabular features plus activity, sine-time, and cosine-time sequence channels | Direct test of whether raw time-aware sequence data improves on existing engineered tabular features |

For N4, the tabular branch excludes features directly derived from the sequence: `inferred_sleep_start_hour`, `longest_evening_inactivity_min`, `nighttime_activity_bursts`, and `dinner_to_sleep_gap`. The raw activity sequence is represented as a three-channel tensor:

```
channel 0: observed binary screen-activity bin
channel 1: sin(2 * pi * bin_index / 288)
channel 2: cos(2 * pi * bin_index / 288)
```

The explicit sine/cosine channels preserve circular time-of-day context, including the midnight boundary, without inventing a participant or target value. N5 retains all 20 tabular features deliberately; comparison of N5 with N1 is the direct incremental-value test beyond the existing engineered contract. N4 answers a different question: whether temporal data can replace duplicated engineered temporal summaries when combined with lifestyle inputs.

## Evaluation Methodology

### Fold plan

1. Keep the current 75-row historical test results as previously examined exploratory context only; do not use them for architecture, loss, output-bound, or ensemble selection.
2. Use the existing 299-row development partition for five-fold outer cross-validation with predeclared shuffled folds.
3. Use the same outer fold assignments for B1-B2 and N1-N5 within each seed.
4. Use an inner validation split drawn only from each outer-training partition for early stopping and candidate selection. No outer-validation target may influence training, early stopping, scaling, loss choice, or output-bound choice.
5. Where valid participant IDs exist in a future linked cohort, replace row-level splitting with grouped outer and inner splits. If participants have repeated nights, all nights remain in one partition.

### Repeated seeds and reproducibility

Use five predeclared seeds: `42`, `1337`, `2027`, `31415`, and `271828`. Record Python, NumPy, PyTorch, data-loader, and model-initialization seeds for every run. Persist only separately versioned experimental configurations, fold assignments, metrics, and checkpoints after explicit implementation approval. Production artifacts must never be overwritten.

### Metrics and uncertainty

For every outer-fold prediction and every seed, report:

- MAE, RMSE, and R2.
- Errors in target ranges below 70, 70 to below 85, and 85 or above sleep-efficiency points.
- Counts and rates of predictions outside 0-100; separately report predictions outside the observed 50-98 training range as a descriptive diagnostic.
- Residual distributions: mean, standard deviation, median, 5th and 95th percentiles, and skew.
- Pairwise residual correlations between RF, XGBoost, N1-N5, and any candidate ensemble.
- Incremental temporal value: paired outer-fold difference between N5 and N1, and between N4 and its lifestyle-only MLP counterpart.

Estimate uncertainty using a stratified bootstrap of complete out-of-fold prediction rows, with at least 1,000 resamples and 95% percentile confidence intervals for MAE, RMSE, R2, and paired metric differences. In a future repeated-participant dataset, resample participants rather than individual nights. Repeated-seed results should be summarized as mean, standard deviation, and range across seeds; no single best seed may be selected for reporting.

## Output-Boundary and Loss Investigation

The current production Hybrid has an unconstrained linear output head and uses MSE. The redesign experiment should compare bounded-output and robust-loss choices as predefined experimental conditions, never as post-hoc clipping.

| Condition | Output / loss | Scientific rationale | Risk to assess |
|---|---|---|---|
| O1 | Linear output with MSE | Historical baseline for reproducibility | Can produce invalid values and over-emphasize large residuals |
| O2 | Linear output with Huber loss, delta derived from outer-training residual scale only | Reduces sensitivity to large errors while retaining continuous regression | Does not itself enforce valid bounds |
| O3 | `100 * sigmoid(z)` output with MSE or Huber loss | Sleep efficiency is naturally bounded from 0 to 100 | Can saturate near boundaries and reduce gradient magnitude |
| O4 | Logit-transform `y / 100` during training, inverse-logit output during evaluation | Enforces a 0-100 output domain through target transformation | Requires an outer-training-only epsilon policy and can overemphasize boundary behavior |

O2-O4 must use the same outer folds and seeds as O1. Huber delta, any target-transform epsilon, and model-selection rule must be set from the outer-training data or predeclared before outer-fold evaluation. Simple inference clipping is not an acceptable experimental condition because it conceals rather than models domain violations.

## Decision Rules

A redesigned Hybrid should be considered for a later ensemble-selection experiment only if all of the following hold on repeated outer-fold results:

1. It improves or is practically non-inferior to the tabular-only MLP under a predeclared margin.
2. It has a positive, uncertainty-supported incremental benefit over N1 or its lifestyle-only counterpart.
3. It has no unexplained output-domain violations under its chosen bounded-output design.
4. It has residual correlation with RF/XGBoost low enough to be complementary **and** standalone error close enough to contribute positively in development-only ensemble selection.
5. The result is reproduced on a new participant-disjoint or external cohort before a production claim.

## Required Approvals Before Implementation

1. Approval to create a separate experimental directory, code, metrics, and non-production checkpoints with a new version identifier.
2. Approval of the five fixed seeds, nested cross-validation protocol, bootstrap procedure, and predeclared decision rules.
3. Approval to train N1-N5 and the bounded-output/loss conditions on the existing data, explicitly labelled exploratory because sequences are synthetic and participant-level validation is unavailable.
4. Approval of the output-bound conditions O1-O4 and the rule for selecting a future candidate without consulting the historical 75-row test split.
5. If a generalizable claim is required, provision of a new linked non-wearable dataset with participant IDs, nightly labels, lifestyle features, and actual screen-activity timestamps, or approval to obtain one.
6. Separate approval before any production artifact, API, inference, ensemble, or frontend change.

## Conclusion

The proposed experiment can rigorously determine whether the redesigned time-aware sequence representation adds predictive value in this project, but current evidence does not support external or participant-level generalization claims. The active data are suitable only for controlled exploratory ablation. Any successful result must remain experimental until confirmed on a newly locked, compatible, participant-aware or external dataset.
