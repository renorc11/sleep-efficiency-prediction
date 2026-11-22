# Coming-Night Forecasting Audit and Evaluation (forecasting_v1)

Date: 2026-11-21. Reproduce from the project root:

```
backend/.venv/bin/python -m ml.forecasting.evaluate        # ~90 s, writes this folder
PYTHONPATH=backend backend/.venv/bin/python -m pytest -q backend/tests ml/tests
```

Raw numbers: `results.json`; out-of-fold predictions: `*_predictions.csv`;
feature tables: `*_features*.csv`. `REST/` was not opened.

## 1. Existing architecture and data flow (verified in code)

| Stage | File / function | What actually happens |
|---|---|---|
| Dataset | `ml/data/raw/source_notebooks/Sleep_health_and_lifestyle_dataset.csv` | Kaggle table, 374 rows, one row per person, no dates |
| Preparation | `ml/scripts/prepare_real_sleep_health_data.py::run`, `generate_sequence` | 9 of 13 lifestyle inputs (caffeine, alcohol, dinner hour, mood, naps, previous sleep, work hours, bedtime consistency, hydration, blue light) are generated from formulas plus random noise. The target is `clip(10*Quality + 2.3*(Duration-6.5) - 0.7*max(0, Stress-6) - 2*Apnoea + N(0, 2^2), 50, 98)`. The 288-bin activity sequence is generated from the same night's sleep duration |
| Preprocessing | `ml/scripts/preprocess.py::infer_sleep_features`, `run` | Sequence features, interaction terms, random stratified 261/38/75 split |
| Training | `train_tabular_models.py` (RandomizedSearchCV on train only), `train_hybrid_model.py` (scaler fit on train only), `compare_and_package.py` → `ml/artifacts/metadata.json` | RF, XGBoost, Hybrid MLP+1D-CNN |
| Inference | `backend/app/services/inactivity_service.py::build_engineered_features` → `model_service.py` (vector built in `metadata.json` order, line 219) → fixed ensemble 0.2/0.3/0.5 | Same 20 features, same order |
| Clients | `frontend/` and `SleepEfficiencyAndroidApp/` → `POST /api/v1/predict` | Unchanged |

## 2. Files inspected

ML pipeline scripts listed above, `ml/artifacts/metadata.json`, `ml/reports/tabular_metrics.json`, `ml/data/processed/{dataset_engineered.csv,split.json}`, backend `inactivity_service.py`, `model_service.py`, `main.py`, `core/config.py`, backend and ML tests, and the raw files of Fitabase, Walch, StudentLife, MMASH, Newcastle and the two Kaggle tables. Their earlier read-only dataset audit is in the conversation history.

## 3. Existing issues, by severity

**Confirmed**

1. **The deployed model is a same-night estimator on constructed data, not a coming-night forecaster.**
   - The target is a formula of self-rated quality, same-night duration and stress.
   - `stress_level` is an input (r = -0.90 with the target).
   - `previous_sleep_hours` is the *same night's* duration plus N(0, 0.5) noise (r = 0.83 with the same-night duration).
   - The activity sequences are generated from the target night's sleep window.
   - Its R² of about 0.93 therefore mostly reflects recovering the formula. It is not evidence of real-world predictive ability.
2. **Duplicate source profiles leak across the split.**
   - The 374 rows hold only 132 distinct source profiles, and 66 of the 75 test rows share a profile with training or validation rows.
   - Re-evaluated with the same hyperparameters, the effect is small: RF R² 0.925 (random 5-fold) vs 0.917 (profile-grouped 5-fold); MAE 2.66 vs 2.80.
   - The inflation comes mainly from item 1, not from duplication.
3. **Train/serve skew in sleep-onset inference (previously deferred by you).** Training scans the whole sequence. `inactivity_service.infer_sleep_from_sequence` masks every bin after `datetime.now()`, so the same request can give different features depending on the time of day it is sent.
4. **Classification metrics reuse the regression output.** `metrics_bundle` maps predictions to a pseudo-probability with `(pred - 50) / 50`; this is not a calibrated probability. ROC-AUC values computed this way should not be reported as classifier performance.

**Potential**

5. Fitabase cannot separate "not worn" from "not moving": minute rows exist with 0 steps either way.
6. Walch's Apple Watch heart rate is sparse outside workouts: on average about 7% (maximum 16%) of minutes in the 24 h before the PSG night have a sample.

## 4. Dataset suitability for coming-night forecasting

| Dataset | Use here | Target | Why |
|---|---|---|---|
| Fitabase | **Training and validation (wearable mode)** | Device sleep-ratio proxy | 20 participants, 776 forecastable nights, minute-level activity, heart rate for 14 participants |
| Walch | **External check only** | PSG sleep efficiency | 31 subjects, one night each; days of watch data before the PSG night |
| StudentLife | **Training and validation (phone mode)** | Self-reported sleep hours | 49 participants, 1,241 nights; phone lock, dark and charging intervals |
| MMASH | Exploratory only | Actigraphy efficiency | One night per person; activity-code meanings not available locally, so not treated as caffeine or alcohol |
| Newcastle | Excluded here | PSG (clinical) | One night, no pre-night wearable data in a usable format |
| Kaggle tables, single-user Fitbit | Excluded as ground truth | — | Unreliable or constructed targets; n = 1 |

The datasets are never joined. Each keeps its own IDs and target.

## 5. Feature engineering (new, versioned, `ml/forecasting/`)

**Cutoffs.** Every feature uses only data before a per-night cutoff.

| Dataset | Cutoff | Reason |
|---|---|---|
| Fitabase | **20:00** primary, 21:00 sensitivity | 40 of 817 main sleeps begin 20:00–21:00, so a 21:00 cutoff would overlap about 5% of target nights. Nights whose sleep starts before the cutoff are excluded as targets but remain history. |
| Walch | 30 min before the PSG start | Relative time only; there are no dates |
| StudentLife | 21:00 America/New_York | The timezone is assumed from the study site. A report counts as history only if it was submitted before the cutoff. |

**Feature groups**

| Group | Fitabase | Walch | StudentLife |
|---|---|---|---|
| Previous-night sleep | Night D-1 main sleep: minutes, asleep, ratio, awake, restless, onset, wake | — | Previous report: hours and rating |
| Personal history | Expanding mean, last-7 mean, count, nights observed in last 7 days | — | Expanding mean, last-7 mean, count |
| Previous-day activity | Steps, stepping minutes, MVPA minutes (00:00–cutoff and last 3 h), coverage | Steps over last 24 h and 3 h; observed 10-min bins | Phone locked, dark and charging fractions and lock events (06:00–cutoff, last 3 h, previous night); longest overnight lock and a cap flag |
| Heart rate | Day mean, 5th percentile, evening mean, previous-sleep mean, coverage | 24-h mean, 5th percentile, SD; evening mean; prior-days mean and 5th percentile; days of history | — |
| Lifestyle / stress | — | — | Stress EMA recoded so higher = more stressed (1 Feeling great … 5 Stressed out), with count |
| Other | Weekend night, nap minutes | — | Weekend night |
| Coverage / quality | Activity and heart-rate coverage | Heart-rate coverage, step bins | Sensing coverage per window |

**Missing data.**
- No observations give NaN plus a coverage of 0, never 0.
- Phone windows with less than 50% sensing coverage are NaN.
- Overlapping source intervals are merged; before merging, some window fractions reached 1.26.
- The models use NaN natively: scikit-learn 1.6.1 random forests and XGBoost 2.1.4.

## 6. Existing model evaluation weaknesses

- **Random row split.** A row split on data with duplicated profiles (item 2), with no participant or chronological structure; the data has no dates.
- **Evaluation data shares the construction.** Synthetic inputs and sequences come from the same generator as the target (item 1).
- **Uncalibrated classification metrics** (item 4).
- **RandomizedSearchCV ran on training rows only**, so there is no tuning leakage into the test set.
- **None of the reported numbers measure coming-night forecasting.**

## 7. Results

**Validation design.**
- Hyperparameters are fixed in `common.py` and were never tuned on evaluation folds.
- Models are fitted on training folds only.
- `evaluate_grouped` asserts that no participant appears in both train and test, except in chronological splits, where every test night is later than every training night.
- The `*_residual` variants (forecast the deviation from personal history) were added **after** the first run. Both versions are reported.

**Metrics.** ΔMAE is the model's MAE minus the MAE of the best baseline, with a paired bootstrap 95% CI over participants.

**Fitabase**, target = sleep-ratio proxy (%):

| Task | Participants / nights | Best baseline MAE | RF | XGBoost | RF residual | ΔMAE, best model vs baseline |
|---|---|---|---|---|---|---|
| Participant-grouped 5-fold, 20:00 | 20 / 776 | **2.28** (personal history) | 3.51 | 4.59 | 2.39 | +0.12 [+0.03, +0.21] (worse) |
| Chronological (test from 2016-04-22), 20:00 | 18 / 242 | **2.18** | 2.34 | 2.42 | 2.37 | +0.16 [-0.08, +0.48] |
| Participant-grouped, 21:00 | 20 / 741 | **2.31** | 3.69 | 3.87 | 2.41 | +0.10 [-0.02, +0.23] |

**Walch**, target = PSG sleep efficiency (%), leave-one-subject-out over 31 subjects:

| Training-mean baseline | RF | XGBoost | XGBoost vs baseline ΔMAE |
|---|---|---|---|
| 4.05 | 4.21 | 3.77 | -0.28 [-1.14, +0.61] (not significant) |

**StudentLife**, target = self-reported hours:

| Task | Participants / nights | Personal-history baseline | RF | RF residual | ΔMAE, best model |
|---|---|---|---|---|---|
| Participant-grouped 5-fold | 49 / 1,241 | 0.785 | 0.780 | 0.776 | -0.008 [-0.032, +0.015] |
| Chronological (test from 2013-04-28) | 39 / 378 | **1.274** | 1.358 | 1.361 | +0.084 [+0.018, +0.169] (worse) |

**Findings.**
- **The models do not beat each person's own history.** On every task the person's average from earlier nights is the strongest predictor or statistically tied with the best model.
- **No ML model shows a significant improvement anywhere.**
- **The activity, heart-rate, phone and stress features add nothing detectable.** A history-only random forest performs about the same as the full one: Fitabase MAE 3.54 vs 3.51.
- **Direct models fail on unseen participants.** In Fitabase they are pulled towards the population mean (R² 0.18). The residual formulation fixes most of this but still doesn't beat the baseline.

## 8. Old versus new model

The deployed model can't be scored on these datasets because its 13 lifestyle inputs don't exist there. A like-for-like comparison is therefore not possible. Its 0.93 R² comes from data built from the target's own formula, while the new tasks measure genuine coming-night forecasting on real data.

**Acceptance criterion: not met for replacement.** The evidence doesn't show that any new model is better than a personal-history baseline. Per the brief:
- the existing implementation and endpoint are retained unchanged;
- no new artifact was packaged and no endpoint was added.

## 9. Changes implemented

| Change | Reason |
|---|---|
| `ml/forecasting/{__init__,common,fitabase,walch,studentlife,evaluate}.py` (new) | Versioned, dataset-specific adapters with explicit cutoffs, coverage features, participant-grouped and chronological evaluation, baselines and bootstrap comparison |
| `ml/tests/test_forecasting.py` (new, 17 tests) | Cutoff and timezone handling; missing data stays NaN; participant separation; history excludes the current night and late reports; interval merging; PSG efficiency definition; dataset smoke tests |
| `ml/experiments/forecasting_v1/` (new outputs) | Results, out-of-fold predictions, feature tables |

No existing file, model artifact, dataset, database or dependency was modified.

## 10. Tests executed

- **Baseline before changes** (project root, `PYTHONPATH=backend`, `backend/tests` + `ml/tests`): 193 passed, 1 skipped (the live BioMistral test).
- **After changes:** 210 passed, 1 skipped.
- **Not re-verified:** the frontend build and the Android build. Neither was touched in this step.

## 11. Remaining limitations

- **Small samples.** 20 Fitabase participants, 31 Walch subjects. The confidence intervals are wide.
- **Fitabase.** The target is a consumer-device proxy, not PSG efficiency. There's no timezone, heart rate exists for 14 people, and non-wear is indistinguishable from inactivity.
- **StudentLife.** The target is self-reported hours (0.5 h resolution, lower bound "< 3 h"). The timezone is assumed. The 600-minute interval cap is reached in only 0.2% of nights.
- **Walch.** One night per subject; sparse heart rate; no calendar context.
- **No dataset links phone, wearable, lifestyle and measured efficiency together.** No cross-modal claim is supported.
- **Not evaluated here.** A forecasting version of the hybrid CNN; a nested or time-series CV with tuning; MMASH and Newcastle.

## 12. Recommendations (not implemented)

1. **For coming-night prediction in the app,** the defensible default is the person's own history (an expanding or 7-night mean of their earlier nights), with "insufficient history" shown until about 3 nights exist. Any learned model should be reported as an adjustment to that baseline and accepted only if it beats it under grouped or chronological validation.
2. **Present the existing model accurately:** a same-night estimator trained on constructed development data. Drop the ROC-AUC claims.
3. **Fix the onset-inference skew** (already deferred by you) before any retraining.
4. **Collect real longitudinal app data** (opt-in, with dates) to evaluate a forecaster on the app's own inputs.

## Rollback

Everything is additive. To roll back, delete `ml/forecasting/`, `ml/tests/test_forecasting.py` and `ml/experiments/forecasting_v1/`; nothing else depends on them. The feature tables in that folder contain per-participant derived data from the research datasets. Don't commit them to a public repository unless the dataset licences allow it (consider adding the folder's CSVs to `.gitignore`).
