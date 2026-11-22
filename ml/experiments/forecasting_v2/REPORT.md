# Personal-History Forecasting, Endpoint Readiness and Documentation (forecasting_v2)

Date: 2026-11-21. Builds on `ml/experiments/forecasting_v1/REPORT.md`, which is unchanged.

> **Historical note (added 2026-11-22).** This report records the state on 2026-11-21. The paths it lists in the "Flagged, not changed" section have since changed:
> - The base-paper comparison (`docs/BasePaper_vs_YourProject_Comparison_*`, `docs/charts/base_vs_project_auc_bar.*`, `docs/charts/generate_base_vs_project_report.py` and related charts) was withdrawn and moved to `docs/withdrawn/base_paper_comparison/`.
> - ROC-AUC is no longer shown by the API or the clients.
> - The Chroma test side effect was fixed by `backend/tests/conftest.py`.
> - `backend/app/services/ml/` was archived outside the repository.
>
> The results and conclusions below are unchanged.

**Reproduce** from the project root:
```
backend/.venv/bin/python -m ml.forecasting.evaluate --out ml/experiments/forecasting_v2/reproduction_of_v1
backend/.venv/bin/python -m ml.forecasting.evaluate_personal      # ~3 s
```

**Outputs** in `personal_baselines/`:
- `results.json` (all metrics, protocol and the dataset manifest)
- per-night forecasts for Fitabase and StudentLife (`*_personal_forecasts.csv`)

## 1. Reproducibility of forecasting_v1

- **Same results:** re-running `ml.forecasting.evaluate` into `reproduction_of_v1/` reproduced `forecasting_v1` exactly. All 294 numeric values in `results.json` match, and all 10 prediction and feature CSVs are byte-identical.
- **Dataset versions:** the 395 raw input files read by the forecasting code are hashed individually in `personal_baselines/results.json` (`dataset_manifest`); their combined SHA-256 is `9e841733…59a43bf`.
- **Data handling:**

  | Dataset | Target | Participant split | Date split |
  |---|---|---|---|
  | Fitabase | `sleep_ratio_proxy_pct` | `GroupKFold(5)` by `Id` | test from 2016-04-22 |
  | Walch | `psg_sleep_efficiency_pct` | leave-one-subject-out | — |
  | StudentLife | `self_reported_sleep_hours` | `GroupKFold(5)` by `uid` | test from 2013-04-28 |

  Fold membership is stored in the `fold` column of each prediction CSV.
- **One expected difference from v1:** the v2 expanding-mean MAE on Fitabase is 2.195, not v1's 2.276. This is not a discrepancy; the evaluation sets differ.
  - v1 scored all 776 nights. Nights with no history fell back to the training mean.
  - v2 scores only nights with at least 3 prior valid nights: 717 nights and 18 participants. The 59 excluded nights would return "insufficient history".

## 2. Methods and protocol

**What a forecast may use.** Each forecast uses only that participant's earlier nights whose outcome was available before the target night's cutoff.
- *Fitabase:* the session had ended by the cutoff.
- *StudentLife:* the report had been submitted by the cutoff.

**Data rules.**
- **Missing nights** are skipped, never zero. "Last k valid nights" may span more than k calendar days. The median gap to the last valid night is 1 day; the 95th percentile is 2 days (Fitabase) and 6 days (StudentLife).
- **Duplicates:** there is one value per night. Fitabase uses the longest main sleep; StudentLife uses the latest report. The code rejects any duplicate night in a history.
- **Minimum history:** 3 valid nights for every method.

**Methods.**
- **Existing baseline:** expanding mean of all earlier valid nights.
- **A / B:** mean of the last 7 / 14 valid nights.
- **C:** exponentially weighted mean (EWMA), with pre-specified half-lives of 3, 7 and 14 valid nights. The half-life is selected on training participants (each grouped fold) or training dates (chronological split) only. This out-of-sample version is `ewma_selected`. Fixed half-lives are reported descriptively.
- **D:** trend-adjusted mean.
  - It needs at least 14 nights and uses the last 14. It adds an ordinary least-squares slope × 0.5 damping × (target day − mean window day).
  - The adjustment is capped at ±1 SD of the window, and the result is clipped to the target's range.
  - With fewer than 14 nights it falls back to the expanding mean. The trend applied to 73% of eligible Fitabase forecasts and 58% of StudentLife forecasts.

**Validation.**
- **Participant-independent:** `GroupKFold(5)`, with the participant sets disjoint (asserted in code).
- **Chronological:** the same date split as v1. Training-period nights choose the EWMA half-life. Test-period nights may use earlier test-period nights as history, as an online forecast would. No training-derived parameter uses test-period data.
- **Uncertainty:** a paired bootstrap over participants (2,000 resamples) of MAE(method) − MAE(expanding mean).

**Decision rule, fixed before results.** A method is adopted only if its 95% CI lies below zero in **both** evaluations **and** the improvement is ≥ 2% of the baseline MAE.

**Protocol correction, disclosed.** After the first run I noticed the rule let the fixed half-lives (h3, h7, h14) compete as separate candidates, which amounts to choosing a half-life on the evaluated data. They were removed as candidates, leaving `ewma_selected`. This made the rule stricter. Without the correction, `ewma_h7` would have been "adopted" on StudentLife.

**Walch is not evaluated:** each subject has one PSG night, so no personal history exists.

## 3. Results

### Fitabase: device-derived sleep-ratio proxy (%)

717 forecasts from 18 participants; 59 nights had insufficient history. History per forecast: median 23 nights (IQR 12–37, range 3–61).

| Method | Grouped MAE [95% CI] | RMSE | ΔMAE vs expanding mean [95% CI] | Chronological MAE (241 / 17) | ΔMAE chronological |
|---|---|---|---|---|---|
| **Expanding mean** | **2.195** [1.84, 2.70] | 3.14 | — | **2.190** | — |
| Last 7 | 2.259 | 3.25 | +0.064 [−0.014, +0.140] | 2.199 | +0.009 [−0.112, +0.132] |
| Last 14 | 2.198 | 3.15 | +0.003 [−0.051, +0.054] | 2.136 | −0.053 [−0.148, +0.042] |
| EWMA, selected out of sample | 2.172 | 3.14 | −0.023 [−0.059, +0.012] | 2.123 | −0.067 [−0.120, −0.015] |
| Trend-adjusted | 2.217 | 3.21 | +0.021 [−0.051, +0.087] | 2.147 | −0.043 [−0.173, +0.080] |
| *EWMA h7 (descriptive only)* | 2.166 | 3.14 | −0.029 [−0.072, +0.011] | 2.093 | −0.097 [−0.170, −0.019] |

- **Participant-level MAE for the expanding mean:** median 2.02, IQR 1.68–2.49, range 1.24–5.43.
- **By history length,** expanding-mean MAE was 1.64 (3–6 nights), 2.43 (7–13), 2.27 (14–27) and 2.18 (≥ 28). No method was consistently better in any band.

### StudentLife: self-reported sleep hours

1,091 forecasts from 47 participants; 150 nights had insufficient history. History per forecast: median 16 (IQR 9–26).

| Method | Grouped MAE [95% CI] | RMSE | ΔMAE vs expanding mean [95% CI] | Chronological MAE (378 / 39) | ΔMAE chronological |
|---|---|---|---|---|---|
| **Expanding mean** | **0.815** [0.71, 0.92] | 1.17 | — | **1.274** | — |
| Last 7 | 0.791 | 1.17 | −0.023 [−0.057, +0.003] | 1.205 | −0.069 [−0.152, +0.003] |
| Last 14 | 0.794 | 1.16 | **−0.021 [−0.045, −0.004]** | 1.218 | **−0.056 [−0.122, −0.011]** |
| EWMA, selected out of sample | 0.790 | 1.17 | −0.025 [−0.060, +0.004] | 1.238 | −0.036 [−0.076, −0.009] |
| Trend-adjusted | 0.792 | 1.17 | −0.022 [−0.055, +0.006] | 1.194 | −0.080 [−0.166, −0.005] |

- **Participant-level MAE for the expanding mean:** median 0.71 h, IQR 0.50–0.93.
- **By history length,** the methods are indistinguishable below 28 nights (0.54 / 0.62 / 0.72 h). At 28 or more nights, the expanding mean is worse (1.36 h) than the windowed methods (1.24–1.28 h). This is consistent with sleep drifting over the academic term, which a long-memory mean doesn't follow.

### Uncertainty

An interval of expanding mean ± 1.28 × the person's historical SD (nominal 80%) covered **76%** of Fitabase outcomes and **67%** of StudentLife outcomes. On StudentLife it covered only 54% with 28 or more nights of history.

**Historical SD alone understates forecast uncertainty.**

## 4. Selected method

- **Fitabase: expanding mean retained.** No candidate met the rule.
- **StudentLife: expanding mean retained.**
  - The 14-night mean formally meets the statistical rule (2.5% relative improvement, both CIs below zero).
  - But the gain is 0.021 h (≈ 1.3 min) across participants and 0.056 h (≈ 3.4 min) chronologically, on a response scale with half-hour steps.
  - Per the brief, a marginal numerical improvement alone is not a reason to switch. The two rules are equally simple; no improvement is claimed.
  - If a windowed rule is ever needed for long histories (≥ 28 nights), the 14-night mean is the candidate the evidence favours.
- **Walch / PSG efficiency:** no personal-history method can be validated, so nothing here supports PSG-measured efficiency.

## 5. Readiness specification (Phase 3)

| Item | Specification |
|---|---|
| Supported dataset / target | Fitabase device sleep-ratio proxy (validated, MAE ≈ 2.2 percentage points); StudentLife self-reported hours (MAE ≈ 0.8 h grouped, 1.3 h on a later period). **Not** validated for PSG-measured efficiency or for the app's constructed index |
| Minimum valid history | 3 nights; below that, insufficient history |
| Window | All earlier valid nights (expanding mean) |
| Missing data | Missing nights skipped, never zero; days since the last valid night reported |
| Variability | Sample SD of the person's valid history (ddof = 1), reported with the count |
| Uncertainty communication | Report history count and SD; describe ± 1.28 SD as "a rough range". It covered about 76% (Fitabase) and 67% (StudentLife), not 80%, so it must not be labelled an 80% interval |
| Insufficient history | Return a structured "insufficient history" status with the count and requirement; no population or synthetic fill-in |
| Limitations | Small cohorts; device proxy and self-report targets; StudentLife time zone assumed; it is a description of the person's past, not a machine-learning prediction or a medical assessment |

## 6. Endpoint (Phase 4): not implemented. Prerequisites documented.

These findings come from inspecting the existing backend.

1. **No authentication or user identity exists.** No auth dependency, user table or user ID is in `backend/app`. `/history` returns every stored record to any caller.
2. **The only table, `predictions`,** stores request payloads and *model predictions*. It has no user column and **no observed sleep outcome**: no measured efficiency, no device sleep ratio, no self-reported sleep.
3. **The app collects no outcome matching a validated target.** The web app and Android collect lifestyle inputs and screen activity, not a device sleep ratio or self-reported hours.

A forecast built on this database would have to treat earlier *predictions* as if they were observations, which would be fabricated history, or pool all users' records, which is a shared history across users. The brief forbids both. Per its instruction, I stopped before adding an endpoint.

**Prerequisites before an endpoint can be built safely** (each needs your approval):
- An authentication mechanism and a user identity.
- A new `sleep_observations` table with a user ID, night date, target type (`device_sleep_ratio` or `self_reported_hours`), value, source and a unique (user, night, target type) constraint. This needs an Alembic migration with a downgrade path; it doesn't change the existing table.
- A client flow that records observed outcomes, for example a morning self-report or an opt-in Health Connect import.
- Then add `GET /api/v2/forecast/next-night`, scoped to the authenticated user and returning:
  - `status` (`ok` or `insufficient_history`)
  - `method` ("personal historical average", not "ML prediction")
  - `target_type`
  - `forecast`
  - `history_count`
  - `history_sd`
  - `days_since_last_observation`
  - `provenance`

  Tests should cover insufficient history, missing nights, duplicates, invalid values and cross-user isolation.

## 7. Documentation changes (Phase 5)

| File | Change |
|---|---|
| `README.md` | Intro note: deployed models are same-night estimators. ROC-AUC column removed, with the reason. Duplicate-profile finding. A "same-night, not coming-night" limitation. New "Coming-Night Forecasting Experiments" section (three targets kept separate, negative results, no endpoint and why). Test counts updated |
| `docs/paper/springer_sleep_efficiency.tex` | Abstract sentence on the forecasting experiments. ROC-AUC column removed from Table 1, with the reason. Duplicate-profile and `previous_sleep_hours` findings. New section "Coming-Night Forecasting on Real Data" with Table 2 and the personal-history comparison. A limitation and conclusion sentence. Data-availability update. Three dataset references added, marked `% VERIFY`. Existing structure and citations preserved. Not compiled: no LaTeX toolchain is installed |
| `docs/presentation/Sleep_Efficiency_Project_PPT_Content.md` | Slide 13 notes (no ROC-AUC; same-night estimator). New slide 14 with the forecasting results, framed as a negative result. Later slides renumbered 15–18. Limitation bullet. New viva question and answer. Test count updated |

**Flagged, not changed** (outside documentation scope, or would change the app):
- The web dashboard (`frontend/components/ResultDashboard.tsx`) and Android (`MainScreen.kt`, `Models.kt`) still **display ROC-AUC** from `/explain/model-metrics` (`ml/artifacts/metadata.json`). Removing it changes the UI and API contract; it needs your decision.
- `docs/BasePaper_vs_YourProject_Comparison_Report.txt` and `docs/charts/generate_base_vs_project_report.py` / `base_vs_project_auc_bar.svg` compare this project's ROC-AUC with a base paper's AUROC for clinical diagnosis. That comparison isn't supported; they should be withdrawn or annotated.
- `backend/app/services/ml/` (an untracked duplicate of `ml/`) contains the same ROC-AUC metrics; cleanup is still pending your decision.
- An existing Phase 3 test (`test_phase3_rag_llm.py::test_chroma_collection_matches_dataset_and_registry`) opens the real Chroma store, and Chroma rewrites `ml/artifacts/chroma_db/*` file contents and timestamps on open. The collection still validates (17 entries), but that test should use a temporary copy.

## 8. Files created or changed in this stage

**Created:**
- `ml/forecasting/personal_baselines.py`
- `ml/forecasting/evaluate_personal.py`
- `ml/tests/test_personal_baselines.py`
- `ml/experiments/forecasting_v2/` (this report, `personal_baselines/`, `reproduction_of_v1/`)

**Modified:**
- `README.md`
- `docs/paper/springer_sleep_efficiency.tex`
- `docs/presentation/Sleep_Efficiency_Project_PPT_Content.md`

Nothing in `backend/`, `frontend/` or `SleepEfficiencyAndroidApp/`, and no model artifact, database or dependency, was changed.

## 9. Verification (Phase 6)

**Tests:**
- Before this stage: 210 passed, 1 skipped.
- After: **220 passed, 1 skipped** (the opt-in live BioMistral test). This includes 10 new personal-baseline tests covering:
  - insufficient history;
  - target night, future nights and late reports excluded;
  - per-participant isolation;
  - missing nights not zero;
  - duplicate rejection;
  - window and EWMA arithmetic;
  - trend minimum and bound;
  - half-life chosen on training rows only;
  - decision rule;
  - participant bootstrap.
- The README's documented command gives the same result.

**Unchanged, verified:**
- **Model artifacts:** SHA-256 identical before and after for all 22 non-Chroma files in `ml/artifacts/`.
- **Database:** `backend/sleep_app.db` schema identical.
- **Code:** `predict.py`, `model_service.py` and `inactivity_service.py` hashes identical, so the existing `/predict` behaviour is unchanged.
- **Clients:** a file-modification scan found no changes under `frontend/` or `SleepEfficiencyAndroidApp/`.

**Not run:** the frontend and Android builds (no client code changed).

No forecast endpoint exists, so there is no cross-user path to test. The temporal guarantees are covered by unit tests and by the code's assertions.

## 10. Deferred

- **The endpoint and its prerequisites** (section 6).
- **ROC-AUC in the UI and API,** and the base-paper comparison files (section 7).
- **The sleep-onset train/serve skew** (deferred earlier by you).
- **Knowledge-base expansion** (deferred earlier).
- **Compiling the paper** (no LaTeX toolchain); verify the three new references.

## Rollback

- **New files:** delete the files and folders created in this stage (section 8).
- **Documentation:** pre-edit copies of the three documents are in this session's scratchpad and are not durable. Use version control: `git diff -- README.md docs/paper docs/presentation` shows every edit made since the last commit, including those from earlier phases.
