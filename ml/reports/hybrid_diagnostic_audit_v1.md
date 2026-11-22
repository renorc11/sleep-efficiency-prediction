# Hybrid MLP + 1D-CNN Diagnostic Audit v1

## Scope and Boundary

This is a read-only diagnostic audit of the deployed Hybrid model family used by `backend/app/services/model_service.py`. It did not retrain a model, change a checkpoint, write a preprocessing artifact, modify production inference, or alter any API, frontend, SHAP, RAG, BioMistral, database, dependency, or `REST/` content.

The audit distinguishes two incompatible artifact families:

1. **Deployed family:** `hybrid.pt` plus `hybrid_tabular_scaler.joblib`. This is the Hybrid used by the FastAPI prediction path.
2. **Stacking family:** `stacking_final_hybrid_cnn.pt` plus `stacking_cnn_scaler_mean.npy` and `stacking_cnn_scaler_std.npy`. This has a different CNN definition and must not be mixed with deployed artifacts.

All findings about production behavior below refer to the deployed family unless explicitly labelled as stacking-family evidence.

## Audited Data and Pipeline

- Engineered dataset: 374 rows, 20 tabular predictors, target range 50.00 to 98.00 sleep-efficiency percentage points.
- Stored split: 261 training rows, 38 validation rows, 75 historical test rows.
- Sequence tensor: 374 x 288 binary values, where one value represents a five-minute activity bin across a day.
- Target handling: `sleep_efficiency` is passed directly to the model in percentage-point units. It is not target-scaled, normalized, bounded, or inverse-transformed.
- Tabular handling: `StandardScaler` is fitted on the 261 training rows and applied to validation/test/inference tabular features. The 288-bin sequence is passed as raw 0/1 values.
- Training: Adam optimizer with learning rate 0.001, MSE loss, maximum 80 epochs, batch size 64, and early stopping after 12 validation-loss epochs without improvement.

## Deployed Architecture

The deployed model has 6,577 trainable parameters:

- Tabular MLP: 20 -> 32 -> 16 with ReLU and dropout 0.1.
- Temporal branch: Conv1D(1, 8, kernel 5, same padding) -> ReLU -> max pooling -> Conv1D(8, 16, kernel 5, same padding) -> ReLU -> adaptive average pooling to 8 positions.
- Fusion head: concatenated 16 tabular and 128 temporal features -> 32 -> 1.
- Output head: unconstrained final linear layer. It can emit any real number.

At inference, RF and XGBoost receive the engineered tabular features; the Hybrid receives the same 20 standardized tabular features plus the raw sequence. The deployed ensemble then calculates `0.2 * RF + 0.3 * XGBoost + 0.5 * Hybrid` and clips only the final ensemble to 50-99. The Hybrid output itself is not clipped.

## Observed Evaluation Evidence

### Regression performance

| Model / artifact family | MAE | RMSE | R2 |
|---|---:|---:|---:|
| Deployed RF | 2.6412 | 3.6059 | 0.9330 |
| Deployed XGBoost | 3.0720 | 4.2481 | 0.9070 |
| Deployed Hybrid | 4.9758 | 6.1650 | 0.8041 |
| Stacking-family Hybrid | 4.7510 | 5.7638 | 0.8288 |

The deployed Hybrid has 2.3346 higher MAE and 2.5591 higher RMSE than deployed RF. The separately trained stacking Hybrid is also materially weaker than its stacking-family RF. This repeated pattern across incompatible Hybrid implementations supports a pipeline/data-representation issue rather than a single deployed-checkpoint anomaly.

### Residual distribution and range behavior

The deployed Hybrid test residuals have MAE 4.9758, RMSE 6.1650, and a broader error spread than RF. In the existing 75-row historical test evaluation, Hybrid MAE is worse in every target range:

| Sleep-efficiency range | Rows | RF MAE | Hybrid MAE | RF RMSE | Hybrid RMSE |
|---|---:|---:|---:|---:|---:|
| Below 70 | 30 | 3.1663 | 4.9923 | 4.0510 | 6.6032 |
| 70 to below 85 | 29 | 2.4475 | 4.8022 | 3.6140 | 5.8333 |
| 85 and above | 16 | 2.0079 | 5.2595 | 2.5508 | 5.8933 |

The four largest domain violations across all stored rows were predictions of 100.74 to 101.94 for actual values of 91.20 to 93.00. Two of these violations occur in the historical test partition. This is evidence of high-end overshoot, not merely a theoretical output-domain risk.

### Train, validation, and test diagnostic performance

| Partition | Rows | MAE | RMSE | R2 | Prediction range | Above 100 |
|---|---:|---:|---:|---:|---|---:|
| Train | 261 | 5.1169 | 6.2270 | 0.7965 | 58.33 to 100.74 | 1 |
| Validation | 38 | 5.8230 | 7.3722 | 0.6823 | 58.43 to 101.94 | 1 |
| Historical test | 75 | 4.9758 | 6.1650 | 0.8041 | 58.96 to 101.33 | 2 |

These are descriptive diagnostics, not new model-selection results. The relatively high training error does not indicate a simple high-capacity memorization failure; it is more consistent with limited fit, representation mismatch, optimization sensitivity, or insufficient signal after duplicate features are accounted for. The audit did not retain epoch-level learning curves, so it cannot distinguish these causes conclusively.

## Temporal Signal Assessment

### Direct association

Descriptive Pearson correlations with target sleep efficiency were modest for raw sequence summaries:

| Sequence summary | Correlation with target |
|---|---:|
| Total active bins | 0.1596 |
| Evening activity, 18:00-24:00 | 0.1233 |
| Night activity, 22:00-07:00 | 0.2167 |
| First-half-day activity | 0.0578 |
| Second-half-day activity | 0.1554 |

After fitting a diagnostic linear model to all 20 engineered tabular features, correlations between these sequence summaries and the remaining target residual were near zero (-0.0368 to 0.0097). This does not prove that no nonlinear temporal signal exists, but it provides no evidence that the currently represented sequence summaries add substantial independent linear information beyond the tabular contract.

### Redundancy

The tabular input already includes `inferred_sleep_start_hour`, `longest_evening_inactivity_min`, and `nighttime_activity_bursts`, all derived from the same sequence. The strongest observed redundancy is exact: the raw sequence's 22:00-07:00 activity-bin count is perfectly correlated with `nighttime_activity_bursts` (correlation 1.0000). Total activity is also correlated with this engineered burst feature (0.5954).

This means RF and XGBoost receive concise, directly usable temporal summaries, while the Hybrid receives both those summaries and the full raw sequence. On only 261 training rows, the CNN must learn useful additional structure beyond information already represented in tabular form. The current audit finds no evidence that it succeeds reliably.

### Data-generation limitation

The stored sequences are synthetic rather than device-observed. `prepare_real_sleep_health_data.py` generates them using expected sleep-start time, sleep duration, stress, and a synthetic blue-light estimate. Sleep efficiency is also generated from sleep-quality score, sleep duration, stress, disorder status, and noise. Therefore sequence and target relationships originate partly from shared generation rules, not independent real-world screen-inactivity measurement. This limits any claim that the current 1D-CNN has demonstrated real device-behavior value.

## Why the Hybrid Is Weaker

### 1. Small sample size relative to multimodal representation

The Hybrid has 6,577 parameters and is trained on 261 rows, with only 38 rows used for early stopping. RF and XGBoost are well suited to small structured data and can exploit the pre-engineered temporal summary features directly. The Hybrid must learn from a 288-step raw sequence in addition to the same tabular inputs, which increases optimization variance without demonstrated added signal.

### 2. Raw temporal information is largely redundant

The strongest screen-inactivity indicators have already been engineered into the tabular feature set. The exact correlation between nighttime raw activity and `nighttime_activity_bursts` is direct evidence of duplication. The low residual associations after controlling for tabular features indicate limited remaining independent information in the currently generated sequence representation.

### 3. Temporal branch may discard timing detail

The CNN uses same-padded convolutions, max pooling, and adaptive average pooling to eight positions. This compresses a full-day 288-bin sequence heavily. Without explicit time-of-day or positional features, patterns occurring at different clock times can become difficult to distinguish after convolution and pooling. This is particularly relevant because sleep onset depends on clock-time context, while the model receives no explicit positional encoding in the sequence branch.

### 4. Training procedure has high variance risk

The deployed training script does not set a random seed, unlike the later stacking training script. With a small dataset, random initialization and shuffled batches can materially affect the selected checkpoint. The script retains only a best validation state; it does not persist learning curves, epoch number, or repeated-seed performance. This prevents a stability assessment.

### 5. MSE prioritizes large errors but does not enforce valid bounds

MSE is valid for continuous regression but squares large deviations, making the fit sensitive to outliers. More importantly, the final linear head is unconstrained while the target domain is 50-98. Nothing in the loss or architecture prevents predictions above 100, and observed outputs confirm this risk. This is the immediate technical reason for values exceeding 100%.

## Why Lower Residual Correlation Did Not Improve the Ensemble

Hybrid residuals are less correlated with RF than RF and XGBoost residuals are:

| Residual pair | Historical test | OOF development |
|---|---:|---:|
| RF and XGBoost | 0.9287 | 0.9395 |
| RF and Hybrid | 0.4917 | 0.5510 |
| XGBoost and Hybrid | 0.5327 | 0.5644 |

Lower residual correlation establishes diversity, but an ensemble improves only when that diversity has adequate standalone accuracy. Here, Hybrid error is too large. The current policy assigns 50% weight to the least accurate predictor, causing development MAE to worsen from RF's 2.6239 to 3.6464. Nested convex blending selected RF/XGBoost weights only (mean RF 0.82, XGBoost 0.18, Hybrid 0.00) in every outer fold. The evidence supports complementary error patterns, but not useful predictive contribution at the Hybrid's current error level.

## Likely Causes Ranked by Evidence

| Likely cause | Evidence strength | Evidence |
|---|---|---|
| Unconstrained output head | High | Four stored predictions exceed 100%, including two historical-test rows; target maximum is 98. |
| Temporal/tabular redundancy | High | Exact correlation of nighttime sequence activity with engineered nighttime bursts; residual associations after tabular adjustment are near zero. |
| Synthetic sequence construction | High | Sequence is programmatically generated from variables also related to target construction, not device-observed behavior. |
| Insufficient data for CNN fusion | Medium-high | 261 training rows for 6,577 parameters and 288-bin sequence input; Hybrid weaker across all ranges and both artifact families. |
| Aggressive temporal compression / missing position context | Medium | Architecture pools to eight positions without explicit positional encoding; plausible from implementation but not proven by an ablation. |
| Optimization instability | Medium | Deployed script has no explicit seed and no learning-curve or multi-seed record; requires controlled experiment to confirm. |
| MSE sensitivity to high residuals | Medium | RMSE gap is large and overshoot exists, but no direct loss-function ablation was performed. |

## Proposed Improvements (Not Implemented)

1. Establish a new locked external or participant-level holdout before any new model selection claim.
2. Run a controlled, separately versioned ablation study: tabular-only MLP, sequence-only CNN, tabular-plus-sequence Hybrid, and RF/XGBoost baselines using identical folds.
3. Compare raw sequences against non-duplicated temporal representations. Remove only duplicate temporal summaries from the Hybrid branch in an experiment; keep the deployed feature contract unchanged until approved.
4. Add explicit time-of-day representation and evaluate whether it improves sleep-onset pattern learning.
5. Evaluate bounded-target approaches, output calibration, and alternative robust regression losses in separately approved experiments. No clipping or calibration should be added directly to production without validation.
6. Set and record deterministic seeds; retain per-epoch train/validation curves, best epoch, parameter count, and repeated-seed mean plus standard deviation.
7. Use genuine device-derived screen activity, with documented observation windows and time zones, before claiming real-world temporal benefit.

## Experimental Validation Plan

The following plan requires approval before implementation:

1. Freeze a new, participant-level or external test cohort and do not inspect it during architecture, loss, or weight selection.
2. On development data only, use repeated grouped cross-validation where participant identity is available. Otherwise document the lack of participant grouping as a limitation.
3. Train the four ablations under fixed seeds and equal folds: RF/XGBoost tabular baseline, tabular MLP, sequence-only CNN, and tabular-plus-sequence Hybrid.
4. Measure MAE, RMSE, R2, target-range errors, prediction-domain violations, residual distributions, residual correlations, and repeated-seed variability.
5. Test incremental temporal value using paired fold-level comparisons between tabular-only and tabular-plus-sequence models. Report confidence intervals or bootstrap intervals rather than only point estimates.
6. Select an ensemble policy using only out-of-fold development predictions. Lock it before a single evaluation on the newly reserved cohort.
7. Keep deployed and stacking artifact families separate throughout every experiment.

## Recorded Integrity Identifiers

The following SHA-256 values were recorded during this audit:

| File | SHA-256 |
|---|---|
| `ml/artifacts/hybrid.pt` | `0636e4064677faf31509361320177e2b7acf600fb03d579bd61e19432c3ce874` |
| `ml/artifacts/hybrid_tabular_scaler.joblib` | `a143e5fe2c112a7987e459c0163ec18c1a7da98fe5e5db5578b47dc46cc3f38e` |
| `ml/data/processed/dataset_engineered.csv` | `5f9ffc3d076d150ff8f0ece6b07ec5d8877e529fda54f0d798b3213cd989a065` |
| `ml/data/processed/sequences.npy` | `e3f515b11e7db70dff428a30bfd462fd75eda580436659cdb8b7dc5c861583bb` |
| `ml/data/processed/split.json` | `d0e01d3430563a3c5dbac601708e74456a3f071fe9bdc6eb499beaa6aec33166` |

## Audit Conclusion

The current Hybrid retains research value as a complementary-error model, but it does not currently provide enough accurate independent temporal information to improve RF or a development-selected blend. Its poorer error, redundant/synthetic temporal representation, compressed timing information, small training sample, and unconstrained regression head together explain the observed performance and output-domain issues. No production change is justified from this audit alone.
