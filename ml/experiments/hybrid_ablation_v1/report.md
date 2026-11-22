# Development-Only Hybrid Ablation Experiment v1

The 288-bin inputs are **synthetic activity/inactivity sequences**. They are not observed device activity or real screen-inactivity measurements. All results are exploratory development evidence only.

## Boundary

- Development rows evaluated: 299
- Historical 75-row test labels were not loaded or used for training, early stopping, tuning, or selection.
- Production checkpoints, scalers, datasets, splits, API behavior, and reports were not modified.

## Feature Availability Audit

| Feature | Decision | Reason |
|---|---|---|
| caffeine_mg_evening | retain_primary | Synthetic proxy; intended self-report before sleep. |
| alcohol_units | retain_primary | Synthetic proxy; intended self-report before sleep; source construction has disorder-confounding risk. |
| dinner_hour | retain_primary | Synthetic proxy; intended known before sleep. |
| stress_level | retain_primary | Source stress value; intended known before sleep. |
| mood_score | exclude_primary | Generated partly from source Quality of Sleep, which also drives the target. |
| physical_activity_min | retain_primary | Derived from source activity; intended accumulated daily value before bedtime. |
| naps_count | exclude_primary | Generated from current source sleep duration, a target-generating variable. |
| naps_total_min | exclude_primary | Generated from naps count and current source sleep duration. |
| previous_sleep_hours | exclude_primary | Generated from current source sleep duration rather than an independently observed prior night. |
| work_hours | retain_primary | Synthetic occupation-based proxy; intended known before sleep. |
| blue_light_minutes | retain_primary | Synthetic proxy from expected sleep timing, stress, and steps; conceptually pre-sleep but not observed device exposure. |
| bedtime_consistency | retain_primary | Synthetic stress/occupation-based proxy; conceptually historical behavior. |
| hydration_liters | retain_primary | Synthetic activity/blood-pressure proxy; intended daily self-report. |
| inferred_sleep_start_hour | exclude_primary | Derived from the full synthetic sequence and uses overnight inactivity unavailable before outcome. |
| longest_evening_inactivity_min | exclude_primary | Derived from the full synthetic sequence, including the imposed sleep window. |
| nighttime_activity_bursts | exclude_primary | Derived from the full synthetic sequence; exactly duplicates a raw nighttime sequence summary. |
| dinner_to_sleep_gap | exclude_primary | Depends on inferred sleep start, which is sequence-derived and post-outcome at intended prediction time. |
| caffeine_stress_interaction | exclude_primary | Deterministic duplicate of retained primary inputs; excluded to keep the primary representation non-redundant. |
| activity_stress_balance | exclude_primary | Deterministic duplicate of retained primary inputs; excluded to keep the primary representation non-redundant. |
| nap_sleep_debt_proxy | exclude_primary | Depends on previous_sleep_hours, which is generated from current source sleep duration. |

Primary models use only the nine predeclared `safe9` features. Legacy-20 models are provenance-risk diagnostics and are not eligible for primary selection.

## Repeated-Seed OOF Results

Metrics are means across five seed-specific 299-row OOF prediction sets. Seed variability is the standard deviation across those five metric values; seeds are not treated as independent datasets. Confidence intervals are 1,000-row bootstrap intervals on the per-row mean OOF prediction across seeds.

| Model | Eligible | Parameters | MAE mean (seed SD) | MAE 95% CI | RMSE mean (seed SD) | R2 mean (seed SD) | MedAE | Bias | Physical violations |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| rf_safe9 | True | 0 | 2.6685 (0.0043) | [2.3599, 3.0007] | 3.9369 (0.0051) | 0.9176 (0.0002) | 1.8511 | -0.0980 | 0 |
| xgb_safe9 | True | 0 | 2.7534 (0.0306) | [2.4041, 3.0302] | 3.9265 (0.0351) | 0.9180 (0.0015) | 1.8040 | -0.0449 | 0 |
| rf_legacy20_provenance_risk | False | 0 | 2.7290 (0.0078) | [2.3992, 3.0452] | 3.8993 (0.0056) | 0.9192 (0.0002) | 1.9133 | -0.1460 | 0 |
| xgb_legacy20_provenance_risk | False | 0 | 2.8847 (0.0257) | [2.5306, 3.1206] | 3.9567 (0.0281) | 0.9168 (0.0012) | 2.0552 | -0.1057 | 0 |
| tabular_mlp_safe9_capacity_matched | True | 5857 | 44.8321 (4.2461) | [42.5801, 46.3455] | 48.5847 (4.1831) | -11.6243 (2.2279) | 48.0598 | -44.5256 | 0 |
| tabular_mlp_legacy20_capacity_matched | False | 6561 | 32.8780 (4.9774) | [30.2053, 33.6299] | 36.8267 (4.8352) | -6.3099 (1.8573) | 32.1660 | -31.7662 | 0 |
| sequence_cnn_synthetic_only | True | 4945 | 17.6818 (2.8193) | [14.6236, 17.1024] | 21.8758 (4.0560) | -1.6143 (0.9847) | 13.3993 | -11.9037 | 0 |
| existing_hybrid_reproduction_legacy20 | False | 6577 | 14.6579 (2.9645) | [12.6397, 14.5554] | 17.7752 (4.1007) | -0.7514 (0.8787) | 11.9581 | -7.9657 | 0 |
| revised_hybrid_safe9_linear_mse | True | 6305 | 11.9390 (0.3302) | [10.9678, 12.4631] | 13.7180 (0.3657) | -0.0011 (0.0537) | 11.9835 | -1.0611 | 0 |
| revised_hybrid_safe9_linear_huber | True | 6305 | 12.0993 (0.1359) | [11.2689, 12.8007] | 14.0677 (0.1527) | -0.0523 (0.0228) | 11.9586 | 1.7080 | 0 |
| revised_hybrid_safe9_sigmoid100_mse | True | 6305 | 8.9744 (0.4586) | [8.2254, 9.3716] | 10.5667 (0.5876) | 0.4049 (0.0658) | 8.3655 | 0.3802 | 0 |
| revised_hybrid_safe9_sigmoid100_huber | True | 6305 | 8.1831 (0.8666) | [7.0687, 8.2068] | 10.1430 (0.8601) | 0.4499 (0.0916) | 6.9869 | -0.1239 | 0 |
| incremental_temporal_hybrid_legacy20 | False | 6657 | 11.3406 (0.5976) | [10.4840, 11.8628] | 13.2463 (0.9725) | 0.0631 (0.1417) | 11.6360 | -0.6960 | 0 |
| temporal_order_negative_control_safe9 | True | 6305 | 8.5155 (0.6984) | [7.5007, 8.6905] | 10.4736 (0.7645) | 0.4143 (0.0815) | 7.2220 | -0.1470 | 0 |

## Target-Range Performance on Mean OOF Predictions

| Model | Below 70 MAE | 70 to below 85 MAE | 85 and above MAE |
|---|---:|---:|---:|
| rf_safe9 | 2.6938 | 2.9298 | 2.0958 |
| xgb_safe9 | 2.8003 | 2.8144 | 2.2854 |
| rf_legacy20_provenance_risk | 2.8393 | 2.8583 | 2.2587 |
| xgb_legacy20_provenance_risk | 3.1569 | 2.7216 | 2.3871 |
| tabular_mlp_safe9_capacity_matched | 27.1678 | 53.1136 | 59.4041 |
| tabular_mlp_legacy20_capacity_matched | 21.4471 | 42.0099 | 31.2520 |
| sequence_cnn_synthetic_only | 7.2598 | 16.4207 | 30.5628 |
| existing_hybrid_reproduction_legacy20 | 8.6646 | 13.3014 | 23.0409 |
| revised_hybrid_safe9_linear_mse | 13.6772 | 6.5344 | 18.4652 |
| revised_hybrid_safe9_linear_huber | 16.8808 | 5.4007 | 16.1589 |
| revised_hybrid_safe9_sigmoid100_mse | 10.9520 | 5.3543 | 11.6587 |
| revised_hybrid_safe9_sigmoid100_huber | 8.5159 | 5.4080 | 10.4533 |
| incremental_temporal_hybrid_legacy20 | 13.4120 | 6.5042 | 16.1877 |
| temporal_order_negative_control_safe9 | 9.1075 | 5.5865 | 11.2486 |

## Incremental Temporal and Negative-Control Tests

| Comparison | MAE difference (candidate - control) | 95% paired bootstrap CI | Interpretation |
|---|---:|---:|---|
| revised temporal value: revised_hybrid_safe9_sigmoid100_huber vs tabular_mlp_safe9_capacity_matched | -36.8865 | [-38.8903, -34.7911] | candidate lower MAE |
| legacy temporal value: incremental_temporal_hybrid_legacy20 vs existing_hybrid_reproduction_legacy20 | -2.4072 | [-3.3683, -1.4331] | candidate lower MAE |
| temporal-order negative control: shuffled order vs revised_hybrid_safe9_sigmoid100_huber | 0.4639 | [0.3527, 0.5863] | candidate higher MAE |

## Residual Correlations

Residual correlations are calculated from mean OOF predictions. Lower correlation is diversity evidence, not sufficient evidence for an ensemble benefit.

| Pair | Correlation |
|---|---:|
| rf_safe9__xgb_safe9 | 0.9549 |
| rf_safe9__revised_hybrid_safe9_sigmoid100_huber | 0.3652 |
| tabular_mlp_safe9_capacity_matched__revised_hybrid_safe9_sigmoid100_huber | 0.7931 |
| existing_hybrid_reproduction_legacy20__incremental_temporal_hybrid_legacy20 | 0.8860 |

## Limitations and Recommendation

- Synthetic sequences cannot establish real-world screen-inactivity benefit.
- Current source data have no retained participant key in active artifacts; this is row-level exploratory validation, not participant-level validation.
- The historical test split was intentionally excluded from all experimental decisions and is already examined, not untouched confirmation data.
- Do not change production or select an ensemble from this report without separate approval and a new locked external or participant-aware cohort.
