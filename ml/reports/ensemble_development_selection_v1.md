# Development-Only Ensemble Selection v1

This evaluation uses only the existing cross-fitted development predictions. The previously explored 75-row test split was not read or used for model or weight selection.

## Cross-Validated Development Results

| Candidate | MAE | RMSE | R2 | Invalid outside 0-100 |
|---|---:|---:|---:|---:|
| rf_oof | 2.6239 | 3.7007 | 0.9272 | 0 |
| xgb_oof | 2.7664 | 3.8134 | 0.9227 | 0 |
| hybrid_oof | 5.3977 | 6.7667 | 0.7566 | 3 |
| current_fixed_0_2_0_3_0_5_oof | 3.6464 | 4.6926 | 0.8829 | 0 |
| equal_weight_oof | 3.1634 | 4.1797 | 0.9071 | 0 |
| nested_linear_stacking_oof | 2.7202 | 3.8235 | 0.9223 | 0 |
| nested_convex_blend_oof | 2.6310 | 3.7022 | 0.9271 | 0 |
| existing_full_oof_linear_stacking_optimistic | 2.6139 | 3.6842 | 0.9278 | 0 |

## Residual Distribution

Residuals are calculated as actual sleep efficiency minus predicted sleep efficiency. Values are percentage points.

| Candidate | Mean | SD | Median | 5th percentile | 95th percentile | Skew |
|---|---:|---:|---:|---:|---:|---:|
| rf_oof | 0.0481 | 3.7066 | 0.2106 | -6.6409 | 4.6836 | -0.0343 |
| xgb_oof | -0.1039 | 3.8183 | -0.1787 | -6.3469 | 5.5371 | -0.1132 |
| hybrid_oof | -0.0187 | 6.7780 | -0.7097 | -9.8673 | 12.0960 | 0.4469 |
| current_fixed_0_2_0_3_0_5_oof | -0.0309 | 4.7004 | -0.3840 | -7.8248 | 8.1331 | 0.2593 |
| equal_weight_oof | -0.0248 | 4.1867 | -0.0155 | -7.1463 | 6.5494 | 0.1557 |
| nested_linear_stacking_oof | -0.0132 | 3.8298 | -0.0462 | -6.7350 | 5.0803 | -0.0613 |
| nested_convex_blend_oof | 0.0116 | 3.7084 | 0.0284 | -6.5865 | 4.8472 | -0.0330 |
| existing_full_oof_linear_stacking_optimistic | 0.0000 | 3.6904 | -0.0128 | -6.4096 | 4.8543 | -0.0553 |

## Residual Correlation of Base Models

Lower correlation indicates more complementary error patterns. This is evidence of potential diversity, not proof that an ensemble improves accuracy.

| Pair | Residual correlation |
|---|---:|
| rf_oof__xgb_oof | 0.9395 |
| rf_oof__hybrid_oof | 0.5510 |
| xgb_oof__hybrid_oof | 0.5644 |

## Performance by Sleep-Efficiency Range

| Range | Rows | Candidate | MAE | RMSE |
|---|---:|---|---:|---:|
| low_under_70 | 114 | rf_oof | 2.9025 | 4.0061 |
| low_under_70 | 114 | xgb_oof | 3.3018 | 4.4305 |
| low_under_70 | 114 | hybrid_oof | 5.0552 | 6.1552 |
| low_under_70 | 114 | current_fixed_0_2_0_3_0_5_oof | 3.6807 | 4.6974 |
| low_under_70 | 114 | nested_linear_stacking_oof | 3.1483 | 4.2993 |
| low_under_70 | 114 | nested_convex_blend_oof | 2.9679 | 4.0588 |
| mid_70_to_under_85 | 123 | rf_oof | 2.5898 | 3.8039 |
| mid_70_to_under_85 | 123 | xgb_oof | 2.5617 | 3.6312 |
| mid_70_to_under_85 | 123 | hybrid_oof | 5.8274 | 7.4046 |
| mid_70_to_under_85 | 123 | current_fixed_0_2_0_3_0_5_oof | 3.7827 | 4.8823 |
| mid_70_to_under_85 | 123 | nested_linear_stacking_oof | 2.5864 | 3.7954 |
| mid_70_to_under_85 | 123 | nested_convex_blend_oof | 2.5497 | 3.7627 |
| high_85_and_above | 62 | rf_oof | 2.1792 | 2.7988 |
| high_85_and_above | 62 | xgb_oof | 2.1883 | 2.8069 |
| high_85_and_above | 62 | hybrid_oof | 5.1748 | 6.5101 |
| high_85_and_above | 62 | current_fixed_0_2_0_3_0_5_oof | 3.3132 | 4.2818 |
| high_85_and_above | 62 | nested_linear_stacking_oof | 2.1981 | 2.8169 |
| high_85_and_above | 62 | nested_convex_blend_oof | 2.1725 | 2.7786 |

## Model Contributions in Development Candidates

| Candidate | RF contribution | XGBoost contribution | Hybrid contribution |
|---|---:|---:|---:|
| Current fixed policy | 0.2000 | 0.3000 | 0.5000 |
| Nested linear stacking, mean coefficient (SD) | 0.7453 (0.1210) | 0.2688 (0.1378) | -0.0112 (0.0235) |
| Nested convex blend, mean weight (SD) | 0.8200 (0.1095) | 0.1800 (0.1095) | 0.0000 (0.0000) |

The nested linear coefficients include an intercept (mean -0.2053; SD 1.5415), so they are not constrained to sum to one. The convex blend is constrained to non-negative weights that sum to one.

## Selection Boundary

- RF, XGBoost, Hybrid, current fixed weights, and equal weights use the saved base-model OOF predictions directly.
- Nested linear stacking and nested convex blending are evaluated with a new outer five-fold split over the existing OOF matrix. Base models are not retrained.
- The full-OOF linear fit is reported only as an optimistic diagnostic for the existing stacking approach; it is not a valid selection result.

## Independent Holdout Assessment

- No fresh independent holdout remains in the existing small dataset. The original test rows have already been examined, and validation rows participate in existing development/training procedures.
- A future definitive comparison requires a newly locked external cohort or a new participant-level holdout established before retraining any candidate models.
