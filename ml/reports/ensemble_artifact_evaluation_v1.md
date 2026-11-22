# Ensemble Artifact Evaluation v1

This report is read-only. It evaluates existing artifacts against the existing held-out test indices; it does not retrain or tune any model.

## Held-Out Test Split

- Test rows: 75
- Target range: 50.00-98.00%

## Comparable Test Results

| Artifact set / prediction | MAE | RMSE | R2 | Range | Invalid outside 0-100 |
|---|---:|---:|---:|---:|---:|
| Deployed: rf | 2.6412 | 3.6059 | 0.9330 | 54.25-93.91 | 0 |
| Deployed: xgb | 3.0720 | 4.2481 | 0.9070 | 51.93-94.46 | 0 |
| Deployed: hybrid | 4.9758 | 6.1650 | 0.8041 | 58.96-101.33 | 2 |
| Deployed: fixed_weight_0_2_0_3_0_5_raw | 3.4842 | 4.4473 | 0.8981 | 56.63-97.60 | 0 |
| Stacking: rf | 2.6666 | 3.6959 | 0.9296 | 53.99-94.00 | 0 |
| Stacking: xgb | 2.9946 | 4.0307 | 0.9163 | 53.03-94.52 | 0 |
| Stacking: hybrid | 4.7510 | 5.7638 | 0.8288 | 58.06-101.63 | 2 |
| Stacking: fixed_weight_0_2_0_3_0_5_raw | 3.3131 | 4.2098 | 0.9087 | 56.72-97.54 | 0 |
| Stacking: stacking_linear_meta_model | 2.6847 | 3.7230 | 0.9286 | 53.63-94.19 | 0 |

## Interpretation

- The deployed fixed-weight ensemble and stacking ensemble use different saved base-model artifacts, so they must not be mixed.
- The stacking meta-model is a linear model over its own out-of-fold-trained artifact set, not a replacement weight vector for the deployed models.
- Output-domain checks are reported separately from model error. No clipping was applied during this evaluation.

## Limitations

- This is a single stored test split, not external validation.
- The deployed fixed-weight models and stacking models were trained with different development-data usage and Hybrid CNN definitions.
- No weight tuning, retraining, calibration, or production-path change was performed.
