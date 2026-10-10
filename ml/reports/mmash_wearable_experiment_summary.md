# MMASH wearable sleep-efficiency experiment

## Objective

Evaluate whether a fixed, domain-driven set of wearable-derived features can predict sleep efficiency in the MMASH dataset.

## Data and quality handling

The dataset contained 22 sleep episodes from 21 participants. One participant contributed two episodes; all validation splits kept that participant’s episodes together.

The core experiment included all 22 episodes and used nine predefined features covering pre-sleep activity, screen exposure, actigraphy, heart rate, and posture.

The HRV extension added SDNN, RMSSD, and pNN50. One record had an unreliable successive-RR pattern and unavailable HRV features. It remained in the core experiment but was excluded from the HRV extension rather than having its HRV values replaced with zero.

## Models and validation

The evaluated models were:

- Dummy mean regressor
- Ridge regression
- Random Forest regression

Primary evaluation used five-fold participant-grouped cross-validation (`GroupKFold`). A leave-one-participant-out analysis was used as a sensitivity check. Metrics were out-of-fold mean absolute error (MAE) and root mean squared error (RMSE), in sleep-efficiency percentage points.

## Results

### Five-fold participant-grouped cross-validation

| Experiment | Model | MAE | RMSE | R² |
|---|---:|---:|---:|---:|
| Core wearable | Dummy mean | 5.796 | 6.684 | -0.028 |
| Core wearable | Ridge | 5.756 | 6.769 | -0.055 |
| Core wearable | Random Forest | 6.274 | 7.266 | -0.215 |
| Core plus HRV | Dummy mean | 5.954 | 6.854 | -0.035 |
| Core plus HRV | Ridge | 8.205 | 9.913 | -1.166 |
| Core plus HRV | Random Forest | 6.148 | 7.473 | -0.231 |

### Leave-one-participant-out sensitivity analysis

| Experiment | Model | MAE | RMSE |
|---|---:|---:|---:|
| Core wearable | Dummy mean | 5.977 | 6.926 |
| Core wearable | Ridge | 6.743 | 7.991 |
| Core wearable | Random Forest | 6.204 | 7.574 |
| Core plus HRV | Dummy mean | 6.217 | 7.097 |
| Core plus HRV | Ridge | 7.687 | 8.688 |
| Core plus HRV | Random Forest | 6.813 | 8.086 |

## Interpretation

Neither wearable-feature model consistently outperformed the mean-prediction baseline. Adding HRV features did not improve predictive performance. Results were unstable across validation folds, particularly for the HRV-augmented Ridge model.

This small sample is therefore insufficient to support a stable, generalizable sleep-efficiency prediction model. The findings should be treated as exploratory feasibility results rather than evidence of clinical or behavioral effects.

## Recommended next steps

Do not tune hyperparameters, select features using this dataset, or interpret model feature importance. These steps would likely overfit the available 22 episodes.

A future predictive study should use a substantially larger participant-level sample, prespecify predictors and evaluation metrics, retain participant-aware validation, and reserve an external cohort or held-out dataset for final assessment.
