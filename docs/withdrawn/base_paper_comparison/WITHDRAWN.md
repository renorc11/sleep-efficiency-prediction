# Withdrawn: Base-Paper Comparison (2026-11-22)

The files in this folder compared the project with **JMIR 2023, e46520** ("Predicting the Risk of Sleep Disorders Using a Machine Learning–Based Simple Questionnaire"). They are kept for the record only and should not be cited.

## Why it is withdrawn

1. **The project's ROC-AUC was not a valid metric.** It was computed by rescaling the regression output, `(prediction - 50) / 50`, and treating the result as a probability. The model is a regression model, not a classifier. The value is no longer returned by `/api/v1/explain/model-metrics` or shown in the web or Android apps.
2. **The tasks are not comparable.**
   - The reference paper classifies clinically assessed sleep-disorder risk (OSA, COMISA, insomnia) in clinical cohorts.
   - This project estimates a sleep-efficiency index *constructed* from a Kaggle table, with synthetic proxies and synthetic activity sequences.
   - Different targets, populations, data and evaluation designs make a side-by-side AUROC comparison unsupported.
3. **No replacement comparison is given,** because no comparable experiment exists.

The `base_paper_disorder_bars` charts only re-plot figures reported by the reference paper. They are withdrawn together with the comparison because their only purpose was that comparison.

## Contents

| File | Status |
|---|---|
| `BasePaper_vs_YourProject_Comparison_Report.txt` | Withdrawal notice added at the top; original text unchanged below it |
| `BasePaper_vs_YourProject_Comparison_Report.pdf`, `..._WithCharts.pdf` | Unchanged binary copies of the withdrawn comparison; covered by this notice |
| `charts/base_vs_project_auc_bar.*`, `charts/results_section_base_vs_project_graph.*`, `charts/base_paper_disorder_bars.*` | Withdrawn charts, unchanged |
| `charts/generate_base_vs_project_report.py`, `charts/generate_results_section_graph.py` | Now exit immediately with a withdrawal message, so the comparison cannot be regenerated |

`docs/base_paper.pdf` (the reference paper itself) is unchanged and was not moved.

## Research context

The project's research reference is the systematic review **"Machine-Learning-Based Approaches for Sleep Stage Classification Utilising a Combination of Physiological Signals: A Systematic Review"** (Almutairi et al., 2023).
- It is background context on sleep-stage classification from physiological signals only.
- It does not validate this project's sleep-efficiency estimator or its coming-night forecasting experiments.
- No numerical comparison with it is made.
- The review is not stored in this repository.

## Where the valid results are

- Regression results of the deployed models: `README.md` ("Results").
- Real-data coming-night forecasting experiments, including negative results: `ml/experiments/forecasting_v1/REPORT.md` and `ml/experiments/forecasting_v2/REPORT.md`.

## Not covered by this withdrawal

Two older dashboard screenshots that showed a ROC_AUC bar, `docs/charts/ui_crops/classification_metrics_crop.png` and `regression_metrics_crop.png`, were removed from the repository on 2026-11-22 (`git rm`; recoverable from Git history). No replacement screenshots were generated.

## Restore

The files were moved with `git mv`, so history is preserved. `git log --follow <file>` shows their history.
