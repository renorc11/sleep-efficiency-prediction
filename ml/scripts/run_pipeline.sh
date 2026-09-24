#!/usr/bin/env bash
set -euo pipefail

export MPLBACKEND=Agg
export OMP_NUM_THREADS=1
export KMP_AFFINITY=disabled

if [ -f "ml/data/raw/Sleep_health_and_lifestyle_dataset.csv" ] || [ -f "ml/data/raw/source_notebooks/Sleep_health_and_lifestyle_dataset.csv" ]; then
  echo "[Pipeline] Found Sleep Health CSV. Preparing real-data training set..."
  python3 ml/scripts/prepare_real_sleep_health_data.py
else
  echo "[Pipeline] Real CSV not found. Falling back to synthetic dataset generation."
  python3 ml/scripts/generate_synthetic_data.py
fi

python3 ml/scripts/preprocess.py
python3 ml/scripts/train_tabular_models.py
python3 ml/scripts/train_hybrid_model.py
python3 ml/scripts/compare_and_package.py
python3 ml/scripts/generate_shap_report.py

echo "Pipeline complete. Artifacts in ml/artifacts"
