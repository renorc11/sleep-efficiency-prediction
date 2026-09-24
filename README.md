# Explainable Machine Learning for Lifestyle-Based Sleep Efficiency Prediction Using Behavioral and Screen Inactivity Signals

Full-stack academic project that predicts sleep efficiency (%) and sleep quality label using only lifestyle behavior and screen inactivity patterns. No EEG/ECG/wearables are used.

## Stack
- Frontend: Next.js 14 + TypeScript + Recharts
- Backend: FastAPI + SQLAlchemy + Alembic + SQLite
- ML: scikit-learn, XGBoost, PyTorch hybrid model, SHAP
- Deployment: Docker + docker-compose (optional)

## Key Capabilities
- Lifestyle + screen inactivity based prediction
- Inferred probable sleep time from >=30 minute inactivity window around nighttime
- 3-model pipeline:
  - Random Forest regressor
  - XGBoost regressor
  - Hybrid model (tabular + screen activity time-series via 1D CNN)
- End-to-end ML workflow:
  - data generation
  - preprocessing + feature engineering
  - train/val/test split
  - hyperparameter tuning (RF, XGB)
  - full metric evaluation
  - model comparison plots
- Explainability:
  - global importance endpoint
  - local SHAP-based top factors in prediction response
- Actionable recommendation generation in natural language

## Folder Structure
```text
.
├── backend/
│   ├── app/
│   │   ├── api/routes/
│   │   ├── core/
│   │   ├── db/
│   │   ├── schemas/
│   │   └── services/
│   ├── alembic/versions/
│   ├── tests/
│   ├── Dockerfile
│   └── requirements.txt
├── frontend/
│   ├── app/
│   ├── components/
│   ├── lib/
│   └── Dockerfile
├── ml/
│   ├── artifacts/
│   ├── data/raw/
│   ├── data/processed/
│   ├── reports/
│   └── scripts/
├── docs/
│   └── architecture.md
├── docker-compose.yml
└── README.md
```

## macOS Apple Silicon Setup (M-Series)
### 1. Create Python environment
```bash
python3 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -r backend/requirements.txt
```

### 2. Run ML pipeline (generates artifacts)
```bash
bash ml/scripts/run_pipeline.sh
```
`run_pipeline.sh` behavior:
- If `ml/data/raw/Sleep_health_and_lifestyle_dataset.csv` (or `ml/data/raw/source_notebooks/Sleep_health_and_lifestyle_dataset.csv`) exists, it uses your real Sleep Health dataset.
- Otherwise it falls back to synthetic data generation.

Outputs:
- Trained models: `ml/artifacts/`
- Comparison report: `ml/reports/model_comparison.json`
- Comparison chart: `ml/reports/model_comparison.png`
- SHAP summary: `ml/reports/shap_summary.png`

### 3. Start backend
```bash
cd backend
uvicorn app.main:app --reload --port 8000
```

### 4. Start frontend (new terminal)
```bash
cd frontend
cp .env.local.example .env.local
npm install
npm run dev
```
Frontend: [http://localhost:3000](http://localhost:3000)
Backend docs: [http://localhost:8000/docs](http://localhost:8000/docs)

## API Endpoints
- `GET /api/v1/health`
- `POST /api/v1/predict`
- `GET /api/v1/history`
- `GET /api/v1/explain/global`

## Prediction Input Format
See example payload:
- `ml/data/raw/example_prediction_payload.json`

## Use Real Sleep Health Dataset
Place the Kaggle CSV (`Sleep_health_and_lifestyle_dataset.csv`) in either:
- `ml/data/raw/`
- `ml/data/raw/source_notebooks/`

Then run:
```bash
bash ml/scripts/run_pipeline.sh
```
The pipeline will automatically build the training matrix from the real dataset and retrain all models.

## Integrate Additional Datasets Safely

The project includes a provenance-preserving integration pipeline for external lifestyle, screen-use, wearable, and clinical datasets. It inventories every source file, standardizes compatible sources, keeps unrelated cohorts separate, and reports missing targets, duplicates, invalid records, and unmatched keys.

```bash
python3 ml/scripts/integrate_external_datasets.py
```

The configured source folder is `~/Desktop/Final_Project_Dataset`. Read [the integration guide](docs/dataset_integration.md) before retraining: the current 20-feature lifestyle-and-screen model must not be trained with fabricated mappings or wearable/PSG fields.

Example cURL:
```bash
curl -X POST http://localhost:8000/api/v1/predict \
  -H 'Content-Type: application/json' \
  -d @ml/data/raw/example_prediction_payload.json
```

## Evaluation Metrics Used
### Classification
- Accuracy
- Precision
- Recall
- F1
- ROC-AUC

### Regression
- MAE
- RMSE
- R²

All metrics are computed in training scripts and saved under `ml/reports/`.

## Frontend UI Description
- Lifestyle input form (numeric behavior factors)
- Browser interaction tracker converts activity into 5-minute bins
- Result dashboard shows:
  - predicted sleep efficiency (%)
  - Good/Poor label
  - inferred sleep time
  - model comparison bars (RF/XGB/Hybrid/Ensemble)
  - top influencing factors
  - personalized recommendations

## Deployment
### Docker
```bash
docker compose up --build
```

### Simple cloud deployment guide
1. Deploy backend container to Render/Railway/Fly.io.
2. Set env vars: `ARTIFACTS_DIR`, `DATABASE_URL`.
3. Attach persistent volume for SQLite or switch to managed Postgres.
4. Deploy frontend on Vercel and point `NEXT_PUBLIC_API_BASE_URL` to backend URL.
5. Enable CORS for deployed frontend domain.

## Academic Notes
- This project is intentionally interpretable: lifestyle + inactivity features are human-understandable.
- SHAP local attributions are exposed per prediction for defensible recommendations.
- Hybrid model architecture demonstrates innovation by combining tabular + temporal behavior data.

## Quick Validation
```bash
pytest backend/tests -q
```
