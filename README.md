# Explainable Machine Learning for Lifestyle-Based Sleep Efficiency Prediction Using Behavioral and Screen Inactivity Signals

Full-stack academic project that estimates a sleep-efficiency index (%) and a Good/Poor label from lifestyle inputs and a 288-bin, five-minute screen-activity sequence. It explains every model's prediction and turns those explanations into evidence-grounded recommendations, using retrieval over a curated sleep knowledge base and a locally hosted BioMistral-7B model. No EEG, ECG or wearable inputs are used.

> **Read before citing results.** The training target is a sleep-efficiency index *constructed* from the Kaggle Sleep Health and Lifestyle dataset, not measured sleep efficiency. Several predictors are synthetic proxies, and the activity sequences are synthetic. The 75-row test split has been examined during development and is not an untouched hold-out. The deployed models are **same-night estimators** of that constructed index, not forecasts of the coming night; separate real-data forecasting experiments found no model better than a person's own sleep history (see [Coming-night forecasting experiments](#coming-night-forecasting-experiments-real-data)). See [Research status and limitations](#research-status-and-limitations).

## Stack
- Frontend: Next.js 14 + TypeScript + Recharts
- Backend: FastAPI + SQLAlchemy + Alembic + SQLite
- ML: scikit-learn (Random Forest), XGBoost, PyTorch (Hybrid MLP + 1D-CNN), SHAP
- Retrieval: BAAI/bge-small-en-v1.5 embeddings + ChromaDB (17-entry, source-traceable knowledge base)
- Language model: BioMistral-7B with a LoRA adapter, served locally by llama.cpp (OpenAI-compatible API)
- Android client: Kotlin + Jetpack Compose
- Deployment: local only (backend on `127.0.0.1:8000`, frontend on `localhost:3000`)

## Key Capabilities
- **Three models, always available:** Random Forest, XGBoost and Hybrid (tabular MLP + 1D-CNN over the 288-bin sequence), plus the deployed fixed ensemble `0.2 RF + 0.3 XGB + 0.5 Hybrid` clipped to 50-99.
- **Sleep-onset inference** from a run of at least 30 minutes of inactivity starting between 20:00 and 03:00. This is an indirect proxy, not proof of sleep.
- **Prediction diagnostics:** raw model outputs, an output-domain check (any value outside 0-100% is flagged, never silently altered), disclosure of ensemble clipping, and model disagreement.
- **Explainability:**
  - RF and XGBoost: TreeSHAP local explanations with reconstruction checks, and global mean-absolute SHAP from training and validation rows only.
  - Hybrid: Integrated Gradients over the tabular and temporal inputs (all 288 five-minute bins) with a completeness check.
  - The two methods are reported separately and never combined.
- **Automatic SHAP -> RAG -> BioMistral recommendations** after every prediction:
  - Factors come only from leakage-screened features where RF and XGBoost agree.
  - Each factor gets its own evidence retrieval.
  - BioMistral output is grammar-constrained (per-factor citations, no numbers) and then validated, with one retry and a deterministic fallback.
- **Sleep Assistant chat:** retrieval-grounded BioMistral answers with citation and safety validation, plus an evidence-only fallback.
- **Degraded operation:** predictions keep working when BioMistral or the knowledge base is unavailable.

## Folder Structure
```text
.
├── backend/                FastAPI app (api/routes, core, db, schemas, services), alembic, tests
├── frontend/               Next.js app (app/, components/, lib/)
├── SleepEfficiencyAndroidApp/  Android client
├── assistant/              BioMistral adapter files and conversion script
├── ml/
│   ├── artifacts/          Deployed models, scalers, SHAP background, ChromaDB, embeddings
│   ├── data/               raw, processed, knowledge base, external-dataset outputs
│   ├── experiments/        Development-only ablation experiments
│   ├── reports/            Metrics, audits, model comparison, MMASH study
│   ├── scripts/            Training, preprocessing, retrieval, evaluation scripts
│   └── tests/
├── docs/                   Architecture, paper, presentation, progress and handover notes
└── README.md
```

A former unused copy of `ml/` at `backend/app/services/ml/` was archived outside the repository on 2026-11-22; the path is ignored by Git.

## macOS Apple Silicon Setup (M-Series)
### 1. Python environment
```bash
python3 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -r backend/requirements.txt
```

### 2. ML pipeline (only if you need to regenerate artifacts)
The repository already contains trained artifacts. Retraining replaces them.
```bash
bash ml/scripts/run_pipeline.sh
```
If `ml/data/raw/Sleep_health_and_lifestyle_dataset.csv` (or `ml/data/raw/source_notebooks/Sleep_health_and_lifestyle_dataset.csv`) exists, the pipeline uses the Kaggle dataset; otherwise it falls back to synthetic data generation. Outputs: models in `ml/artifacts/` and reports in `ml/reports/`.

### 3. BioMistral server (optional but needed for generated text)
Serve the BioMistral GGUF model with llama.cpp's OpenAI-compatible server on `http://127.0.0.1:8081`, using the model name `BioMistral-Sleep-Assistant-V3`. Without it, the API still works and returns deterministic, evidence-backed fallbacks.

### 4. Backend
```bash
cd backend
uvicorn app.main:app --reload --port 8000
```
Startup loads the models and the knowledge base (about 5 s). `GET /api/v1/health` reports `knowledge_service` and `models` availability. The SQLite database is always `backend/sleep_app.db`, whichever directory you start from.

Useful environment variables:

| Variable | Default | Purpose |
|---|---|---|
| `DATABASE_URL` | `sqlite:///./sleep_app.db` (anchored to `backend/`) | Database |
| `ARTIFACTS_DIR` | `ml/artifacts` | Model artifacts |
| `BIOMISTRAL_SERVER_URL` | `http://127.0.0.1:8081` | llama.cpp server |
| `BIOMISTRAL_RECOMMENDATION_TIMEOUT` | `45` | Seconds allowed for recommendation generation |
| `BIOMISTRAL_CHAT_MAX_TOKENS` | `700` | Chat generation limit |
| `KNOWLEDGE_SERVICE_REQUIRED` | `false` | `true` makes startup fail if retrieval cannot load |
| `KNOWLEDGE_MIN_SIMILARITY` | `0.65` | Retrieval cosine-similarity threshold |
| `MODEL_WARMUP` | `true` | Load models at startup |
| `CORS_ORIGINS` | `http://localhost:3000,http://127.0.0.1:3000` | Comma-separated allowed frontend origins (e.g. your deployed URL) |
| `CORS_ORIGIN_REGEX` | any port on `localhost`/`127.0.0.1` | Local-development origins; set to empty to disable |

### 5. Frontend (new terminal)
```bash
cd frontend
cp .env.local.example .env.local
npm install
npm run dev
```
Frontend: [http://localhost:3000](http://localhost:3000) · API docs: [http://localhost:8000/docs](http://localhost:8000/docs)

## API Endpoints
- `GET /api/v1/health`: status plus knowledge-service and model availability.
- `POST /api/v1/predict`: prediction, model outputs, XGBoost TreeSHAP top factors, rule-based tips, `prediction_diagnostics`, `personalized_recommendations`, and `generated_explanation`/`explanation_status`. It takes about 20-45 s when BioMistral is running.
- `GET /api/v1/history?limit=20`: recent predictions (`limit` 1-200).
- `GET /api/v1/explain/global`: global XGBoost TreeSHAP importance (development rows only).
- `POST /api/v1/explain/local`: RF/XGBoost TreeSHAP and Hybrid Integrated Gradients for one input.
- `GET /api/v1/explain/model-metrics`: historical metrics.
- `POST /api/v1/chat`: Sleep Assistant (returns 503 when the knowledge base is unavailable).

Example payload: `ml/data/raw/example_prediction_payload.json`
```bash
curl -X POST http://localhost:8000/api/v1/predict \
  -H 'Content-Type: application/json' \
  -d @ml/data/raw/example_prediction_payload.json
```
`screen_activity_sequence` must contain exactly 288 values, each 0 or 1.

## Results

Regression metrics are primary. Accuracy thresholds both the constructed index and the regression output at 85 (Good). ROC-AUC is not reported: earlier versions computed it from a linear rescaling of the regression output, `(prediction - 50) / 50`, not from a classifier probability. `/explain/model-metrics` withholds it (the stored artifact metadata is unchanged) and reports regression and threshold-derived metrics as separate groups. An earlier comparison with a clinical base paper's AUROC has been withdrawn (`docs/withdrawn/base_paper_comparison/WITHDRAWN.md`).

**Historical test split** (75 rows; examined during development, so not an untouched hold-out):

| Model | MAE | RMSE | R² | Accuracy |
|---|---:|---:|---:|---:|
| Random Forest | 2.64 | 3.61 | 0.933 | 0.960 |
| XGBoost | 3.07 | 4.25 | 0.907 | 0.947 |
| Hybrid | 4.98 | 6.17 | 0.804 | 0.920 |

The 374 rows hold only 132 distinct source profiles, and 66 of the 75 test rows share a profile with training rows. Grouping cross-validation by profile moves RF only from MAE 2.66 to 2.80 (R² 0.925 to 0.917), so duplication inflates these figures slightly; the constructed target is the main reason they are high.

**Development cross-validation** (299 development rows, test split unused; `ml/reports/ensemble_development_selection_v1.md`):
- RF has MAE 2.62, XGBoost 2.77 and the Hybrid 5.40.
- The **deployed fixed ensemble has MAE 3.65, worse than RF alone.** It is kept for backward compatibility, not as the best validated policy.

**Leakage ablation** (`ml/experiments/hybrid_ablation_v2/report.md`):
- RF scores MAE 2.67 with 9 leakage-screened features and 2.73 with all 20.
- A reproduction of the Hybrid scores MAE 14.66, and adding temporal input did not beat a tabular-only network.

## Research Status and Limitations
- **The target is constructed:** `10 x Quality of Sleep + 2.3 x (Sleep Duration - 6.5)`, minus stress and apnoea terms, plus noise. High accuracy mainly reflects that this index is close to a function of the source stress and sleep-quality fields; it is not real-world predictive validity.
- **Same-night, not coming-night:** `previous_sleep_hours` was generated from the *same* night's duration (r = 0.83) and the activity sequences from the same night's sleep window, so the deployed models estimate the night described by their inputs; they do not forecast a future night.
- **Leakage-prone predictors:** `previous_sleep_hours`, `naps_count`, `naps_total_min`, `nap_sleep_debt_proxy`, `mood_score`, and all sequence-derived features. They remain in the deployed 20-feature contract for artifact compatibility but are never used to drive recommendations.
- **Activity sequences are synthetic.** The models have not been trained or validated on real browser or Android activity.
- **No participant-level validation:** there are no repeated participants or retained IDs, and no external validation.
- **Hybrid output is unconstrained** (4 stored predictions exceed 100%). Out-of-range values are reported with a warning, not clipped.
- **Sleep-onset inference differs between training and serving**, and serving depends on the request time. The measured effect is small (RF test MAE 2.641 vs 2.648), but it is a known open issue. Investigation and proposed fix: `docs/SLEEP_ONSET_SKEW_INVESTIGATION.md`.
- **No authentication or per-user isolation:** `/history` returns all records. Design (not implemented): `docs/AUTH_PRIVACY_DESIGN.md`.
- **Retrieval:** the knowledge base has 17 entries; recall@3 was 0.967 on 10 judged questions, a regression check only. Some factors (work hours, hydration) have no evidence entry.
- **Language-model safety checks are rule-based**, not a guarantee of medical safety. Recommendation confidence is a rule, not a calibrated probability.
- Feature attributions describe model behaviour, not causes. Nothing here is a diagnosis.

## MMASH Wearable Feasibility Study (exploratory)
The deployed application does not accept wearable data. Optional wearable input (for example through Android Health Connect) is a possible future direction only: it is not implemented, and its consent and permission requirements are described in [docs/AUTH_PRIVACY_DESIGN.md](docs/AUTH_PRIVACY_DESIGN.md). A separate exploratory MMASH experiment (22 episodes from 21 participants, participant-grouped validation) found that wearable models did not consistently outperform a mean-prediction baseline. It is not a validation of the main system. The MMASH data are not version-controlled; to reproduce, place an authorized copy at `ml/data/processed/mmash/mmash_wearable_features_final.csv` and run:
```bash
.venv/bin/python ml/scripts/train_mmash_wearable.py
```

## Coming-Night Forecasting Experiments (real data)
Separate from the deployed models, `ml/forecasting/` asks whether the coming night can be forecast from information available before it begins. Each dataset keeps its own participants and target, and they are never pooled:

| Dataset | Target | Cohort | Cutoff |
|---|---|---|---|
| Fitabase (Fitbit exports, 2016) | Device-derived sleep ratio (asleep / session minutes), a proxy, **not** PSG efficiency | 20 participants, 776 nights | 20:00 local (21:00 sensitivity) |
| Walch Apple Watch + PSG | **PSG-measured** sleep efficiency, one lab night each | 31 subjects | 30 min before the PSG recording |
| StudentLife (2013) | **Self-reported** sleep hours | 49 participants, 1,241 nights | 21:00 America/New_York (assumed) |

Missing observations stay missing (with coverage indicators), validation is participant-grouped and chronological, hyperparameters are fixed in advance, and differences are tested with a participant-level bootstrap.

**Findings** (`ml/experiments/forecasting_v1/REPORT.md`, `ml/experiments/forecasting_v2/REPORT.md`):
- No Random Forest or XGBoost model reliably beat the participant's own historical average. Fitabase: personal mean MAE 2.28 vs best model 2.39 (worse, 95% CI +0.03 to +0.21). StudentLife: 0.785 h vs 0.776 h (no difference). Walch: inconclusive (31 single nights).
- Activity, heart-rate, phone and stress features added no detectable value over sleep history.
- Among personal-history rules (expanding mean, last-7/14 means, exponentially weighted mean, bounded trend), none met the pre-specified adoption rule on Fitabase. On StudentLife the 14-night mean met it, but the gain is 1-3 minutes on a half-hour response scale, so the expanding mean is retained.
- Results on one target say nothing about another: the Fitabase proxy does not validate PSG efficiency.
- A forecast endpoint was **not** added: the application stores no user identity and no observed sleep outcomes, so there is no per-user history to forecast from safely.

Reproduce: `backend/.venv/bin/python -m ml.forecasting.evaluate` and `backend/.venv/bin/python -m ml.forecasting.evaluate_personal` (datasets located through `SLEEP_DATASET_DIR`; see [Research dataset location](docs/dataset_integration.md#research-dataset-location)).

## Integrate Additional Datasets Safely
`python3 ml/scripts/integrate_external_datasets.py` inventories, standardizes and quality-reports external lifestyle, screen-use, wearable and clinical datasets, keeping unrelated cohorts separate. The dataset folder is configured locally (`--source-root`, `SLEEP_DATASET_DIR`, or the untracked link `ml/data/external_raw/final_project_dataset`); no machine-specific path is stored in Git. Read [the integration guide](docs/dataset_integration.md) before retraining: the 20-feature model must not be trained with fabricated mappings or wearable/PSG fields.

## Testing
From the project root:
```bash
PYTHONPATH=backend:. .venv/bin/python -m pytest backend/tests ml/tests -q
# Optional: call the live BioMistral server
RUN_LIVE_BIOMISTRAL=1 PYTHONPATH=backend:. .venv/bin/python -m pytest backend/tests/test_phase3_rag_llm.py -k live -q
# Frontend type check and production build
cd frontend && npx next build
```
Latest run (2026-11-22): 232 passed, 1 skipped (the opt-in live test, which passes when enabled), including 27 forecasting tests in `ml/tests/test_forecasting.py` and `ml/tests/test_personal_baselines.py` and 8 report-path tests in `ml/tests/test_integration_report_paths.py`. The forecasting smoke tests run only when the research datasets are configured (see [Research dataset location](docs/dataset_integration.md#research-dataset-location)). The frontend production build passed on 2026-11-22. The Android build has not been verified offline (Gradle dependencies are not cached).

## Deployment
The project runs locally as a single-user application: start the backend and frontend as described above. Docker files were removed on 2026-11-22 because they were unused and untested. There is no hosted deployment. Hosting it for other users would first require the multi-user controls described in [docs/AUTH_PRIVACY_DESIGN.md](docs/AUTH_PRIVACY_DESIGN.md).

This is an academic prototype: it has not been security-reviewed for production use.

## Further Documentation
- [docs/architecture.md](docs/architecture.md): system and recommendation-pipeline architecture
- [docs/PROJECT_PROGRESS.md](docs/PROJECT_PROGRESS.md): phase-by-phase verified findings and tests
- [docs/CODEX_HANDOVER.md](docs/CODEX_HANDOVER.md): handover and open items
- [docs/paper/springer_sleep_efficiency.tex](docs/paper/springer_sleep_efficiency.tex): research paper
