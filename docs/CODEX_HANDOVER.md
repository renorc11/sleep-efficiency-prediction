# Handover

Last updated: 2026-11-21. All seven master-plan phases are complete; see
`docs/PROJECT_PROGRESS.md` for the verified, phase-by-phase evidence.

## Ground Rules (unchanged)
- Branch `main`, base commit `b941546`. The working tree is intentionally dirty
  and uncommitted. Do not reset, clean, stash or overwrite it.
- Never inspect or modify the top-level `REST/` directory.
- Do not retrain models, change checkpoints, scalers, datasets or splits,
  change ensemble weights, add dependencies, or change the DB schema without
  explicit approval.
- Never present synthetic activity sequences as real device data, and never
  call Integrated Gradients "SHAP".

## Run
```bash
cd "<repository root>"
source .venv/bin/activate
# Optional: llama.cpp server for BioMistral on http://127.0.0.1:8081
cd backend && uvicorn app.main:app --reload --port 8000     # DB: backend/sleep_app.db
cd frontend && npm run dev
```
Tests (from the project root):
`PYTHONPATH=backend:. .venv/bin/python -m pytest backend/tests ml/tests -q`.
Last run (2026-11-22): 232 passed, 1 skipped. Add `RUN_LIVE_BIOMISTRAL=1` for
the live test, which passed earlier. The frontend production build passes; the
Android build is not verified offline.

## Architecture (see docs/architecture.md)
- **Prediction** (`services/model_service.py`): RF, XGBoost, Hybrid, fixed
  ensemble 0.2/0.3/0.5 clipped to 50-99, plus `build_prediction_diagnostics`
  (domain, clipping, disagreement). Inputs are validated: finite values and a
  binary 288-bin sequence.
- **Explanations**: `explain_rf`/`explain_xgb` (TreeSHAP + reconstruction),
  `explain_hybrid` (Integrated Gradients + completeness), and
  `global_tree_shap` (train and validation rows only, via `.iloc`).
- **Recommendations**:
  - `recommendation_context_service.py` picks the signals: the safe9
    features, RF/XGB agreement, exclusions.
  - `personalized_recommendation_service.py` handles retrieval, BioMistral
    JSON-schema generation, validation, repair, retry and the fallback;
    `recommend_for_prediction` is the entry point.
  - Invoked by `routes/predict.py` after the record is saved.
  - `prediction_explanation_service.py` is legacy: kept and tested, but no
    longer called by the route.
- **Chat**: `chat_service.py` with `llm_output_validation.py`; invalid replies
  become an evidence-only fallback.
- **Startup**: the `main.py` lifespan does create_all, loads the knowledge
  service (degraded mode unless `KNOWLEDGE_SERVICE_REQUIRED=true`) and warms
  the models.

## Open Items (ordered)
1. **Sleep-onset training/serving skew (user-deferred; must be fixed before
   completion).** `ml/scripts/preprocess.py` and
   `backend/app/services/inactivity_service.py` use different rules. Serving
   depends on `datetime.now()`, so identical inputs gave 64.87/64.62/64.58 at
   different times. Fixing it needs approval for an inference change or
   retraining. Related: the Android sequence starts at midnight, and the
   Android blue-light estimate uses a third inference variant.
2. **Knowledge-base expansion (user-deferred).** Add real, verifiable
   evidence (for example phone use, blue light, work hours, hydration, sleep
   aids) as a new dataset version, then re-embed and refresh the relevance
   judgments.
3. **Android build verification.** The offline Gradle build fails because
   AndroidX artifacts were never cached on this machine (this was true before
   any change). `ApiClient.kt` and `MainScreen.kt` edits are not yet
   compiled; the collector was compiled standalone. Build once in Android
   Studio, or approve a dependency download. The app ignores
   `personalized_recommendations`.
4. **Paper.** `docs/paper/springer_sleep_efficiency.tex` was rewritten but
   not compiled (no LaTeX installed here). Verify the Kaggle citation (marked
   `% VERIFY`) and fill in the author placeholders.
5. **Cleanup decisions for the user:**
   - ~~Delete the unused `backend/app/services/ml/` copy~~ Resolved 2026-11-22:
     archived outside the repository and ignored by Git. Still open: the stray
     root `sleep_app.db` (1 row)?
   - Delete build outputs: `frontend/.next`, Android `build/`, `tmp-icons/`?
   - ~~Docker: mount `./ml:/ml`~~ Resolved 2026-11-22: the unused Docker
     files were removed; the project runs locally only.
6. **Research follow-ups (need approval):** retrain and re-select the
   ensemble on leakage-screened features with a pre-registered hold-out;
   constrain or replace the Hybrid output; persist personalized
   recommendations (requires a migration).
