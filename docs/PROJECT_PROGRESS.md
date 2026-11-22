# Project Progress

## Current Status

- All seven master-plan phases completed (2026-11-20/21). Two user-deferred
  items remain open (see Current Work), plus Android build and paper
  compilation checks.
- Repository branch: `main`.
- Base commit: `b941546 Integrate BioMistral sleep assistant`.
- Working tree: intentionally dirty; existing user changes are preserved.

## Completed Modules

- Three-model prediction pipeline: RF, XGBoost, and Hybrid MLP + 1D-CNN.
- Screen-inactivity inference and engineered feature construction.
- SQLite prediction history.
- RF and XGBoost TreeSHAP local explanations, reconstruction checks, and
  deterministic development-data global mean-absolute SHAP rankings.
- Hybrid MLP + 1D-CNN direct PyTorch Integrated Gradients with separate
  20-feature tabular and 288-bin temporal attributions, five-minute time
  mapping, and combined-branch completeness checks.
- Knowledge retrieval service and BioMistral explanation integration.
- Next.js user interface and Sleep Assistant.
- Ensemble, Hybrid, dataset-suitability, and development-only ablation reports.

## Current Work

- Master-plan Phases 1-7 completed. Open items are listed in
  `docs/CODEX_HANDOVER.md`.
- **Deferred by user decision:** expand the knowledge base with real,
  verifiable evidence (for example phone use and blue light, and sleep aids).
  This needs a new dataset version, re-embedding, and new relevance
  judgments. It is not part of the current phases.
- **Deferred by user decision (must be fixed before completion):** the
  training/inference sleep-time skew (Phase 1 defect 2). Current behaviour
  is kept and documented; the fix needs approval for an inference change or
  retraining.

## Master-Plan Phase 7: Testing and Research Documentation (2026-11-21)

### Final test runs

- `PYTHONPATH=backend:. .venv/bin/python -m pytest backend/tests ml/tests -q`:
  **188 passed, 1 skipped**.
- `RUN_LIVE_BIOMISTRAL=1 ... -k live`: **1 passed** against the running
  BioMistral server.
- The frontend `next build`, including the type check, passes.
- Not run: Android build (dependencies not cached), LaTeX compilation (no
  TeX installed), browser rendering.

### Documentation updated

- `docs/paper/springer_sleep_efficiency.tex`: rewritten.
  - The previous source used doubled backslashes on every command and could
    not compile; the structure is now checked (balanced environments and
    braces, every citation has a reference).
  - Content now covers: target construction, predictor provenance and
    leakage, synthetic sequences, real sleep-inference rules and the skew,
    historical vs development vs ablation results, ensemble and Hybrid
    limitations, explainability methods, RAG/BioMistral pipeline and
    validation, development observations, MMASH as exploratory only, and
    limitations.
  - New references are limited to established method papers; the Kaggle
    dataset reference is marked `% VERIFY`.
- `README.md`:
  - Reorganized, with a prominent data caveat and an environment-variable
    table.
  - Lists all endpoints and the historical, development and ablation
    results.
  - Adds the limitations, MMASH as exploratory, testing commands, and the
    Docker degraded-mode note.
- `docs/architecture.md`: the new flow, degraded modes, explainability
  methods, and the DB schema note.
- `docs/presentation/...PPT_Content.md`: slides 7-16 corrected:
  - Real vs synthetic data stated accurately.
  - Results table with caveats; Hybrid no longer presented as a result.
  - Viva answers corrected, including that the full input payload is stored.
- `docs/CODEX_HANDOVER.md` and `docs/PROJECT_COMPLETION_PLAN.md`: current
  state and ordered open items.
- Not changed: the existing model-comparison charts and PDF reports in
  `docs/` and `ml/reports/`. Regenerating them was out of scope; they show
  historical numbers and should be read with the caveats above.

## Master-Plan Phase 6: Frontend and Android Integration (2026-11-20)

No dependency was installed, upgraded, or downloaded.

### Frontend (Next.js 14)

- New `lib/types.ts` mirrors the backend Pydantic models; new fields are
  optional, so an older backend still renders. `ResultDashboard` re-exports
  `PredictionResult`, so existing imports are unchanged.
- New `components/PersonalizedInsights.tsx`, shown below the existing model
  chart:
  - A generated or fallback status banner, and the summary.
  - A model-agreement panel: raw RF/XGB/Hybrid/unclipped-ensemble values,
    spread and level, the fixed ensemble policy, and out-of-domain or
    clipping warnings.
  - Recommendation cards: priority, confidence, action, reason, the user's
    value, RF/XGB TreeSHAP and Hybrid IG shown separately, evidence ID,
    strength and sources, wording source, and the safety note.
  - A TreeSHAP chart of agreed factors (RF vs XGBoost), with disagreements
    listed.
  - A separate Hybrid Integrated Gradients panel (lifestyle inputs and
    influential hours), with the synthetic-data disclosure.
  - Excluded factors with reasons, linked sources, and the uncertainty list.
- Existing charts and metrics are kept. Additions:
  - A disclosure on the historical metrics: examined 75-row split,
    constructed target, synthetic sequences, regression metrics primary,
    85% threshold.
  - The heading names XGBoost TreeSHAP for the legacy factors.
  - Rule-based tips are relabelled when personalized recommendations exist.
- `page.tsx` shows a loading note (up to about 45 s, with the fallback
  explained) and an alert role on errors.
- `ScreenTracker` discloses that activity comes from this browser tab only
  and that the models were trained on synthetic sequences.
- `SleepAssistant` takes the latest prediction and posts its explanation
  and recommendations into the chat once, labelled as BioMistral-validated
  or deterministic.

### Frontend verification

- `next build` (including the type check) passes before and after the
  changes. The `/` route grew from 109 kB to 113 kB.
- Every field path the new UI reads was checked against a real
  `POST /predict` response from the live backend (BioMistral generated):
  none missing.
- Not done: visual rendering in a browser, and browser end-to-end tests
  (none exist in the project).

### Android findings and changes

- **Critical regression fixed.** OkHttp's default 10 s read timeout would
  fail every `/predict` now that it takes 20-45 s. The client now uses
  connect 10 s, read 90 s, and call 100 s.
- **Privacy.** `HttpLoggingInterceptor` logged full bodies (usage timeline
  and lifestyle answers) in every build. It now logs at BASIC level in debug
  builds only.
- **Collector undercounting.** Only the bin where an app came to the
  foreground was marked, so one hour in an app looked like 55 minutes of
  inactivity. Full foreground intervals (resume to pause; open intervals to
  now) are now marked. A pause with no resume today is ignored rather than
  assumed active.
- The usage-access card now discloses what is collected, where it is sent,
  and the synthetic-training limitation.
- Not changed (documented): the sequence starts at local midnight, so a
  bedtime before midnight is missed. The app's own blue-light estimate uses
  a third variant of sleep inference. Phone and server time zones must
  match for the backend's clock masking. The app ignores
  `personalized_recommendations`.

### Android verification

- `./gradlew --offline :app:compileDebugKotlin` (JDK 21 from Android
  Studio) **fails before any change**: about 20 declared AndroidX artifacts
  are not in the local Gradle cache. Building needs a one-time download of
  the project's already-declared dependencies; **not done without
  approval**.
- `ScreenActivityCollector.kt` and `UsageAccessHelper.kt` were compiled
  successfully with the Kotlin 2.2.0 compiler bundled in Gradle 9.1.0
  against `android-35/android.jar` (output in the scratchpad; only
  deprecation warnings).
- `ApiClient.kt` and `MainScreen.kt` could not be compiled here: OkHttp and
  Compose are not cached.

## Master-Plan Phase 5: Backend Integration (2026-11-20)

No dependency, migration, or DB-schema change. Neither existing database
file was modified (verified: `backend/sleep_app.db` still has 73 rows; the
root `sleep_app.db` still has 1).

### Findings and changes

- **Two databases.** `sqlite:///./sleep_app.db` resolved against the current
  directory, so starting from `backend/` (as documented) and from the root
  used different files: `backend/sleep_app.db` (73 predictions,
  2026-02-09 to 2026-11-20) and a stray root `sleep_app.db` (1 row).
  `config.resolve_database_url` now anchors relative SQLite paths to
  `backend/`; absolute and non-SQLite URLs are unchanged. Both startup
  directories now resolve to `backend/sleep_app.db`. The root file is left
  in place for the user to decide.
- **Alembic.** The DB has no `alembic_version` table (it was built by
  `create_all`). The `0001` migration columns match `PredictionRecord`
  exactly, so there is no drift. Startup still uses `create_all`.
- **Startup.** The deprecated `@app.on_event("startup")` hook is replaced by
  a `lifespan` handler (`initialize_app_state`) with the same behaviour.
  Model artifacts and the RF explainer are warmed at startup
  (`MODEL_WARMUP`, default true); a failure is logged and never blocks
  startup.
- `/health` adds `models: loaded|not_loaded` (additive).
- `/history` `limit` is bounded to 1-200 (previously unbounded, including
  negative values).
- **Docker note (no change made, untested here).** docker-compose mounts
  only `backend/` and `ml/artifacts`, so the knowledge dataset and
  `ml/scripts` are absent in the container and the API starts in degraded
  mode. Mounting `./ml:/ml` would likely fix this, but it was not tested.
  BioMistral is not part of compose.

### Tests executed

- Full suite: **188 passed, 1 skipped**. New file:
  `backend/tests/test_phase5_api_integration.py` exercises every endpoint
  through one lifespan with the real models: predict -> history round trip,
  persistence, five invalid payloads (422), explain global/local, model
  metrics, chat, OpenAPI coverage, and DB URL resolution.
- Live, with `cd backend && uvicorn app.main:app` (scratch DB), live
  BioMistral and ChromaDB:
  - Ready in about 5 s; `/health` reported knowledge available and models
    loaded.
  - `/predict` returned 200 twice (32 s fallback; 34 s generated).
  - `/chat` returned 200 in 6 s; `/history` returned both records.

## Master-Plan Phase 4: Automatic SHAP -> RAG -> BioMistral Recommendations (2026-11-20)

The pipeline runs inside `POST /predict` after the ML prediction has been
computed and persisted. It has a 45 s budget
(`BIOMISTRAL_RECOMMENDATION_TIMEOUT`) and returns the new additive
`personalized_recommendations` field. It also fills the existing
`generated_explanation`/`explanation_status` fields, replacing the old
free-text call, which failed validation in 3 of 3 live runs. That leaves one
LLM call per prediction instead of two. `prediction_explanation_service.py`
is kept, with its tests, but the route no longer calls it. No dependency,
model, knowledge-data, or DB-schema change was made. The DB still persists
only the deterministic `recommendations`.

### Architecture

1. **Context** (`recommendation_context_service.py`): runs RF/XGB TreeSHAP and
   Hybrid IG (64 steps) and checks each explanation reproduces its prediction.
   - Factors must be in the ablation-v2 `safe9` set, have RF and XGB agreeing
     in direction, and have a mean |TreeSHAP| of at least 0.1.
   - Target-derived features (previous sleep, naps, mood, sleep-debt proxy)
     and sequence-derived features are reported as excluded, never used as
     drivers.
   - IG is reported beside TreeSHAP and never combined with it. The temporal
     IG is summarised per hour with a synthetic-data and "not proof of sleep"
     note.
   - Up to 5 candidates; "maintain" items are added when fewer than 3 factors
     lower the prediction.
2. **Retrieval**: one query per factor. Evidence is accepted only if the
   factor's own knowledge entry is retrieved above the similarity threshold.
   Neighbouring topics are rejected because bge similarities are compressed.
3. **Generation**: BioMistral words the top 3 factors under a llama.cpp JSON
   schema with one slot per factor. Each slot's `source_ids` enum holds only
   that factor's own retrieved sources. Text fields use the JSON-safe ASCII
   pattern `^[A-Za-z ,.;:'()/&-]{1,N}$`, so there are no digits, percent
   signs, or quotes. The model receives no numbers and no assistant-facing
   `practical_implication` text. Factors 4-5 use template wording
   (`wording_source`).
4. **Validation**:
   - A controlled repair drops unsupported summary sentences.
   - The output is rejected for any of: cross-factor or invented citations,
     placeholder or letter format, medication, diagnosis, extreme advice,
     risky numbers, blank artifacts, unsupported inputs (age, BMI, ...),
     confidence claims, comparisons with an average, repetition, an action
     that is off its factor's topic, or a "maintain" reason that talks about
     lowering.
   - One retry follows a rejection, HTTP 500, or malformed JSON. A timeout
     or connection error goes straight to the fallback.
5. **Assembly** (code only): predictions, attributions, priority,
   confidence, scientific-evidence records, sources, uncertainty, and the
   safety note. A truncated single-sentence field takes template text
   (`wording_source: mixed`).

### Live findings that shaped the design

The local model is BioMistral Q4_K_M under llama.cpp with `n_ctx` 4096.

- Output without a schema looped ("...not a diagnosis" x4). A 5-factor
  generation exceeded 45 s.
- With a free schema the model cited other factors' sources and invented
  numbers: wrong predictions, "an average of 82.59", "a survey of 1,342
  adults".
- A permissive pattern (`[^%]`) let the model emit raw quotes, causing
  llama.cpp HTTP 500 ("does not match the expected peg-native format").
- A digit ban with values still in the context produced blanks ("your
  stress level of  is").
- The model copied the knowledge base's assistant-facing guidance into user
  advice. This also affected the Phase 3 chat fallback, which now shows
  evidence text instead.

### Results

- Final live evaluation over 8 input variants: **8 of 8 generated**, in
  16.8-44.3 s (median about 21 s). Three needed summary repair and two
  needed a retry. Predictions were unchanged in every run.
- One real `POST /predict` through uvicorn returned 200 with structured
  recommendations, in 37 s on that run (fallback).
- Repeated identical inputs gave slightly different predictions
  (64.87 -> 64.62 -> 64.58). This is the deferred wall-clock sleep-time
  skew observed live.
- Tests: **173 passed, 1 skipped**, including the new
  `backend/tests/test_phase4_recommendations.py`. The two
  `test_predict_route.py` tests now patch `recommend_for_prediction`; their
  assertions on existing fields and persistence are unchanged.

### Limitations

- These are pattern-based safety checks, not a guarantee of medical
  safety.
- Generated reasons are generic by design, because the model is given no
  numbers; user values appear in `model_evidence`.
- `confidence` is a rule (agreement, evidence strength, disagreement
  level), not a calibrated probability.
- Factors without a knowledge-base entry (work hours, hydration) get no
  recommendation. That is a knowledge-base gap and is deferred.
- About 20-45 s is added to `/predict` when BioMistral is running.

## Master-Plan Phase 3: RAG and BioMistral Reliability (2026-11-20)

No knowledge data, embeddings, ChromaDB contents, model weights, or
dependencies were changed.

### Live verification findings

- The ChromaDB collection `sleep_knowledge_v0_3_1` (17 chunks, bge-small,
  384-d) uses cosine distance. Reported `similarity` equals true cosine
  (verified against raw embeddings), so the 0.65 threshold is meaningful.
  Off-topic queries score 0.39-0.56 and are filtered out.
- Chroma chunk IDs and `source_ids` match the JSON dataset exactly, and every
  cited source exists in the registry. Retrieval reproduces the historical
  v0.3.1 metrics (recall@3 0.967, MRR@3 0.95) as a regression check only;
  10 questions do not establish general accuracy.
- The single concatenated prediction query dilutes retrieval. With stress as
  the top factor, CAFF-001 ranked first and SCREEN-001 was not retrieved for
  blue light. bge similarities are compressed (unrelated chunks score
  0.70-0.74). Phase 4 should use per-factor queries.
- BioMistral runs locally under llama.cpp with Q4_K_M quantization and
  `n_ctx` 4096.
  - **Prediction explanation: 0 of 3 live calls passed validation.** The
    model wrote emails ("Subject: ... Dear [USER_NAME]") and invented a
    "sleep tracker". Users therefore always receive the deterministic
    fallback.
  - **Chat: 5 of 6 live replies passed.** The melatonin question was
    correctly rejected.
  - A live probe confirmed the server supports JSON-schema-constrained
    decoding (`response_format: json_schema`). With `source_ids` restricted
    by an enum, the output was valid JSON in 11 s. This is the basis for
    Phase 4.

### Changes

- New `services/llm_output_validation.py`: checks for placeholders, letter
  format, citations that were not retrieved, medication or dosage,
  diagnosis, "inactivity proves sleep", guaranteed improvement, and extreme
  advice, plus cosmetic stripping of leaked "Answer:" prefixes. These are
  conservative pattern checks, not a guarantee of medical safety.
- `chat_service.py`:
  - `max_tokens` 30000 -> `BIOMISTRAL_CHAT_MAX_TOKENS` (default 700).
  - A malformed JSON body or unexpected structure now raises
    `RuntimeError`, which the route maps to 503 (previously an unhandled
    500).
  - Replies are validated. An invalid reply is replaced by a deterministic
    answer built only from retrieved chunks and their real source titles.
  - The `{"reply": ...}` contract is unchanged.
- `prediction_explanation_service.py`: adds citation and safety checks,
  handles `knowledge_service=None`, and no longer logs the full LLM text or
  HTTP bodies above debug level.
- `main.py`: **behaviour change.** If the knowledge service fails to load,
  the API now starts in degraded mode (predictions work; chat returns 503;
  explanations fall back). `KNOWLEDGE_SERVICE_REQUIRED=true` restores the
  previous fail-fast behaviour, and the existing test now sets this flag.
- `/health` adds `knowledge_service: available|unavailable` (additive).

### Tests executed

Full suite: **138 passed, 1 skipped** (the opt-in live LLM test). New file:
`backend/tests/test_phase3_rag_llm.py`. Its live ChromaDB and embedding
tests ran (not skipped). `RUN_LIVE_BIOMISTRAL=1`: live chat test **passed**
(22 s).

## Master-Plan Phase 2: ML and Explainability Reliability (2026-11-20)

No model, checkpoint, scaler, dataset, split, dependency, or DB schema was
changed. Deployed prediction values are unchanged; a test asserts the
ensemble equals the original inline formula exactly.

Changes:

- `knowledge_service.py`: `ensure_project_root_importable()` appends the
  project root to `sys.path`, so `cd backend && uvicorn app.main:app` starts.
- `schemas/prediction.py`: rejects non-binary `screen_activity_sequence`
  values with 422. Adds optional `prediction_diagnostics` to
  `PredictionResponse` and `LocalExplanationResponse`. `model_outputs` is
  unchanged because the web and Android clients iterate every key in it.
- `model_service.py`:
  - Finite-value check on features and a binary check on sequences.
    `predict_all` now uses `_validate_sequence`.
  - Double-checked locks for `get_model_bundle()` and the RF explainer.
  - Global SHAP selects with `.iloc`, asserts no test or development
    overlap, and reports `sample_row_positions` and
    `excluded_test_row_count`.
  - Ensemble constants are named. The new `build_prediction_diagnostics`
    reports raw outputs, the 0-100 domain check, whether ensemble clipping
    applied, and model disagreement. The low/moderate/high bands (5 and 10
    points) are descriptive, not validated.
  - Hybrid IG output adds `standardized_values` beside the raw values, and
    warns when the Hybrid prediction falls outside 0-100.
- `routes/predict.py`, `routes/explain.py`: model `ValueError` -> HTTP 422.
  `/predict` returns `prediction_diagnostics`.

Output-domain policy: individual model outputs are never clipped or altered.
A value outside 0-100 is returned as-is with a warning. The deployed
ensemble keeps its 50-99 clip, and any clipping is disclosed.

Tests executed (project root,
`PYTHONPATH=backend:. .venv/bin/python -m pytest backend/tests ml/tests -q`):
**104 passed**, including 21 new tests in
`backend/tests/test_phase2_reliability.py`.

Live smoke test: `cd backend && uvicorn app.main:app` with offline Hugging
Face, a scratch SQLite DB, and BioMistral pointed at a closed port.
Startup completed. `/health` returned ok. `/predict` returned 65.89 (Poor)
with diagnostics and `explanation_status: fallback`. A non-binary sequence
returned 422.

## Master-Plan Phase 1 Verification (2026-11-20)

Read-only verification; no code, model, data, or artifact was changed.

### Tests actually executed

`PYTHONPATH=backend:. .venv/bin/python -m pytest backend/tests ml/tests -q`
from the project root: **83 passed** (12 backend files, 1 ml file) in 18 s.
BioMistral and ChromaDB are mocked in these tests; no live LLM call was made.
Only third-party and FastAPI/SQLAlchemy deprecation warnings were emitted.

### Phase 3 explainability: confirmed present

RF and XGBoost local TreeSHAP with reconstruction checks, Hybrid two-branch
Integrated Gradients with completeness and 288 five-minute bins,
`POST /explain/local`, and global mean-absolute TreeSHAP over train plus
validation rows only. The test split is excluded.

### Verified defects

1. **Backend startup fails from the documented directory.**
   `cd backend && uvicorn app.main:app` raises `ModuleNotFoundError: ml` in
   the startup hook (`knowledge_service.create_knowledge_service` imports
   `ml.scripts.search_knowledge`). Tests pass only because they set
   `PYTHONPATH` to include the project root.
2. **Training/inference skew in sleep-time inference.** Training
   (`ml/scripts/preprocess.py`) uses no clock masking, no 20:00-03:00 window,
   and no "started after activity" rule; the API
   (`inactivity_service.py`) uses all three plus `datetime.now()`. With
   stored sequences at a 23:59 clock, 54% of rows receive a different
   inferred sleep start; at 09:00, 82%. The training rule can select morning
   inactivity (row 0: 07:10). Measured prediction effect is small: all-row
   MAE RF 1.765 -> 1.853, XGB 0.923 -> 1.188; historical-test RF MAE
   2.641 -> 2.648. Predictions also depend on the wall-clock time of the
   request.
3. **Non-binary activity values are accepted** by `/predict` (for example
   5 or -3) and passed to the Hybrid. NaN and infinity are rejected by the
   Pydantic range constraints.
4. `predict_all` does not use `_validate_sequence`; `get_model_bundle()` and
   `_get_rf_explainer()` are not lock-protected; global SHAP uses `.loc` with
   positional split indices (currently equivalent only because the CSV has a
   RangeIndex).
5. Hybrid output is unconstrained (4 stored rows above 100%, max 101.94);
   reconfirmed.

### Target construction and leakage

`sleep_efficiency` is not measured. `prepare_real_sleep_health_data.py`
constructs it as `10 x Quality of Sleep + 2.3 x (Sleep Duration - 6.5)`
minus stress and apnea terms plus noise. Feature provenance:

- Source-observed: `stress_level` (also a term in the target formula),
  `physical_activity_min` (scaled source value).
- Synthetic proxies from non-target source columns: caffeine, alcohol,
  dinner hour, work hours, bedtime consistency, hydration, blue light.
- Target-derived (leakage-prone): `previous_sleep_hours` (current Sleep
  Duration plus noise), `naps_count` and `naps_total_min` (from Sleep
  Duration), `nap_sleep_debt_proxy`, `mood_score` (small Quality term), and
  the 288-bin sequence with every sequence-derived feature (its imposed
  inactivity window equals Sleep Duration).

Existing evidence (`ml/experiments/hybrid_ablation_v2/report.md`): RF with
the 9 leakage-screened features has OOF MAE 2.67, against 2.73 with all 20.
High tabular accuracy therefore does not depend on the leakage-prone
features. It mainly reflects that the constructed target is close to a
function of Kaggle stress and sleep-quality fields. The reproduced legacy
Hybrid has OOF MAE 14.66 (seed SD 2.96), far from the deployed historical
4.98, so the Hybrid result is not reproducible under development CV.

### Other findings

- `backend/app/services/ml/` duplicates `ml/`. No code imports it. Only
  `chroma_db` differs, and the runtime uses `ml/artifacts/chroma_db`.
- Frontend `lib/api.ts` is untyped, so it does not conflict with the backend
  schemas. The Android `PredictionResponse` matches the existing fields.
- The Android collector builds bins from local midnight to now. A bedtime
  before midnight on the previous evening is not in the sequence.

---

# Historical Notes (pre-master-plan Phase 3, superseded)

The sections below were written before the master-plan phases and are kept
for history. Their pending items and known issues have since been addressed
or moved to `docs/CODEX_HANDOVER.md`.

## Pending Modules

- External dataset assessment.
- Automatic SHAP-driven, RAG-backed BioMistral recommendations.
- RAG/BioMistral operational validation.
- Backend startup reliability and comprehensive test execution.
- Frontend integration validation and final documentation.

## Known Issues

- Deployed fixed ensemble has documented evidence of underperforming RF.
- Hybrid output is unconstrained before final ensemble clipping.
- Read-only artifact validation loaded RF, XGBoost, Hybrid, and the Hybrid
  scaler successfully. All expect the same ordered 20-feature contract.
- Stored sequence input is binary with shape `374 x 288`; deployed Hybrid
  inference expects `[n, 1, 288]`.
- Four stored Hybrid predictions exceed 100% (maximum 101.9400). The raw
  fixed ensemble did not exceed 0-100 on stored rows, but production clips its
  final value to 50-99 and does not expose invalid Hybrid output handling.
- Full-row descriptive metrics are not validation metrics: RF MAE 1.7652,
  XGBoost MAE 0.9231, Hybrid MAE 5.1603, fixed ensemble MAE 3.0127.
- The existing `/predict` response deliberately remains XGBoost-TreeSHAP-only
  for backward compatibility. The new `/explain/local` route provides RF,
  XGBoost, and Hybrid explanations without changing the prediction contract.
- Hybrid temporal attributions concern synthetic activity/inactivity data and
  must not be claimed as observed device or wearable evidence.
- No combined ensemble attribution is returned because TreeSHAP and Integrated
  Gradients are not mathematically interchangeable.
- Current automatic recommendations are deterministic; BioMistral generates a
  separate explanation rather than structured automatic recommendations.
- The active data lacks retained participant identifiers and uses synthetic
  activity/inactivity sequences.

## Safest Current Prediction Policy

Preserve the deployed API behavior and retain RF, XGBoost, Hybrid, and ensemble
outputs for research comparison and backward compatibility. Do not describe the
fixed `0.2/0.3/0.5` ensemble as the best validated policy, and do not replace it
with RF or new weights without a separately approved validation and production
change. Surface model disagreement and the synthetic-data limitation wherever
the prediction is demonstrated.

## Tests Executed In This Phase

- Read-only deployed-artifact validation script: passed model loading, strict
  Hybrid checkpoint loading, feature-order compatibility, scaler compatibility,
  binary sequence validation, and prediction-domain measurement.
- Phase 3 targeted explainability tests: `7 passed`. They validate RF/XGBoost
  reconstruction, Hybrid two-branch Integrated Gradients completeness and time
  mapping, global SHAP sampling, sequence validation, and route construction.

## Resume Command

```bash
cd "<repository root>"
source .venv/bin/activate
```

## Files Changed In This Phase

- `backend/app/services/model_service.py`
- `backend/app/api/routes/explain.py`
- `backend/app/schemas/prediction.py`
- `backend/tests/test_model_explainability.py`
- `backend/tests/test_explain_route.py`
- `docs/PROJECT_COMPLETION_PLAN.md`
- `docs/PROJECT_PROGRESS.md`
- `docs/CODEX_HANDOVER.md`

No model, scaler, checkpoint, dataset, split, database schema, frontend, or
production prediction calculation was changed.
