# Project Completion Plan

## Objective

Complete the existing sleep-efficiency prediction project without replacing its
mandatory Random Forest, XGBoost, Hybrid MLP + 1D-CNN, inactivity inference,
SHAP, RAG, or BioMistral components. Synthetic 288-bin activity/inactivity
sequences must never be presented as real device-observed screen activity.

## Master-Plan Phases (status 2026-11-21)

| Phase | Scope | Status |
|---|---|---|
| 1 | Repository and implementation integrity audit | Done |
| 2 | ML and explainability reliability (validation, thread safety, global SHAP, output-domain policy) | Done |
| 3 | RAG and BioMistral reliability (validation, fallbacks, degraded startup) | Done |
| 4 | Automatic SHAP -> RAG -> BioMistral recommendations | Done |
| 5 | Backend integration (DB path, lifespan, warm-up, endpoint integration tests) | Done |
| 6 | Frontend and Android integration | Done (Android build not verified) |
| 7 | Testing and research documentation | Done (paper not compiled) |

Evidence for each phase: `docs/PROJECT_PROGRESS.md`.

## Remaining Work Before the Project Is Complete

1. Fix the sleep-onset training/serving skew (deferred by the user; needs
   approval for an inference change or retraining).
2. Expand the knowledge base with verified evidence (deferred by the user).
3. Verify the Android build and compile the LaTeX paper.
4. User cleanup decisions: stray database and build outputs. (Resolved
   2026-11-22: the duplicate `ml/` copy was archived outside the repository
   and the unused Docker infrastructure was removed.)

## Non-Negotiable Boundaries

- Do not inspect or modify `REST/`.
- Preserve uncommitted work and existing artifacts.
- Do not change dependencies, checkpoints, production ensemble weights, API
  contracts, database schema, or deployed prediction behavior without explicit
  approval.
- The historical 75-row split is already examined and is not untouched final
  confirmation data.
- The 374-row development data and its activity/inactivity sequences are
  synthetic-development evidence only.
