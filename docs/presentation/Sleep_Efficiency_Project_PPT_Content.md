# Slide 1: Title
Explainable Machine Learning for Lifestyle-Based Sleep Efficiency Prediction Using Behavioral and Screen Inactivity Signals  
Final Year Project Presentation  
Student: [Your Name] | Department: [Your Dept] | Guide: [Guide Name]

---

# Slide 2: Motivation (5 Marks)
- Sleep quality strongly affects cognition, mood, and productivity.
- Most sleep tools rely on wearables/clinical signals that are costly or intrusive.
- Need a practical and low-cost solution based on daily behavior and phone usage patterns.
- Goal: Predict sleep efficiency and provide understandable guidance.

---

# Slide 3: Literature Review (10 Marks) - Existing Work
- Wearable-based methods (EEG/ECG/PPG) provide high accuracy but poor accessibility.
- Tabular ML in healthcare shows strong performance on structured behavioral data.
- Explainable AI (SHAP, feature importance) improves trust and actionability.
- Research gap: Integrating lifestyle tabular data with temporal screen inactivity in an interpretable system.

---

# Slide 4: Literature Review - Gap and Takeaways
- Gap 1: Many models are black-box and difficult to explain to end users.
- Gap 2: Most datasets/tools ignore practical device behavior patterns.
- Gap 3: Few end-to-end systems combine UI + backend + ML + explainability.
- Our takeaway: Build a full-stack, interpretable, deployable solution with actionable recommendations.

---

# Slide 5: Problem Identification (10 Marks)
- Problem: How to estimate sleep efficiency without wearable sensors?
- Challenges:
  - Behavioral data is noisy and incomplete.
  - Sleep timing inference from device activity can be unstable.
  - Recommendations must be human-readable and trustworthy.

---

# Slide 6: Objectives (10 Marks)
- Predict sleep efficiency (%) and classify Good/Poor sleep.
- Use only lifestyle + device activity/inactivity behavior.
- Infer probable sleep time from inactivity patterns.
- Compare 3 models: Random Forest, XGBoost, Hybrid (MLP+1D CNN).
- Provide explainability (SHAP + feature narratives + recommendations).
- Deliver full web + Android application.

---

# Slide 7: System Architecture (5 Marks)
- Frontend: Next.js (web) + Android (Jetpack Compose).
- Backend: FastAPI + SQLite + model, explanation, retrieval and recommendation services.
- ML: Random Forest, XGBoost, Hybrid MLP + 1D-CNN; TreeSHAP and Integrated Gradients.
- Knowledge: 17-entry, source-traceable sleep knowledge base (bge-small embeddings + ChromaDB).
- Language model: BioMistral-7B (LoRA adapter) served locally by llama.cpp.
- Flow:
  1. Lifestyle inputs + 288-bin activity sequence
  2. Sleep-onset inference + 20-feature engineering
  3. RF, XGBoost and Hybrid predictions + fixed ensemble (prediction saved first)
  4. Per-model explanations -> recommendation context -> per-factor retrieval
  5. BioMistral wording under a JSON schema -> validation -> retry or deterministic fallback
  6. Dashboard: predictions, disagreement, explanations, recommendations, sources, limitations

---

# Slide 8: Module Description (5 Marks) - Data Layer
- Source: Kaggle Sleep Health and Lifestyle dataset (374 rows, 374 unique people).
- Target: a sleep-efficiency **index constructed** from "Quality of Sleep" and "Sleep Duration" (not measured).
- Predictors: stress and activity from the source; caffeine, alcohol, dinner time, work hours, bedtime consistency, hydration and blue light are **synthetic proxies**.
- Activity sequences: **synthetic** 288-bin sequences (imposed sleep window); never presented as real phone use.
- Leakage audit: previous sleep, naps, mood and sequence features are derived from target fields.
- Split: 261 train / 38 validation / 75 test (stratified).

---

# Slide 9: Module Description - ML Layer
- Model 1: Random Forest Regressor (strongest model throughout).
- Model 2: XGBoost Regressor.
- Model 3: Hybrid: MLP (tabular) + 1D-CNN (activity sequence), late fusion, unconstrained output.
- Ensemble: fixed 0.2 RF + 0.3 XGB + 0.5 Hybrid, clipped 50-99 (kept for compatibility).
- Diagnostics: raw outputs, 0-100% domain warnings, clipping disclosure, model disagreement.

---

# Slide 10: Module Description - Explainability and Recommendation Layer
- RF and XGBoost: TreeSHAP with baseline + reconstruction check; global SHAP from train/validation rows only.
- Hybrid: Integrated Gradients over tabular inputs and all 288 time bins, with a completeness check.
- The two methods are shown separately and never combined into an "ensemble SHAP".
- Recommendations use only leakage-screened factors on which RF and XGBoost agree.
- Each factor is matched to its own retrieved evidence; BioMistral writes the wording only.
- Code fills in every number and source; output is validated, otherwise a deterministic fallback is used.

---

# Slide 11: Innovation (5 Marks)
- One traceable pipeline: prediction -> per-model explanation -> per-factor evidence -> constrained LLM -> validation.
- Grammar-constrained BioMistral: citations limited to each factor's own retrieved sources; no numbers from the LLM.
- Method-faithful explainability across tree models and a neural hybrid.
- Explicit data-provenance and leakage audit built into the recommendation policy.
- Degrades safely: predictions work without the LLM or the knowledge base.
- Web and Android clients; Android derives activity from app-usage intervals.

---

# Slide 12: Implementation Highlights
- FastAPI endpoints: /health, /predict, /history, /explain/global, /explain/local, /explain/model-metrics, /chat
- Automated tests: 232 pass across backend and ML, including the forecasting experiments (plus an opt-in live BioMistral test).
- Android: UsageStats foreground intervals, consent text, debug-only metadata logging, 90 s timeout.
- Web: dashboard with model agreement, TreeSHAP and Integrated Gradients views, recommendation cards, sources and limitations; Sleep Assistant chat.

---

# Slide 13: Results and Evaluation
Regression metrics are primary; Good = index >= 85.

| Model | Historical test MAE | R² | Dev. CV MAE |
|---|---:|---:|---:|
| Random Forest | 2.64 | 0.933 | 2.62 |
| XGBoost | 3.07 | 0.907 | 2.77 |
| Hybrid | 4.98 | 0.804 | 5.40 |
| Fixed ensemble | – | – | 3.65 |

- The historical test split was examined during development; it is **not** an untouched hold-out.
- RF is best; the fixed ensemble is worse than RF; the Hybrid adds no measurable value on this data.
- Leakage ablation: RF MAE 2.67 (9 screened features) vs 2.73 (all 20), so accuracy reflects the constructed target.
- Retrieval: recall@3 0.967 on 10 judged questions (a regression check only).
- BioMistral recommendations: validated generation for 8/8 development inputs (16.8-44.3 s). This is an engineering check, not an evaluation.
- ROC-AUC is not reported: it was computed from rescaled regression output, not a classifier probability.
- These models are **same-night estimators** of a constructed index, not coming-night forecasts.

---

# Slide 14: Coming-Night Forecasting on Real Data (experiment)
Question: can the coming night be forecast from data available before it starts?

| Dataset | Target (not interchangeable) | Best simple baseline | Best ML model |
|---|---|---:|---:|
| Fitabase, 20 people, 776 nights | Device sleep-ratio proxy (%) | 2.28 (own history) | 2.39 |
| Walch, 31 people, 1 night each | PSG sleep efficiency (%) | 4.05 (mean) | 3.77 (CI crosses 0) |
| StudentLife, 49 people, 1,241 nights | Self-reported hours | 0.785 h (own history) | 0.776 h (no difference) |

- Leakage-free features only (cutoff before each night); participant-grouped and chronological validation.
- **Negative result:** ML did not beat a person's own sleep history; activity, heart-rate and phone features added nothing detectable.
- Simple personal-history rules (7/14-night means, weighted mean, trend) gave no meaningful gain either.
- Not integrated into the app: it stores no user identity or observed sleep outcomes.

---

# Slide 15: Limitations and Future Scope
- Constructed target, synthetic proxies and synthetic activity sequences; no participant-level or external validation.
- Training and serving use different sleep-onset rules; serving depends on request time (open issue).
- Hybrid output unconstrained and not reproducible under cross-validation.
- Small knowledge base (17 entries; no evidence for work hours or hydration); rule-based safety checks.
- Coming-night forecasting on real data did not beat personal history; small cohorts, a device proxy and a self-reported target.
- Future work: measured sleep efficiency with real device data; aligned sleep inference; retraining on screened features with a pre-registered hold-out; a larger verified knowledge base; user studies.

---

# Slide 16: Conclusion
- Built an explainable, end-to-end sleep-efficiency system across web and Android.
- Predictions, per-model attributions, retrieved evidence and LLM output stay separately traceable.
- The LLM cannot change numbers or cite unretrieved sources; unsafe output falls back safely.
- The data demonstrate architecture and safeguards, not real-world predictive validity.

---

# Slide 17: Viva Preparation (10 Marks)
Expected Questions:
1. Why keep the Hybrid model when RF is better?
2. How is sleep time inferred, and what are its limits?
3. Why TreeSHAP for trees but Integrated Gradients for the Hybrid?
4. How do you stop BioMistral from inventing facts or numbers?
5. What does high accuracy mean when the target is constructed?
6. Can the system forecast tonight's sleep?
7. What data are stored, and what privacy safeguards exist?

Short Answers:
- It is a required research component for fusing temporal behaviour; on this synthetic data it does not add value, and we report that.
- Longest >= 30-min inactivity run starting 20:00-03:00 after activity; it is a proxy, and training used a simpler rule (documented open issue).
- TreeSHAP is exact for tree ensembles; the Hybrid is a neural network, so Integrated Gradients is used; the two are never combined.
- Schema-constrained decoding (per-factor citation lists, digit-free text), code-filled numbers, rule-based validation, retry and a deterministic fallback.
- It shows the models fit an index built from stress and sleep-quality fields; it is not evidence of real-world performance.
- Not with demonstrated skill: on three real datasets, no model beat the person's own sleep history, which is the honest baseline to report.
- The input payload (lifestyle values and binary activity bins, no app names or content) and predictions are stored in local SQLite; Android logs only request metadata in debug builds.

---

# Slide 18: Thank You
Thank you.  
Questions?
