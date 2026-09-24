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
- Frontend: React/Next.js + Android (Jetpack Compose).
- Backend: FastAPI + SQLite + model service.
- ML: preprocessing, training, comparison, explainability.
- Flow:
  1. User input + screen activity collection
  2. Feature engineering + sleep-time inference
  3. Multi-model prediction + ensemble
  4. SHAP insights + recommendations
  5. Results shown on dashboard

---

# Slide 8: Module Description (5 Marks) - Data Layer
- Data sources:
  - Sleep Health and Lifestyle Dataset (real)
  - Device-derived screen activity sequences (5-minute bins)
- Preprocessing:
  - Missing handling and normalization
  - Engineered features (stress interactions, inactivity windows, bedtime consistency)
  - Train/validation/test split

---

# Slide 9: Module Description - ML Layer
- Model 1: Random Forest Regressor (strong tabular baseline).
- Model 2: XGBoost Regressor (high-performance boosted trees).
- Model 3: Hybrid model:
  - MLP branch for tabular features
  - 1D CNN branch for activity time series
  - Late fusion head for final regression output.
- Ensemble combines all model outputs.

---

# Slide 10: Module Description - Explainability Layer
- Global explainability: model feature importance.
- Local explainability: SHAP contributions per prediction.
- Humanized insights:
  - Risk factors are explained with context
  - Avoids misleading language like “more blue light is better”
- Personalized recommendation generation from high-impact factors.

---

# Slide 11: Innovation (5 Marks)
- Hybrid fusion of behavioral tabular + temporal device activity.
- Automatic blue-light proxy from device usage with optional manual override.
- Optional user sleep-time hint to stabilize sleep inference.
- Humanized explainability text, not raw machine-style output.
- End-to-end deployable system across web and Android.

---

# Slide 12: Implementation Highlights
- FastAPI APIs:
  - /predict
  - /history
  - /explain/global
  - /explain/model-metrics
- Android app:
  - UsageStats-based sequence extraction
  - Permission-aware UX
  - Optional bedtime hint
- Web app:
  - Input forms, charts, model comparison, recommendations.

---

# Slide 13: Results and Evaluation
(Replace with your final run metrics from ml/reports/model_comparison.json)
- Regression: MAE, RMSE, R²
- Classification: Accuracy, Precision, Recall, F1, ROC-AUC
- Comparative analysis:
  - RF: strongest tabular baseline
  - XGB: robust and explainable
  - Hybrid: captures temporal behavior and supports research novelty

---

# Slide 14: Limitations and Future Scope
- Current system uses proxy estimation for some features (e.g., blue-light minutes).
- Real-world behavior drift may require periodic retraining.
- Future work:
  - Personalized longitudinal user models
  - Better sleep stage proxies
  - Broader clinical validation

---

# Slide 15: Conclusion
- Built a full-stack, interpretable sleep prediction platform.
- Achieved practical prediction without wearables.
- Delivered explainable insights and actionable recommendations.
- Project demonstrates both research novelty and implementation readiness.

---

# Slide 16: Viva Preparation (10 Marks)
Expected Questions:
1. Why use hybrid model when tabular models are strong?
2. How is inferred sleep time computed and stabilized?
3. Why SHAP, and how do you avoid misleading explanations?
4. How does your system remain sensor-free yet useful?
5. What are ethical/privacy safeguards for device usage data?

Short Answers:
- Hybrid adds temporal behavior signals from activity sequences.
- Inference uses inactivity windows + realistic window + optional user hint.
- SHAP gives local transparency; wording layer converts technical output to safe, human explanations.
- Uses low-cost lifestyle/device behavior only; no clinical sensors.
- Stores only derived features and predictions; no raw personal app content.

---

# Slide 17: Thank You
Thank you.  
Questions?
