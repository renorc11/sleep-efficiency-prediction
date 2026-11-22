// TypeScript mirror of the backend Pydantic response models in
// backend/app/schemas/prediction.py. New fields are optional so responses
// from an older backend still render.

export type TopFactor = {
  feature: string;
  impact: number;
  value: number;
  direction?: string;
  feature_key?: string;
  insight?: string;
};

export type ModelDisagreement = {
  spread: number;
  rf_xgb_abs_difference: number;
  level: "low" | "moderate" | "high";
  bands: Record<string, number>;
  note: string;
};

export type PredictionDiagnostics = {
  raw_outputs: { rf: number; xgb: number; hybrid: number; ensemble_unclipped: number };
  valid_domain: { min: number; max: number; out_of_domain_models: string[] };
  ensemble_policy: { weights: Record<string, number>; clip_range: number[]; clipping_applied: boolean; note: string };
  model_disagreement: ModelDisagreement;
  warnings: string[];
};

export type AttributionRow = { feature_key: string; feature: string; value: number; attribution: number };

export type FactorSignal = {
  feature_key: string;
  feature: string;
  user_value: number | null;
  unit: string;
  rf_treeshap: number;
  xgb_treeshap: number;
  hybrid_integrated_gradients: number | null;
  mean_abs_treeshap: number;
  direction: "lowers" | "raises";
  rf_xgb_agree: boolean;
  hybrid_agrees: boolean | null;
};

export type HourlyAttribution = { hour: string; attribution: number; active_five_minute_bins: number };

export type ModelExplanation = {
  validity: Record<string, { method: string; valid: boolean; absolute_error: number; integration_steps?: number }>;
  rf_treeshap: { baseline: number; top_negative: AttributionRow[]; top_positive: AttributionRow[] };
  xgb_treeshap: { baseline: number; top_negative: AttributionRow[]; top_positive: AttributionRow[] };
  hybrid_integrated_gradients: null | {
    baseline_prediction: number;
    baseline_description: { tabular: string; temporal: string };
    top_negative_tabular: AttributionRow[];
    top_positive_tabular: AttributionRow[];
    temporal_summary: {
      method: string;
      hours_lowering_prediction: HourlyAttribution[];
      hours_raising_prediction: HourlyAttribution[];
      note: string;
    } | null;
  };
  agreed_lowering_factors: FactorSignal[];
  agreed_raising_factors: FactorSignal[];
  rf_xgb_disagreements: FactorSignal[];
  excluded_features: Array<{ feature_key: string; feature: string; reason: string }>;
  method_note: string;
};

export type ScientificEvidence = {
  knowledge_id: string;
  title: string | null;
  text: string | null;
  evidence_type: string | null;
  evidence_strength: string | null;
  limitations: string[];
  similarity: number | null;
  retrieval_rank: number;
};

export type PersonalizedRecommendation = {
  title: string;
  priority: number;
  wording_source: "biomistral" | "mixed" | "template";
  factor_key: string;
  related_features: string[];
  kind: "improve" | "maintain";
  model_evidence: {
    user_value: number | null;
    unit: string;
    direction: string;
    rf_treeshap: number;
    xgb_treeshap: number;
    hybrid_integrated_gradients: number | null;
    rf_xgb_agree: boolean;
    hybrid_agrees: boolean | null;
    note: string;
  };
  scientific_evidence: ScientificEvidence | null;
  source_ids: string[];
  reason: string;
  action: string;
  confidence: "low" | "moderate" | "high";
  safety_note: string;
};

export type RetrievedSource = {
  source_id: string;
  title: string | null;
  authors?: string[];
  organization?: string | null;
  year?: number | null;
  journal?: string | null;
  doi?: string | null;
  url?: string | null;
  supports: string[];
};

export type PersonalizedRecommendationResult = {
  summary: string;
  prediction_context: {
    model_predictions: Record<string, number>;
    predicted_sleep_efficiency: number;
    sleep_quality_label: string;
    inferred_sleep_time: string;
    model_disagreement: Partial<ModelDisagreement>;
  };
  model_explanation: ModelExplanation;
  recommendations: PersonalizedRecommendation[];
  retrieved_sources: RetrievedSource[];
  uncertainty: string[];
  generation_status: "generated" | "fallback";
  fallback_used: boolean;
  fallback_reason?: string | null;
  validation_issues?: string[];
  warnings?: string[];
};

export type PredictionResult = {
  predicted_sleep_efficiency: number;
  sleep_quality_label: string;
  inferred_sleep_time: string;
  model_outputs: Record<string, number>;
  top_influencing_factors: TopFactor[];
  recommendations: string[];
  generated_explanation?: string;
  explanation_status?: string;
  prediction_diagnostics?: PredictionDiagnostics | null;
  personalized_recommendations?: PersonalizedRecommendationResult | null;
};
