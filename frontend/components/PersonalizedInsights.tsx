"use client";

import {
  Bar,
  BarChart,
  CartesianGrid,
  Legend,
  ReferenceLine,
  ResponsiveContainer,
  Tooltip,
  XAxis,
  YAxis
} from "recharts";
import type {
  FactorSignal,
  PersonalizedRecommendation,
  PersonalizedRecommendationResult,
  PredictionDiagnostics
} from "../lib/types";

const RF = "#00b894";
const XGB = "#0984e3";
const HYBRID = "#e1a915";

function signed(value: number | null | undefined, digits = 2) {
  if (value === null || value === undefined) return "n/a";
  return `${value >= 0 ? "+" : ""}${value.toFixed(digits)}`;
}

function formatValue(value: number | null, unit: string) {
  if (value === null || value === undefined) return "n/a";
  const shown = Number.isInteger(value) ? String(value) : value.toFixed(2);
  return `${shown} ${unit}`.trim();
}

function DisagreementPanel({ diagnostics }: { diagnostics: PredictionDiagnostics }) {
  const raw = diagnostics.raw_outputs;
  const level = diagnostics.model_disagreement.level;
  return (
    <div className="insight-panel">
      <h4>Model agreement</h4>
      <div className="model-raw-grid">
        <div><span>RF</span><strong>{raw.rf.toFixed(2)}%</strong></div>
        <div><span>XGBoost</span><strong>{raw.xgb.toFixed(2)}%</strong></div>
        <div><span>Hybrid</span><strong>{raw.hybrid.toFixed(2)}%</strong></div>
        <div><span>Ensemble (unclipped)</span><strong>{raw.ensemble_unclipped.toFixed(2)}%</strong></div>
      </div>
      <p className="small">
        Spread between the three models: <strong>{diagnostics.model_disagreement.spread.toFixed(2)} points</strong>{" "}
        <span className={`level level-${level}`}>{level} disagreement</span>. Ensemble = fixed{" "}
        {Object.entries(diagnostics.ensemble_policy.weights).map(([k, w]) => `${w} × ${k.toUpperCase()}`).join(" + ")},
        clipped to {diagnostics.ensemble_policy.clip_range.join("–")}%. {diagnostics.ensemble_policy.note}
      </p>
      {diagnostics.warnings.length > 0 && (
        <ul className="warning-list">
          {diagnostics.warnings.map((w) => <li key={w}>{w}</li>)}
        </ul>
      )}
    </div>
  );
}

function FactorChart({ factors }: { factors: FactorSignal[] }) {
  const data = factors.map((f) => ({ feature: f.feature, RF: f.rf_treeshap, XGBoost: f.xgb_treeshap }));
  return (
    <div style={{ width: "100%", height: Math.max(180, 48 * data.length + 70) }}>
      <ResponsiveContainer>
        <BarChart data={data} layout="vertical" margin={{ top: 8, right: 24, left: 8, bottom: 8 }}>
          <CartesianGrid strokeDasharray="4 4" stroke="#dbe7ff" />
          <XAxis type="number" tickLine={false} axisLine={false} tickFormatter={(v) => signed(Number(v), 1)} />
          <YAxis type="category" dataKey="feature" width={170} tickLine={false} axisLine={false} />
          <ReferenceLine x={0} stroke="#9aa9bd" />
          <Tooltip formatter={(v, name) => [`${signed(Number(v))} points`, `${name} TreeSHAP`]} />
          <Legend />
          <Bar dataKey="RF" fill={RF} radius={4} maxBarSize={16} />
          <Bar dataKey="XGBoost" fill={XGB} radius={4} maxBarSize={16} />
        </BarChart>
      </ResponsiveContainer>
    </div>
  );
}

function RecommendationCard({ rec, titles }: { rec: PersonalizedRecommendation; titles: Record<string, string> }) {
  const ev = rec.model_evidence;
  return (
    <article className="rec-card">
      <header>
        <span className="rec-priority">{rec.priority}</span>
        <h4>{rec.title}</h4>
        <span className={`conf conf-${rec.confidence}`}>{rec.confidence} confidence</span>
      </header>
      <p className="rec-action">{rec.action}</p>
      <p className="small">{rec.reason}</p>
      <details>
        <summary>Model evidence and sources</summary>
        <p className="small">
          Your value: <strong>{formatValue(ev.user_value, ev.unit)}</strong> — {ev.direction}.
        </p>
        <ul className="evidence-list">
          <li><span className="dot" style={{ background: RF }} />RF TreeSHAP: {signed(ev.rf_treeshap)} points</li>
          <li><span className="dot" style={{ background: XGB }} />XGBoost TreeSHAP: {signed(ev.xgb_treeshap)} points</li>
          <li>
            <span className="dot" style={{ background: HYBRID }} />Hybrid Integrated Gradients: {signed(ev.hybrid_integrated_gradients)}
            {ev.hybrid_agrees === false && " (opposite direction)"}
          </li>
        </ul>
        <p className="small">{ev.note}</p>
        {rec.scientific_evidence ? (
          <p className="small">
            Evidence <strong>{rec.scientific_evidence.knowledge_id}</strong> “{rec.scientific_evidence.title}”
            {rec.scientific_evidence.evidence_strength && <> · strength: {rec.scientific_evidence.evidence_strength.replace(/_/g, " ")}</>}
            {rec.source_ids.length > 0 && <> · sources: {rec.source_ids.map((id) => (titles[id] ? `${id} (${titles[id]})` : id)).join("; ")}</>}
          </p>
        ) : (
          <p className="small">No knowledge-base evidence was retrieved for this factor; it rests on model attribution only.</p>
        )}
        <p className="small">
          Wording: {rec.wording_source === "biomistral" ? "BioMistral" : rec.wording_source === "mixed" ? "BioMistral with template text" : "template"}.{" "}
          {rec.safety_note}
        </p>
      </details>
    </article>
  );
}

export default function PersonalizedInsights({
  insights,
  diagnostics
}: {
  insights: PersonalizedRecommendationResult | null | undefined;
  diagnostics: PredictionDiagnostics | null | undefined;
}) {
  if (!insights && !diagnostics) return null;
  const explanation = insights?.model_explanation;
  const hybrid = explanation?.hybrid_integrated_gradients;
  const temporal = hybrid?.temporal_summary;
  const titles = Object.fromEntries((insights?.retrieved_sources ?? []).map((s) => [s.source_id, s.title ?? ""]));
  const agreed = [...(explanation?.agreed_lowering_factors ?? []), ...(explanation?.agreed_raising_factors ?? [])].slice(0, 8);
  const temporalData = temporal
    ? [...temporal.hours_lowering_prediction, ...temporal.hours_raising_prediction]
        .sort((a, b) => a.hour.localeCompare(b.hour))
        .map((h) => ({ hour: h.hour.slice(0, 5), attribution: h.attribution }))
    : [];

  return (
    <section className="insights" aria-labelledby="insights-title">
      <h3 id="insights-title">Personalized Explanation & Recommendations</h3>

      {insights && (
        <div className={`status-banner ${insights.fallback_used ? "status-fallback" : "status-generated"}`}>
          <strong>{insights.fallback_used ? "Deterministic recommendations" : "Generated by BioMistral, validated"}</strong>
          <span>
            {insights.fallback_used
              ? " BioMistral output was unavailable or did not pass validation, so recommendations were built directly from the model explanations and retrieved evidence."
              : " Wording from BioMistral; every number, attribution and source comes from the models and the knowledge base."}
          </span>
        </div>
      )}

      {insights && <p className="insight-summary">{insights.summary}</p>}

      {diagnostics && <DisagreementPanel diagnostics={diagnostics} />}

      {insights && insights.recommendations.length > 0 && (
        <>
          <h4>Recommendations</h4>
          <div className="rec-grid">
            {insights.recommendations.map((rec) => <RecommendationCard key={rec.factor_key} rec={rec} titles={titles} />)}
          </div>
        </>
      )}

      {explanation && agreed.length > 0 && (
        <div className="chart-card">
          <h4>What moved the prediction (TreeSHAP)</h4>
          <p className="chart-caption">
            Percentage-point contributions where Random Forest and XGBoost agree on direction. Negative values lowered
            the predicted sleep efficiency. These describe model behavior, not causes.
          </p>
          <FactorChart factors={agreed} />
          {explanation.rf_xgb_disagreements.length > 0 && (
            <p className="chart-note">
              RF and XGBoost disagree on: {explanation.rf_xgb_disagreements.map((f) => f.feature).join(", ")} (not used for recommendations).
            </p>
          )}
        </div>
      )}

      {hybrid && (
        <div className="chart-card">
          <h4>Hybrid model (Integrated Gradients)</h4>
          <p className="chart-caption">
            The Hybrid model is explained with Integrated Gradients, a different method from SHAP, so its values are
            shown separately and never added to the TreeSHAP values. Baseline prediction: {hybrid.baseline_prediction.toFixed(2)}%.
          </p>
          <div className="ig-columns">
            <div>
              <strong className="small">Lowering (lifestyle inputs)</strong>
              <ul className="evidence-list">
                {hybrid.top_negative_tabular.map((r) => <li key={r.feature_key}>{r.feature}: {signed(r.attribution)}</li>)}
              </ul>
            </div>
            <div>
              <strong className="small">Raising (lifestyle inputs)</strong>
              <ul className="evidence-list">
                {hybrid.top_positive_tabular.map((r) => <li key={r.feature_key}>{r.feature}: {signed(r.attribution)}</li>)}
              </ul>
            </div>
          </div>
          {temporalData.length > 0 && (
            <>
              <strong className="small">Activity timeline: most influential hours</strong>
              <div style={{ width: "100%", height: 200 }}>
                <ResponsiveContainer>
                  <BarChart data={temporalData} margin={{ top: 10, right: 16, left: 0, bottom: 4 }}>
                    <CartesianGrid strokeDasharray="4 4" stroke="#dbe7ff" />
                    <XAxis dataKey="hour" tickLine={false} axisLine={false} />
                    <YAxis tickLine={false} axisLine={false} tickFormatter={(v) => signed(Number(v), 1)} />
                    <ReferenceLine y={0} stroke="#9aa9bd" />
                    <Tooltip formatter={(v) => [signed(Number(v)), "Integrated Gradients"]} />
                    <Bar dataKey="attribution" fill={HYBRID} radius={4} maxBarSize={36} />
                  </BarChart>
                </ResponsiveContainer>
              </div>
              <p className="disclosure">{temporal?.note}</p>
            </>
          )}
        </div>
      )}

      {explanation && explanation.excluded_features.length > 0 && (
        <details className="insight-panel">
          <summary>Factors shown by the models but not used for advice ({explanation.excluded_features.length})</summary>
          <ul className="evidence-list">
            {explanation.excluded_features.map((f) => <li key={f.feature_key}><strong>{f.feature}</strong>: {f.reason}</li>)}
          </ul>
        </details>
      )}

      {insights && insights.retrieved_sources.length > 0 && (
        <details className="insight-panel">
          <summary>Scientific sources ({insights.retrieved_sources.length})</summary>
          <ul className="evidence-list">
            {insights.retrieved_sources.map((s) => (
              <li key={s.source_id}>
                <strong>{s.source_id}</strong> {s.title}
                {s.organization && <> — {s.organization}</>}
                {s.year && <> ({s.year})</>}
                {s.url && <> · <a href={s.url} target="_blank" rel="noopener noreferrer">link</a></>}
              </li>
            ))}
          </ul>
        </details>
      )}

      {insights && insights.uncertainty.length > 0 && (
        <div className="insight-panel">
          <h4>Limitations and uncertainty</h4>
          <ul className="evidence-list">
            {insights.uncertainty.map((u) => <li key={u}>{u}</li>)}
          </ul>
        </div>
      )}
    </section>
  );
}
