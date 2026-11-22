"use client";

import {
  Bar,
  BarChart,
  CartesianGrid,
  Cell,
  LabelList,
  Legend,
  PolarAngleAxis,
  PolarGrid,
  Radar,
  RadarChart,
  ResponsiveContainer,
  Tooltip,
  XAxis,
  YAxis
} from "recharts";
import PersonalizedInsights from "./PersonalizedInsights";
import type { PredictionResult } from "../lib/types";

// Re-exported so existing imports from this module keep working.
export type { PredictionResult };

type ModelMetrics = {
  mae: number;
  rmse: number;
  r2: number;
  accuracy: number;
  precision: number;
  recall: number;
  f1: number;
};

export type ModelMetricsSummary = Record<string, ModelMetrics>;

const MODEL_COLORS: Record<string, string> = {
  rf: "#00b894",
  xgb: "#0984e3",
  hybrid: "#fdcb6e",
  ensemble: "#6c5ce7"
};

function toPercent(v: number) {
  return `${(v * 100).toFixed(1)}%`;
}

function formatMetricName(metric: string) {
  return metric.replace("_", "-");
}

function percentBarLabel(value: number) {
  return `${value.toFixed(1)}%`;
}

function metricValueLabel(value: number) {
  return value.toFixed(3);
}

export default function ResultDashboard({ result, metrics }: { result: PredictionResult | null; metrics: ModelMetricsSummary | null }) {
  const predChartData = result
    ? Object.entries(result.model_outputs).map(([model, value]) => ({ model: model.toUpperCase(), value }))
    : [];

  const clsChartData = metrics
    ? ["accuracy", "precision", "recall", "f1"].map((metric) => ({
        metric: formatMetricName(metric.toUpperCase()),
        rf: (metrics.rf?.[metric as keyof ModelMetrics] ?? 0) * 100,
        xgb: (metrics.xgb?.[metric as keyof ModelMetrics] ?? 0) * 100,
        hybrid: (metrics.hybrid?.[metric as keyof ModelMetrics] ?? 0) * 100
      }))
    : [];

  const regChartData = metrics
    ? ["mae", "rmse", "r2"].map((metric) => ({
        metric: formatMetricName(metric.toUpperCase()),
        rf: metrics.rf?.[metric as keyof ModelMetrics] ?? 0,
        xgb: metrics.xgb?.[metric as keyof ModelMetrics] ?? 0,
        hybrid: metrics.hybrid?.[metric as keyof ModelMetrics] ?? 0
      }))
    : [];

  const radarData = metrics
    ? ["accuracy", "precision", "recall", "f1"].map((metric) => ({
        metric: formatMetricName(metric.toUpperCase()),
        RF: (metrics.rf?.[metric as keyof ModelMetrics] ?? 0) * 100,
        XGB: (metrics.xgb?.[metric as keyof ModelMetrics] ?? 0) * 100,
        HYBRID: (metrics.hybrid?.[metric as keyof ModelMetrics] ?? 0) * 100
      }))
    : [];

  return (
    <div className="card vibrant" style={{ marginTop: 16 }}>
      <h2>Results & Model Comparison</h2>

      {result && (
        <>
          <div className="stats-row">
            <div className="stat-pill">
              <span>Predicted Sleep Efficiency</span>
              <strong>{result.predicted_sleep_efficiency.toFixed(2)}%</strong>
            </div>
            <div className="stat-pill">
              <span>Predicted Quality</span>
              <strong className={result.sleep_quality_label === "Good" ? "badge-good" : "badge-poor"}>{result.sleep_quality_label}</strong>
            </div>
            <div className="stat-pill">
              <span>Inferred Sleep Time</span>
              <strong>{result.inferred_sleep_time}</strong>
            </div>
          </div>

          <h3>Prediction Output by Model</h3>
          <div className="chart-card">
            <p className="chart-caption">
              This chart compares the sleep-efficiency score predicted by each model for the same input. The ensemble
              score is the final combined result shown to the user.
            </p>
            <div style={{ width: "100%", height: 340 }}>
              <ResponsiveContainer>
                <BarChart data={predChartData} margin={{ top: 26, right: 18, left: 0, bottom: 6 }}>
                  <CartesianGrid strokeDasharray="4 4" stroke="#dbe7ff" />
                  <XAxis dataKey="model" tickLine={false} axisLine={false} />
                  <YAxis domain={[50, 100]} tickLine={false} axisLine={false} />
                  <Tooltip formatter={(v) => [`${Number(v).toFixed(2)}%`, "Predicted Efficiency"]} />
                  <Legend />
                  <Bar dataKey="value" name="Predicted Efficiency" radius={[10, 10, 0, 0]} maxBarSize={180}>
                    <LabelList dataKey="value" position="top" formatter={percentBarLabel} fill="#102236" fontSize={12} />
                    {predChartData.map((entry) => (
                      <Cell key={entry.model} fill={MODEL_COLORS[entry.model.toLowerCase()] ?? "#00b894"} />
                    ))}
                  </Bar>
                </BarChart>
              </ResponsiveContainer>
            </div>
            <p className="chart-note">Higher values indicate a better predicted sleep-efficiency outcome for the current record.</p>
          </div>

          <PersonalizedInsights
            insights={result.personalized_recommendations}
            diagnostics={result.prediction_diagnostics}
          />
        </>
      )}

      {metrics && (
        <>
          <h3>Detailed Metrics by Model</h3>
          <p className="disclosure">
            Historical results on the stored 75-row test split. That split was examined during development, so it is
            not an untouched hold-out, and there is no external validation. The target is a sleep-efficiency index
            constructed from the Kaggle Sleep Health dataset rather than a measured value, and the screen-activity
            sequences used in training are synthetic. Regression metrics (MAE, RMSE, R2) are primary. The
            threshold-derived metrics are secondary: both the index and the regression output are cut at 85, so they
            measure agreement on that band, not a separately trained classifier. ROC-AUC is not shown because it was
            computed from a rescaled regression output rather than a classifier probability.
          </p>
          <div className="metrics-grid">
            {Object.entries(metrics).map(([model, m]) => (
              <div key={model} className="metric-card">
                <h4>{model.toUpperCase()}</h4>
                <p><em>Regression (primary)</em></p>
                <p>MAE: <strong>{m.mae.toFixed(3)}</strong></p>
                <p>RMSE: <strong>{m.rmse.toFixed(3)}</strong></p>
                <p>R2: <strong>{m.r2.toFixed(3)}</strong></p>
                <p><em>Threshold-derived at 85 (secondary)</em></p>
                <p>Accuracy: <strong>{toPercent(m.accuracy)}</strong></p>
                <p>Precision: <strong>{toPercent(m.precision)}</strong></p>
                <p>Recall: <strong>{toPercent(m.recall)}</strong></p>
                <p>F1: <strong>{toPercent(m.f1)}</strong></p>
              </div>
            ))}
          </div>

          <h3>Threshold-Derived Metrics at 85 (%, secondary)</h3>
          <div className="chart-card">
            <p className="chart-caption">
              Accuracy, precision, recall and F1 after cutting both the index and each model's regression output at
              85. They describe agreement on that band and are secondary to the regression metrics.
            </p>
            <div style={{ width: "100%", height: 360 }}>
              <ResponsiveContainer>
                <BarChart data={clsChartData} margin={{ top: 28, right: 20, left: 0, bottom: 6 }}>
                  <CartesianGrid strokeDasharray="4 4" stroke="#dbe7ff" />
                  <XAxis dataKey="metric" tickLine={false} axisLine={false} />
                  <YAxis domain={[0, 100]} tickLine={false} axisLine={false} />
                  <Tooltip formatter={(v) => `${Number(v).toFixed(2)}%`} />
                  <Legend />
                  <Bar dataKey="rf" fill="#00b894" name="RF" radius={[8, 8, 0, 0]} maxBarSize={62}>
                    <LabelList dataKey="rf" position="top" formatter={percentBarLabel} fill="#102236" fontSize={11} />
                  </Bar>
                  <Bar dataKey="xgb" fill="#0984e3" name="XGB" radius={[8, 8, 0, 0]} maxBarSize={62}>
                    <LabelList dataKey="xgb" position="top" formatter={percentBarLabel} fill="#102236" fontSize={11} />
                  </Bar>
                  <Bar dataKey="hybrid" fill="#fdcb6e" name="HYBRID" radius={[8, 8, 0, 0]} maxBarSize={62}>
                    <LabelList dataKey="hybrid" position="top" formatter={percentBarLabel} fill="#102236" fontSize={11} />
                  </Bar>
                </BarChart>
              </ResponsiveContainer>
            </div>
          </div>

          <h3>Regression Metrics Comparison</h3>
          <div className="chart-card">
            <p className="chart-caption">
              Lower MAE and RMSE are better, while higher R2 is better. This chart makes the numerical prediction error
              of each model visible without hovering.
            </p>
            <div style={{ width: "100%", height: 340 }}>
              <ResponsiveContainer>
                <BarChart data={regChartData} margin={{ top: 28, right: 20, left: 0, bottom: 6 }}>
                  <CartesianGrid strokeDasharray="4 4" stroke="#dbe7ff" />
                  <XAxis dataKey="metric" tickLine={false} axisLine={false} />
                  <YAxis tickLine={false} axisLine={false} />
                  <Tooltip formatter={(v) => Number(v).toFixed(3)} />
                  <Legend />
                  <Bar dataKey="rf" fill="#00b894" name="RF" radius={[8, 8, 0, 0]} maxBarSize={72}>
                    <LabelList dataKey="rf" position="top" formatter={metricValueLabel} fill="#102236" fontSize={11} />
                  </Bar>
                  <Bar dataKey="xgb" fill="#0984e3" name="XGB" radius={[8, 8, 0, 0]} maxBarSize={72}>
                    <LabelList dataKey="xgb" position="top" formatter={metricValueLabel} fill="#102236" fontSize={11} />
                  </Bar>
                  <Bar dataKey="hybrid" fill="#fdcb6e" name="HYBRID" radius={[8, 8, 0, 0]} maxBarSize={72}>
                    <LabelList dataKey="hybrid" position="top" formatter={metricValueLabel} fill="#102236" fontSize={11} />
                  </Bar>
                </BarChart>
              </ResponsiveContainer>
            </div>
          </div>

          <h3>Model Radar (Threshold-Derived Metrics, secondary)</h3>
          <div className="chart-card">
            <p className="chart-caption">
              The radar view summarizes balance across the threshold-derived metrics. Exact values are also printed above in
              the detailed model cards and bar charts for easy review.
            </p>
            <div style={{ width: "100%", height: 380 }}>
              <ResponsiveContainer>
                <RadarChart data={radarData} margin={{ top: 16, right: 20, left: 20, bottom: 12 }}>
                  <PolarGrid />
                  <PolarAngleAxis dataKey="metric" />
                  <Tooltip formatter={(v) => `${Number(v).toFixed(2)}%`} />
                  <Legend />
                  <Radar name="RF" dataKey="RF" stroke="#00b894" fill="#00b894" fillOpacity={0.2} />
                  <Radar name="XGB" dataKey="XGB" stroke="#0984e3" fill="#0984e3" fillOpacity={0.2} />
                  <Radar name="HYBRID" dataKey="HYBRID" stroke="#fdcb6e" fill="#fdcb6e" fillOpacity={0.2} />
                </RadarChart>
              </ResponsiveContainer>
            </div>
          </div>
        </>
      )}

      {result && (
        <>
          <h3>Top Influencing Factors (XGBoost TreeSHAP)</h3>
          <ul>
            {result.top_influencing_factors.map((f) => (
              <li key={`${f.feature}-${f.impact}`}>
                {f.insight ?? `${f.feature} is currently ${f.direction ?? (f.impact >= 0 ? "supporting better sleep" : "reducing sleep quality")}.`}
              </li>
            ))}
          </ul>

          <h3>{result.personalized_recommendations ? "Quick Rule-based Tips" : "Recommendations"}</h3>
          <ul>
            {result.recommendations.map((r, i) => (
              <li key={i}>{r}</li>
            ))}
          </ul>
        </>
      )}
    </div>
  );
}
