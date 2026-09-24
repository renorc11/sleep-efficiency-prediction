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

export type PredictionResult = {
  predicted_sleep_efficiency: number;
  sleep_quality_label: string;
  inferred_sleep_time: string;
  model_outputs: Record<string, number>;
  top_influencing_factors: Array<{ feature: string; impact: number; value: number; direction?: string; feature_key?: string; insight?: string }>;
  recommendations: string[];
};

type ModelMetrics = {
  mae: number;
  rmse: number;
  r2: number;
  accuracy: number;
  precision: number;
  recall: number;
  f1: number;
  roc_auc: number;
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
    ? ["accuracy", "precision", "recall", "f1", "roc_auc"].map((metric) => ({
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
    ? ["accuracy", "precision", "recall", "f1", "roc_auc"].map((metric) => ({
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
        </>
      )}

      {metrics && (
        <>
          <h3>Detailed Metrics by Model</h3>
          <div className="metrics-grid">
            {Object.entries(metrics).map(([model, m]) => (
              <div key={model} className="metric-card">
                <h4>{model.toUpperCase()}</h4>
                <p>Accuracy: <strong>{toPercent(m.accuracy)}</strong></p>
                <p>Precision: <strong>{toPercent(m.precision)}</strong></p>
                <p>Recall: <strong>{toPercent(m.recall)}</strong></p>
                <p>F1: <strong>{toPercent(m.f1)}</strong></p>
                <p>ROC-AUC: <strong>{toPercent(m.roc_auc)}</strong></p>
                <p>MAE: <strong>{m.mae.toFixed(3)}</strong></p>
                <p>RMSE: <strong>{m.rmse.toFixed(3)}</strong></p>
                <p>R2: <strong>{m.r2.toFixed(3)}</strong></p>
              </div>
            ))}
          </div>

          <h3>Classification Metrics Comparison (%)</h3>
          <div className="chart-card">
            <p className="chart-caption">
              Accuracy, precision, recall, F1, and ROC-AUC are shown together so each model can be compared in a more
              complete way than a single score.
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

          <h3>Model Radar (Classification Strength)</h3>
          <div className="chart-card">
            <p className="chart-caption">
              The radar view summarizes balance across classification metrics. Exact values are also printed above in
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
          <h3>Top Influencing Factors</h3>
          <ul>
            {result.top_influencing_factors.map((f) => (
              <li key={`${f.feature}-${f.impact}`}>
                {f.insight ?? `${f.feature} is currently ${f.direction ?? (f.impact >= 0 ? "supporting better sleep" : "reducing sleep quality")}.`}
              </li>
            ))}
          </ul>

          <h3>Recommendations</h3>
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
