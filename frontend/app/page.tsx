"use client";

import { useEffect, useState } from "react";
import LifestyleForm, { LifestylePayload } from "../components/LifestyleForm";
import ResultDashboard, { PredictionResult, ModelMetricsSummary } from "../components/ResultDashboard";
import ScreenTracker, { useScreenActivitySequence } from "../components/ScreenTracker";
import { fetchModelMetrics, predict } from "../lib/api";
import SleepAssistant from "../components/SleepAssistant";

export default function Home() {
  const [loading, setLoading] = useState(false);
  const [result, setResult] = useState<PredictionResult | null>(null);
  const [metrics, setMetrics] = useState<ModelMetricsSummary | null>(null);
  const [error, setError] = useState<string | null>(null);
  const { sequence, inactivitySummary } = useScreenActivitySequence();

  useEffect(() => {
    fetchModelMetrics()
      .then((data) => setMetrics((data.metrics_summary ?? null) as ModelMetricsSummary | null))
      .catch(() => setMetrics(null));
  }, []);

  async function handleSubmit(lifestyle: LifestylePayload) {
    setLoading(true);
    setError(null);
    try {
      const res = await predict({ lifestyle, screen_activity_sequence: sequence });
      setResult(res as PredictionResult);
    } catch (e) {
      setError((e as Error).message);
    } finally {
      setLoading(false);
    }
  }

  return (
    <main>
      <div className="hero">
        <div className="hero-copy">
          <span className="hero-kicker">Behavioral sleep intelligence</span>
          <h1>Explainable Machine Learning for Lifestyle-Based Sleep Efficiency Prediction Using Behavioral and Screen Inactivity Signals</h1>
          <p className="small">Lifestyle behavior + 5-minute screen inactivity patterns. No EEG, ECG, or wearables.</p>
        </div>
      </div>
      <ScreenTracker summary={inactivitySummary} />
      <LifestyleForm onSubmit={handleSubmit} loading={loading} />
      {error && <p className="badge-poor">{error}</p>}
      {(result || metrics) && <ResultDashboard result={result} metrics={metrics} />}
      <SleepAssistant />
    </main>
  );
}
