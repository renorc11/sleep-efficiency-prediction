const API_BASE = process.env.NEXT_PUBLIC_API_BASE_URL || "http://localhost:8000/api/v1";

export async function predict(payload: unknown) {
  const res = await fetch(`${API_BASE}/predict`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(payload)
  });
  if (!res.ok) {
    const text = await res.text();
    throw new Error(`Prediction failed: ${text}`);
  }
  return res.json();
}

export async function fetchHistory() {
  const res = await fetch(`${API_BASE}/history`);
  if (!res.ok) throw new Error("History fetch failed");
  return res.json();
}

export async function fetchModelMetrics() {
  const res = await fetch(`${API_BASE}/explain/model-metrics`);
  if (!res.ok) throw new Error("Model metrics fetch failed");
  return res.json();
}
