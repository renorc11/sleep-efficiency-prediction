# SleepEfficiencyAndroidApp

Standalone Android Studio project for lifestyle-based sleep efficiency prediction.

## What this app does
- Collects lifestyle inputs from user.
- Automatically builds 288-bin screen activity sequence from Android `UsageStatsManager` (no manual sequence entry).
- Automatically computes `blue_light_minutes` from recent device usage (2 hours before inferred sleep start).
- Provides optional blue-light override input for user correction.
- Calls your existing FastAPI backend:
  - `POST /api/v1/predict`
  - `GET /api/v1/explain/model-metrics`
- Displays prediction, explainability factors, recommendations, and per-model metrics.

## Requirements
- Android Studio Jellyfish or newer
- Android SDK 35
- Backend running on host machine at `http://localhost:8000`

## Setup
1. Open this folder in Android Studio:
   - `<repository root>/SleepEfficiencyAndroidApp`
2. Let Gradle sync and install missing SDK packages.
3. Ensure backend is running.
4. Run app on emulator or physical device.

## API URL
Configured in `app/build.gradle.kts`:
- `API_BASE_URL = "http://10.0.2.2:8000/api/v1/"`

For physical device testing, replace this with your Mac local IP, e.g.:
- `http://192.168.1.20:8000/api/v1/`

## First-run steps in app
1. Tap **Open Settings** in Usage Access card.
2. Enable usage access for **Sleep Efficiency** app.
3. Return to app and tap **I Enabled It**.
4. Fill form and tap **Predict Sleep Efficiency**.

## Notes
- This does not use EEG/ECG/wearables.
- Screen data is inferred from app foreground usage events. Each foreground interval
  (resume to pause) is marked active, so continuous use of one app is no longer
  recorded as inactivity.
- The deployed models were trained on **synthetic** activity sequences. They have not
  been trained or validated on Android-observed usage, so phone-based predictions are
  exploratory. Inactivity is an indirect proxy, not a measurement of sleep.
- The sequence covers local midnight to now. A bedtime before midnight on the
  previous evening is not included; this is a known limitation shared with the
  backend's sleep-time inference.
- `POST /predict` can take up to about 45 s while the backend generates
  recommendations; the HTTP client uses a 90 s read timeout.
- Request logging is debug-only and metadata-only (no request bodies).
- The app shows the original prediction fields. The backend's newer
  `personalized_recommendations` field is ignored by this client for now.
- If metrics are missing, run ML training pipeline in backend project first.
