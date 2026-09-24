from __future__ import annotations

import json
from pathlib import Path


def run() -> None:
    seq = [1] * 288
    # likely sleep from 23:00 to 06:30 => inactivity
    for i in list(range(23 * 12, 24 * 12)) + list(range(0, 6 * 12 + 6)):
        seq[i] = 0

    payload = {
        "lifestyle": {
            "caffeine_mg_evening": 140,
            "alcohol_units": 1,
            "dinner_hour": 20.5,
            "stress_level": 6,
            "mood_score": 0,
            "physical_activity_min": 35,
            "naps_count": 1,
            "naps_total_min": 25,
            "previous_sleep_hours": 6.8,
            "work_hours": 9,
            "blue_light_minutes": 220,
            "bedtime_consistency": 0.65,
            "hydration_liters": 2.1,
        },
        "screen_activity_sequence": seq,
    }

    out = Path("ml/data/raw/example_prediction_payload.json")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(f"Wrote {out}")


if __name__ == "__main__":
    run()
