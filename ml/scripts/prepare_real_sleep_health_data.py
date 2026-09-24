from __future__ import annotations

from pathlib import Path
import numpy as np
import pandas as pd

SEQ_LEN = 288
RNG = np.random.default_rng(42)


def pick_source_csv() -> Path:
    candidates = [
        Path("ml/data/raw/Sleep_health_and_lifestyle_dataset.csv"),
        Path("ml/data/raw/source_notebooks/Sleep_health_and_lifestyle_dataset.csv"),
    ]
    for p in candidates:
        if p.exists():
            return p
    raise FileNotFoundError(
        "Sleep_health_and_lifestyle_dataset.csv not found. Place it at ml/data/raw/ or ml/data/raw/source_notebooks/."
    )


def parse_bp(bp: str) -> tuple[float, float]:
    try:
        s, d = str(bp).split("/")
        return float(s), float(d)
    except Exception:
        return 125.0, 80.0


def generate_sequence(sleep_start_hour: float, sleep_duration_h: float, stress: float, blue_light_minutes: float) -> np.ndarray:
    seq = np.ones(SEQ_LEN, dtype=np.int8)

    sleep_start_idx = int((sleep_start_hour % 24) * 12)
    wake_idx = int(((sleep_start_hour + sleep_duration_h) % 24) * 12)

    # Base activity pattern
    for i in range(SEQ_LEN):
        hh = i / 12.0
        if 8 <= hh <= 22:
            seq[i] = 1 if RNG.random() > 0.10 else 0
        else:
            seq[i] = 0 if RNG.random() > 0.18 else 1

    # Sleep window inactivity
    idx = sleep_start_idx
    while idx != wake_idx:
        seq[idx] = 0
        idx = (idx + 1) % SEQ_LEN

    # Fragmentation linked to stress + blue light
    sleep_span = (wake_idx - sleep_start_idx) % SEQ_LEN
    frag = np.clip(0.02 + 0.015 * stress + 0.0009 * blue_light_minutes, 0.01, 0.35)
    bursts = int(frag * sleep_span)
    if sleep_span > 0 and bursts > 0:
        burst_positions = RNG.choice(sleep_span, size=min(bursts, sleep_span), replace=False)
        for b in burst_positions:
            seq[(sleep_start_idx + int(b)) % SEQ_LEN] = 1

    return seq


def run() -> None:
    source_csv = pick_source_csv()
    df = pd.read_csv(source_csv)

    required = [
        "Age",
        "Occupation",
        "Sleep Duration",
        "Quality of Sleep",
        "Physical Activity Level",
        "Stress Level",
        "BMI Category",
        "Blood Pressure",
        "Heart Rate",
        "Daily Steps",
        "Sleep Disorder",
    ]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"Dataset missing expected columns: {missing}")

    rows = []
    sequences = np.zeros((len(df), SEQ_LEN), dtype=np.int8)

    work_base = {
        "Doctor": 10.0,
        "Nurse": 9.5,
        "Engineer": 9.0,
        "Lawyer": 9.0,
        "Teacher": 8.0,
        "Accountant": 8.5,
        "Salesperson": 8.5,
        "Sales Representative": 8.5,
        "Scientist": 8.5,
        "Software Engineer": 9.0,
        "Manager": 9.0,
    }

    for i, r in df.iterrows():
        age = float(r["Age"])
        sleep_h = float(r["Sleep Duration"])
        quality = float(r["Quality of Sleep"])
        activity = float(r["Physical Activity Level"])
        stress = float(r["Stress Level"])
        steps = float(r["Daily Steps"])
        hr = float(r["Heart Rate"])
        occ = str(r["Occupation"])
        bmi_cat = str(r["BMI Category"])
        disorder = str(r["Sleep Disorder"])
        sys_bp, dia_bp = parse_bp(r["Blood Pressure"])

        # Map to project feature space (some fields approximated from available columns).
        alcohol = float(np.clip(0.3 + 0.18 * (stress - 4) + (0.7 if "Insomnia" in disorder else 0.2) + RNG.normal(0, 0.35), 0, 5))
        caffeine = float(np.clip(80 + 11 * (stress - 4) + 2.5 * (hr - 70) + RNG.normal(0, 35), 0, 400))
        dinner_hour = float(np.clip(19.0 + 0.12 * (work_base.get(occ, 8.5) - 8) + RNG.normal(0, 0.8), 17.0, 23.5))
        mood = float(np.clip(1.0 - 0.22 * stress + 0.015 * (quality - 5) + RNG.normal(0, 0.45), -2, 2))

        physical_activity_min = float(np.clip(activity * 1.2, 0, 300))
        naps_count = int(np.clip(np.round((7 - sleep_h) * 0.7 + RNG.normal(0.2, 0.5)), 0, 3))
        naps_total = float(np.clip(naps_count * (20 + 3 * max(0, 7 - sleep_h)) + RNG.normal(0, 8), 0, 180))
        prev_sleep = float(np.clip(sleep_h + RNG.normal(0, 0.5), 3.5, 10.5))
        work_hours = float(np.clip(work_base.get(occ, 8.5) + RNG.normal(0, 0.8), 4, 14))

        bedtime_consistency = float(np.clip(0.82 - 0.05 * max(0, stress - 5) - 0.06 * ("Shift" in occ) + RNG.normal(0, 0.08), 0, 1))
        hydration = float(np.clip(2.0 + 0.004 * activity - 0.0015 * sys_bp + RNG.normal(0, 0.25), 0.8, 5.0))

        sleep_eff = float(np.clip(quality * 10 + 2.3 * (sleep_h - 6.5) - 0.7 * max(0, stress - 6) - 0.5 * ("Sleep Apnea" in disorder) * 4 + RNG.normal(0, 2.0), 50, 98))

        expected_sleep_start_hour = float(np.clip(23.0 + 0.22 * alcohol + 0.0025 * caffeine + 0.08 * stress - 0.007 * physical_activity_min - 0.45 * bedtime_consistency + RNG.normal(0, 0.45), 20.0, 26.0))

        # Auto blue-light estimate from behavior and stress (later refined by model with sequence features).
        blue_light = float(np.clip(90 + 8 * stress + 0.03 * steps + 12 * max(0, expected_sleep_start_hour % 24 - 22) + RNG.normal(0, 40), 20, 480))

        seq = generate_sequence(expected_sleep_start_hour % 24, sleep_h, stress, blue_light)
        sequences[i] = seq

        rows.append(
            {
                "caffeine_mg_evening": caffeine,
                "alcohol_units": alcohol,
                "dinner_hour": dinner_hour,
                "stress_level": stress,
                "mood_score": mood,
                "physical_activity_min": physical_activity_min,
                "naps_count": naps_count,
                "naps_total_min": naps_total,
                "previous_sleep_hours": prev_sleep,
                "work_hours": work_hours,
                "blue_light_minutes": blue_light,
                "bedtime_consistency": bedtime_consistency,
                "hydration_liters": hydration,
                "sleep_efficiency": sleep_eff,
                "sleep_quality_label": int(sleep_eff >= 85),
            }
        )

    out_raw = Path("ml/data/raw")
    out_raw.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(out_raw / "lifestyle_sleep.csv", index=False)
    np.save(out_raw / "screen_activity_sequences.npy", sequences)
    print(f"Prepared real dataset from {source_csv} -> {out_raw / 'lifestyle_sleep.csv'}")


if __name__ == "__main__":
    run()
