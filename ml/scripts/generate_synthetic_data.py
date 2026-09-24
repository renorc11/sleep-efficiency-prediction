from __future__ import annotations

from pathlib import Path
import numpy as np
import pandas as pd

RNG = np.random.default_rng(42)
SEQ_LEN = 288


def generate_sequence(sleep_start_idx: int, wake_idx: int, fragmentation: float) -> np.ndarray:
    seq = np.ones(SEQ_LEN, dtype=np.int8)

    # daytime mostly active with sporadic inactivity
    for i in range(SEQ_LEN):
        hh = i / 12.0
        if 8 <= hh <= 22:
            seq[i] = 1 if RNG.random() > 0.12 else 0
        else:
            seq[i] = 0 if RNG.random() > 0.15 else 1

    idx = sleep_start_idx
    while idx != wake_idx:
        seq[idx] = 0
        idx = (idx + 1) % SEQ_LEN

    sleep_span = (wake_idx - sleep_start_idx) % SEQ_LEN
    bursts = int(fragmentation * sleep_span)
    if sleep_span > 0 and bursts > 0:
        burst_positions = RNG.choice(sleep_span, size=min(bursts, sleep_span), replace=False)
        for b in burst_positions:
            seq[(sleep_start_idx + int(b)) % SEQ_LEN] = 1

    return seq


def run(n_samples: int = 3000) -> None:
    rows = []
    sequences = np.zeros((n_samples, SEQ_LEN), dtype=np.int8)

    for i in range(n_samples):
        caffeine = float(np.clip(RNG.normal(130, 70), 0, 400))
        alcohol = float(np.clip(RNG.normal(1.0, 1.0), 0, 5))
        dinner_hour = float(np.clip(RNG.normal(20.0, 1.2), 17.0, 23.5))
        stress = float(np.clip(RNG.normal(5.5, 2.0), 1, 10))
        mood = float(np.clip(RNG.normal(0.1, 1.0), -2, 2))
        activity = float(np.clip(RNG.normal(45, 25), 0, 180))
        naps_count = int(np.clip(np.round(RNG.normal(0.7, 0.8)), 0, 3))
        naps_total = float(np.clip(naps_count * RNG.normal(25, 10), 0, 120))
        prev_sleep = float(np.clip(RNG.normal(7.0, 1.0), 4, 10))
        work_hours = float(np.clip(RNG.normal(8.0, 2.0), 2, 14))
        blue_light = float(np.clip(RNG.normal(180, 80), 20, 500))
        bedtime_consistency = float(np.clip(RNG.beta(4, 2), 0, 1))
        hydration = float(np.clip(RNG.normal(2.2, 0.7), 0.8, 5))

        expected_sleep_hour = np.clip(
            22.7
            + 0.003 * caffeine
            + 0.15 * alcohol
            + 0.08 * stress
            - 0.01 * activity
            - 0.4 * bedtime_consistency
            + RNG.normal(0, 0.5),
            21,
            26,
        )
        sleep_start_idx = int((expected_sleep_hour % 24) * 12)
        sleep_duration_h = np.clip(7.8 - 0.3 * alcohol - 0.05 * stress + 0.02 * activity / 10 + RNG.normal(0, 0.6), 5.0, 9.5)
        wake_idx = int(((expected_sleep_hour + sleep_duration_h) % 24) * 12)

        fragmentation = float(np.clip(0.03 + 0.012 * stress + 0.002 * alcohol * 10 + 0.001 * blue_light + RNG.normal(0, 0.01), 0.01, 0.35))
        seq = generate_sequence(sleep_start_idx, wake_idx, fragmentation)
        sequences[i] = seq

        nighttime_idxs = list(range(22 * 12, 24 * 12)) + list(range(0, 7 * 12))
        night_bursts = int(np.sum(seq[nighttime_idxs]))
        sleep_eff = (
            89
            - 0.03 * caffeine
            - 1.7 * alcohol
            - 1.8 * max(0, 2 - ((expected_sleep_hour % 24) - dinner_hour) % 24)
            - 1.9 * stress
            + 1.7 * mood
            + 0.04 * activity
            - 0.02 * naps_total
            + 1.8 * bedtime_consistency
            + 0.9 * (prev_sleep - 7)
            - 0.35 * night_bursts
            - 45 * fragmentation
            + RNG.normal(0, 2.2)
        )
        sleep_eff = float(np.clip(sleep_eff, 50, 98))
        label = int(sleep_eff >= 85)

        rows.append(
            {
                "caffeine_mg_evening": caffeine,
                "alcohol_units": alcohol,
                "dinner_hour": dinner_hour,
                "stress_level": stress,
                "mood_score": mood,
                "physical_activity_min": activity,
                "naps_count": naps_count,
                "naps_total_min": naps_total,
                "previous_sleep_hours": prev_sleep,
                "work_hours": work_hours,
                "blue_light_minutes": blue_light,
                "bedtime_consistency": bedtime_consistency,
                "hydration_liters": hydration,
                "sleep_efficiency": sleep_eff,
                "sleep_quality_label": label,
            }
        )

    out_raw = Path("ml/data/raw")
    out_raw.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(out_raw / "lifestyle_sleep.csv", index=False)
    np.save(out_raw / "screen_activity_sequences.npy", sequences)
    print(f"Saved dataset with {n_samples} samples to {out_raw}")


if __name__ == "__main__":
    run()
