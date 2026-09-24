from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

SEQ_LEN = 288


def infer_sleep_features(seq: np.ndarray) -> tuple[float, float, int]:
    evening_start = 18 * 12
    morning_end = 8 * 12
    extended = np.concatenate([seq, seq[:morning_end]])

    best_len = 0
    best_idx = 23 * 12
    i = evening_start
    while i < len(extended):
        if extended[i] == 0:
            j = i
            while j < len(extended) and extended[j] == 0:
                j += 1
            run_len = j - i
            if run_len >= 6 and run_len > best_len:
                best_len = run_len
                best_idx = i % SEQ_LEN
            i = j
        i += 1

    night_idxs = list(range(22 * 12, 24 * 12)) + list(range(0, 7 * 12))
    night_bursts = int(np.sum(seq[night_idxs]))
    return best_idx / 12.0, best_len * 5.0, night_bursts


def run(test_size: float = 0.2, val_size: float = 0.1) -> None:
    raw_dir = Path("ml/data/raw")
    proc_dir = Path("ml/data/processed")
    proc_dir.mkdir(parents=True, exist_ok=True)

    lifestyle_path = raw_dir / "lifestyle_sleep.csv"
    seq_path = raw_dir / "screen_activity_sequences.npy"
    if not lifestyle_path.exists() or not seq_path.exists():
        raise FileNotFoundError(
            "Raw dataset missing. Run `bash ml/scripts/run_pipeline.sh` or prepare raw data first."
        )

    df = pd.read_csv(lifestyle_path)
    seq = np.load(seq_path)

    inferred = np.array([infer_sleep_features(s) for s in seq])
    df["inferred_sleep_start_hour"] = inferred[:, 0]
    df["longest_evening_inactivity_min"] = inferred[:, 1]
    df["nighttime_activity_bursts"] = inferred[:, 2]
    df["dinner_to_sleep_gap"] = (df["inferred_sleep_start_hour"] - df["dinner_hour"]) % 24
    df["caffeine_stress_interaction"] = df["caffeine_mg_evening"] * df["stress_level"]
    df["activity_stress_balance"] = df["physical_activity_min"] / (df["stress_level"] + 1e-6)
    df["nap_sleep_debt_proxy"] = (8 - df["previous_sleep_hours"]).clip(lower=0) + 0.01 * df["naps_total_min"]

    # Keep classification target balanced enough for stratified splits.
    quality_threshold = 85.0
    df["sleep_quality_label"] = (df["sleep_efficiency"] >= quality_threshold).astype(int)
    counts = df["sleep_quality_label"].value_counts()
    if counts.min() < 2:
        quality_threshold = float(df["sleep_efficiency"].median())
        df["sleep_quality_label"] = (df["sleep_efficiency"] >= quality_threshold).astype(int)

    feature_cols = [c for c in df.columns if c not in ["sleep_efficiency", "sleep_quality_label"]]

    idx = np.arange(len(df))
    train_idx, test_idx = train_test_split(idx, test_size=test_size, random_state=42, stratify=df["sleep_quality_label"])
    train_idx, val_idx = train_test_split(
        train_idx,
        test_size=val_size / (1 - test_size),
        random_state=42,
        stratify=df.iloc[train_idx]["sleep_quality_label"],
    )

    split = {
        "train_idx": train_idx.tolist(),
        "val_idx": val_idx.tolist(),
        "test_idx": test_idx.tolist(),
        "feature_columns": feature_cols,
        "quality_threshold": quality_threshold,
    }

    df.to_csv(proc_dir / "dataset_engineered.csv", index=False)
    np.save(proc_dir / "sequences.npy", seq)
    with (proc_dir / "split.json").open("w", encoding="utf-8") as f:
        json.dump(split, f, indent=2)

    print(f"Saved engineered dataset and split metadata to {proc_dir}")


if __name__ == "__main__":
    run()
