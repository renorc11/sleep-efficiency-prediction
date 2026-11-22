from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime


@dataclass
class InactivityFeatures:
    inferred_sleep_start_idx: int
    inferred_sleep_time: str
    longest_evening_inactivity_min: float
    nighttime_activity_bursts: int


def idx_to_hhmm(idx: int) -> str:
    total_minutes = idx * 5
    hh = (total_minutes // 60) % 24
    mm = total_minutes % 60
    return f"{hh:02d}:{mm:02d}"


def hhmm_to_idx(hhmm: str) -> int:
    hh, mm = hhmm.split(":")
    return int(hh) * 12 + int(mm) // 5


def circular_distance(a: int, b: int, mod: int) -> int:
    d = abs(a - b)
    return min(d, mod - d)


def infer_sleep_from_sequence(sequence: list[int], sleep_time_hint: str | None = None) -> InactivityFeatures:
    evening_start = 18 * 12
    morning_end = 8 * 12

    # Ignore future bins to avoid false "sleep at 18:00" in daytime predictions.
    now = datetime.now()
    current_idx = (now.hour * 60 + now.minute) // 5
    observed = sequence[:]
    for i in range(current_idx + 1, len(observed)):
        observed[i] = 1

    best_idx = 23 * 12
    best_len = 0
    candidates: list[tuple[int, int]] = []

    extended = observed + observed[:morning_end]
    i = evening_start
    while i < len(extended):
        if extended[i] == 0:
            j = i
            while j < len(extended) and extended[j] == 0:
                j += 1
            run_len = j - i
            candidate_idx = i % len(sequence)

            # Only accept inactivity runs that genuinely begin after screen activity.
            started_after_activity = sequence[(candidate_idx - 1) % len(sequence)] == 1

            # Only consider realistic bedtime candidates (20:00-03:00).
            in_bedtime_window = (candidate_idx >= 20 * 12) or (candidate_idx <= 3 * 12)

            if (
                run_len >= 6
                and in_bedtime_window
                and candidate_idx <= current_idx
                and started_after_activity
            ):
                candidates.append((candidate_idx, run_len))
                if run_len > best_len:
                    best_len = run_len
                    best_idx = candidate_idx
            i = j
        else:
            i += 1

    # If user gives approximate sleep time, prefer a nearby inactivity candidate.
    if sleep_time_hint and candidates:
        hint_idx = hhmm_to_idx(sleep_time_hint)
        nearby = [c for c in candidates if circular_distance(c[0], hint_idx, len(sequence)) <= 24]
        if nearby:
            sleep_start_idx = max(nearby, key=lambda it: it[1])[0]
        else:
            sleep_start_idx = min(candidates, key=lambda it: circular_distance(it[0], hint_idx, len(sequence)))[0]
    elif candidates:
        sleep_start_idx = max(candidates, key=lambda it: it[1])[0]
    else:
        # Fallback to hint if provided, otherwise default around 23:00.
        sleep_start_idx = hhmm_to_idx(sleep_time_hint) if sleep_time_hint else best_idx

    sleep_time = idx_to_hhmm(sleep_start_idx)

    # 22:00 to 07:00 window
    night_idxs = list(range(22 * 12, 24 * 12)) + list(range(0, 7 * 12))
    bursts = sum(1 for i in night_idxs if i <= current_idx and sequence[i] == 1)

    return InactivityFeatures(
        inferred_sleep_start_idx=sleep_start_idx,
        inferred_sleep_time=sleep_time,
        longest_evening_inactivity_min=max(best_len, max((c[1] for c in candidates), default=0)) * 5,
        nighttime_activity_bursts=bursts,
    )


def build_engineered_features(
    lifestyle: dict,
    sequence: list[int],
    sleep_time_hint: str | None = None,
) -> tuple[dict, InactivityFeatures]:
    inactivity = infer_sleep_from_sequence(sequence, sleep_time_hint=sleep_time_hint)

    sleep_hour = inactivity.inferred_sleep_start_idx / 12.0
    dinner_gap = (sleep_hour - lifestyle["dinner_hour"]) % 24

    features = {
        **lifestyle,
        "inferred_sleep_start_hour": sleep_hour,
        "longest_evening_inactivity_min": inactivity.longest_evening_inactivity_min,
        "nighttime_activity_bursts": inactivity.nighttime_activity_bursts,
        "dinner_to_sleep_gap": dinner_gap,
        "caffeine_stress_interaction": lifestyle["caffeine_mg_evening"] * lifestyle["stress_level"],
        "activity_stress_balance": lifestyle["physical_activity_min"] / (lifestyle["stress_level"] + 1e-6),
        "nap_sleep_debt_proxy": max(0.0, 8 - lifestyle["previous_sleep_hours"]) + 0.01 * lifestyle["naps_total_min"],
    }
    return features, inactivity
