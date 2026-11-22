from datetime import datetime as RealDateTime

import pytest

from app.services import inactivity_service as service


class FixedDateTime:
    """Provides a predictable current time for deterministic tests."""

    @classmethod
    def now(cls):
        return RealDateTime(2026, 11, 16, 23, 59)


@pytest.fixture
def fixed_time(monkeypatch):
    monkeypatch.setattr(service, "datetime", FixedDateTime)


def make_sequence(default=1):
    return [default] * 288


def test_idx_to_hhmm():
    assert service.idx_to_hhmm(0) == "00:00"
    assert service.idx_to_hhmm(12) == "01:00"
    assert service.idx_to_hhmm(276) == "23:00"
    assert service.idx_to_hhmm(288) == "00:00"


def test_hhmm_to_idx():
    assert service.hhmm_to_idx("00:00") == 0
    assert service.hhmm_to_idx("01:00") == 12
    assert service.hhmm_to_idx("23:00") == 276
    assert service.hhmm_to_idx("23:59") == 287


def test_circular_distance_across_midnight():
    assert service.circular_distance(287, 1, 288) == 2


def test_detects_continuous_overnight_inactivity(fixed_time):
    sequence = make_sequence()
    # Inactive from 23:00 through 01:00, crossing midnight.
    for i in list(range(276, 288)) + list(range(0, 12)):
        sequence[i] = 0

    result = service.infer_sleep_from_sequence(sequence)

    assert result.inferred_sleep_time == "23:00"
    assert result.longest_evening_inactivity_min == 120


def test_uses_sleep_time_hint_when_no_inactivity_is_detected(fixed_time):
    sequence = make_sequence()

    result = service.infer_sleep_from_sequence(
        sequence,
        sleep_time_hint="22:30",
    )

    assert result.inferred_sleep_time == "22:30"
    assert result.longest_evening_inactivity_min == 0


def test_defaults_to_2300_when_no_candidate_or_hint(fixed_time):
    sequence = make_sequence()

    result = service.infer_sleep_from_sequence(sequence)

    assert result.inferred_sleep_time == "23:00"
    assert result.longest_evening_inactivity_min == 0


def test_hint_selects_nearby_candidate(fixed_time):
    sequence = make_sequence()

    # Candidate 1: 22:00–22:55 (60 minutes).
    for i in range(264, 276):
        sequence[i] = 0

    # Activity at 23:00 separates the two candidates.
    sequence[276] = 1

    # Candidate 2: 23:05–00:30 (90 minutes).
    for i in range(277, 295):
        sequence[i % 288] = 0

    result = service.infer_sleep_from_sequence(
        sequence,
        sleep_time_hint="22:30",
    )

    # Both candidates are within two hours of the hint.
    # The longer inactivity period should be selected.
    assert result.inferred_sleep_time == "23:05"
    assert result.longest_evening_inactivity_min == 90

def test_future_night_bins_are_not_counted_as_activity(monkeypatch):
    class FixedDateTime:
        @classmethod
        def now(cls):
            return RealDateTime(2026, 11, 16, 21, 0)

    monkeypatch.setattr(service, "datetime", FixedDateTime)

    # No activity has been recorded in any time slot.
    sequence = make_sequence(default=0)

    result = service.infer_sleep_from_sequence(sequence)

    # At 21:00, future nighttime slots must not count as observed activity.
    assert result.nighttime_activity_bursts == 0

def test_all_zero_sequence_uses_2300_sleep_time_hint(fixed_time):
    # Entire 24-hour sequence is marked inactive.
    sequence = make_sequence(default=0)

    result = service.infer_sleep_from_sequence(
        sequence,
        sleep_time_hint="23:00",
    )

    assert result.inferred_sleep_time == "23:00"
    assert result.inferred_sleep_start_idx == 276