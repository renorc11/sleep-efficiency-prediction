from importlib.util import module_from_spec, spec_from_file_location
from pathlib import Path

import pandas as pd


MODULE_PATH = Path(__file__).resolve().parents[1] / "scripts" / "integrate_external_datasets.py"
SPEC = spec_from_file_location("dataset_integration", MODULE_PATH)
integration = module_from_spec(SPEC)
assert SPEC and SPEC.loader
SPEC.loader.exec_module(integration)


def test_normalize_efficiency_accepts_fraction_and_percent():
    assert integration.normalize_efficiency(0.88) == 88.0
    assert integration.normalize_efficiency(88) == 88.0
    assert integration.normalize_efficiency(110) is None


def test_calculates_efficiency_only_for_valid_durations():
    assert integration.calculate_sleep_efficiency(6, 8) == 75.0
    assert integration.calculate_sleep_efficiency(6, 0) is None
    assert integration.calculate_sleep_efficiency(10, 8) is None


def test_duration_handles_overnight_values_without_date_rollover():
    start = pd.Series(pd.to_datetime(["2021-01-01 23:00:00"]))
    end = pd.Series(pd.to_datetime(["2021-01-01 07:00:00"]))
    assert integration.duration_hours(start, end).iloc[0] == 8


def test_filename_participant_extraction():
    value = integration.source_participant_from_filename(Path("running_app_u09.csv"), r"running_app_(u\d+)\.csv")
    assert value == "u09"
