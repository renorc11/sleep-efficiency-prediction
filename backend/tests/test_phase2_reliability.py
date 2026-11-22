"""Phase 2 reliability checks: input validation, output-domain policy,
global SHAP sampling, thread safety, and startup import paths.

Run from the project root (paths below are project-root relative).
"""

import json
import subprocess
import sys
import threading
import time
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from pydantic import ValidationError

import app.services.model_service as model_service
from app.schemas.prediction import PredictionRequest
from app.services.model_service import (
    ENSEMBLE_CLIP_RANGE,
    ModelBundle,
    build_prediction_diagnostics,
)

PROJECT_ROOT = Path(__file__).resolve().parents[2]

LIFESTYLE = {
    "caffeine_mg_evening": 100,
    "alcohol_units": 0,
    "dinner_hour": 20,
    "stress_level": 5,
    "mood_score": 1,
    "physical_activity_min": 30,
    "naps_count": 1,
    "naps_total_min": 20,
    "previous_sleep_hours": 7,
    "work_hours": 8,
    "blue_light_minutes": 60,
    "bedtime_consistency": 0.8,
    "hydration_liters": 2,
}


@pytest.fixture(scope="module")
def bundle_and_rows():
    bundle = ModelBundle()
    frame = pd.read_csv(PROJECT_ROOT / "ml/data/processed/dataset_engineered.csv")
    sequences = np.load(PROJECT_ROOT / "ml/data/processed/sequences.npy")
    return bundle, frame, sequences


# ---------------------------------------------------------------- validation


def test_request_rejects_non_binary_sequence():
    with pytest.raises(ValidationError, match="only 0"):
        PredictionRequest(lifestyle=LIFESTYLE, screen_activity_sequence=[0] * 287 + [5])


def test_request_accepts_binary_sequence():
    request = PredictionRequest(lifestyle=LIFESTYLE, screen_activity_sequence=[0, 1] * 144)
    assert len(request.screen_activity_sequence) == 288


@pytest.mark.parametrize(
    "sequence, message",
    [
        ([0] * 287, "288"),
        (np.zeros((2, 288)), "shape"),
        ([0] * 287 + [2], "binary"),
        ([0] * 287 + [0.5], "binary"),
    ],
)
def test_model_rejects_invalid_sequences(bundle_and_rows, sequence, message):
    bundle, frame, _ = bundle_and_rows
    features = frame.iloc[0][bundle.features].to_dict()
    with pytest.raises(ValueError, match=message):
        bundle.predict_all(features, sequence)


@pytest.mark.parametrize("bad_value", [float("nan"), float("inf")])
def test_model_rejects_non_finite_features(bundle_and_rows, bad_value):
    bundle, frame, sequences = bundle_and_rows
    features = frame.iloc[0][bundle.features].to_dict()
    features["stress_level"] = bad_value
    with pytest.raises(ValueError, match="finite"):
        bundle.predict_all(features, sequences[0].tolist())


def test_model_rejects_missing_features(bundle_and_rows):
    bundle, frame, sequences = bundle_and_rows
    features = frame.iloc[0][bundle.features].to_dict()
    del features["work_hours"]
    with pytest.raises(ValueError, match="work_hours"):
        bundle.predict_all(features, sequences[0].tolist())


# ------------------------------------------------- prediction behaviour kept


def test_deployed_prediction_values_are_unchanged(bundle_and_rows):
    """The refactor must reproduce the original inline ensemble formula exactly."""
    bundle, frame, sequences = bundle_and_rows
    for row in (0, 50, 200):
        features = frame.iloc[row][bundle.features].to_dict()
        pred = bundle.predict_all(features, sequences[row].tolist())
        legacy = float(np.clip(0.2 * pred["rf"] + 0.3 * pred["xgb"] + 0.5 * pred["hybrid"], 50, 99))
        assert pred["ensemble"] == legacy
        assert pred["sleep_quality"] == ("Good" if legacy >= bundle.quality_threshold else "Poor")
        assert pred["diagnostics"]["raw_outputs"]["hybrid"] == pred["hybrid"]


def test_out_of_domain_hybrid_is_reported_not_altered():
    diag = build_prediction_diagnostics(95.0, 96.0, 101.94, 99.6, 99.0)
    assert diag["raw_outputs"]["hybrid"] == 101.94
    assert diag["valid_domain"]["out_of_domain_models"] == ["hybrid"]
    assert any("hybrid output 101.94" in w for w in diag["warnings"])
    assert diag["ensemble_policy"]["clipping_applied"] is True


def test_in_domain_agreeing_models_produce_no_warnings():
    diag = build_prediction_diagnostics(80.0, 81.0, 82.0, 81.3, 81.3)
    assert diag["warnings"] == []
    assert diag["model_disagreement"]["level"] == "low"
    assert diag["ensemble_policy"]["clip_range"] == list(ENSEMBLE_CLIP_RANGE)


@pytest.mark.parametrize("hybrid, level", [(85.0, "low"), (89.0, "moderate"), (95.0, "high")])
def test_disagreement_levels(hybrid, level):
    diag = build_prediction_diagnostics(80.0, 80.0, hybrid, 0.0, 0.0)
    assert diag["model_disagreement"]["level"] == level
    assert diag["model_disagreement"]["spread"] == pytest.approx(hybrid - 80.0)


def test_hybrid_explanation_separates_raw_and_standardized_values(bundle_and_rows):
    bundle, frame, sequences = bundle_and_rows
    features = frame.iloc[0][bundle.features].to_dict()
    tabular = bundle.explain_hybrid(features, sequences[0].tolist(), steps=8)["tabular"]
    raw = np.array([tabular["feature_values"]])
    expected = bundle.hybrid_scaler.transform(raw)[0]
    assert np.allclose(tabular["standardized_values"], expected, atol=1e-5)
    assert not np.allclose(tabular["standardized_values"], tabular["feature_values"])


# ---------------------------------------------------------- global SHAP


@pytest.mark.parametrize("model_id", ["rf", "xgb"])
def test_global_shap_excludes_test_rows_and_uses_positions(bundle_and_rows, model_id):
    bundle, frame, _ = bundle_and_rows
    split = json.loads((PROJECT_ROOT / "ml/data/processed/split.json").read_text())
    result = bundle.global_tree_shap(model_id, sample_size=40)

    positions = result["sample_row_positions"]
    assert len(positions) == 40
    assert set(positions).isdisjoint(split["test_idx"])
    assert set(positions) <= set(split["train_idx"]) | set(split["val_idx"])
    assert result["excluded_test_row_count"] == len(split["test_idx"])


# ------------------------------------------------------------ thread safety


def test_rf_explainer_is_built_once_under_concurrency(bundle_and_rows, monkeypatch):
    bundle, _, _ = bundle_and_rows
    bundle.rf_explainer = None
    built = []
    real_explainer = model_service.shap.TreeExplainer

    def slow_explainer(*args, **kwargs):
        built.append(1)
        time.sleep(0.05)
        return real_explainer(*args, **kwargs)

    monkeypatch.setattr(model_service.shap, "TreeExplainer", slow_explainer)
    results = []
    threads = [threading.Thread(target=lambda: results.append(bundle._get_rf_explainer())) for _ in range(8)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()

    assert len(built) == 1
    assert all(result is results[0] for result in results)


def test_model_bundle_is_loaded_once_under_concurrency(monkeypatch):
    built = []

    class SlowBundle:
        def __init__(self):
            built.append(1)
            time.sleep(0.05)

    monkeypatch.setattr(model_service, "ModelBundle", SlowBundle)
    monkeypatch.setattr(model_service, "_model_bundle", None)
    results = []
    threads = [
        threading.Thread(target=lambda: results.append(model_service.get_model_bundle()))
        for _ in range(8)
    ]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()

    assert len(built) == 1
    assert all(result is results[0] for result in results)


# ------------------------------------------------------------ startup path


def test_ml_package_importable_from_documented_backend_directory():
    """`cd backend && uvicorn app.main:app` must be able to import ml.scripts."""
    code = (
        "import importlib.util, sys\n"
        "from app.services.knowledge_service import ensure_project_root_importable\n"
        "assert importlib.util.find_spec('ml') is None, 'precondition: ml not importable yet'\n"
        "ensure_project_root_importable()\n"
        "assert importlib.util.find_spec('ml.scripts.search_knowledge') is not None\n"
    )
    completed = subprocess.run(
        [sys.executable, "-c", code],
        cwd=PROJECT_ROOT / "backend",
        env={"PATH": "/usr/bin:/bin"},
        capture_output=True,
        text=True,
        timeout=120,
    )
    assert completed.returncode == 0, completed.stderr
