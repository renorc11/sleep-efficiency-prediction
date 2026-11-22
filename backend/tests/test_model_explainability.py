import numpy as np
import pandas as pd
import pytest

from app.services.model_service import ModelBundle


@pytest.fixture(scope="module")
def deployed_inputs():
    bundle = ModelBundle()
    frame = pd.read_csv("ml/data/processed/dataset_engineered.csv")
    sequence = np.load("ml/data/processed/sequences.npy")[0].tolist()
    return bundle, frame.loc[0, bundle.features].to_dict(), sequence


@pytest.mark.parametrize("model_name", ["rf", "xgb"])
def test_tree_shap_reconstructs_deployed_prediction(deployed_inputs, model_name):
    bundle, features, _ = deployed_inputs
    explanation = getattr(bundle, f"explain_{model_name}")(features)

    assert explanation["method"] == "TreeSHAP"
    assert explanation["feature_names"] == bundle.features
    assert len(explanation["attributions"]) == 20
    assert explanation["reconstruction"]["passed"]


def test_hybrid_integrated_gradients_preserves_two_input_structure(deployed_inputs):
    bundle, features, sequence = deployed_inputs
    explanation = bundle.explain_hybrid(features, sequence, steps=16)

    assert explanation["method"] == "IntegratedGradients"
    assert len(explanation["tabular"]["attributions"]) == 20
    assert len(explanation["temporal"]["positions"]) == 288
    assert explanation["temporal"]["positions"][0]["time_interval"] == "00:00-00:05"
    assert explanation["temporal"]["positions"][-1]["time_interval"] == "23:55-24:00"
    assert explanation["completeness"]["passed"]
    assert any("synthetic" in warning.lower() for warning in explanation["warnings"])


def test_hybrid_integrated_gradients_rejects_invalid_sequence_length(deployed_inputs):
    bundle, features, _ = deployed_inputs
    with pytest.raises(ValueError, match="288"):
        bundle.explain_hybrid(features, [0] * 287, steps=8)


@pytest.mark.parametrize("model_name", ["rf", "xgb"])
def test_global_tree_shap_uses_development_rows_only(deployed_inputs, model_name):
    bundle, _, _ = deployed_inputs
    explanation = bundle.global_tree_shap(model_name, sample_size=12)

    assert explanation["sample_count"] == 12
    assert len(explanation["feature_importance"]) == 20
    assert explanation["feature_importance"][0]["rank"] == 1
    assert "train plus validation" in explanation["sample_source"]
