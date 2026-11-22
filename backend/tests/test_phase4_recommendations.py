"""Phase 4: automatic SHAP -> RAG -> BioMistral personalized recommendations.

BioMistral and retrieval are faked; the deployed RF/XGBoost/Hybrid models and
their real explanations are used. Run from the project root.
"""

import copy
import json
import re
from pathlib import Path
from unittest.mock import patch

import httpx
import numpy as np
import pandas as pd
import pytest
from fastapi.testclient import TestClient
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from sqlalchemy.pool import StaticPool

import app.services.personalized_recommendation_service as prs
from app.db.session import Base, get_db
from app.main import app
from app.schemas.prediction import PersonalizedRecommendationResult
from app.services.inactivity_service import build_engineered_features
from app.services.model_service import ModelBundle
from app.services.recommendation_context_service import (
    RECOMMENDATION_TOPICS,
    TARGET_DERIVED,
    build_recommendation_context,
)

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATASET = json.loads((PROJECT_ROOT / "ml/data/knowledge/sleep_knowledge_dataset_v0.3.1.json").read_text())
CHUNKS = {chunk["chunk_id"]: chunk for chunk in DATASET["knowledge_chunks"]}
REGISTRY = {source["source_id"]: source for source in DATASET["source_registry"]}
PAYLOAD = json.loads((PROJECT_ROOT / "ml/data/raw/example_prediction_payload.json").read_text())
HIGH_RISK = {
    **PAYLOAD["lifestyle"],
    "caffeine_mg_evening": 350,
    "alcohol_units": 3,
    "stress_level": 8,
    "blue_light_minutes": 300,
    "physical_activity_min": 10,
}


class TopicKnowledge:
    """Returns the real knowledge entry whose topic query was asked, or a neighbour."""

    def __init__(self, neighbour_only=False):
        self.neighbour_only = neighbour_only
        self.queries = []

    def retrieve(self, question, top_k=3):
        self.queries.append(question)
        chunk_id = next(t["chunk_id"] for t in RECOMMENDATION_TOPICS.values() if t.get("query") == question)
        if self.neighbour_only:
            chunk_id = "HYGIENE-001"
        chunk = CHUNKS[chunk_id]
        return {
            "dataset_version": "0.3.1",
            "chunks": [{**chunk, "similarity": 0.8}],
            "sources": [{"source_id": sid, "title": REGISTRY[sid]["title"]} for sid in chunk["source_ids"]],
        }


@pytest.fixture(scope="module")
def bundle():
    return ModelBundle()


def make_context(bundle, lifestyle=HIGH_RISK):
    sequence = PAYLOAD["screen_activity_sequence"]
    features, inactivity = build_engineered_features(lifestyle, sequence, PAYLOAD.get("sleep_time_hint"))
    prediction = bundle.predict_all(features, sequence)
    explanations = {
        "rf": bundle.explain_rf(features),
        "xgb": bundle.explain_xgb(features),
        "hybrid": bundle.explain_hybrid(features, sequence, steps=16),
    }
    context = build_recommendation_context(
        lifestyle=lifestyle,
        features=features,
        prediction=prediction,
        inferred_sleep_time=inactivity.inferred_sleep_time,
        explanations=explanations,
    )
    return context, prediction, features, explanations


def good_llm_output(candidates):
    words = {
        "stress_level": ("Build a calm wind-down routine", "Try slow breathing before bed."),
        "physical_activity_min": ("Move more during the day", "Try a brisk walk in the afternoon."),
        "blue_light_minutes": ("Put screens away earlier", "Try a screen-free hour before bed."),
        "caffeine_mg_evening": ("Have caffeine earlier", "Try switching to decaf after lunch."),
        "alcohol_units": ("Limit alcohol near bedtime", "Try finishing any drink early in the evening."),
        "bedtime_consistency": ("Keep a regular schedule", "Try waking at the same time daily."),
        "dinner_hour": ("Eat dinner earlier", "Try having your main meal earlier."),
    }
    return {
        "summary": "Your stress and activity levels were the main influences on this estimate.",
        "recommendations": {
            c["feature_key"]: {
                "title": words[c["feature_key"]][0],
                "reason": "Your current level is associated with a lower predicted sleep efficiency in the models.",
                "action": words[c["feature_key"]][1],
                "source_ids": [],
            }
            for c in candidates
        },
    }


class FakeLLMClient:
    """Stands in for httpx.Client; returns queued responses or raises queued errors."""

    def __init__(self, outcomes):
        self.outcomes = list(outcomes)
        self.payloads = []

    def __call__(self, *args, **kwargs):
        return self

    def __enter__(self):
        return self

    def __exit__(self, *args):
        return None

    def post(self, url, json):
        self.payloads.append(json)
        outcome = self.outcomes.pop(0)
        if isinstance(outcome, Exception):
            raise outcome
        if isinstance(outcome, int):
            request = httpx.Request("POST", url)
            response = httpx.Response(outcome, request=request)
            return response
        content = outcome if isinstance(outcome, str) else __import__("json").dumps(outcome)
        return httpx.Response(
            200,
            request=httpx.Request("POST", url),
            json={"choices": [{"message": {"content": content}}]},
        )


def run_generation(context, outcomes, knowledge=None):
    client = FakeLLMClient(outcomes)
    with patch.object(prs.httpx, "Client", client):
        result = prs.generate_personalized_recommendations(
            context=context, knowledge_service=knowledge or TopicKnowledge()
        )
    PersonalizedRecommendationResult(**result)  # schema-valid in every path
    return result, client


# ------------------------------------------------------- context service


def test_target_derived_features_never_drive_recommendations(bundle):
    context, *_ = make_context(bundle)
    candidate_keys = {c["feature_key"] for c in context["candidates"]}
    assert candidate_keys
    assert not candidate_keys & set(TARGET_DERIVED)
    assert candidate_keys <= {k for k, t in RECOMMENDATION_TOPICS.items() if t["chunk_id"]}
    excluded = {e["feature_key"] for e in context["model_explanation"]["excluded_features"]}
    assert excluded & set(TARGET_DERIVED), "target-derived attributions are reported as excluded"


def test_candidates_require_rf_xgb_agreement_and_use_real_attributions(bundle):
    context, _, _, explanations = make_context(bundle)
    rf = dict(zip(explanations["rf"]["feature_names"], explanations["rf"]["attributions"]))
    for candidate in context["candidates"]:
        assert candidate["rf_xgb_agree"]
        assert candidate["rf_treeshap"] == rf[candidate["feature_key"]]
    assert "not combined" in context["model_explanation"]["method_note"]


def test_failed_reconstruction_produces_no_candidates(bundle):
    _, prediction, features, explanations = make_context(bundle)
    broken = copy.deepcopy(explanations)
    broken["rf"]["reconstruction"]["passed"] = False
    context = build_recommendation_context(
        lifestyle=HIGH_RISK, features=features, prediction=prediction,
        inferred_sleep_time="23:00", explanations=broken,
    )
    assert context["candidates"] == []
    assert any("failed validation" in w for w in context["warnings"])


def test_explanation_prediction_mismatch_is_rejected(bundle):
    _, prediction, features, explanations = make_context(bundle)
    tampered = copy.deepcopy(prediction)
    tampered["xgb"] += 5.0
    context = build_recommendation_context(
        lifestyle=HIGH_RISK, features=features, prediction=tampered,
        inferred_sleep_time="23:00", explanations=explanations,
    )
    assert context["model_explanation"]["validity"]["xgb"]["valid"] is False
    assert context["candidates"] == []


def test_temporal_summary_is_hourly_and_flagged_synthetic(bundle):
    context, *_ = make_context(bundle)
    temporal = context["model_explanation"]["hybrid_integrated_gradients"]["temporal_summary"]
    assert "Integrated Gradients" in temporal["method"]
    assert "synthetic" in temporal["note"] and "not proof of sleep" in temporal["note"]
    for row in temporal["hours_lowering_prediction"]:
        assert re.fullmatch(r"\d{2}:00-\d{2}:00", row["hour"]) and row["attribution"] < 0


# ----------------------------------------------------------- generation


def test_valid_generation_uses_code_numbers_and_own_sources(bundle):
    context, prediction, *_ = make_context(bundle)
    llm = prs.LLM_MAX_FACTORS
    result, client = run_generation(context, [good_llm_output(context["candidates"][:llm])])

    assert result["generation_status"] == "generated" and not result["fallback_used"]
    assert result["summary"].startswith(f"Your predicted sleep efficiency is {prediction['ensemble']:.2f}%")
    for rec in result["recommendations"]:
        candidate = next(c for c in context["candidates"] if c["feature_key"] == rec["factor_key"])
        assert rec["model_evidence"]["rf_treeshap"] == candidate["rf_treeshap"]
        own = set(CHUNKS[RECOMMENDATION_TOPICS[rec["factor_key"]]["chunk_id"]]["source_ids"])
        assert set(rec["source_ids"]) <= own and rec["source_ids"]
        assert rec["scientific_evidence"]["knowledge_id"] == RECOMMENDATION_TOPICS[rec["factor_key"]]["chunk_id"]
    sources = [r["wording_source"] for r in result["recommendations"]]
    assert sources[:llm] == ["biomistral"] * min(llm, len(sources))
    assert set(sources[llm:]) <= {"template"}


def test_schema_restricts_sources_per_factor_and_forbids_digits(bundle):
    context, *_ = make_context(bundle)
    candidates = context["candidates"][: prs.LLM_MAX_FACTORS]
    evidence, _ = prs.retrieve_factor_evidence(candidates, TopicKnowledge())
    schema = prs._response_schema(candidates, evidence)
    for c in candidates:
        slot = schema["properties"]["recommendations"]["properties"][c["feature_key"]]
        expected = CHUNKS[RECOMMENDATION_TOPICS[c["feature_key"]]["chunk_id"]]["source_ids"]
        assert slot["properties"]["source_ids"]["items"]["enum"] == expected
        pattern = re.compile(slot["properties"]["action"]["pattern"])
        assert pattern.match("Try a screen-free hour before bed.")
        for bad in ("Aim for 30 minutes.", "Cut by 10%.", 'Say "hi".'):
            assert not pattern.match(bad)


VALIDATION_CASES = [
    ("summary", "Your estimate is 64.87% today.", "percentage"),
    ("summary", "Your estimate fell by 3.42 points.", "numbers not in the inputs"),
    ("summary", "A survey of 1,342 adults supports this.", "numbers not in the inputs"),
    ("action", "Try to get at least .... minutes.", "blanks"),
    ("reason", "Your stress level of  is associated with a lower prediction.", "blanks"),
    ("action", "Try taking melatonin before bed.", "Medication"),
    ("summary", "Dear [USER_NAME], here is your report.", "placeholder"),
    ("summary", "The model also considers your age and BMI.", "characteristics"),
    ("summary", "The model is high-confidence in this prediction.", "confidence"),
    ("summary", "Your sleep efficiency is lower than average.", "average"),
    ("summary", "Your naps were the main influence.", "not supplied"),
    ("summary", "Stress matters a lot here. Stress matters a lot here.", "repeats"),
    ("action", "Try to eat more vegetables at lunch.", "not about its factor"),
]


@pytest.mark.parametrize("field, text, expected", VALIDATION_CASES)
def test_generation_validator_rejects_unsafe_or_invented_content(bundle, field, text, expected):
    context, *_ = make_context(bundle)
    candidates = context["candidates"][: prs.LLM_MAX_FACTORS]
    llm_context = {**context, "candidates": candidates}
    evidence, _ = prs.retrieve_factor_evidence(candidates, TopicKnowledge())
    data = prs._normalize(good_llm_output(candidates), candidates)
    assert prs.validate_generation(data, llm_context, evidence) == []
    if field == "summary":
        data["summary"] = text
    else:
        data["recommendations"][0][field] = text
    problems = prs.validate_generation(data, llm_context, evidence)
    assert any(expected.lower() in p.lower() for p in problems), problems


def test_cross_factor_citation_is_rejected(bundle):
    context, *_ = make_context(bundle)
    candidates = context["candidates"][: prs.LLM_MAX_FACTORS]
    evidence, _ = prs.retrieve_factor_evidence(candidates, TopicKnowledge())
    data = prs._normalize(good_llm_output(candidates), candidates)
    other = next(s for s in REGISTRY if s not in {x["source_id"] for x in evidence[candidates[0]["feature_key"]]["sources"]})
    data["recommendations"][0]["source_ids"] = [other]
    problems = prs.validate_generation(data, {**context, "candidates": candidates}, evidence)
    assert any("Source IDs not retrieved" in p for p in problems)


def test_small_advice_integers_and_user_values_are_allowed(bundle):
    context, *_ = make_context(bundle)
    candidates = context["candidates"][: prs.LLM_MAX_FACTORS]
    evidence, _ = prs.retrieve_factor_evidence(candidates, TopicKnowledge())
    data = prs._normalize(good_llm_output(candidates), candidates)
    data["recommendations"][0]["action"] = "Try a calming routine for 10 to 15 minutes before bed."
    assert prs.validate_generation(data, {**context, "candidates": candidates}, evidence) == []


def test_timeout_falls_back_without_retry(bundle):
    context, *_ = make_context(bundle)
    result, client = run_generation(context, [httpx.ReadTimeout("slow")])
    assert result["generation_status"] == "fallback" and result["fallback_used"]
    assert len(client.payloads) == 1
    assert all(r["wording_source"] == "template" for r in result["recommendations"])
    assert "timed out" in result["fallback_reason"]


def test_connection_error_falls_back(bundle):
    context, *_ = make_context(bundle)
    result, _ = run_generation(context, [httpx.ConnectError("refused")])
    assert result["generation_status"] == "fallback"
    assert all(r["source_ids"] for r in result["recommendations"])


def test_http_500_and_malformed_json_are_retried(bundle):
    context, *_ = make_context(bundle)
    good = good_llm_output(context["candidates"][: prs.LLM_MAX_FACTORS])
    result, client = run_generation(context, [500, good])
    assert result["generation_status"] == "generated" and len(client.payloads) == 2
    result, client = run_generation(context, ["{not json", good])
    assert result["generation_status"] == "generated" and len(client.payloads) == 2


def test_two_invalid_generations_fall_back_with_issues_recorded(bundle):
    context, *_ = make_context(bundle)
    bad = good_llm_output(context["candidates"][: prs.LLM_MAX_FACTORS])
    bad["summary"] = "Your estimate is 99.00% and you have insomnia."
    result, client = run_generation(context, [bad, bad])
    assert result["generation_status"] == "fallback" and len(client.payloads) == 2
    assert any("Attempt 2" in issue for issue in result["validation_issues"])


def test_summary_sentences_with_unsupported_claims_are_repaired(bundle):
    context, *_ = make_context(bundle)
    output = good_llm_output(context["candidates"][: prs.LLM_MAX_FACTORS])
    output["summary"] = (
        "Your sleep efficiency is lower than average. "
        "Stress and activity were the main influences on this estimate."
    )
    result, client = run_generation(context, [output])
    assert result["generation_status"] == "generated" and len(client.payloads) == 1
    assert "average" not in result["summary"]
    assert "Stress and activity were the main influences" in result["summary"]
    assert any("Summary repaired" in issue for issue in result["validation_issues"])


def test_fully_unsupported_summary_falls_back_to_code_summary(bundle):
    context, prediction, *_ = make_context(bundle)
    output = good_llm_output(context["candidates"][: prs.LLM_MAX_FACTORS])
    output["summary"] = "You sleep worse than most people."
    result, _ = run_generation(context, [output])
    assert result["generation_status"] == "generated"
    assert "machine-learning estimate" in result["summary"]
    assert "most people" not in result["summary"]


def test_truncated_single_sentence_uses_template_field(bundle):
    context, *_ = make_context(bundle)
    output = good_llm_output(context["candidates"][: prs.LLM_MAX_FACTORS])
    first = next(iter(output["recommendations"]))
    # An on-topic single sentence cut off by the grammar's length limit.
    action = output["recommendations"][first]["action"]
    output["recommendations"][first]["action"] = action.rstrip(".").rsplit(" ", 1)[0]
    result, _ = run_generation(context, [output])
    rec = result["recommendations"][0]
    assert rec["wording_source"] == "mixed"
    assert rec["action"] == RECOMMENDATION_TOPICS[rec["factor_key"]]["action"]


def test_missing_knowledge_service_skips_llm(bundle):
    context, *_ = make_context(bundle)
    client = FakeLLMClient([])
    with patch.object(prs.httpx, "Client", client):
        result = prs.generate_personalized_recommendations(context=context, knowledge_service=None)
    assert result["generation_status"] == "fallback" and client.payloads == []
    assert all(r["scientific_evidence"] is None and r["source_ids"] == [] for r in result["recommendations"])
    assert any("knowledge base is unavailable" in u for u in result["uncertainty"])


def test_neighbouring_topic_is_not_accepted_as_evidence(bundle):
    context, *_ = make_context(bundle)
    evidence, notes = prs.retrieve_factor_evidence(context["candidates"], TopicKnowledge(neighbour_only=True))
    assert evidence == {}
    assert len(notes) == len(context["candidates"])


def test_llm_context_contains_no_numbers_or_assistant_guidance(bundle):
    context, *_ = make_context(bundle)
    candidates = context["candidates"][: prs.LLM_MAX_FACTORS]
    evidence, _ = prs.retrieve_factor_evidence(candidates, TopicKnowledge())
    prompt = prs._user_prompt({**context, "candidates": candidates}, evidence)
    factors = json.loads(prompt.split("<context>\n")[1].split("\n</context>")[0])["factors"]
    for factor in factors:
        assert "user_value" not in factor
        assert not re.search(r"\d", factor["model_association"])
        assert "practical_implication" not in (factor["evidence"] or {})


def test_pipeline_never_modifies_prediction(bundle):
    sequence = PAYLOAD["screen_activity_sequence"]
    features, inactivity = build_engineered_features(HIGH_RISK, sequence, PAYLOAD.get("sleep_time_hint"))
    prediction = bundle.predict_all(features, sequence)
    before = copy.deepcopy((prediction, features, HIGH_RISK))
    bad = {"summary": "Efficiency is really 99.", "recommendations": {}}
    with patch.object(prs.httpx, "Client", FakeLLMClient([bad, bad])):
        out = prs.recommend_for_prediction(
            bundle=bundle, lifestyle=HIGH_RISK, features=features, prediction=prediction,
            sequence=sequence, inferred_sleep_time=inactivity.inferred_sleep_time,
            knowledge_service=TopicKnowledge(), legacy_top_factors=[], legacy_recommendations=[],
        )
    assert (prediction, features, HIGH_RISK) == before
    context_predictions = out["personalized_recommendations"]["prediction_context"]["model_predictions"]
    assert context_predictions == {k: prediction[k] for k in ("rf", "xgb", "hybrid", "ensemble")}
    assert out["explanation_status"] == "fallback"


# ------------------------------------------------------ /predict integration


@pytest.fixture
def memory_db():
    engine = create_engine("sqlite://", connect_args={"check_same_thread": False}, poolclass=StaticPool)
    Session = sessionmaker(bind=engine)
    Base.metadata.create_all(bind=engine)

    def override():
        db = Session()
        try:
            yield db
        finally:
            db.close()

    app.dependency_overrides[get_db] = override
    yield
    app.dependency_overrides.clear()
    engine.dispose()


@pytest.mark.parametrize("llm_up", [True, False])
def test_predict_returns_structured_recommendations(memory_db, llm_up):
    payload = {**PAYLOAD, "lifestyle": HIGH_RISK}

    def outcomes_for(client):
        return client

    knowledge = TopicKnowledge()
    with patch("app.main.create_knowledge_service", return_value=knowledge):
        with TestClient(app) as test_client:
            # Build a valid LLM response from the same context the route will compute.
            candidates_holder = {}
            real_generate = prs.generate_personalized_recommendations

            def capture(*, context, knowledge_service, **kwargs):
                candidates_holder["c"] = context["candidates"][: prs.LLM_MAX_FACTORS]
                outcomes = (
                    [good_llm_output(candidates_holder["c"])] if llm_up else [httpx.ConnectError("down")]
                )
                with patch.object(prs.httpx, "Client", FakeLLMClient(outcomes)):
                    return real_generate(context=context, knowledge_service=knowledge_service, **kwargs)

            with patch.object(prs, "generate_personalized_recommendations", capture):
                response = test_client.post("/api/v1/predict", json=payload)

    assert response.status_code == 200
    body = response.json()
    result = PersonalizedRecommendationResult(**body["personalized_recommendations"])
    assert result.generation_status == ("generated" if llm_up else "fallback")
    assert body["explanation_status"] == result.generation_status
    assert body["generated_explanation"].startswith(result.summary)
    assert body["predicted_sleep_efficiency"] == round(result.prediction_context["predicted_sleep_efficiency"], 2)
    assert set(body["model_outputs"]) == {"rf", "xgb", "hybrid", "ensemble"}
    assert result.recommendations and all(r.source_ids for r in result.recommendations)
