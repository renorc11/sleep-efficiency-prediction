from app.services.recommendation_service import generate_recommendations


def healthy_features():
    return {
        "caffeine_mg_evening": 50,
        "alcohol_units": 0,
        "stress_level": 3,
        "dinner_to_sleep_gap": 3,
        "blue_light_minutes": 60,
        "physical_activity_min": 45,
        "nighttime_activity_bursts": 2,
    }


def test_default_recommendation_for_healthy_features():
    recommendations = generate_recommendations(
        healthy_features(), []
    )

    assert len(recommendations) == 1
    assert "routine looks stable" in recommendations[0]


def test_caffeine_recommendation():
    features = healthy_features()
    features["caffeine_mg_evening"] = 150

    recommendations = generate_recommendations(features, [])

    assert any("caffeine" in r.lower() for r in recommendations)


def test_alcohol_recommendation():
    features = healthy_features()
    features["alcohol_units"] = 2

    recommendations = generate_recommendations(features, [])

    assert any("alcohol" in r.lower() for r in recommendations)


def test_stress_recommendation():
    features = healthy_features()
    features["stress_level"] = 8

    recommendations = generate_recommendations(features, [])

    assert any("stress" in r.lower() for r in recommendations)


def test_dinner_gap_recommendation():
    features = healthy_features()
    features["dinner_to_sleep_gap"] = 1.5

    recommendations = generate_recommendations(features, [])

    assert any("dinner" in r.lower() for r in recommendations)


def test_blue_light_recommendation():
    features = healthy_features()
    features["blue_light_minutes"] = 200

    recommendations = generate_recommendations(features, [])

    assert any("screen exposure" in r.lower() for r in recommendations)


def test_low_activity_recommendation():
    features = healthy_features()
    features["physical_activity_min"] = 20

    recommendations = generate_recommendations(features, [])

    assert any("activity level" in r.lower() for r in recommendations)


def test_nighttime_activity_recommendation():
    features = healthy_features()
    features["nighttime_activity_bursts"] = 10

    recommendations = generate_recommendations(features, [])

    assert any("nighttime phone" in r.lower() for r in recommendations)


def test_shap_insights_are_included():
    factors = [
        {"feature": "stress_level", "impact": -2.0, "insight": "Stress may reduce sleep efficiency."},
        {"feature": "physical_activity_min", "impact": 1.5, "insight": "Activity may support better sleep."},
    ]

    recommendations = generate_recommendations(
        healthy_features(), factors
    )

    assert any("Stress may reduce sleep efficiency." in r for r in recommendations)
    assert any("Activity may support better sleep." in r for r in recommendations)


def test_recommendations_are_limited_to_six():
    features = {
        "caffeine_mg_evening": 200,
        "alcohol_units": 3,
        "stress_level": 9,
        "dinner_to_sleep_gap": 1,
        "blue_light_minutes": 240,
        "physical_activity_min": 10,
        "nighttime_activity_bursts": 15,
    }

    factors = [
        {"feature": "stress_level", "impact": -2.0},
        {"feature": "caffeine_mg_evening", "impact": -1.5},
    ]

    recommendations = generate_recommendations(features, factors)

    assert len(recommendations) == 6

def test_recommendations_do_not_trigger_at_exclusive_boundaries():
    boundary_cases = [
        ("caffeine_mg_evening", 120),
        ("alcohol_units", 1),
        ("dinner_to_sleep_gap", 2),
        ("blue_light_minutes", 180),
        ("physical_activity_min", 30),
        ("nighttime_activity_bursts", 8),
    ]

    for feature, value in boundary_cases:
        features = healthy_features()
        features[feature] = value

        recommendations = generate_recommendations(features, [])

        assert len(recommendations) == 1, (
            f"Unexpected recommendation at boundary: {feature}={value}"
        )
        assert "routine looks stable" in recommendations[0]


def test_stress_recommendation_triggers_at_exact_threshold():
    features = healthy_features()
    features["stress_level"] = 7

    recommendations = generate_recommendations(features, [])

    assert any("stress" in r.lower() for r in recommendations)
