def generate_recommendations(features: dict, top_factors: list[dict]) -> list[str]:
    recs: list[str] = []

    if features["caffeine_mg_evening"] > 120:
        recs.append("Your evening caffeine is likely affecting sleep depth. Try keeping post-6 PM caffeine under about 100 mg.")
    if features["alcohol_units"] > 1:
        recs.append("Alcohol close to bedtime may fragment sleep. If possible, limit intake to 0-1 units at night.")
    if features["stress_level"] >= 7:
        recs.append("Stress is elevated today. A short 10-minute wind-down routine (breathing, journaling, light stretching) can help.")
    if features["dinner_to_sleep_gap"] < 2:
        recs.append("Dinner is a bit close to bedtime. Aim for a 2-3 hour gap before sleep.")
    if features["blue_light_minutes"] > 180:
        recs.append("Screen exposure before sleep is high. Reducing bright-screen time in the last 2 hours can improve sleep quality.")
    if features["physical_activity_min"] < 30:
        recs.append("Activity level is on the lower side today. Try to include at least 30 minutes of movement during the day.")
    if features["nighttime_activity_bursts"] > 8:
        recs.append("Frequent nighttime phone checks may be disrupting sleep continuity. Try reducing late-night unlocks/checks.")

    if not recs:
        recs.append("Your routine looks stable today. Keep your bedtime consistent and stay hydrated.")

    for factor in top_factors[:2]:
        if factor.get("insight"):
            recs.append(f"Key driver: {factor['insight']}")
        else:
            signal = "supporting better sleep" if factor["impact"] >= 0 else "pulling sleep efficiency down"
            recs.append(f"Key driver: {factor['feature']} is currently {signal}.")

    return recs[:6]
