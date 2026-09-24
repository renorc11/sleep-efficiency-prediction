package com.renotech.sleepefficiency.data.model

import kotlinx.serialization.SerialName
import kotlinx.serialization.Serializable

@Serializable
data class LifestyleInput(
    val caffeine_mg_evening: Double,
    val alcohol_units: Double,
    val dinner_hour: Double,
    val stress_level: Double,
    val mood_score: Double,
    val physical_activity_min: Double,
    val naps_count: Int,
    val naps_total_min: Double,
    val previous_sleep_hours: Double,
    val work_hours: Double,
    val blue_light_minutes: Double,
    val bedtime_consistency: Double,
    val hydration_liters: Double
)

@Serializable
data class PredictionRequest(
    val lifestyle: LifestyleInput,
    val screen_activity_sequence: List<Int>,
    val sleep_time_hint: String? = null
)

@Serializable
data class TopFactor(
    val feature_key: String? = null,
    val feature: String,
    val impact: Double,
    val direction: String? = null,
    val insight: String? = null,
    val value: Double
)

@Serializable
data class PredictionResponse(
    val predicted_sleep_efficiency: Double,
    val sleep_quality_label: String,
    val inferred_sleep_time: String,
    val model_outputs: Map<String, Double>,
    @SerialName("top_influencing_factors")
    val topInfluencingFactors: List<TopFactor>,
    val recommendations: List<String>
)

@Serializable
data class ModelMetric(
    val mae: Double,
    val rmse: Double,
    val r2: Double,
    val accuracy: Double,
    val precision: Double,
    val recall: Double,
    val f1: Double,
    val roc_auc: Double
)

@Serializable
data class MetricsResponse(
    val metrics_summary: Map<String, ModelMetric>
)
