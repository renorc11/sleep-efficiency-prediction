package com.renotech.sleepefficiency

import android.app.Application
import androidx.lifecycle.AndroidViewModel
import androidx.lifecycle.viewModelScope
import com.renotech.sleepefficiency.data.model.LifestyleInput
import com.renotech.sleepefficiency.data.model.ModelMetric
import com.renotech.sleepefficiency.data.model.PredictionRequest
import com.renotech.sleepefficiency.data.model.PredictionResponse
import com.renotech.sleepefficiency.data.network.ApiClient
import com.renotech.sleepefficiency.data.repository.SleepRepository
import com.renotech.sleepefficiency.domain.LifestyleState
import com.renotech.sleepefficiency.domain.TimeUnit
import com.renotech.sleepefficiency.usage.ScreenActivityCollector
import com.renotech.sleepefficiency.usage.UsageAccessHelper
import com.renotech.sleepefficiency.util.toDoubleSafe
import com.renotech.sleepefficiency.util.toIntSafe
import kotlinx.coroutines.flow.MutableStateFlow
import kotlinx.coroutines.flow.StateFlow
import kotlinx.coroutines.flow.asStateFlow
import kotlinx.coroutines.launch

class MainViewModel(app: Application) : AndroidViewModel(app) {
    private val repository = SleepRepository(ApiClient.api)
    private val collector = ScreenActivityCollector(app.applicationContext)

    private val _lifestyle = MutableStateFlow(LifestyleState())
    val lifestyle: StateFlow<LifestyleState> = _lifestyle.asStateFlow()

    private val _prediction = MutableStateFlow<PredictionResponse?>(null)
    val prediction: StateFlow<PredictionResponse?> = _prediction.asStateFlow()

    private val _metrics = MutableStateFlow<Map<String, ModelMetric>>(emptyMap())
    val metrics: StateFlow<Map<String, ModelMetric>> = _metrics.asStateFlow()

    private val _loading = MutableStateFlow(false)
    val loading: StateFlow<Boolean> = _loading.asStateFlow()

    private val _error = MutableStateFlow<String?>(null)
    val error: StateFlow<String?> = _error.asStateFlow()

    private val _hasUsagePermission = MutableStateFlow(UsageAccessHelper.hasUsageAccess(app.applicationContext))
    val hasUsagePermission: StateFlow<Boolean> = _hasUsagePermission.asStateFlow()

    fun refreshUsagePermission() {
        _hasUsagePermission.value = UsageAccessHelper.hasUsageAccess(getApplication())
    }

    fun updateLifestyle(update: (LifestyleState) -> LifestyleState) {
        _lifestyle.value = update(_lifestyle.value)
    }

    fun setLifestyle(newState: LifestyleState) {
        _lifestyle.value = newState
    }

    fun loadMetrics() {
        viewModelScope.launch {
            runCatching { repository.fetchMetrics() }
                .onSuccess { _metrics.value = it.metrics_summary }
                .onFailure { _error.value = "Metrics not available yet. Train models first." }
        }
    }

    fun predict() {
        if (!hasUsagePermission.value) {
            _error.value = "Enable Usage Access to auto-collect screen inactivity sequence."
            return
        }

        _loading.value = true
        _error.value = null

        viewModelScope.launch {
            val form = _lifestyle.value
            val activityMin = if (form.activityUnit == TimeUnit.HOURS) form.activityValue.toDoubleSafe() * 60 else form.activityValue.toDoubleSafe()
            val sequence = collector.collectTodaySequence()
            val autoBlueMin = collector.estimateBlueLightMinutes(sequence)
            val sleepHint = form.sleepTimeHint.trim().takeIf { Regex("^([01]\\d|2[0-3]):[0-5]\\d$").matches(it) }
            val blueMin = if (form.useBlueLightOverride) {
                form.blueLightOverrideMinutes.toDoubleSafe(autoBlueMin)
            } else {
                autoBlueMin
            }

            val payload = PredictionRequest(
                lifestyle = LifestyleInput(
                    caffeine_mg_evening = if (form.hadCaffeine) form.caffeineMg.toDoubleSafe() else 0.0,
                    alcohol_units = if (form.hadAlcohol) form.alcoholUnits.toDoubleSafe() else 0.0,
                    dinner_hour = form.dinnerHour.toDoubleSafe(),
                    stress_level = form.stressLevel.toDoubleSafe(),
                    mood_score = form.moodScore.toDoubleSafe(),
                    physical_activity_min = activityMin,
                    naps_count = form.napsCount.toIntSafe(),
                    naps_total_min = form.napsTotalMin.toDoubleSafe(),
                    previous_sleep_hours = form.previousSleepHours.toDoubleSafe(),
                    work_hours = form.workHours.toDoubleSafe(),
                    blue_light_minutes = blueMin,
                    bedtime_consistency = form.bedtimeConsistency.toDoubleSafe(),
                    hydration_liters = form.hydrationLiters.toDoubleSafe()
                ),
                screen_activity_sequence = sequence,
                sleep_time_hint = sleepHint
            )

            runCatching { repository.predict(payload) }
                .onSuccess { _prediction.value = it }
                .onFailure { _error.value = it.message ?: "Prediction failed" }

            _loading.value = false
        }
    }
}
