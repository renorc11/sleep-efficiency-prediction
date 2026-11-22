package com.renotech.sleepefficiency.ui.screens

import androidx.compose.foundation.background
import androidx.compose.foundation.layout.Arrangement
import androidx.compose.foundation.layout.Column
import androidx.compose.foundation.layout.Row
import androidx.compose.foundation.layout.fillMaxSize
import androidx.compose.foundation.layout.fillMaxWidth
import androidx.compose.foundation.layout.padding
import androidx.compose.foundation.rememberScrollState
import androidx.compose.foundation.shape.RoundedCornerShape
import androidx.compose.foundation.verticalScroll
import androidx.compose.material3.Button
import androidx.compose.material3.Card
import androidx.compose.material3.CardDefaults
import androidx.compose.material3.DropdownMenuItem
import androidx.compose.material3.DropdownMenu
import androidx.compose.material3.ExperimentalMaterial3Api
import androidx.compose.material3.ExposedDropdownMenuBox
import androidx.compose.material3.ExposedDropdownMenuDefaults
import androidx.compose.material3.MaterialTheme
import androidx.compose.material3.OutlinedTextField
import androidx.compose.material3.Switch
import androidx.compose.material3.Text
import androidx.compose.material3.TextButton
import androidx.compose.runtime.Composable
import androidx.compose.runtime.getValue
import androidx.compose.runtime.mutableStateOf
import androidx.compose.runtime.remember
import androidx.compose.runtime.setValue
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.graphics.Brush
import androidx.compose.ui.unit.dp
import com.renotech.sleepefficiency.data.model.ModelMetric
import com.renotech.sleepefficiency.data.model.PredictionResponse
import com.renotech.sleepefficiency.domain.LifestyleState
import com.renotech.sleepefficiency.domain.TimeUnit
import com.renotech.sleepefficiency.ui.theme.Blue
import com.renotech.sleepefficiency.ui.theme.Green
import com.renotech.sleepefficiency.ui.theme.Orange

@Composable
fun MainScreen(
    form: LifestyleState,
    metrics: Map<String, ModelMetric>,
    prediction: PredictionResponse?,
    loading: Boolean,
    hasUsagePermission: Boolean,
    error: String?,
    onPermissionClick: () -> Unit,
    onRefreshPermission: () -> Unit,
    onUpdate: (LifestyleState) -> Unit,
    onPredict: () -> Unit,
    onLoadMetrics: () -> Unit,
) {
    Column(
        modifier = Modifier
            .fillMaxSize()
            .background(Brush.verticalGradient(listOf(Orange.copy(alpha = 0.06f), Blue.copy(alpha = 0.03f))))
            .verticalScroll(rememberScrollState())
            .padding(16.dp),
        verticalArrangement = Arrangement.spacedBy(12.dp)
    ) {
        HeaderCard()

        if (!hasUsagePermission) {
            PermissionCard(
                hasPermission = hasUsagePermission,
                onPermissionClick = onPermissionClick,
                onRefreshPermission = onRefreshPermission
            )
        }

        LifestyleFormCard(form = form, onUpdate = onUpdate)

        Button(
            onClick = onPredict,
            enabled = !loading && hasUsagePermission,
            modifier = Modifier.fillMaxWidth()
        ) {
            Text(if (loading) "Predicting..." else "Predict Sleep Efficiency")
        }

        TextButton(onClick = onLoadMetrics, modifier = Modifier.align(Alignment.End)) {
            Text("Refresh Model Metrics")
        }

        error?.let { Text(text = it, color = MaterialTheme.colorScheme.error) }

        prediction?.let { PredictionCard(it) }

        if (metrics.isNotEmpty()) {
            MetricsCard(metrics)
        }
    }
}

@Composable
private fun HeaderCard() {
    Card(
        shape = RoundedCornerShape(18.dp),
        colors = CardDefaults.cardColors(containerColor = MaterialTheme.colorScheme.surface)
    ) {
        Column(Modifier.padding(16.dp), verticalArrangement = Arrangement.spacedBy(6.dp)) {
            Text(
                "Explainable Machine Learning for Lifestyle-Based Sleep Efficiency Prediction Using Behavioral and Screen Inactivity Signals",
                style = MaterialTheme.typography.headlineSmall
            )
            Text("Lifestyle + automatic screen inactivity analysis with explainable ML results.")
        }
    }
}

@Composable
private fun PermissionCard(
    hasPermission: Boolean,
    onPermissionClick: () -> Unit,
    onRefreshPermission: () -> Unit,
) {
    Card {
        Column(Modifier.padding(14.dp), verticalArrangement = Arrangement.spacedBy(8.dp)) {
            Text("Usage Access Permission", style = MaterialTheme.typography.titleMedium)
            Text(if (hasPermission) "Permission granted. Screen sequence collection enabled." else "Permission required for automatic screen activity collection.")
            Text(
                "What is collected: only whether any app was in use in each five-minute period since " +
                    "midnight today (no app names or content). It is sent to your configured backend " +
                    "with each prediction. Inactivity is an indirect proxy, not a measurement of sleep, " +
                    "and the models were trained on synthetic activity data, not real phone usage.",
                style = MaterialTheme.typography.bodySmall
            )
            Row(horizontalArrangement = Arrangement.spacedBy(8.dp)) {
                Button(onClick = onPermissionClick) { Text("Open Settings") }
                TextButton(onClick = onRefreshPermission) { Text("I Enabled It") }
            }
        }
    }
}

@Composable
private fun LifestyleFormCard(form: LifestyleState, onUpdate: (LifestyleState) -> Unit) {
    Card {
        Column(Modifier.padding(14.dp), verticalArrangement = Arrangement.spacedBy(10.dp)) {
            Text("Lifestyle Inputs", style = MaterialTheme.typography.titleMedium)

            ToggleAndInput(
                title = "Had caffeine?",
                checked = form.hadCaffeine,
                value = form.caffeineMg,
                inputLabel = "Caffeine (mg)",
                onChecked = { onUpdate(form.copy(hadCaffeine = it)) },
                onValue = { onUpdate(form.copy(caffeineMg = it)) }
            )

            ToggleAndInput(
                title = "Had alcohol?",
                checked = form.hadAlcohol,
                value = form.alcoholUnits,
                inputLabel = "Alcohol (units)",
                onChecked = { onUpdate(form.copy(hadAlcohol = it)) },
                onValue = { onUpdate(form.copy(alcoholUnits = it)) }
            )

            NumberField("Dinner hour (24h)", form.dinnerHour) { onUpdate(form.copy(dinnerHour = it)) }
            NumberField("Stress level (1-10)", form.stressLevel) { onUpdate(form.copy(stressLevel = it)) }
            NumberField("Mood score (-2 to 2)", form.moodScore) { onUpdate(form.copy(moodScore = it)) }

            TimeUnitField(
                label = "Physical activity",
                value = form.activityValue,
                unit = form.activityUnit,
                onValue = { onUpdate(form.copy(activityValue = it)) },
                onUnit = { onUpdate(form.copy(activityUnit = it)) }
            )

            NumberField("Naps count", form.napsCount) { onUpdate(form.copy(napsCount = it)) }
            NumberField("Naps total (min)", form.napsTotalMin) { onUpdate(form.copy(napsTotalMin = it)) }
            NumberField("Previous sleep hours", form.previousSleepHours) { onUpdate(form.copy(previousSleepHours = it)) }
            NumberField("Work hours", form.workHours) { onUpdate(form.copy(workHours = it)) }
            NumberField("Approx sleep time (optional, HH:MM)", form.sleepTimeHint) { onUpdate(form.copy(sleepTimeHint = it)) }

            OptionalBlueLightOverrideField(
                enabled = form.useBlueLightOverride,
                value = form.blueLightOverrideMinutes,
                onToggle = { onUpdate(form.copy(useBlueLightOverride = it)) },
                onValue = { onUpdate(form.copy(blueLightOverrideMinutes = it)) }
            )

            NumberField("Bedtime consistency (0-1)", form.bedtimeConsistency) { onUpdate(form.copy(bedtimeConsistency = it)) }
            NumberField("Hydration liters", form.hydrationLiters) { onUpdate(form.copy(hydrationLiters = it)) }
        }
    }
}

@Composable
private fun NumberField(label: String, value: String, onValue: (String) -> Unit) {
    OutlinedTextField(
        value = value,
        onValueChange = onValue,
        label = { Text(label) },
        modifier = Modifier.fillMaxWidth(),
        singleLine = true
    )
}

@Composable
private fun OptionalBlueLightOverrideField(
    enabled: Boolean,
    value: String,
    onToggle: (Boolean) -> Unit,
    onValue: (String) -> Unit
) {
    Column(verticalArrangement = Arrangement.spacedBy(6.dp)) {
        Row(
            verticalAlignment = Alignment.CenterVertically,
            horizontalArrangement = Arrangement.SpaceBetween,
            modifier = Modifier.fillMaxWidth()
        ) {
            Text("Override auto blue-light minutes?")
            Switch(checked = enabled, onCheckedChange = onToggle)
        }
        Text(
            "Blue-light minutes are auto-estimated from device usage (2h before inferred sleep).",
            style = MaterialTheme.typography.bodySmall
        )
        if (enabled) {
            OutlinedTextField(
                value = value,
                onValueChange = onValue,
                label = { Text("Blue-light override (minutes)") },
                modifier = Modifier.fillMaxWidth(),
                singleLine = true
            )
        }
    }
}

@Composable
private fun ToggleAndInput(
    title: String,
    checked: Boolean,
    value: String,
    inputLabel: String,
    onChecked: (Boolean) -> Unit,
    onValue: (String) -> Unit
) {
    Column(verticalArrangement = Arrangement.spacedBy(6.dp)) {
        Row(verticalAlignment = Alignment.CenterVertically, horizontalArrangement = Arrangement.SpaceBetween, modifier = Modifier.fillMaxWidth()) {
            Text(title)
            Switch(checked = checked, onCheckedChange = onChecked)
        }
        OutlinedTextField(
            value = value,
            onValueChange = onValue,
            enabled = checked,
            label = { Text(inputLabel) },
            modifier = Modifier.fillMaxWidth(),
            singleLine = true
        )
    }
}

@OptIn(ExperimentalMaterial3Api::class)
@Composable
private fun TimeUnitField(
    label: String,
    value: String,
    unit: TimeUnit,
    onValue: (String) -> Unit,
    onUnit: (TimeUnit) -> Unit
) {
    var expanded by remember { mutableStateOf(false) }

    Column(verticalArrangement = Arrangement.spacedBy(6.dp)) {
        OutlinedTextField(
            value = value,
            onValueChange = onValue,
            label = { Text("$label value") },
            modifier = Modifier.fillMaxWidth(),
            singleLine = true
        )

        ExposedDropdownMenuBox(expanded = expanded, onExpandedChange = { expanded = !expanded }) {
            OutlinedTextField(
                value = if (unit == TimeUnit.MINUTES) "Minutes" else "Hours",
                onValueChange = {},
                readOnly = true,
                label = { Text("$label unit") },
                trailingIcon = { ExposedDropdownMenuDefaults.TrailingIcon(expanded = expanded) },
                modifier = Modifier
                    .menuAnchor()
                    .fillMaxWidth(),
                singleLine = true
            )
            DropdownMenu(expanded = expanded, onDismissRequest = { expanded = false }) {
                DropdownMenuItem(text = { Text("Minutes") }, onClick = {
                    onUnit(TimeUnit.MINUTES)
                    expanded = false
                })
                DropdownMenuItem(text = { Text("Hours") }, onClick = {
                    onUnit(TimeUnit.HOURS)
                    expanded = false
                })
            }
        }
    }
}

@Composable
private fun PredictionCard(result: PredictionResponse) {
    Card(colors = CardDefaults.cardColors(containerColor = Green.copy(alpha = 0.12f))) {
        Column(Modifier.padding(14.dp), verticalArrangement = Arrangement.spacedBy(8.dp)) {
            Text("Prediction", style = MaterialTheme.typography.titleMedium)
            Text("Sleep efficiency: ${"%.2f".format(result.predicted_sleep_efficiency)}%")
            Text("Sleep quality: ${result.sleep_quality_label}")
            Text("Inferred sleep time: ${result.inferred_sleep_time}")

            Text("Model outputs:")
            result.model_outputs.forEach { (model, value) ->
                Text("- ${model.uppercase()}: ${"%.2f".format(value)}")
            }

            Text("Top factors:")
            result.topInfluencingFactors.take(5).forEach {
                val fallback = if (it.impact >= 0) "supporting better sleep" else "pulling sleep quality down"
                Text("- ${it.insight ?: (it.feature + " is currently " + (it.direction ?: fallback) + ".")}")
            }

            Text("Recommendations:")
            result.recommendations.forEach { Text("- $it") }
        }
    }
}

@Composable
private fun MetricsCard(metrics: Map<String, ModelMetric>) {
    Card(colors = CardDefaults.cardColors(containerColor = Blue.copy(alpha = 0.08f))) {
        Column(Modifier.padding(14.dp), verticalArrangement = Arrangement.spacedBy(8.dp)) {
            Text("Model Metrics", style = MaterialTheme.typography.titleMedium)
            metrics.forEach { (name, m) ->
                Card {
                    Column(Modifier.padding(10.dp), verticalArrangement = Arrangement.spacedBy(2.dp)) {
                        Text(name.uppercase(), style = MaterialTheme.typography.titleSmall)
                        Text("Regression (primary)", style = MaterialTheme.typography.labelMedium)
                        Text("MAE: ${"%.3f".format(m.mae)} | RMSE: ${"%.3f".format(m.rmse)} | R2: ${"%.3f".format(m.r2)}")
                        Text("Threshold-derived at 85 (secondary)", style = MaterialTheme.typography.labelMedium)
                        Text("Accuracy: ${"%.2f".format(m.accuracy * 100)}% | F1: ${"%.2f".format(m.f1 * 100)}%")
                        Text("Precision: ${"%.2f".format(m.precision * 100)}% | Recall: ${"%.2f".format(m.recall * 100)}%")
                    }
                }
            }
        }
    }
}
