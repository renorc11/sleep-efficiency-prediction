package com.renotech.sleepefficiency

import android.os.Bundle
import androidx.activity.ComponentActivity
import androidx.activity.compose.setContent
import androidx.activity.enableEdgeToEdge
import androidx.activity.viewModels
import androidx.compose.runtime.collectAsState
import androidx.compose.runtime.getValue
import com.renotech.sleepefficiency.ui.screens.MainScreen
import com.renotech.sleepefficiency.ui.theme.SleepEfficiencyTheme
import com.renotech.sleepefficiency.usage.UsageAccessHelper

class MainActivity : ComponentActivity() {
    private val vm: MainViewModel by viewModels()

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        enableEdgeToEdge()
        vm.loadMetrics()

        setContent {
            SleepEfficiencyTheme {
                val form by vm.lifestyle.collectAsState()
                val prediction by vm.prediction.collectAsState()
                val metrics by vm.metrics.collectAsState()
                val loading by vm.loading.collectAsState()
                val error by vm.error.collectAsState()
                val hasUsagePermission by vm.hasUsagePermission.collectAsState()

                MainScreen(
                    form = form,
                    metrics = metrics,
                    prediction = prediction,
                    loading = loading,
                    hasUsagePermission = hasUsagePermission,
                    error = error,
                    onPermissionClick = { UsageAccessHelper.openUsageSettings(this) },
                    onRefreshPermission = { vm.refreshUsagePermission() },
                    onUpdate = { updated -> vm.setLifestyle(updated) },
                    onPredict = { vm.predict() },
                    onLoadMetrics = { vm.loadMetrics() }
                )
            }
        }
    }

    override fun onResume() {
        super.onResume()
        vm.refreshUsagePermission()
    }
}
