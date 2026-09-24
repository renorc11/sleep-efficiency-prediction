package com.renotech.sleepefficiency.data.repository

import com.renotech.sleepefficiency.data.model.MetricsResponse
import com.renotech.sleepefficiency.data.model.PredictionRequest
import com.renotech.sleepefficiency.data.model.PredictionResponse
import com.renotech.sleepefficiency.data.network.SleepApi

class SleepRepository(private val api: SleepApi) {
    suspend fun predict(payload: PredictionRequest): PredictionResponse = api.predict(payload)
    suspend fun fetchMetrics(): MetricsResponse = api.modelMetrics()
}
