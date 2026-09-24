package com.renotech.sleepefficiency.data.network

import com.renotech.sleepefficiency.data.model.MetricsResponse
import com.renotech.sleepefficiency.data.model.PredictionRequest
import com.renotech.sleepefficiency.data.model.PredictionResponse
import retrofit2.http.Body
import retrofit2.http.GET
import retrofit2.http.POST

interface SleepApi {
    @POST("predict")
    suspend fun predict(@Body payload: PredictionRequest): PredictionResponse

    @GET("explain/model-metrics")
    suspend fun modelMetrics(): MetricsResponse
}
