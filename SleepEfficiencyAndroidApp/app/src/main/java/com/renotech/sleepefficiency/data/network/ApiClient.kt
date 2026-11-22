package com.renotech.sleepefficiency.data.network

import com.jakewharton.retrofit2.converter.kotlinx.serialization.asConverterFactory
import com.renotech.sleepefficiency.BuildConfig
import kotlinx.serialization.json.Json
import okhttp3.MediaType.Companion.toMediaType
import okhttp3.OkHttpClient
import okhttp3.logging.HttpLoggingInterceptor
import retrofit2.Retrofit
import java.util.concurrent.TimeUnit

object ApiClient {
    private val json = Json {
        ignoreUnknownKeys = true
        prettyPrint = false
    }

    private val client: OkHttpClient = OkHttpClient.Builder()
        .connectTimeout(10, TimeUnit.SECONDS)
        // POST /predict runs three models, their explanations, evidence retrieval and
        // BioMistral (server budget about 45 s). OkHttp's 10 s default read timeout
        // would make every prediction fail.
        .readTimeout(90, TimeUnit.SECONDS)
        .callTimeout(100, TimeUnit.SECONDS)
        .apply {
            // BODY logging wrote the usage timeline and lifestyle answers to logcat in
            // every build. Log request metadata only, and only in debug builds.
            if (BuildConfig.DEBUG) {
                addInterceptor(HttpLoggingInterceptor().apply { level = HttpLoggingInterceptor.Level.BASIC })
            }
        }
        .build()

    val api: SleepApi = Retrofit.Builder()
        .baseUrl(BuildConfig.API_BASE_URL)
        .client(client)
        .addConverterFactory(json.asConverterFactory("application/json".toMediaType()))
        .build()
        .create(SleepApi::class.java)
}
