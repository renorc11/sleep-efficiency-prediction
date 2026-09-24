package com.renotech.sleepefficiency.domain

data class LifestyleState(
    val hadCaffeine: Boolean = true,
    val caffeineMg: String = "120",
    val hadAlcohol: Boolean = true,
    val alcoholUnits: String = "1",
    val dinnerHour: String = "20",
    val stressLevel: String = "5",
    val moodScore: String = "0",
    val activityValue: String = "40",
    val activityUnit: TimeUnit = TimeUnit.MINUTES,
    val napsCount: String = "1",
    val napsTotalMin: String = "20",
    val previousSleepHours: String = "7",
    val workHours: String = "8",
    val sleepTimeHint: String = "",
    val useBlueLightOverride: Boolean = false,
    val blueLightOverrideMinutes: String = "",
    val bedtimeConsistency: String = "0.7",
    val hydrationLiters: String = "2.2"
)

enum class TimeUnit { MINUTES, HOURS }
