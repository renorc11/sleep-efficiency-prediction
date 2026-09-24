package com.renotech.sleepefficiency.util

fun String.toDoubleSafe(default: Double = 0.0): Double = this.toDoubleOrNull() ?: default
fun String.toIntSafe(default: Int = 0): Int = this.toIntOrNull() ?: default
