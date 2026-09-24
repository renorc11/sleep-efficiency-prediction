package com.renotech.sleepefficiency.ui.theme

import androidx.compose.material3.MaterialTheme
import androidx.compose.material3.darkColorScheme
import androidx.compose.material3.lightColorScheme
import androidx.compose.runtime.Composable

private val LightColors = lightColorScheme(
    primary = Orange,
    secondary = Blue,
    tertiary = Green
)

private val DarkColors = darkColorScheme(
    primary = Orange,
    secondary = Blue,
    tertiary = Green
)

@Composable
fun SleepEfficiencyTheme(darkTheme: Boolean = false, content: @Composable () -> Unit) {
    MaterialTheme(
        colorScheme = if (darkTheme) DarkColors else LightColors,
        typography = Typography,
        content = content
    )
}
