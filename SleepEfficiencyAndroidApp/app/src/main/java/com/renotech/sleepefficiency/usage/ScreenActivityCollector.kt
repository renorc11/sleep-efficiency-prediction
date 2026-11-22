package com.renotech.sleepefficiency.usage

import android.app.usage.UsageEvents
import android.app.usage.UsageStatsManager
import android.content.Context
import java.util.Calendar

class ScreenActivityCollector(private val context: Context) {
    companion object {
        const val BINS = 288
        private const val BIN_MS = 5 * 60 * 1000L
    }

    fun collectTodaySequence(): List<Int> {
        val usageStatsManager = context.getSystemService(Context.USAGE_STATS_SERVICE) as UsageStatsManager

        val now = System.currentTimeMillis()
        val calendar = Calendar.getInstance().apply {
            timeInMillis = now
            set(Calendar.HOUR_OF_DAY, 0)
            set(Calendar.MINUTE, 0)
            set(Calendar.SECOND, 0)
            set(Calendar.MILLISECOND, 0)
        }
        val startOfDay = calendar.timeInMillis

        val bins = MutableList(BINS) { 0 }

        // Mark every bin overlapped by [start, end], clamped to today so far.
        fun markInterval(start: Long, end: Long) {
            val first = ((maxOf(start, startOfDay) - startOfDay) / BIN_MS).toInt()
            val last = ((minOf(end, now) - startOfDay) / BIN_MS).toInt()
            for (index in first..last) {
                if (index in 0 until BINS) bins[index] = 1
            }
        }

        val events = usageStatsManager.queryEvents(startOfDay, now)
        val event = UsageEvents.Event()
        // Foreground start time per activity. Previously only the bin in which an app
        // came to the foreground was marked, so an hour in one app looked like 55
        // minutes of inactivity. Each foreground interval is now marked until the
        // matching pause/background event. (MOVE_TO_FOREGROUND/BACKGROUND share their
        // values with ACTIVITY_RESUMED/PAUSED; both names are checked for clarity.)
        val openSince = HashMap<String, Long>()

        while (events.hasNextEvent()) {
            events.getNextEvent(event)
            val key = "${event.packageName}/${event.className}"
            val type = event.eventType
            if (type == UsageEvents.Event.MOVE_TO_FOREGROUND || type == UsageEvents.Event.ACTIVITY_RESUMED) {
                openSince[key] = event.timeStamp
            } else if (type == UsageEvents.Event.MOVE_TO_BACKGROUND || type == UsageEvents.Event.ACTIVITY_PAUSED) {
                // A pause without a resume today began before midnight; it is ignored
                // rather than assumed active, to avoid marking sleep time as activity.
                openSince.remove(key)?.let { started -> markInterval(started, event.timeStamp) }
            }
        }
        // Activities still in the foreground when the query ran.
        openSince.values.forEach { started -> markInterval(started, now) }

        val currentIndex = ((now - startOfDay) / BIN_MS).toInt().coerceIn(0, BINS - 1)
        bins[currentIndex] = 1
        return bins
    }

    fun estimateBlueLightMinutes(sequence: List<Int>): Double {
        if (sequence.size != BINS) return 0.0

        val sleepStart = inferLikelySleepStartIdx(sequence)
        val windowBins = 24 // 2 hours * 12 bins/hour
        val activeBins = (0 until windowBins).count { offset ->
            val idx = (sleepStart - offset - 1 + BINS) % BINS
            sequence[idx] == 1
        }
        return activeBins * 5.0
    }

    private fun inferLikelySleepStartIdx(sequence: List<Int>): Int {
        val eveningStart = 18 * 12
        val morningEnd = 8 * 12
        val extended = sequence + sequence.take(morningEnd)

        var bestIdx = 23 * 12
        var bestLen = 0
        var i = eveningStart
        while (i < extended.size) {
            if (extended[i] == 0) {
                var j = i
                while (j < extended.size && extended[j] == 0) j++
                val runLen = j - i
                if (runLen >= 6 && runLen > bestLen) {
                    bestLen = runLen
                    bestIdx = i % BINS
                }
                i = j
            } else {
                i++
            }
        }
        return bestIdx
    }
}
