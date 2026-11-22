# Sleep-Onset Train/Serve Skew: Investigation and Specification `activity_window/2`

Status: investigation plus **draft specification**, documentation only (2026-11-22). Supersedes `activity_window/1`.
- No production preprocessing, `/predict`, training script, model artifact, forecast endpoint or database schema was changed.
- Every default marked **[assumption]** must be validated against data (section E8) before implementation.
- Implementation needs separate approval.

## 1. Verified findings (unchanged from v1)

| Stage | Location | Behaviour |
|---|---|---|
| Training sequences | `ml/scripts/prepare_real_sleep_health_data.py::generate_sequence` | Synthetic. The **target night's** sleep window, built from the same `Sleep Duration` that defines the target, is marked inactive on a 24-hour circle. |
| Training features | `ml/scripts/preprocess.py::infer_sleep_features` | Whole-sequence scan; no time masking |
| Web collector | `frontend/components/ScreenTracker.tsx` | A 60-second timer marks the current bin active even when the user is idle. Bins before page load read 0. No midnight reset; no timezone sent. |
| Android collector | `.../usage/ScreenActivityCollector.kt` | Covers local midnight → now only. Future bins are 0; the current bin is forced to 1. No timezone sent. |
| Serving features | `backend/app/services/inactivity_service.py::infer_sleep_from_sequence` | Uses the server's naive `datetime.now()`. Bins after it become **active**. Bins 00:00–08:00 of the same day are appended as if they were the next morning. |

**Measured effect, unchanged.** On the example payload, the training rule gives a longest evening inactivity of 450 min and 18 night-time bursts. The serving rule gives 0, 0, 35, 0 and 0 min, and 6, 6, 18, 0 and 6 bursts, at simulated server times 09:00, 21:00, 23:30, 02:00 and 07:30.

**Problems**
- **Target leakage by construction:** the training sequences encode the night being predicted.
- **"0" has several meanings:** inactive, unobserved and in the future all look the same.
- **Server-clock and timezone dependence.**
- **Inactivity is treated as sleep.**

**The specification below defines a representation for coming-night forecasting.** It doesn't repair the existing same-night estimator, which would need retraining.

## E1. Time definitions

| Symbol | Definition |
|---|---|
| `t_req` | When the client assembles the request, as UTC integer milliseconds. Sent by the client. |
| `Z` | The device's IANA timezone at `t_req`, e.g. `Asia/Kolkata`. Sent by the client and validated against the IANA database. |
| `B` | Night boundary: **05:00 local [assumption]** |
| `D` | Target sleep-night date (local date on which the night begins) = `local_date_Z(t_req − B)` |
| `c_D` | Cutoff: **20:00 local on D [assumption]**, resolved to a UTC instant in `Z` (E6) |
| `observed_until` | `min(t_req, c_D)`, the latest instant from which observations may be used |
| `W0` | Window start: `c_D − 24 h` (absolute) |

**The server is the authority.** It derives `D`, `c_D`, `observed_until` and the window from `t_req`, `Z` and raw intervals. Client-computed values, if sent, are ignored or must match.

**No data at or after `observed_until` is used.** Every bin from `observed_until` onward has `observed = 0` and `activity = 0`.

The server clock is used **only** to reject requests where `t_req > server_now + 5 min` (future) or `t_req < server_now − 24 h` (stale) **[assumption]**. It never affects features.

## E2. Target-night rules and examples

| Local request time | `D` | Status | Data used |
|---|---|---|---|
| 05:00 – 19:59 | today | `early` | Up to `t_req`; bins `[t_req, c_D)` unobserved |
| 20:00 – 23:59 | today | `post_cutoff` | Up to `c_D` only |
| 00:00 – 04:59 | yesterday | `post_cutoff` + `night_in_progress` | Up to `c_D` of yesterday only |

Worked examples, computed with Python `zoneinfo`:

| Zone, local request | `D` | `c_D` (UTC) | `W0` (UTC) | `observed_until` (UTC) | Observable bins | Status |
|---|---|---|---|---|---|---|
| Asia/Kolkata, 2026-11-22 15:10 | 11-22 | 2026-11-22T14:30Z | 2026-11-21T14:30Z | 09:40Z | 0–229 (230–287 unobserved) | early |
| Asia/Kolkata, 2026-11-22 21:30 | 11-22 | 14:30Z | 2026-11-21T14:30Z | 14:30Z | 0–287 | post_cutoff |
| Asia/Kolkata, 2026-11-23 01:30 | **11-22** | 2026-11-22T14:30Z | 2026-11-21T14:30Z | 2026-11-22T14:30Z | 0–287 | post_cutoff, night_in_progress |
| Asia/Kolkata, 2026-11-23 09:00 | **11-23** | 2026-11-23T14:30Z | 2026-11-22T14:30Z | 03:30Z | 0–155 (156–287 unobserved) | early |

## E3. Observation states and coverage

Each 5-minute bin is in exactly one state:

| State | `observed` | `activity` | Meaning |
|---|---|---|---|
| Observed active | 1 | 1 | The collector was able to observe, and an activity interval overlaps the bin |
| Observed inactive | 1 | 0 | The collector was able to observe, and no activity was seen |
| Unobserved | 0 | 0 | The collector couldn't observe: not running, no permission, device off, suspended, or after `observed_until`. **Unknown, not inactive.** |

- **The absence of events never implies inactivity.** A bin is inactive only if it is covered.
- **Coverage is defined per source (E4).** `observed` = union of the source's coverage intervals and its activity intervals; an activity event is itself evidence of observation.
- **No state means sleep.** Phone lock, a dark screen or inactivity are behavioural signals. They aren't sleep labels and are never used as sleep targets.

## E4. Source-specific semantics

| `source` | Activity intervals | Coverage intervals | Caveats |
|---|---|---|---|
| `web_input` | Input events (pointer, key, touch, click) as point events | Heartbeats every 60 s while `document.visibilityState === "visible"`. Coverage runs from one heartbeat to the next if the gap is ≤ 90 s **[assumption]**. Longer gaps (suspend, throttling, tab hidden) are unobserved. | **Covers only this tab.** The heartbeat's reliability is unvalidated, and it isn't continuous sensing. |
| `android_foreground` | `ACTIVITY_RESUMED` → `ACTIVITY_PAUSED` intervals from `UsageStatsManager` | From `max(W0, usage-access grant time)` to `t_req`, excluding `DEVICE_SHUTDOWN` → `DEVICE_STARTUP` (API 28+) | Event availability and retention vary by device and OS version: unvalidated |
| `android_screen_interactive` | `SCREEN_INTERACTIVE` → `SCREEN_NON_INTERACTIVE` (API 28+) | As above | A screen that is on isn't necessarily being used |
| `studentlife_unlocked` | Complement of merged `phonelock` intervals within coverage | Activity-inference samples no more than *g* minutes apart **[assumption: g = 10]** | The 600-minute cap (E5); US/Eastern assumed |
| `studentlife_screen_on` | Complement of merged `dark` intervals within coverage | As above | A dark screen may also mean a covered sensor |
| `fitabase_steps` | Minutes with steps > 0 | Minutes with a row present | Not worn and not moving can't be told apart; local time without a zone |

**Sources are never pooled into one model** until their equivalence is shown on data.
- Features carry their `source`.
- Training and evaluation are per source.
- Cross-platform parity of *collectors* is claimed only after the collector tests (E9) pass on that platform.

## E5. Interval rules

- **Timestamps:** UTC integer milliseconds since the Unix epoch.
- **Intervals:** half-open `[start, end)`.
  - `end < start`: rejected.
  - `end == start`: a **point event**.
- **Bins:** 288 bins aligned to the cutoff. Bin *k* = `[W0 + 300000·k, W0 + 300000·(k+1))`, *k* = 0…287.
  - An interval marks bin *k* active if it overlaps the bin with positive length.
  - A point event at *t* marks bin *k* active if `bin_start ≤ t < bin_end`.
- **Boundaries:**
  - An interval ending exactly at a bin start doesn't touch that bin.
  - An event exactly at `observed_until` (or `c_D`) is **excluded**.
- **Ordering, duplicates and overlaps:** sort by start, then merge intervals that overlap or touch. Exact duplicates are idempotent. The output doesn't depend on input order.
- **Clipping:** intervals are clipped to `[W0, observed_until)`.
- **Long intervals:** an interval longer than 24 h is clipped and flagged `long_interval`.
- **Censored intervals (StudentLife):**
  - a `phonelock`, `dark` or `phonecharge` interval of at least 599 minutes is right-censored by the 600-minute collection cap;
  - from its end until that same sensor's next interval starts, the sensor's channel is `observed = 0`.

## E6. Timezones and DST

- **Window length:** always 24 absolute hours = 288 bins, regardless of how long the local day is.
- **Resolving `c_D` in `Z`:**
  - a **non-existent** local time (spring forward) uses the pre-transition offset, which moves it forward by the gap;
  - an **ambiguous** local time (fall back) takes the earlier instant.

  These are Python `fold=0` semantics, and `java.time` behaves the same by default. 20:00 is unaffected in common zones, but the rule covers configurable cutoffs.
- **Verified examples** (Python `zoneinfo`):

  | Zone and night | Cutoff | Window starts (local) |
  |---|---|---|
  | America/New_York, night 2026-03-08 (spring forward that morning) | 2026-03-09T00:00Z | 2026-03-07 19:00 EST |
  | America/New_York, night 2026-11-01 (fall back) | 2026-11-02T01:00Z | 2026-10-31 21:00 EDT |
  | Asia/Kathmandu (+05:45) | 14:15Z | 20:00 local; bins stay aligned to the cutoff, so 15-minute offsets are handled |
  | Australia/Lord_Howe (30-minute DST), night 2026-04-05: offset +11:00 → +10:30 | 09:30Z | 20:30 local |
  | Australia/Lord_Howe, night 2026-10-04: +10:30 → +11:00 | 09:00Z | 19:30 local |

  So the local span of a window can be 23, 23.5, 24.5 or 25 hours. **Never assume a local day has 24 hours.**
- **Timezone change within the window** (travel):
  - Bins are UTC, so the arrays are unaffected.
  - Local-time features (E7) use `Z` at `t_req` for every bin.
  - The client sends `tz_changed_in_window: true` when it knows of a change, and those features are flagged approximate.
  - Training data assumes a single timezone per participant.

## E7. Coverage-aware features

`MIN_COV = 0.5` **[provisional; validate per E8]**. Bin ranges refer to E5 indices.

| Feature | Definition | Missing-value rule |
|---|---|---|
| `obs_frac_24h` | mean(`observed[0:288]`) | Always computed |
| `obs_frac_last3h` | mean(`observed[252:288]`) | Always computed |
| `active_frac_last3h` | Σ`activity[252:288]` / Σ`observed[252:288]` | Null if `obs_frac_last3h < MIN_COV` |
| `longest_observed_inactive_run_last12h` | Longest run of consecutive bins in [144, 288) with `observed = 1` and `activity = 0`, × 5 min. **An unobserved bin ends a run.** | Null if the observed fraction of [144, 288) is below `MIN_COV` |
| `active_frac_prev_night` | Over bins whose local time in `Z` is in [22:00, 06:00) on the night before *D*: active ÷ observed | Null if the observed fraction of those bins is below `MIN_COV`, or no such bins exist |
| `minutes_since_last_activity` | `(observed_until − end of the last activity interval)` in minutes | Null if any bin after the last activity is unobserved. **1440** plus flag `no_activity_in_window` if the window is fully observed with no activity. |

- **Every feature is per source.**
- **Unobserved bins never enter a numerator or denominator as inactive.**
- **The coverage features are always reported,** so models can learn from missingness explicitly.

## E8. Validation requirements (before implementation)

| # | Check | Data |
|---|---|---|
| 1 | Is the 20:00 cutoff suitable for StudentLife: how often do long overnight lock or dark intervals start before 20:00? | StudentLife `phonelock`, `dark` |
| 2 | The gap *g* between activity-inference samples, and how much coverage various values of *g* yield | StudentLife `activity` |
| 3 | Frequency and position of capped `phonelock`, `dark` and `phonecharge` intervals | StudentLife sensing |
| 4 | The US/Eastern assumption: plausibility of the local-time activity pattern | StudentLife sensing |
| 5 | Timing of morning EMA sleep reports relative to the 05:00 boundary | StudentLife EMA Sleep |
| 6 | Sensitivity of features and forecasts to `MIN_COV` (0.3 / 0.5 / 0.7) | StudentLife, Fitabase |
| 7 | Android event availability: `ACTIVITY_*`, `SCREEN_*` and `DEVICE_*` events across devices and OS versions, and retention over 24 h | Consented pilot devices |
| 8 | Reliability of the web heartbeat: gap distribution under visible, hidden and suspended states across browsers | Consented pilot |

## E9. Test vectors

**Architecture (proposed)**
- **One canonical window builder, in Python,** used by both training and serving.
- Platform collectors only convert native events into raw UTC intervals plus coverage intervals, with `source`, `Z` and `t_req`.

There are two separate test levels:
1. **Platform collector tests,** per platform: native events → normalised UTC intervals. Run in TypeScript (web) and Kotlin (Android) on their own toolchains. **No cross-platform parity is claimed until these have run on each platform.** The Android toolchain currently can't build offline in this environment.
2. **Canonical builder tests,** Python: normalised intervals → `observed` and `activity` arrays and E7 features. Integers must match exactly; floats within 1e-9. Each vector records `spec_version: "activity_window/2"`.

**Required vector coverage**

| Area | Cases |
|---|---|
| Bin boundaries | An interval ending exactly at a bin start; a point event at a bin start; a 1 ms interval |
| Cutoff boundaries | An event at `c_D − 1 ms` (included) and at `c_D` (excluded); an interval spanning `c_D` |
| Duplicates and overlaps | Exact duplicates; overlapping and touching intervals → identical output |
| Out-of-order input | A shuffled interval list → identical output |
| Kathmandu | +05:45 cutoff and bin alignment |
| Lord Howe DST | Nights 2026-04-05 and 2026-10-04 (E6) |
| New York DST | Nights 2026-03-08 and 2026-11-01 |
| Midnight requests | Requests at 00:30 and 04:59 (→ yesterday) and at 05:00 (→ today) |
| Future or stale `t_req` | `server_now + 6 min` and `server_now − 25 h` rejected; `+4 min` accepted |
| Timezone change | `tz_changed_in_window` sets the approximate-feature flag; the arrays are unchanged |
| Android shutdown/startup | A shutdown → startup gap is unobserved |
| Long intervals | An interval longer than 24 h is clipped and flagged |
| Censored intervals | A 600-minute lock interval: bins after it, up to the next lock event, are unobserved |
| Missing observations | A fully unobserved window (features null); a fully observed inactive window (1440 + flag); web heartbeat gaps of 89 s (observed) and 91 s (unobserved) |
| Early request | The Kolkata 15:10 and 09:00 examples (E2) |
| Invalid input | Unknown IANA name; `end < start`; non-integer timestamps → rejected |
