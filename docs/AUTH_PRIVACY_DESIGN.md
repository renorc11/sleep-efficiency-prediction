# Authentication and Data-Privacy Design, v2 (design only)

Status: design decision record, 2026-11-22. Supersedes v1 of 2026-11-22.
- **Nothing in this document is implemented** unless marked "current behaviour".
- No authentication, schema change, migration, endpoint restriction or Health Connect integration was added.
- Each implementation step needs separate approval.

## D1. Deployment model

| | Current (approved) | Future (out of scope) |
|---|---|---|
| Users | **One local owner** | Several users |
| Hosting | The owner's own machine | Hosted service |
| Registration and login | None: no public registration, no first-party passwords | OIDC sign-in (D9) |
| Backend network exposure | **Localhost only.** Current behaviour: the documented start command `uvicorn app.main:app --reload --port 8000` binds `127.0.0.1`. The Android emulator reaches it at `10.0.2.2`. The Docker files that bound `0.0.0.0` were removed. | TLS-terminated public endpoint |
| CORS | Restricted to the intended local frontend origins. Current behaviour: `localhost` / `127.0.0.1` on any port, configurable with `CORS_ORIGINS` and `CORS_ORIGIN_REGEX`. Proposed: narrow to the ports actually used, and drop `allow_credentials` while no credentials exist. | Exact deployed origins only |
| `/history` | **Retained** for the single owner. All local records belong to that owner. | Scoped per authenticated user (D9) |

**Consequences of the single-user model**
- Any process or person able to reach `127.0.0.1:8000` on the owner's machine can read the history.
- The owner must not bind the backend to a LAN or public address. A physical Android device on the LAN would require that, so it's out of scope until the multi-user controls exist.

## D2. Device identifiers

**Hashed device identifiers are not authentication credentials.**
- A hashed device or installation ID is guessable, copyable, and not secret.
- It must never be used as proof of identity or as an authorization decision, in either deployment model.
- In a future multi-user service, an installation ID may be stored only *after* authentication, as a label for that user's devices, for example to list devices or de-duplicate uploads.

## D3. Existing records

- `backend/sleep_app.db` holds **74 ownerless legacy rows** in `predictions` (count on 2026-11-22).
- **Now:** they stay unowned local records belonging implicitly to the single owner. Don't assign them to a user, delete them, infer ownership, or change their database representation.
- **If multi-user support is introduced later:**
  - add a nullable `user_id`, with legacy rows keeping `NULL`;
  - make `NULL`-owner rows invisible to every user-facing endpoint;
  - offer the local owner a one-time **export** of the legacy rows;
  - delete them only on the owner's explicit request.

  No automatic backfill.

## D4. Observation provenance (future `sleep_observations`)

Observed sleep outcomes are kept **strictly separate from model predictions**.
- A prediction is never written to, or read as, an observation.
- Forecasts read observations only.

| Requirement | Rule |
|---|---|
| Versioning | Append-only rows. A correction adds a new version (`version`, `supersedes_id`); nothing is updated in place. |
| Source identification | `source` (`self_report`, `health_connect:<origin package>`, …), `source_record_id`, `target_type` (`self_reported_hours` first; `device_sleep_ratio` only after its definition is shown to match the validated Fitabase proxy, asleep minutes ÷ session minutes) |
| Morning self-report → night date | A report answered at local time *T* describes the night ending that morning: wake date = *T*'s local date (or the previous date if *T* is before 05:00), and night date = wake date − 1. Same boundary as the sleep-onset spec, `activity_window/2`. Store the IANA timezone with each report. |
| Conflicts on the same night | Resolution is deterministic and recorded: (1) the latest version within a source; (2) across sources, a configured precedence, for example `health_connect` over `self_report` for `device_sleep_ratio`. `target_type`s are never mixed. |
| Reproducibility | Each version has `available_at_utc`. A forecast for night *D* uses only versions with `available_at_utc < c_D` (the night's cutoff). The forecast stores `inputs_as_of = c_D`, so it can be recomputed exactly. |

Proposed columns, in addition to v1's: `id`, `night_date`, `timezone`, `sleep_start_utc`, `sleep_end_utc`, `target_type`, `value`, `unit`, `source`, `source_record_id`, `version`, `supersedes_id`, `recorded_at_utc`, `available_at_utc`, `quality_flags`, `consent_id`.

In single-user mode there's no `user_id`, or a constant owner.

## D5. Health Connect (not implemented)

- **Optional and permission-based:** request read permissions only for consented record types (sleep sessions, steps, heart rate), each as a separate choice.
- **Recheck permissions at every sync:** call `getGrantedPermissions()` before reading. Revocation isn't pushed to the app.
- **On revocation:** stop importing that type immediately and record the revocation time. Ask the owner whether to **keep or delete** previously imported records of that type, and act on the answer.
- **Stale change tokens:** a changes token unused for about 30 days expires. On expiry, do a bounded full re-read (Health Connect's history window) and de-duplicate by `source_record_id`.
- **Provenance:** keep the origin app package as part of `source`.

## D6. Privacy and retention (proposed policy, not implemented)

| Data | Proposed retention |
|---|---|
| Predictions (`predictions` rows) | 90 days, then deleted by a scheduled job |
| Observations | While the local application is actively used, or until the owner deletes them |
| Backups | 30-day rolling retention |
| Deletions | A deletion log (record type and ID, no contents) is kept and **re-applied after any backup restore**, so restored data doesn't resurrect deleted records |

**Minimisation (proposed)**
- Keep the raw 288-bin activity sequence only if the owner opts in; otherwise store derived features.
- Never collect app names or content (current behaviour).
- Logs exclude payloads, health values and tokens. Current behaviour: the Android client logs request metadata only, in debug builds. A search of the backend found no payload logging at info level, but generated LLM text is logged at debug level.

## D7. Encryption

**Current single-user local deployment (owner actions, proposed)**
- Full-disk encryption on the owner's machine (FileVault on macOS).
- File permissions limited to the owner on `backend/sleep_app.db`, `ml/artifacts/` and local dataset folders.
- Backups encrypted (for example an encrypted disk image or encrypted backup tool) and kept under the 30-day policy.

**Future multi-user service**
- TLS 1.2 or later with HSTS.
- A managed database with encryption at rest.
- Keys in a managed key service with rotation.
- A least-privilege database role.
- Encrypted backups with their own key.
- Consider field-level encryption for observation values and free text.

## D8. Jurisdiction and age

- **India's Digital Personal Data Protection Act, 2023 and its rules may be relevant** once personal data of individuals in India is processed. Its requirements include notice, consent, purpose limitation, erasure, and verifiable parental consent for children under 18.
- **Applicability needs legal confirmation and is not settled here:** the Act's commencement and the phasing of its rules, and whether exemptions (for example research or personal and domestic use) cover a single-owner academic prototype.
- Other laws, for example GDPR, may apply if the service is ever offered in other jurisdictions.
- Until that's confirmed, the design assumes adult users only and consent-based collection (D5, D6).

## D9. Future multi-user architecture (not current scope)

- **Authentication:** an external OIDC provider; no first-party passwords. Android uses the authorization-code flow with **PKCE** (AppAuth or an equivalent).
- **Sessions:** **opaque, server-side sessions** stored in the database, so revocation is immediate.
  - Web: `HttpOnly`, `Secure`, `SameSite=Lax` cookies through a Next.js server layer, with **CSRF tokens** on state-changing requests.
  - Android: tokens in Keystore-backed storage.
- **Refresh tokens:** rotated on use, with **reuse detection**; a reused token revokes the whole token family.
- **Logout** revokes the session and its refresh-token family. Account deletion revokes all sessions.
- **Ownership:** `user_id` comes only from the verified session. All queries go through repository helpers that always filter by owner. Other users' IDs return 404. Client-supplied `user_id` fields are ignored.
- **Data isolation:** per-user scoping on predictions, observations, consents, exports and deletions. No admin or bulk endpoints without a separate review.
- **Export isolation:** the export contains only the caller's rows and is generated server-side for the authenticated user.
- **Deletion and restores:** account deletion cascades, and the D6 deletion log is re-applied after restores.
- **Migrations:**
  0. Back up the database. Fix the relative SQLite URL in `alembic.ini`. Compare the live schema with revision `0001` column by column, then `alembic stamp 0001`.
  1. Additive revisions: `users`/`sessions`/`consents`; `sleep_observations`; a nullable `predictions.user_id`. On SQLite these need `batch_alter_table`.
  2. Test upgrade → downgrade → upgrade on a copy, with checksums.

  A downgrade that drops observations must export them first.

## D10. Required future security tests (before enabling personal-history forecasting)

1. Access after logout is rejected: 401 on every personal route.
2. Replay of a captured session or access token after logout or expiry is rejected.
3. Cross-user access: user B can't list, read, update, delete or export user A's records; A's IDs return 404. Tests are generated from the route table so new routes are covered.
4. Client-supplied ownership fields (`user_id` in the body or query) are ignored.
5. CSRF: a state-changing request without a valid CSRF token is rejected in cookie mode.
6. Refresh-token reuse revokes the token family.
7. Backup restore after deletion: deleted records stay deleted once the deletion log is re-applied.
8. Logs contain no tokens, payloads or health values (log-capture test).
9. Health Connect permission revocation stops imports, and the keep-or-delete choice is honoured.
10. Forecasts read only the caller's observations with `available_at_utc < c_D`; predictions are never used; fewer than 3 valid nights returns `insufficient_history`.
11. Migration round-trip test (D9).
