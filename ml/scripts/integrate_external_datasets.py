"""Inventory and integrate external sleep datasets without fabricating joins.

The project models lifestyle and screen-inactivity signals.  This script keeps
wearable/PSG sources separate, because they use different cohorts and would
otherwise violate both provenance and the deployed application's data contract.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import re
import shutil
import sys
import zipfile
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG = ROOT / "ml" / "config" / "dataset_integration.json"
DATA_DIR = ROOT / "ml" / "data"

CANONICAL_SLEEP_COLUMNS = [
    "participant_id",
    "source_participant_id",
    "sleep_date",
    "bedtime",
    "wakeup_time",
    "total_sleep_duration_hours",
    "time_in_bed_hours",
    "sleep_efficiency",
    "sleep_efficiency_calculation_source",
    "sleep_quality_score",
    "caffeine_consumption_level",
    "alcohol_consumption_level",
    "exercise_frequency_per_week",
    "physical_activity_level_source",
    "stress_level_source",
    "daily_steps",
    "awakenings",
    "age",
    "gender",
    "occupation",
    "source_dataset",
    "source_file",
    "source_row_number",
    "record_status",
    "quality_flags",
]


def load_config(config_path: Path) -> dict[str, Any]:
    with config_path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def snake_case(value: str) -> str:
    value = value.replace("%", " percent ").replace("/", " ")
    value = re.sub(r"[^0-9A-Za-z]+", "_", value).strip("_")
    return re.sub(r"_+", "_", value).lower()


def normalized_missing(value: Any, missing_values: set[str]) -> bool:
    return value is None or str(value).strip().lower() in missing_values


def normalize_efficiency(value: Any) -> float | None:
    """Return sleep efficiency in percent, accepting source fractions or percents."""
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    if np.isnan(number):
        return None
    if 0 <= number <= 1:
        number *= 100
    return number if 0 <= number <= 100 else None


def calculate_sleep_efficiency(total_sleep_hours: Any, time_in_bed_hours: Any) -> float | None:
    try:
        sleep = float(total_sleep_hours)
        bed = float(time_in_bed_hours)
    except (TypeError, ValueError):
        return None
    if not np.isfinite(sleep) or not np.isfinite(bed) or sleep < 0 or bed <= 0:
        return None
    efficiency = sleep / bed * 100
    return efficiency if 0 <= efficiency <= 100 else None


def duration_hours(start: pd.Series, end: pd.Series) -> pd.Series:
    delta = (end - start).dt.total_seconds() / 3600
    # Some sources store the next-morning wake time without advancing its date.
    return delta.where(delta >= 0, delta + 24)


def source_participant_from_filename(path: Path, pattern: str) -> str | None:
    match = re.search(pattern, path.name)
    return match.group(1) if match else None


def count_newlines(path: Path) -> int:
    with path.open("rb") as handle:
        return sum(chunk.count(b"\n") for chunk in iter(lambda: handle.read(1024 * 1024), b""))


def profile_csv(
    path: Path,
    missing_values: set[str],
    duplicate_hash_limit: int = 250_000,
    full_scan_max_bytes: int = 1_000_000,
    sample_rows: int = 25_000,
) -> dict[str, Any]:
    """Stream a CSV so inventorying many raw files does not change them."""
    is_sampled = path.stat().st_size > full_scan_max_bytes
    with path.open("r", encoding="utf-8-sig", errors="replace", newline="") as handle:
        sample = handle.read(4096)
        handle.seek(0)
        try:
            dialect = csv.Sniffer().sniff(sample, delimiters=",;\t|")
        except csv.Error:
            dialect = csv.excel
        reader = csv.reader(handle, dialect)
        header = next(reader, [])
        missing = [0] * len(header)
        rows = 0
        duplicates = 0
        duplicate_check_status = "exact"
        seen: set[bytes] = set()
        date_candidates = [h for h in header if re.search(r"date|time|timestamp|bed|wake", h, re.I)]
        id_candidates = [h for h in header if re.search(r"(^id$|_id$|person|participant|subject|uid|user)", h, re.I)]
        for row in reader:
            rows += 1
            row = row + [""] * max(0, len(header) - len(row))
            for index, value in enumerate(row[: len(header)]):
                if normalized_missing(value, missing_values):
                    missing[index] += 1
            digest = hashlib.blake2b("\x1f".join(row).encode("utf-8", "replace"), digest_size=16).digest()
            if rows <= duplicate_hash_limit:
                if digest in seen:
                    duplicates += 1
                else:
                    seen.add(digest)
            elif duplicate_check_status == "exact":
                # A full hash set for minute-level wearable exports can consume
                # several GB. Preserve the row/missing audit and disclose the limit.
                duplicate_check_status = f"not_computed_after_{duplicate_hash_limit}_rows"
            if is_sampled and rows >= sample_rows:
                break
    reported_rows = max(0, count_newlines(path) - 1) if is_sampled else rows
    return {
        "format": "csv",
        "rows": reported_rows,
        "quality_profile_rows_scanned": rows,
        "quality_profile_scope": "sampled_first_rows" if is_sampled else "complete_file",
        "columns": header,
        "column_count": len(header),
        "missing_values": {header[i]: missing[i] for i in range(len(header))},
        "duplicate_rows": duplicates if duplicate_check_status == "exact" else None,
        "duplicate_check_status": duplicate_check_status,
        "date_time_fields": date_candidates,
        "participant_identifier_candidates": id_candidates,
    }


def profile_json(path: Path) -> dict[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8", errors="replace"))
    except (json.JSONDecodeError, OSError) as error:
        return {"format": "json", "read_error": str(error)}
    if isinstance(payload, dict):
        return {"format": "json", "top_level_type": "object", "keys": sorted(payload.keys()), "rows": 1}
    if isinstance(payload, list):
        first = payload[0] if payload else None
        columns = sorted(first.keys()) if isinstance(first, dict) else []
        return {"format": "json", "top_level_type": "array", "rows": len(payload), "columns": columns}
    return {"format": "json", "top_level_type": type(payload).__name__, "rows": 1}


def profile_file(path: Path, root: Path, missing_values: set[str]) -> dict[str, Any]:
    record: dict[str, Any] = {
        "relative_path": str(path.relative_to(root)),
        "size_bytes": path.stat().st_size,
        "extension": path.suffix.lower() or "[no_extension]",
    }
    try:
        if path.suffix.lower() == ".csv":
            record.update(profile_csv(path, missing_values))
        elif path.suffix.lower() == ".json":
            record.update(profile_json(path))
        elif path.suffix.lower() == ".zip":
            with zipfile.ZipFile(path) as archive:
                members = [member for member in archive.infolist() if not member.is_dir()]
            record.update({
                "format": "zip",
                "archive_member_count": len(members),
                "archive_member_extensions": dict(Counter(Path(m.filename).suffix.lower() or "[no_extension]" for m in members)),
            })
        else:
            record["format"] = "text" if path.suffix.lower() in {".txt", ".md"} else "binary_or_unknown"
    except Exception as error:  # The report must expose unreadable files rather than omit them.
        record["read_error"] = str(error)
    return record


def inventory_source(source_root: Path, output_path: Path, missing_values: set[str]) -> dict[str, Any]:
    files = [path for path in source_root.rglob("*") if path.is_file() and path.name != ".DS_Store"]
    entries: list[dict[str, Any]] = []
    if output_path.exists():
        try:
            previous = json.loads(output_path.read_text(encoding="utf-8"))
            if previous.get("source_root") == str(source_root):
                if previous.get("inventory_complete", False):
                    print(f"[Inventory] using completed inventory for {len(previous.get('files', []))}/{len(files)} files", flush=True)
                    return previous
                entries = previous.get("files", [])
                print(f"[Inventory] resuming at {len(entries)}/{len(files)} files", flush=True)
        except json.JSONDecodeError:
            entries = []

    def checkpoint(complete: bool) -> dict[str, Any]:
        report = {
            "generated_at_utc": datetime.now(timezone.utc).isoformat(),
            "source_root": str(source_root),
            "file_count": len(files),
            "profiled_file_count": len(entries),
            "inventory_complete": complete,
            "files": entries,
            "formats": dict(Counter(entry["format"] for entry in entries)),
        }
        output_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
        return report

    for index, path in enumerate(sorted(files)[len(entries) :], start=len(entries) + 1):
        entries.append(profile_file(path, source_root, missing_values))
        if index % 100 == 0 or index == len(files):
            checkpoint(complete=index == len(files))
            print(f"[Inventory] profiled {index}/{len(files)} files", flush=True)
    return checkpoint(complete=len(entries) == len(files))


def ensure_raw_reference(source_root: Path, external_raw: Path, link_name: str) -> Path:
    """Create a non-copying raw-data reference; the originals remain untouched."""
    external_raw.mkdir(parents=True, exist_ok=True)
    link = external_raw / link_name
    if link.is_symlink() or link.exists():
        if link.is_symlink() and link.resolve() == source_root.resolve():
            return link
        raise FileExistsError(f"Raw-data reference already exists with a different target: {link}")
    link.symlink_to(source_root.resolve(), target_is_directory=True)
    return link


def read_source(path: Path, missing_values: Iterable[str]) -> pd.DataFrame:
    return pd.read_csv(path, na_values=list(missing_values), keep_default_na=True)


def build_sleep_efficiency_source(source_root: Path, spec: dict[str, Any], missing_values: Iterable[str]) -> pd.DataFrame:
    source_path = source_root / spec["path"]
    raw = read_source(source_path, missing_values)
    output = pd.DataFrame(index=raw.index)
    for original, canonical in spec["column_map"].items():
        if original in raw:
            output[canonical] = raw[original]
    output["participant_id"] = spec["participant_id_prefix"] + "_" + output["source_participant_id"].astype(str)
    output["bedtime"] = pd.to_datetime(output.get("bedtime"), errors="coerce")
    output["wakeup_time"] = pd.to_datetime(output.get("wakeup_time"), errors="coerce")
    output["sleep_date"] = output["bedtime"].dt.date.astype("string")
    output["time_in_bed_hours"] = duration_hours(output["bedtime"], output["wakeup_time"])
    output["sleep_efficiency"] = output["sleep_efficiency"].map(normalize_efficiency)
    calculated = [calculate_sleep_efficiency(a, b) for a, b in zip(output["total_sleep_duration_hours"], output["time_in_bed_hours"])]
    output["sleep_efficiency_calculation_source"] = np.where(output["sleep_efficiency"].notna(), "reported", "calculated_from_duration_and_time_in_bed")
    output["sleep_efficiency"] = output["sleep_efficiency"].fillna(pd.Series(calculated, index=output.index))
    output["source_dataset"] = "sleep_efficiency"
    output["source_file"] = spec["path"]
    output["source_row_number"] = np.arange(2, len(output) + 2)
    output["record_status"] = np.where(output["sleep_efficiency"].notna(), "target_available", "missing_target")
    output["quality_flags"] = np.where(output["sleep_efficiency"].between(0, 100), "", "invalid_or_missing_sleep_efficiency")
    return output


def build_sleep_health_source(source_root: Path, spec: dict[str, Any], missing_values: Iterable[str]) -> pd.DataFrame:
    source_path = source_root / spec["path"]
    raw = read_source(source_path, missing_values)
    output = pd.DataFrame(index=raw.index)
    for original, canonical in spec["column_map"].items():
        if original in raw:
            output[canonical] = raw[original]
    output["participant_id"] = spec["participant_id_prefix"] + "_" + output["source_participant_id"].astype(str)
    output["source_dataset"] = "sleep_health_lifestyle"
    output["source_file"] = spec["path"]
    output["source_row_number"] = np.arange(2, len(output) + 2)
    output["record_status"] = "no_sleep_efficiency_target"
    output["quality_flags"] = "no_observation_date;not_joined_to_other_cohorts"
    return output


def build_fitbit_sleep_score_source(source_root: Path, spec: dict[str, Any], missing_values: Iterable[str]) -> pd.DataFrame:
    source_path = source_root / spec["path"]
    raw = read_source(source_path, missing_values)
    output = pd.DataFrame(index=raw.index)
    for original, canonical in spec["column_map"].items():
        if original in raw:
            output[canonical] = raw[original]
    output["timestamp"] = pd.to_datetime(output["timestamp"], errors="coerce", utc=True)
    output["sleep_date"] = output["timestamp"].dt.date.astype("string")
    output["participant_id"] = "fitbit_sleep_score_unknown_participant"
    output["source_dataset"] = "fitbit_sleep_score"
    output["source_file"] = spec["path"]
    output["source_row_number"] = np.arange(2, len(output) + 2)
    output["record_status"] = "wearable_benchmark_not_app_training"
    output["quality_flags"] = "participant_id_not_provided;wearable_source_excluded_from_app_model"
    return output


def longest_inactivity_minutes(timestamps: pd.Series) -> float | None:
    if len(timestamps) < 2:
        return None
    values = np.sort(timestamps.dropna().astype("int64").unique())
    if len(values) < 2:
        return None
    return round(float(np.diff(values).max() / 60), 2)


def build_student_screen_features(source_root: Path, spec: dict[str, Any], missing_values: Iterable[str]) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for source_path in sorted(source_root.glob(spec["glob"])):
        raw = read_source(source_path, missing_values)
        participant = source_participant_from_filename(source_path, spec["participant_id_pattern"])
        if not participant or spec["timestamp_column"] not in raw:
            continue
        timestamps = pd.to_datetime(raw[spec["timestamp_column"]], unit="s", errors="coerce", utc=True)
        raw = raw.assign(_timestamp=timestamps).dropna(subset=["_timestamp"])
        raw["_sleep_date"] = raw["_timestamp"].dt.date.astype("string")
        for sleep_date, group in raw.groupby("_sleep_date", dropna=True):
            seconds = group["_timestamp"].astype("int64") // 10**9
            night = group[(group["_timestamp"].dt.hour >= 22) | (group["_timestamp"].dt.hour < 7)]
            evening = group[(group["_timestamp"].dt.hour >= 18) | (group["_timestamp"].dt.hour < 8)]
            rows.append({
                "participant_id": f"student_{participant}",
                "source_participant_id": participant,
                "sleep_date": sleep_date,
                "screen_activity_events": int(len(group)),
                "nighttime_activity_bursts": int(night["_timestamp"].dt.floor("5min").nunique()),
                "longest_evening_inactivity_min": longest_inactivity_minutes(evening["_timestamp"].astype("int64") // 10**9),
                "first_screen_event_utc": group["_timestamp"].min().isoformat(),
                "last_screen_event_utc": group["_timestamp"].max().isoformat(),
                "source_dataset": "student_app_usage",
                "source_file": str(source_path.relative_to(source_root)),
                "source_row_count": int(len(group)),
                "record_status": "screen_features_only_no_sleep_target",
                "quality_flags": "timestamps_grouped_as_utc;app_events_are_not_direct_screen_on_duration",
            })
    return pd.DataFrame(rows)


def response_to_score(value: Any) -> float | None:
    key = str(value).strip().lower()
    mapping = {
        "never": 0, "almost never": 1, "sometime": 2, "sometimes": 2,
        "fairly often": 3, "very often": 4,
    }
    return mapping.get(key)


def build_student_stress(source_root: Path, spec: dict[str, Any], missing_values: Iterable[str]) -> pd.DataFrame:
    source_path = source_root / spec["path"]
    raw = read_source(source_path, missing_values)
    item_columns = [column for column in raw.columns if re.match(r"^\d+\.", str(column))]
    scores = raw[item_columns].map(response_to_score)
    # PSS reverse-scored items are 4, 5, 7 and 8 (one-indexed).
    for position in [3, 4, 6, 7]:
        if position < len(item_columns):
            scores.iloc[:, position] = scores.iloc[:, position].map(lambda value: 4 - value if pd.notna(value) else value)
    output = pd.DataFrame({
        "participant_id": "student_" + raw[spec["participant_id_column"]].astype(str),
        "source_participant_id": raw[spec["participant_id_column"]].astype(str),
        "survey_wave": raw.get("type", pd.Series(pd.NA, index=raw.index)),
        "perceived_stress_score": scores.sum(axis=1, min_count=len(item_columns)),
        "stress_items_answered": scores.notna().sum(axis=1),
        "source_dataset": "student_perceived_stress",
        "source_file": spec["path"],
        "source_row_number": np.arange(2, len(raw) + 2),
        "record_status": "survey_only_no_observation_date",
        "quality_flags": "survey_wave_not_daily;not_joined_to_other_cohorts",
    })
    return output


def build_student_psqi(source_root: Path, spec: dict[str, Any], missing_values: Iterable[str]) -> pd.DataFrame:
    source_path = source_root / spec["path"]
    raw = read_source(source_path, missing_values)
    columns = {snake_case(column): column for column in raw.columns}
    quality_column = next((original for normalized, original in columns.items() if "sleep_quality_overall" in normalized), None)
    sleep_hours_column = next((original for normalized, original in columns.items() if "hours_of_actual_sleep" in normalized), None)
    output = pd.DataFrame({
        "participant_id": "student_" + raw[spec["participant_id_column"]].astype(str),
        "source_participant_id": raw[spec["participant_id_column"]].astype(str),
        "survey_wave": raw.get("type", pd.Series(pd.NA, index=raw.index)),
        "psqi_overall_sleep_quality": raw[quality_column] if quality_column else pd.NA,
        "psqi_actual_sleep_hours_text": raw[sleep_hours_column] if sleep_hours_column else pd.NA,
        "source_dataset": "student_psqi",
        "source_file": spec["path"],
        "source_row_number": np.arange(2, len(raw) + 2),
        "record_status": "survey_only_no_observation_date",
        "quality_flags": "free_text_sleep_duration_not_coerced;survey_wave_not_daily",
    })
    return output


def add_missing_canonical_columns(frame: pd.DataFrame) -> pd.DataFrame:
    frame = frame.copy()
    for column in CANONICAL_SLEEP_COLUMNS:
        if column not in frame:
            frame[column] = pd.NA
    return frame[CANONICAL_SLEEP_COLUMNS]


def quality_report(
    sleep_records: pd.DataFrame,
    screen_features: pd.DataFrame,
    model_contract: pd.DataFrame,
    config: dict[str, Any],
) -> dict[str, Any]:
    target_rows = sleep_records[sleep_records["sleep_efficiency"].notna()]
    duplicate_keys = int(target_rows.duplicated(["participant_id", "sleep_date"], keep=False).sum())
    return {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "join_policy": "No cross-dataset participant joins were performed because attached cohorts use unrelated identifiers and collection periods.",
        "sleep_efficiency_records": int(len(sleep_records)),
        "available_sleep_efficiency_targets": int(len(target_rows)),
        "missing_sleep_efficiency_targets": int(sleep_records["sleep_efficiency"].isna().sum()),
        "duplicate_target_participant_date_rows": duplicate_keys,
        "student_screen_feature_records": int(len(screen_features)),
        "screen_participants": int(screen_features["participant_id"].nunique()) if not screen_features.empty else 0,
        "project_model_feature_coverage": {
            feature: int(model_contract[feature].notna().sum()) for feature in config["project_model_features"]
        },
        "complete_rows_for_existing_20_feature_model": int(model_contract[config["project_model_features"]].notna().all(axis=1).sum()),
        "model_readiness": "not_retrainable_without_fabrication: external cohorts do not provide all 20 project inputs and cannot be cross-joined safely",
    }


def dataframe_to_csv(frame: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(path, index=False)


def markdown_inventory(report: dict[str, Any], path: Path) -> None:
    rows = [
        "# External Dataset Inventory",
        "",
        f"Source files inspected: {report['file_count']}",
        "",
        "| Format | Files |",
        "|---|---:|",
    ]
    for kind, count in sorted(report["formats"].items()):
        rows.append(f"| {kind} | {count} |")
    rows.extend(["", "The machine-readable inventory contains columns, row counts, missing values, duplicates, date/time fields, ID candidates, and read errors for each file."])
    path.write_text("\n".join(rows) + "\n", encoding="utf-8")


def run(config_path: Path, source_root_override: Path | None = None) -> dict[str, Any]:
    config = load_config(config_path)
    source_root = (source_root_override or Path(config["source_root"])).expanduser()
    if not source_root.is_dir():
        raise FileNotFoundError(f"External dataset folder does not exist: {source_root}")

    external_raw = DATA_DIR / "external_raw"
    interim = DATA_DIR / "interim"
    inventory_dir = interim / "inventory"
    standardized = interim / "standardized"
    processed = DATA_DIR / "processed"
    reports = DATA_DIR / "reports"
    for directory in [inventory_dir, standardized, processed, reports]:
        directory.mkdir(parents=True, exist_ok=True)

    raw_reference = ensure_raw_reference(source_root, external_raw, config["raw_link_name"])
    missing_values = set(config["missing_values"])
    inventory = inventory_source(source_root, inventory_dir / "dataset_inventory.json", missing_values)
    markdown_inventory(inventory, inventory_dir / "dataset_inventory.md")

    specs = config["sources"]
    sleep_efficiency = add_missing_canonical_columns(build_sleep_efficiency_source(source_root, specs["sleep_efficiency"], missing_values))
    sleep_health = add_missing_canonical_columns(build_sleep_health_source(source_root, specs["sleep_health_lifestyle"], missing_values))
    sleep_records = pd.concat([sleep_efficiency, sleep_health], ignore_index=True)
    fitbit = build_fitbit_sleep_score_source(source_root, specs["fitbit_sleep_score"], missing_values)
    screen_features = build_student_screen_features(source_root, specs["student_app_usage"], missing_values)
    stress = build_student_stress(source_root, specs["student_stress"], missing_values)
    psqi = build_student_psqi(source_root, specs["student_psqi"], missing_values)

    dataframe_to_csv(sleep_efficiency, standardized / "sleep_efficiency_nightly.csv")
    dataframe_to_csv(sleep_health, standardized / "sleep_health_lifestyle.csv")
    dataframe_to_csv(fitbit, standardized / "fitbit_sleep_score.csv")
    dataframe_to_csv(screen_features, standardized / "student_screen_features_daily.csv")
    dataframe_to_csv(stress, standardized / "student_perceived_stress.csv")
    dataframe_to_csv(psqi, standardized / "student_psqi.csv")
    dataframe_to_csv(sleep_records, processed / "integrated_sleep_records.csv")

    # This file exposes the deployed 20-feature contract but intentionally leaves
    # unobserved values empty. It is an auditable compatibility check, not fake training data.
    model_contract = sleep_efficiency[["participant_id", "sleep_date", "sleep_efficiency", "source_dataset", "source_file", "source_row_number"]].copy()
    for feature in config["project_model_features"]:
        model_contract[feature] = pd.NA
    dataframe_to_csv(model_contract, processed / "model_contract_compatibility.csv")

    unmatched = pd.concat([
        sleep_records.loc[sleep_records["record_status"] != "target_available", ["participant_id", "source_dataset", "source_file", "record_status", "quality_flags"]],
        screen_features.loc[:, ["participant_id", "source_dataset", "source_file", "record_status", "quality_flags"]] if not screen_features.empty else pd.DataFrame(),
    ], ignore_index=True)
    dataframe_to_csv(unmatched, reports / "unmatched_keys.csv")
    dataframe_to_csv(sleep_records.loc[sleep_records["sleep_efficiency"].isna()], reports / "missing_targets.csv")
    dataframe_to_csv(sleep_records.loc[sleep_records["quality_flags"].fillna("") != ""], reports / "invalid_records.csv")

    quality = quality_report(sleep_records, screen_features, model_contract, config)
    quality["raw_data_reference"] = str(raw_reference)
    quality["outputs"] = {
        "integrated_sleep_records": str(processed / "integrated_sleep_records.csv"),
        "model_contract_compatibility": str(processed / "model_contract_compatibility.csv"),
        "inventory": str(inventory_dir / "dataset_inventory.json"),
    }
    (reports / "integration_quality_report.json").write_text(json.dumps(quality, indent=2), encoding="utf-8")
    return quality


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--source-root", type=Path, help="Override the source_root in the config file.")
    args = parser.parse_args()
    print(json.dumps(run(args.config, args.source_root), indent=2))


if __name__ == "__main__":
    main()
