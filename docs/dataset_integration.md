# External Dataset Integration

## Purpose

`ml/scripts/integrate_external_datasets.py` inventories and standardizes the datasets in the configured external folder. It does not alter the originals. A symbolic link under `ml/data/external_raw/final_project_dataset` points to the supplied folder so the raw source remains visible from the project without duplicating large wearable and PSG files.

The existing application remains a lifestyle-and-screen-inactivity predictor. Wearable, heart-rate, accelerometer, and PSG sources are retained as documented benchmark sources only; they are not passed to the deployed model or API.

## Run

From the project root:

```bash
source .venv/bin/activate
python3 ml/scripts/integrate_external_datasets.py
```

Use another source folder without modifying code:

```bash
python3 ml/scripts/integrate_external_datasets.py --source-root "/path/to/new/dataset_folder"
```

## Outputs

- `ml/data/interim/inventory/dataset_inventory.json`: one entry for every file, including format, columns, row count, missing values, duplicates, date/time fields, participant-ID candidates, and archive details. Files up to 1 MB receive complete checks. Larger high-frequency exports receive a 25,000-row quality sample and complete line count; the report explicitly records this scope. Duplicate checking is exact for files up to 250,000 rows and otherwise explicitly marked as not computed.
- `ml/data/interim/standardized/`: source-specific standardized tables.
- `ml/data/processed/integrated_sleep_records.csv`: vertically integrated canonical sleep records with provenance.
- `ml/data/processed/model_contract_compatibility.csv`: the existing 20-feature model schema, with only observed values populated. Empty values remain empty; no behaviour is invented.
- `ml/data/reports/integration_quality_report.json`: coverage, target availability, duplicate-key count, and model-readiness decision.
- `ml/data/reports/unmatched_keys.csv`, `missing_targets.csv`, and `invalid_records.csv`: explicit audit outputs.

## Join Policy

The supplied sources use unrelated participant identifiers such as numeric IDs, `u00`-style student IDs, anonymous Fitbit IDs, and clinical subject IDs. They are not joined across sources. The pipeline only permits joins on a shared, source-confirmed `participant_id` and `sleep_date`; otherwise it records the source as unmatched. This prevents fabricated relationships and silent row loss.

## Adding a Dataset

1. Place the new source under the configured external folder; do not edit the original file.
2. Add a source entry and `column_map` to `ml/config/dataset_integration.json`.
3. Implement a small source-specific standardizer in `ml/scripts/integrate_external_datasets.py` when the source needs aggregation or unit conversion.
4. Rerun the command and review the generated inventory and quality report before using the output for modelling.

Map only fields whose source units and meaning are known. For example, an exercise-frequency category must not be converted to minutes, and a caffeine category must not be converted to milligrams without source documentation.
