"""Locate the local research datasets without machine-specific paths in Git.

Resolution order:
1. The ``SLEEP_DATASET_DIR`` environment variable.
2. The local, untracked link ``ml/data/external_raw/final_project_dataset``
   (created by ``ml/scripts/integrate_external_datasets.py``; ignored by Git).

The folder is expected to contain the extracted datasets as distributed
("archive (8)" for Fitabase, "dataset" for StudentLife, and the Walch
PhysioNet folder). Nothing is copied into the repository.
"""

from __future__ import annotations

import os
from pathlib import Path

ENV_VAR = "SLEEP_DATASET_DIR"
PROJECT_ROOT = Path(__file__).resolve().parents[2]
LOCAL_LINK = PROJECT_ROOT / "ml" / "data" / "external_raw" / "final_project_dataset"


def dataset_root() -> Path:
    """Configured dataset folder. It may not exist; callers check with ``require_dir``."""
    configured = os.environ.get(ENV_VAR, "").strip()
    if configured:
        return Path(configured).expanduser().resolve()
    # Resolving the link keeps file paths identical to the pre-existing
    # behaviour (e.g. in the dataset manifest of forecasting_v2).
    return LOCAL_LINK.resolve() if LOCAL_LINK.exists() else LOCAL_LINK


def require_dir(path: Path, description: str) -> Path:
    if not Path(path).is_dir():
        raise FileNotFoundError(
            f"{description} not found at {path}. Set {ENV_VAR} to the folder that contains the "
            f"research datasets, or create the local link {LOCAL_LINK.relative_to(PROJECT_ROOT)} "
            f"(see docs/dataset_integration.md)."
        )
    return Path(path)
