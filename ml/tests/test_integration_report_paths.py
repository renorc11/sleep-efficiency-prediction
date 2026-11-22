"""Portable report paths and inventory-checkpoint matching (synthetic folders only)."""

import json
from importlib.util import module_from_spec, spec_from_file_location
from pathlib import Path

import pytest

MODULE_PATH = Path(__file__).resolve().parents[1] / "scripts" / "integrate_external_datasets.py"
SPEC = spec_from_file_location("dataset_integration_paths", MODULE_PATH)
integration = module_from_spec(SPEC)
assert SPEC and SPEC.loader
SPEC.loader.exec_module(integration)

PLACEHOLDER = integration.DATASET_PLACEHOLDER


# ---------- report_path ----------

def test_repository_paths_become_project_relative():
    root = integration.ROOT
    assert integration.report_path(root / "ml/data/processed/x.csv") == "ml/data/processed/x.csv"
    # The dataset link lives inside the repository and stays a repository path,
    # even though it resolves to the external folder.
    link = root / "ml/data/external_raw/final_project_dataset"
    assert integration.report_path(link, Path("/elsewhere")) == "ml/data/external_raw/final_project_dataset"


def test_dataset_paths_use_the_placeholder(tmp_path):
    dataset = tmp_path / "datasets"
    (dataset / "sub").mkdir(parents=True)
    assert integration.report_path(dataset, dataset) == PLACEHOLDER
    assert integration.report_path(dataset / "sub" / "a.csv", dataset) == f"{PLACEHOLDER}/sub/a.csv"


def test_paths_outside_repository_and_dataset(tmp_path):
    home_path = Path.home() / "some_other_folder" / "file.txt"
    assert integration.report_path(home_path, tmp_path / "datasets") == "~/some_other_folder/file.txt"
    assert integration.report_path("/opt/shared/file.txt", tmp_path) == "/opt/shared/file.txt"


# ---------- checkpoint matching ----------

def _make_dataset(root: Path, names):
    for name in names:
        path = root / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("a,b\n1,2\n", encoding="utf-8")


def _stub_profiler(monkeypatch):
    calls = []

    def fake_profile(path, root, missing_values):
        calls.append(path)
        return {"relative_path": str(path.relative_to(root)), "format": "csv"}

    monkeypatch.setattr(integration, "profile_file", fake_profile)
    return calls


def _saved_inventory(saved_root, relative_paths, complete=True):
    return {
        "generated_at_utc": "2026-01-01T00:00:00+00:00",
        "source_root": saved_root,
        "file_count": len(relative_paths),
        "profiled_file_count": len(relative_paths),
        "inventory_complete": complete,
        "files": [{"relative_path": p, "format": "csv", "marker": "saved"} for p in relative_paths],
        "formats": {"csv": len(relative_paths)},
    }


@pytest.mark.parametrize("legacy", [True, False])
def test_matching_checkpoint_is_reused_without_profiling(tmp_path, monkeypatch, legacy):
    dataset = tmp_path / "datasets"
    names = ["a.csv", "sub/b.csv"]
    _make_dataset(dataset, names)
    output = tmp_path / "inventory.json"
    saved_root = str(dataset.resolve()) if legacy else PLACEHOLDER
    saved = _saved_inventory(saved_root, names)
    output.write_text(json.dumps(saved), encoding="utf-8")
    calls = _stub_profiler(monkeypatch)

    result = integration.inventory_source(dataset, output, set())

    assert calls == []                      # no re-profiling
    assert result == saved                  # checkpoint returned unchanged
    assert json.loads(output.read_text()) == saved  # file not rewritten


def test_checkpoint_for_a_different_file_inventory_is_rejected(tmp_path, monkeypatch):
    dataset = tmp_path / "datasets"
    _make_dataset(dataset, ["a.csv", "c.csv"])
    output = tmp_path / "inventory.json"
    output.write_text(json.dumps(_saved_inventory(PLACEHOLDER, ["a.csv", "b.csv"])), encoding="utf-8")
    calls = _stub_profiler(monkeypatch)

    result = integration.inventory_source(dataset, output, set())

    assert sorted(p.name for p in calls) == ["a.csv", "c.csv"]   # profiled from scratch
    assert all("marker" not in entry for entry in result["files"])
    written = json.loads(output.read_text())
    assert written["source_root"] == PLACEHOLDER                  # portable root written
    assert str(tmp_path) not in output.read_text()


def test_checkpoint_with_unrelated_root_is_rejected(tmp_path, monkeypatch):
    dataset = tmp_path / "datasets"
    _make_dataset(dataset, ["a.csv"])
    output = tmp_path / "inventory.json"
    output.write_text(json.dumps(_saved_inventory("/some/other/root", ["a.csv"])), encoding="utf-8")
    calls = _stub_profiler(monkeypatch)
    integration.inventory_source(dataset, output, set())
    assert len(calls) == 1


def test_partial_checkpoint_resumes_only_on_matching_prefix(tmp_path, monkeypatch):
    dataset = tmp_path / "datasets"
    _make_dataset(dataset, ["a.csv", "b.csv", "c.csv"])
    output = tmp_path / "inventory.json"
    output.write_text(json.dumps(_saved_inventory(PLACEHOLDER, ["a.csv"], complete=False)), encoding="utf-8")
    calls = _stub_profiler(monkeypatch)
    result = integration.inventory_source(dataset, output, set())
    assert [p.name for p in calls] == ["b.csv", "c.csv"]          # resumed after a.csv
    assert result["files"][0]["marker"] == "saved"

    output.write_text(json.dumps(_saved_inventory(PLACEHOLDER, ["x.csv"], complete=False)), encoding="utf-8")
    calls.clear()
    integration.inventory_source(dataset, output, set())
    assert len(calls) == 3                                         # mismatched prefix: start over
