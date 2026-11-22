"""Isolate the knowledge-base vector store from production data during tests.

Opening a ChromaDB PersistentClient rewrites files in its directory, so any
test that loaded the real store (``ml/artifacts/chroma_db``) modified
production data. This module runs when pytest imports it, before any test
module in this directory is imported or any app lifespan starts:

1. The production store's file inventory and SHA-256 hashes are recorded.
2. The store is copied into a fresh temporary directory.
3. ``ml.scripts.search_knowledge.DATABASE_DIR`` (read by every
   ``SleepKnowledgeRetriever``, including the one built by
   ``create_knowledge_service`` at app startup) points to the copy.
4. ``chromadb.PersistentClient`` is wrapped to refuse the production path,
   so no other initialization route can reach it.
5. At session end the production inventory and hashes are compared, and the
   session fails if anything changed. The copy is always removed, also when
   tests fail.
"""

from __future__ import annotations

import hashlib
import shutil
import sys
import tempfile
from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[2]
PRODUCTION_STORE = (PROJECT_ROOT / "ml" / "artifacts" / "chroma_db").resolve()


def snapshot(directory: Path) -> dict[str, str]:
    """Relative path -> SHA-256 for every file (inventory and contents)."""
    if not directory.exists():
        return {}
    return {
        str(p.relative_to(directory)): hashlib.sha256(p.read_bytes()).hexdigest()
        for p in sorted(directory.rglob("*"))
        if p.is_file()
    }


def _is_production(path) -> bool:
    try:
        resolved = Path(path).resolve()
    except TypeError:
        return False
    return resolved == PRODUCTION_STORE or PRODUCTION_STORE in resolved.parents


_BEFORE = snapshot(PRODUCTION_STORE)
_TEMP = tempfile.TemporaryDirectory(prefix="test_chroma_")
TEST_STORE = Path(_TEMP.name) / "chroma_db"
if PRODUCTION_STORE.exists():
    shutil.copytree(PRODUCTION_STORE, TEST_STORE)
else:
    TEST_STORE.mkdir()

if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))

try:
    import chromadb

    _real_persistent_client = chromadb.PersistentClient

    def _guarded_persistent_client(path=None, *args, **kwargs):
        if path is None or _is_production(path):
            raise RuntimeError(f"Tests must not open the production vector store: {path!r}")
        return _real_persistent_client(path, *args, **kwargs)

    chromadb.PersistentClient = _guarded_persistent_client
except ImportError:  # pragma: no cover - retrieval stack not installed
    chromadb = None

try:
    import ml.scripts.search_knowledge as search_knowledge

    search_knowledge.DATABASE_DIR = TEST_STORE
except ImportError:  # pragma: no cover - same dependencies as above
    search_knowledge = None


def pytest_sessionfinish(session, exitstatus):
    after = snapshot(PRODUCTION_STORE)
    session.config._production_store_unchanged = after == _BEFORE
    if after != _BEFORE:
        changed = sorted(k for k in set(after) | set(_BEFORE) if after.get(k) != _BEFORE.get(k))
        session.config._production_store_changes = changed
        session.exitstatus = pytest.ExitCode.TESTS_FAILED


def pytest_terminal_summary(terminalreporter, exitstatus, config):
    changes = getattr(config, "_production_store_changes", None)
    if changes:
        terminalreporter.write_line(f"PRODUCTION VECTOR STORE CHANGED: {changes}", red=True)


def pytest_unconfigure(config):
    _TEMP.cleanup()


@pytest.fixture
def production_store_snapshot():
    return dict(_BEFORE)
