import json
import os
from pathlib import Path
import shutil


def _write_plan(path: Path, data: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data), encoding="utf-8")


def test_dry_run_no_changes(tmp_path: Path):
    root = tmp_path
    # Create sample files
    (root / "patterns").mkdir(parents=True)
    legacy = root / "patterns" / "patterns_legacy_trim1.csv"
    legacy.write_text("a,b\n1,2\n", encoding="utf-8")

    autosave_dir = root / "reports"
    autosave_dir.mkdir(parents=True)
    autosave = autosave_dir / "tmp_autosave.csv"
    autosave.write_text("x\n", encoding="utf-8")

    plan = {
        "keep": ["patterns/patterns_20.csv", "patterns/patterns_60.csv"],
        "referenced": [{"path": "patterns/patterns_20.csv"}],
        "archiveCandidates": [
            {"glob": "patterns/patterns_*_trim*.csv", "archiveTo": "_archive/patterns/"}
        ],
        "deleteCandidates": [
            {"path": "reports/tmp_autosave.csv", "action": "delete"}
        ],
    }
    plan_path = root / "reports" / "cleanup_dry_run.json"
    _write_plan(plan_path, plan)

    # Import and run
    import importlib.util
    spec = importlib.util.spec_from_file_location("cleanup_apply", str(Path.cwd() / "scripts" / "cleanup_apply.py"))
    mod = importlib.util.module_from_spec(spec)
    assert spec and spec.loader
    import sys as _sys
    _sys.modules["cleanup_apply"] = mod  # ensure dataclasses can resolve module during exec
    spec.loader.exec_module(mod)  # type: ignore

    summary, code, log_path = mod.run_cleanup(
        repo_root=root, plan_path=plan_path, apply=False, retention_days=30, keep_n=15
    )
    assert code == 0
    assert legacy.exists()
    assert autosave.exists()
    assert summary.moved == 0
    assert summary.deleted == 0
    assert log_path.exists()


def test_apply_moves_and_delete_if_empty(tmp_path: Path):
    root = tmp_path
    # Create sample legacy pattern file
    (root / "patterns").mkdir(parents=True)
    legacy = root / "patterns" / "patterns_legacy_trim1.csv"
    legacy.write_text("a,b\n1,2\n", encoding="utf-8")

    # Create empty reports/runs dir
    empty_runs = root / "reports" / "runs"
    empty_runs.mkdir(parents=True)

    plan = {
        "keep": ["patterns/patterns_20.csv", "patterns/patterns_60.csv"],
        "referenced": [{"path": "patterns/patterns_20.csv"}],
        "archiveCandidates": [
            {"glob": "patterns/patterns_*_trim*.csv", "archiveTo": "_archive/patterns/"}
        ],
        "deleteCandidates": [
            {"path": "reports/runs", "action": "delete-if-empty"}
        ],
    }
    plan_path = root / "reports" / "cleanup_dry_run.json"
    _write_plan(plan_path, plan)

    # Import and run
    import importlib.util
    spec = importlib.util.spec_from_file_location("cleanup_apply", str(Path.cwd() / "scripts" / "cleanup_apply.py"))
    mod = importlib.util.module_from_spec(spec)
    assert spec and spec.loader
    import sys as _sys
    _sys.modules["cleanup_apply"] = mod  # ensure dataclasses can resolve module during exec
    spec.loader.exec_module(mod)  # type: ignore

    summary, code, log_path = mod.run_cleanup(
        repo_root=root, plan_path=plan_path, apply=True, retention_days=30, keep_n=15
    )
    assert code in (0, 2)
    # File moved
    assert not legacy.exists()
    archived = root / "_archive" / "patterns" / "patterns_legacy_trim1.csv"
    assert archived.exists()
    # Empty dir deleted
    assert not empty_runs.exists()
    # Some changes recorded
    assert (summary.moved + summary.deleted) >= 1
    assert log_path.exists()
