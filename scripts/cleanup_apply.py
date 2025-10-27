from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
from dataclasses import dataclass
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Set, Tuple, Union


@dataclass
class Summary:
    moved: int = 0
    deleted: int = 0
    ignored: int = 0


def _now_ts() -> str:
    return datetime.now().strftime("%Y%m%d_%H%M%S")


def _ensure_dir(p: Path) -> None:
    p.mkdir(parents=True, exist_ok=True)


def _load_plan(plan_path: Path) -> Dict[str, Any]:
    if not plan_path.exists():
        raise FileNotFoundError(f"Cleanup plan not found: {plan_path}")
    with plan_path.open("r", encoding="utf-8") as f:
        return json.load(f)


def _normalize_ref_list(ref_list: Iterable[Union[str, Dict[str, Any]]]) -> Set[str]:
    out: Set[str] = set()
    for item in ref_list:
        if isinstance(item, str):
            out.add(item.replace("\\", "/"))
        elif isinstance(item, dict) and "path" in item:
            out.add(str(item["path"]).replace("\\", "/"))
    return out


def _is_protected(rel: Path, protected_roots: List[Path], protected_files: Set[Path]) -> bool:
    # Protect exact files
    if rel in protected_files:
        return True
    # Protect directories (prefix rule)
    for root in protected_roots:
        try:
            rel.relative_to(root)
            return True
        except ValueError:
            continue
    return False


def _glob_paths(root: Path, pattern: str) -> List[Path]:
    # Use glob from root; support both files and directories
    return [p for p in root.glob(pattern)]


def _literal_prefix(glob_pattern: str) -> Path:
    # Return the literal prefix of a glob pattern up to the first wildcard
    parts = Path(glob_pattern).parts
    literal_parts: List[str] = []
    for part in parts:
        if any(ch in part for ch in ("*", "?", "[")):
            break
        literal_parts.append(part)
    return Path(*literal_parts)


def _preserve_structure_dest(root: Path, match: Path, glob_pattern: str, archive_to: Path) -> Path:
    base = _literal_prefix(glob_pattern)
    # If base is empty, preserve full relative path
    rel_to_base = match
    if base.parts:
        try:
            rel_to_base = match.relative_to(base)
        except ValueError:
            # If match is not under base (edge case), fallback to full rel
            rel_to_base = match
    return archive_to / rel_to_base


def run_cleanup(
    *,
    repo_root: Path,
    plan_path: Path,
    apply: bool,
    retention_days: int,
    keep_n: int,
) -> Tuple[Summary, int, Path]:
    repo_root = repo_root.resolve()
    plan = _load_plan(plan_path)

    # Build protected sets
    default_protected_dirs = [
        Path("src"), Path("scripts"), Path("tools"), Path("configs"), Path("catalogs"),
        Path(".git"), Path(".venv"), Path(".vscode"), Path("tests"),
    ]
    default_protected_files = {Path("requirements.txt")}

    plan_keep = {Path(p) for p in plan.get("keep", [])}
    plan_ref = _normalize_ref_list(plan.get("referenced", []))
    plan_ref_paths = {Path(p) for p in plan_ref}

    protected_roots: List[Path] = default_protected_dirs + list(plan_keep)
    protected_files: Set[Path] = default_protected_files | plan_ref_paths

    # Logging setup
    reports_dir = repo_root / "reports"
    _ensure_dir(reports_dir)
    log_path = reports_dir / f"cleanup_apply_{_now_ts()}.log"
    log = []  # accumulate log lines and write once at end

    def _log(msg: str) -> None:
        ts = datetime.now().strftime("%H:%M:%S")
        line = f"[{ts}] {msg}"
        log.append(line)

    _log(f"Root: {repo_root}")
    _log(f"Plan: {plan_path}")
    _log(f"Mode: {'APPLY' if apply else 'DRY-RUN'}; retention_days={retention_days}; keep_n={keep_n}")

    summary = Summary()

    # Process archive candidates
    for item in plan.get("archiveCandidates", []):
        glob_pattern = item.get("glob") or item.get("path")
        archive_to_raw = item.get("archiveTo", "_archive")
        if not glob_pattern:
            _log("[SKIP] archive candidate missing 'glob' or 'path'")
            summary.ignored += 1
            continue
        archive_root = (repo_root / archive_to_raw).resolve()
        matches = _glob_paths(repo_root, glob_pattern)

        # Special handling for runs/train retention
        is_runs_train = Path(glob_pattern).as_posix().startswith("runs/train/") or glob_pattern.startswith("runs/train*")
        if is_runs_train and (repo_root / "runs" / "train").exists():
            runs_dir = repo_root / "runs" / "train"
            subdirs = [p for p in runs_dir.iterdir() if p.is_dir()]
            # Sort by mtime desc
            subdirs.sort(key=lambda p: p.stat().st_mtime, reverse=True)
            # Keep newest keep_n
            keep_set = set(subdirs[: max(keep_n, 0)])
            # Keep those newer than retention_days
            cutoff = datetime.now() - timedelta(days=max(retention_days, 0))
            for p in subdirs:
                if datetime.fromtimestamp(p.stat().st_mtime) >= cutoff:
                    keep_set.add(p)
            # Candidates are others
            matches = [p for p in subdirs if p not in keep_set]

        for m in matches:
            rel = m.relative_to(repo_root)
            if _is_protected(rel, protected_roots, protected_files):
                _log(f"[IGNORE] Protected from archive: {rel}")
                summary.ignored += 1
                continue

            dest = _preserve_structure_dest(repo_root, rel, glob_pattern, Path(archive_to_raw))
            dest_abs = (repo_root / dest).resolve()
            _log(f"[ARCHIVE] {rel} -> {dest}")
            if apply:
                _ensure_dir(dest_abs.parent)
                try:
                    shutil.move(str(repo_root / rel), str(dest_abs))
                    summary.moved += 1
                except Exception as e:
                    _log(f"[ERROR] Move failed for {rel}: {e}")
                    summary.ignored += 1
            else:
                # dry-run: count as ignored change (no-op)
                summary.ignored += 1

    # Process delete candidates
    for item in plan.get("deleteCandidates", []):
        action = item.get("action", "delete")
        patterns: List[str] = []
        if "path" in item:
            patterns = [item["path"]]
        elif "glob" in item:
            patterns = [item["glob"]]
        else:
            _log("[SKIP] delete candidate missing 'path' or 'glob'")
            summary.ignored += 1
            continue

        for patt in patterns:
            matches = _glob_paths(repo_root, patt)
            for m in matches:
                rel = m.relative_to(repo_root)
                if _is_protected(rel, protected_roots, protected_files):
                    _log(f"[IGNORE] Protected from delete: {rel}")
                    summary.ignored += 1
                    continue

                if action == "delete":
                    if m.is_dir():
                        _log(f"[IGNORE] Not deleting directory via 'delete': {rel}")
                        summary.ignored += 1
                        continue
                    _log(f"[DELETE] {rel}")
                    if apply:
                        try:
                            m.unlink(missing_ok=True)
                            summary.deleted += 1
                        except Exception as e:
                            _log(f"[ERROR] Delete failed for {rel}: {e}")
                            summary.ignored += 1
                    else:
                        summary.ignored += 1
                elif action == "delete-if-empty":
                    if m.is_dir():
                        try:
                            is_empty = not any(m.iterdir())
                        except FileNotFoundError:
                            is_empty = True
                        if not is_empty:
                            _log(f"[SKIP] Not empty: {rel}")
                            summary.ignored += 1
                            continue
                        _log(f"[DELETE-EMPTY] {rel}")
                        if apply:
                            try:
                                m.rmdir()
                                summary.deleted += 1
                            except Exception as e:
                                _log(f"[ERROR] rmdir failed for {rel}: {e}")
                                summary.ignored += 1
                        else:
                            summary.ignored += 1
                    else:
                        _log(f"[IGNORE] 'delete-if-empty' expects directory: {rel}")
                        summary.ignored += 1
                else:
                    _log(f"[SKIP] Unknown action '{action}' for {rel}")
                    summary.ignored += 1

    # Write log
    with log_path.open("w", encoding="utf-8") as f:
        for line in log:
            f.write(line + "\n")
        f.write(f"SUMMARY moved={summary.moved} deleted={summary.deleted} ignored={summary.ignored}\n")

    # Exit code
    if not apply:
        exit_code = 0
    else:
        exit_code = 2 if (summary.moved + summary.deleted) > 0 else 0

    return summary, exit_code, log_path


def parse_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Apply or dry-run repository cleanup plan")
    p.add_argument("--apply", action="store_true", help="Apply changes (default: dry-run)")
    p.add_argument("--retention-days", type=int, default=30, help="Keep runs newer than N days")
    p.add_argument("--keep-n", type=int, default=15, help="Keep latest N runs regardless of age")
    p.add_argument("--plan", type=str, default="reports/cleanup_dry_run.json", help="Path to cleanup plan JSON")
    p.add_argument("--root", type=str, default=".", help="Repository root (for testing)")
    return p.parse_args(argv)


def main(argv: Optional[List[str]] = None) -> int:
    ns = parse_args(argv)
    try:
        summary, code, log_path = run_cleanup(
            repo_root=Path(ns.root),
            plan_path=Path(ns.plan),
            apply=ns.apply,
            retention_days=ns.retention_days,
            keep_n=ns.keep_n,
        )
        print(f"Cleanup {'APPLIED' if ns.apply else 'DRY-RUN'} | moved={summary.moved} deleted={summary.deleted} ignored={summary.ignored} | log={log_path}")
        return code
    except Exception as e:
        print(f"ERROR: {e}", file=sys.stderr)
        return 3


if __name__ == "__main__":
    raise SystemExit(main())
