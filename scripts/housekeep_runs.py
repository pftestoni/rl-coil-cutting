"""
Housekeep run folders to avoid clutter in reports/runs.

Features:
- Select root (default: reports/runs)
- Keep only the most recent N ppo_* folders in-place
- Move or zip older folders to an archive directory
- Dry-run mode to preview actions

Usage examples:
  python scripts/housekeep_runs.py --keep 8 --action move --dry-run
  python scripts/housekeep_runs.py --keep 5 --action zip

Notes:
- Only directories matching pattern 'ppo_*' are targeted. Folders like 'train' and 'deploy' are ignored.
- Sorting is by modification time (newest kept).
"""

from __future__ import annotations

import argparse
import os
import shutil
import sys
from pathlib import Path
from typing import List, Tuple


def list_run_dirs(root: Path, pattern: str = "ppo_*", recursive: bool = False) -> List[Path]:
    if recursive:
        return [p for p in root.rglob(pattern) if p.is_dir()]
    return [p for p in root.glob(pattern) if p.is_dir()]


def sort_by_mtime(paths: List[Path]) -> List[Path]:
    return sorted(paths, key=lambda p: p.stat().st_mtime, reverse=True)


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def move_dir(src: Path, dst_dir: Path, dry_run: bool) -> None:
    ensure_dir(dst_dir)
    dst = dst_dir / src.name
    if dry_run:
        print(f"[DRY-RUN] MOVE {src} -> {dst}")
        return
    if dst.exists():
        print(f"[WARN] Destination already exists, removing: {dst}")
        if dst.is_dir():
            shutil.rmtree(dst)
        else:
            dst.unlink()
    print(f"[MOVE] {src} -> {dst}")
    shutil.move(str(src), str(dst))


def zip_dir(src: Path, dst_dir: Path, dry_run: bool) -> None:
    ensure_dir(dst_dir)
    zip_base = dst_dir / src.name
    zip_path = Path(shutil.make_archive(str(zip_base), 'zip', root_dir=str(src))) if not dry_run else zip_base.with_suffix('.zip')
    if dry_run:
        print(f"[DRY-RUN] ZIP {src} -> {zip_path}")
        return
    # Remove original dir after zipping
    print(f"[ZIP] {src} -> {zip_path}")
    shutil.rmtree(src)


def human_dt(ts: float) -> str:
    from datetime import datetime
    return datetime.fromtimestamp(ts).strftime('%Y-%m-%d %H:%M:%S')


def plan_actions(
    root: Path,
    keep: int,
    pattern: str,
    exclude: Tuple[str, ...] = ("train", "deploy", "archive"),
    recursive: bool = False,
    archive_dir: Path | None = None,
) -> Tuple[List[Path], List[Path]]:
    runs = list_run_dirs(root, pattern, recursive=recursive)
    # Exclude non-ppo root markers explicitly and anything under archive_dir
    runs = [p for p in runs if p.name not in exclude]
    if archive_dir is not None:
        runs = [p for p in runs if archive_dir not in p.parents]
    runs_sorted = sort_by_mtime(runs)
    keep_list = runs_sorted[:keep]
    archive_list = runs_sorted[keep:]
    return keep_list, archive_list


def main(argv: List[str]) -> int:
    parser = argparse.ArgumentParser(description="Housekeep ppo_* runs under reports/runs")
    parser.add_argument("--root", type=str, default=str(Path("reports") / "runs"), help="Root runs directory")
    parser.add_argument("--keep", type=int, default=8, help="Number of most recent ppo_* folders to keep in-place")
    parser.add_argument("--pattern", type=str, default="ppo_*", help="Glob pattern to match run folders")
    parser.add_argument(
        "--action",
        type=str,
        choices=["move", "zip", "delete"],
        default="move",
        help="What to do with older folders: move to archive, zip to archive (and remove original), or delete",
    )
    parser.add_argument("--archive-dir", type=str, default=str(Path("reports") / "runs" / "archive"), help="Archive directory for move/zip actions")
    parser.add_argument("--recursive", action="store_true", help="Search for ppo_* folders recursively under root (excluding archive)")
    parser.add_argument("--prune-empty", action="store_true", help="Remove empty directories left behind after moves")
    parser.add_argument("--dry-run", action="store_true", help="Preview actions without making changes")

    args = parser.parse_args(argv)

    root = Path(args.root)
    if not root.exists():
        print(f"[ERROR] Root not found: {root}")
        return 2

    archive_dir = Path(args.archive_dir)
    keep_list, archive_list = plan_actions(
        root=root,
        keep=args.keep,
        pattern=args.pattern,
        recursive=args.recursive,
        archive_dir=archive_dir,
    )

    print(f"[INFO] Root: {root}")
    print(f"[INFO] Pattern: {args.pattern}")
    print(f"[INFO] Total matched: {len(keep_list) + len(archive_list)} | Keep: {len(keep_list)} | Archive: {len(archive_list)}")
    if keep_list:
        print("[KEEP]")
        for p in keep_list:
            print(f"  - {p.name} (mtime {human_dt(p.stat().st_mtime)})")
    if archive_list:
        print("[OLDER]")
        for p in archive_list:
            print(f"  - {p.name} (mtime {human_dt(p.stat().st_mtime)})")

    if not archive_list:
        print("[INFO] Nothing to archive.")
        return 0

    if args.action == "delete":
        for p in archive_list:
            if args.dry_run:
                print(f"[DRY-RUN] DELETE {p}")
            else:
                print(f"[DELETE] {p}")
                shutil.rmtree(p)
        return 0

    if args.action == "move":
        for p in archive_list:
            move_dir(p, archive_dir, args.dry_run)
        # Optionally prune empty dirs
        if args.prune_empty and not args.dry_run:
            for parent in sorted({p.parent for p in archive_list}, key=lambda x: len(str(x)), reverse=True):
                try:
                    next(parent.iterdir())
                except StopIteration:
                    # Don't remove root itself or archive_dir
                    if parent != root and parent != archive_dir:
                        try:
                            parent.rmdir()
                        except Exception:
                            pass
        return 0

    if args.action == "zip":
        for p in archive_list:
            zip_dir(p, archive_dir, args.dry_run)
        return 0

    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
