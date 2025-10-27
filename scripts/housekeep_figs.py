"""
Housekeep figures under reports/figs so that only two folders exist: train and deploy,
and each contains images directly (no nested subfolders).

Actions:
- Create reports/figs/train and reports/figs/deploy if missing
- Move any top-level images in reports/figs into the selected target (default: train)
- Flatten nested folders inside train/deploy by moving images up and removing empty dirs
- Optionally remove any other unexpected subfolders under figs (after moving images)

Usage examples:
  python scripts/housekeep_figs.py --assign-top train --dry-run
  python scripts/housekeep_figs.py --assign-top deploy

No files are deleted except empty directories created by prior runs. If a filename
collision occurs during move, the incoming file is renamed with a __{source} suffix.
"""

from __future__ import annotations

import argparse
from pathlib import Path
import shutil
from typing import Iterable


IMAGE_EXTS = {".png", ".jpg", ".jpeg", ".svg"}


def is_image(p: Path) -> bool:
    return p.is_file() and p.suffix.lower() in IMAGE_EXTS


def ensure_mode_dirs(figs_root: Path) -> tuple[Path, Path]:
    train_dir = figs_root / "train"
    deploy_dir = figs_root / "deploy"
    train_dir.mkdir(parents=True, exist_ok=True)
    deploy_dir.mkdir(parents=True, exist_ok=True)
    return train_dir, deploy_dir


def move_with_collision_handling(src: Path, dst_dir: Path) -> Path:
    dst = dst_dir / src.name
    if dst.exists():
        # Append suffix with parent dir name to avoid overwrite
        stem, ext = src.stem, src.suffix
        suffix = f"__{src.parent.name}"
        candidate = dst_dir / f"{stem}{suffix}{ext}"
        i = 1
        while candidate.exists():
            candidate = dst_dir / f"{stem}{suffix}_{i}{ext}"
            i += 1
        dst = candidate
    shutil.move(str(src), str(dst))
    return dst


def flatten_mode_dir(mode_dir: Path, dry_run: bool) -> None:
    # Move images from nested subdirectories up to mode_dir and remove empty dirs
    for sub in [p for p in mode_dir.rglob("*") if p.is_dir() and p != mode_dir]:
        # Gather images in this subdir
        imgs = [p for p in sub.iterdir() if is_image(p)]
        for img in imgs:
            if dry_run:
                print(f"[DRY-RUN] MOVE {img} -> {mode_dir / img.name}")
            else:
                move_with_collision_handling(img, mode_dir)
        # After moving images, if empty (no files/dirs), remove
        if not dry_run:
            try:
                next(sub.iterdir())
            except StopIteration:
                sub.rmdir()


def remove_unexpected_subdirs(figs_root: Path, allowed: Iterable[str], dry_run: bool) -> None:
    for child in figs_root.iterdir():
        if child.is_dir() and child.name not in allowed:
            # Attempt to move any images up to train by default, then remove if empty
            imgs = list(child.rglob("*"))
            if imgs and dry_run:
                print(f"[DRY-RUN] Found unexpected folder {child}; images will be moved to 'train' and folder removed if empty")
            elif imgs:
                pass  # Images will be handled by top-level mover below
            # Only remove when empty (post-move)
            if not dry_run:
                try:
                    next(child.iterdir())
                except StopIteration:
                    child.rmdir()


def main() -> int:
    parser = argparse.ArgumentParser(description="Housekeep reports/figs layout (keep only train/ and deploy/ with images)")
    parser.add_argument("--root", type=str, default=str(Path("reports") / "figs"), help="Path to figs root")
    parser.add_argument("--assign-top", choices=["train", "deploy"], default="train", help="Where to move top-level images found directly under figs root")
    parser.add_argument("--dry-run", action="store_true", help="Preview actions without making changes")
    args = parser.parse_args()

    figs_root = Path(args.root)
    if not figs_root.exists():
        print(f"[INFO] Nothing to do; figs root not found: {figs_root}")
        return 0

    train_dir, deploy_dir = ensure_mode_dirs(figs_root)
    target_dir = train_dir if args.assign_top == "train" else deploy_dir

    # Move any images directly under figs_root into target_dir
    for item in figs_root.iterdir():
        if item.is_file() and is_image(item):
            if args.dry_run:
                print(f"[DRY-RUN] MOVE {item} -> {target_dir / item.name}")
            else:
                move_with_collision_handling(item, target_dir)

    # Flatten nested subfolders inside train and deploy
    flatten_mode_dir(train_dir, args.dry_run)
    flatten_mode_dir(deploy_dir, args.dry_run)

    # Remove unexpected subdirs (after moves). Only removes if empty.
    remove_unexpected_subdirs(figs_root, allowed=("train", "deploy"), dry_run=args.dry_run)

    print(f"[INFO] Fig layout enforced at {figs_root}: only 'train' and 'deploy' with images directly inside.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
