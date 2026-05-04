#!/usr/bin/env python3
"""
BDD100K unseen batch downloader + extractor + manifest builder

Purpose
-------
This script is for the next capstone stage after the annotated-video collection.
Instead of keeping videos whose BDD_ID appears in the annotated metadata, this
script keeps videos that are NOT already in the annotated/training set.

Default behavior:
1) Scrape the BDD100K video_parts index page
2) Download batch 20 only by default
3) Extract the zip
4) Locate the extracted videos/train folder
5) Keep unseen .mov files whose BDD_ID has not already been used/annotated
6) Copy or move them into data/unseen_batches/batch_020/videos
7) Write a manifest CSV for model inference and manual audit
8) Optionally clean up the downloaded zip and extraction folder

Recommended first run:
    python bdd100k_pull_unseen_batch20.py --dry-run

Then real run:
    python bdd100k_pull_unseen_batch20.py

Notes
-----
- This script is intentionally conservative.
- If no seen IDs are loaded, real mode stops unless --allow-zero-seen-ids is set.
- Dry run downloads/extracts if needed so it can accurately inspect the batch,
  but it does not copy/move videos into the final unseen folder.
"""

from __future__ import annotations

import argparse
import os
import re
import shutil
import time
import zipfile
from pathlib import Path
from urllib.parse import urljoin

import pandas as pd
import requests


# -------------------------
# Defaults for ARC/project
# -------------------------
DEFAULT_BASE_URL = "http://128.32.162.150/bdd100k/video_parts/"
DEFAULT_BATCH = 20
DEFAULT_DATA_DIR = Path("data")
DEFAULT_OUT_DIR = Path("data/unseen_batches/batch_020")
DEFAULT_SEEN_META = [
    Path("downloaded_videos_meta.csv"),
    Path("meta.csv"),
    Path("data/downloaded_videos_meta.csv"),
    Path("data/meta.csv"),
    Path("data/bdd_sce.csv"),
]
DEFAULT_SEEN_VIDEO_DIR = Path("data/annotated_videos_only")

REQUEST_TIMEOUT = 60
CHUNK_SIZE = 1024 * 1024 * 16  # 16 MB
VIDEO_EXTENSIONS = {".mov", ".mp4", ".mkv"}


def log(msg: str) -> None:
    print(msg, flush=True)


def safe_mkdir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def normalize_id(value: object) -> str:
    return str(value).strip()


def scrape_available_train_parts(base_url: str) -> dict[int, str]:
    """Return {batch_number: zip_url} from the BDD100K video_parts index page."""
    log(f"Reading index page: {base_url}")
    resp = requests.get(base_url, timeout=REQUEST_TIMEOUT)
    resp.raise_for_status()

    matches = re.findall(r'href="(bdd100k_videos_train_(\d+)\.zip)"', resp.text)
    parts: dict[int, str] = {}
    for href, idx_str in matches:
        parts[int(idx_str)] = urljoin(base_url, href)

    if not parts:
        raise RuntimeError(f"No BDD100K train zip links found at: {base_url}")

    return dict(sorted(parts.items()))


def download_file(url: str, dest_path: Path) -> None:
    safe_mkdir(dest_path.parent)
    tmp_path = dest_path.with_suffix(dest_path.suffix + ".part")

    log(f"Downloading: {url}")
    with requests.get(url, stream=True, timeout=REQUEST_TIMEOUT) as r:
        r.raise_for_status()
        total = 0
        with open(tmp_path, "wb") as f:
            for chunk in r.iter_content(chunk_size=CHUNK_SIZE):
                if chunk:
                    f.write(chunk)
                    total += len(chunk)
                    if total and total % (1024**3) < CHUNK_SIZE:
                        log(f"  downloaded ~{total / (1024**3):.2f} GB")

    tmp_path.replace(dest_path)
    log(f"Saved zip: {dest_path} ({dest_path.stat().st_size / (1024**3):.2f} GB)")


def validate_zip(zip_path: Path) -> None:
    if not zip_path.exists():
        raise FileNotFoundError(f"Zip does not exist: {zip_path}")

    log(f"Validating zip: {zip_path}")
    with zipfile.ZipFile(zip_path, "r") as zf:
        bad_member = zf.testzip()
        if bad_member is not None:
            raise zipfile.BadZipFile(f"Corrupt member inside zip: {bad_member}")


def extract_zip(zip_path: Path, extract_root: Path) -> None:
    validate_zip(zip_path)
    safe_mkdir(extract_root)
    log(f"Extracting {zip_path} into {extract_root}")
    with zipfile.ZipFile(zip_path, "r") as zf:
        zf.extractall(extract_root)


def find_train_folder(root: Path) -> Path:
    """Locate the extracted train folder containing video files."""
    common = [
        root / "bdd100k" / "videos" / "train",
        root / "videos" / "train",
        root / "train",
    ]

    for candidate in common:
        if candidate.exists() and candidate.is_dir():
            if any(p.is_file() and p.suffix.lower() in VIDEO_EXTENSIONS for p in candidate.iterdir()):
                return candidate

    for dirpath, _, filenames in os.walk(root):
        path = Path(dirpath)
        if path.name == "train" and any(Path(name).suffix.lower() in VIDEO_EXTENSIONS for name in filenames):
            return path

    # Fallback: directory with the most video files
    best_dir = None
    best_count = 0
    for dirpath, _, filenames in os.walk(root):
        count = sum(1 for name in filenames if Path(name).suffix.lower() in VIDEO_EXTENSIONS)
        if count > best_count:
            best_count = count
            best_dir = Path(dirpath)

    if best_dir is not None and best_count > 0:
        return best_dir

    raise FileNotFoundError(f"Could not locate extracted videos under: {root}")


def load_seen_ids_from_csvs(paths: list[Path]) -> set[str]:
    seen: set[str] = set()

    for csv_path in paths:
        if not csv_path.exists():
            log(f"[WARN] Seen metadata file does not exist, skipping: {csv_path}")
            continue

        try:
            df = pd.read_csv(csv_path)
        except Exception as exc:
            log(f"[WARN] Could not read {csv_path}: {exc}")
            continue

        if "BDD_ID" not in df.columns:
            log(f"[WARN] {csv_path} has no BDD_ID column, skipping.")
            continue

        ids = set(df["BDD_ID"].dropna().map(normalize_id))
        seen.update(ids)
        log(f"Loaded {len(ids)} seen IDs from {csv_path}")

    return seen


def load_seen_ids_from_video_dir(video_dir: Path) -> set[str]:
    if not video_dir.exists():
        log(f"[WARN] Seen video directory does not exist, skipping: {video_dir}")
        return set()

    ids = set()
    for p in video_dir.iterdir():
        if p.is_file() and not p.name.startswith(".") and p.suffix.lower() in VIDEO_EXTENSIONS:
            ids.add(p.stem)

    log(f"Loaded {len(ids)} seen IDs from existing videos in {video_dir}")
    return ids


def list_batch_videos(batch_folder: Path) -> list[Path]:
    videos = []
    for p in batch_folder.iterdir():
        if p.is_file() and not p.name.startswith(".") and p.suffix.lower() in VIDEO_EXTENSIONS:
            videos.append(p)
    return sorted(videos)


def build_manifest_rows(
    kept_videos: list[Path],
    out_video_dir: Path,
    batch: int,
    status: str,
) -> list[dict[str, object]]:
    rows = []
    for src in kept_videos:
        dest = out_video_dir / src.name
        rows.append(
            {
                "BATCH": batch,
                "BDD_ID": src.stem,
                "SOURCE_FILENAME": src.name,
                "VIDEO_PATH": str(dest),
                "AUTO_LABEL_STATUS": status,
                "MODEL_PRED_EVENT_TYPE": "",
                "MODEL_PRED_CONFLICT_TYPE": "",
                "MODEL_CONFIDENCE": "",
                "PROB_CONFLICT": "",
                "PROB_BUMP": "",
                "PROB_HARD_BRAKE": "",
                "PROB_NOT_SCE": "",
                "HUMAN_AUDIT_REQUIRED": "",
                "HUMAN_AUDIT_EVENT_TYPE": "",
                "HUMAN_AUDIT_CONFLICT_TYPE": "",
                "NOTES": "",
            }
        )
    return rows


def transfer_unseen_videos(
    batch_videos: list[Path],
    seen_ids: set[str],
    out_video_dir: Path,
    mode: str,
    dry_run: bool,
) -> tuple[list[Path], dict[str, int]]:
    safe_mkdir(out_video_dir)

    kept = []
    stats = {
        "total_video_files": 0,
        "already_seen_skipped": 0,
        "already_in_output_skipped": 0,
        "unseen_kept": 0,
    }

    for src in batch_videos:
        stats["total_video_files"] += 1
        video_id = src.stem

        if video_id in seen_ids:
            stats["already_seen_skipped"] += 1
            continue

        dest = out_video_dir / src.name
        kept.append(src)

        if dest.exists():
            stats["already_in_output_skipped"] += 1
            continue

        stats["unseen_kept"] += 1

        if dry_run:
            continue

        if mode == "copy":
            shutil.copy2(src, dest)
        elif mode == "move":
            shutil.move(str(src), str(dest))
        else:
            raise ValueError(f"Unsupported mode: {mode}")

    return kept, stats


def cleanup_path(path: Path) -> None:
    if not path.exists():
        return
    if path.is_dir():
        shutil.rmtree(path)
    else:
        path.unlink()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Download BDD100K batch 20 and keep unseen videos for inference/audit.")

    parser.add_argument("--base-url", type=str, default=DEFAULT_BASE_URL)
    parser.add_argument("--batch", type=int, default=DEFAULT_BATCH)
    parser.add_argument("--data-dir", type=Path, default=DEFAULT_DATA_DIR)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    parser.add_argument("--seen-meta", type=Path, nargs="*", default=DEFAULT_SEEN_META)
    parser.add_argument("--seen-video-dir", type=Path, default=DEFAULT_SEEN_VIDEO_DIR)
    parser.add_argument("--mode", choices=["copy", "move"], default="copy")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--keep-zip", action="store_true", help="Do not delete downloaded zip after processing.")
    parser.add_argument("--keep-extracted", action="store_true", help="Do not delete extracted folder after processing.")
    parser.add_argument(
        "--allow-zero-seen-ids",
        action="store_true",
        help="Allow real run even if no seen IDs are loaded. Not recommended.",
    )

    return parser.parse_args()


def main() -> None:
    args = parse_args()

    data_dir: Path = args.data_dir
    out_dir: Path = args.out_dir
    out_video_dir = out_dir / "videos"
    manifest_path = out_dir / f"batch_{args.batch:03d}_unseen_manifest.csv"
    zip_name = f"bdd100k_videos_train_{args.batch:02d}.zip"
    zip_path = data_dir / zip_name
    extract_root = data_dir / f"extracted_train_{args.batch:02d}"

    safe_mkdir(data_dir)
    safe_mkdir(out_dir)
    safe_mkdir(out_video_dir)

    seen_ids = set()
    seen_ids.update(load_seen_ids_from_csvs(args.seen_meta))
    seen_ids.update(load_seen_ids_from_video_dir(args.seen_video_dir))

    log("\n" + "=" * 72)
    log("BDD100K unseen batch pull")
    log(f"Batch:                 {args.batch:02d}")
    log(f"Base URL:              {args.base_url}")
    log(f"Zip path:              {zip_path}")
    log(f"Extract root:          {extract_root}")
    log(f"Output video dir:      {out_video_dir}")
    log(f"Manifest path:         {manifest_path}")
    log(f"Seen ID count:         {len(seen_ids)}")
    log(f"Mode:                  {args.mode}")
    log(f"Dry run:               {args.dry_run}")
    log("=" * 72 + "\n")

    if len(seen_ids) == 0 and not args.dry_run and not args.allow_zero_seen_ids:
        raise RuntimeError(
            "No seen IDs were loaded. This is unsafe because every clip would be treated as unseen.\n"
            "Check --seen-meta and --seen-video-dir paths, or use --allow-zero-seen-ids only if intentional."
        )

    parts = scrape_available_train_parts(args.base_url)
    if args.batch not in parts:
        available = ", ".join(f"{idx:02d}" for idx in sorted(parts))
        raise RuntimeError(f"Batch {args.batch:02d} not found at index. Available batches: {available}")

    zip_url = parts[args.batch]

    if zip_path.exists():
        log(f"Zip already exists locally. Reusing: {zip_path}")
    else:
        download_file(zip_url, zip_path)

    if extract_root.exists():
        log(f"Extraction folder already exists. Reusing: {extract_root}")
    else:
        extract_zip(zip_path, extract_root)

    batch_folder = find_train_folder(extract_root)
    batch_videos = list_batch_videos(batch_folder)
    log(f"Located batch video folder: {batch_folder}")
    log(f"Found {len(batch_videos)} video files in batch {args.batch:02d}")

    kept_videos, stats = transfer_unseen_videos(
        batch_videos=batch_videos,
        seen_ids=seen_ids,
        out_video_dir=out_video_dir,
        mode=args.mode,
        dry_run=args.dry_run,
    )

    rows = build_manifest_rows(
        kept_videos=kept_videos,
        out_video_dir=out_video_dir,
        batch=args.batch,
        status="DRY_RUN_UNSEEN_CANDIDATE" if args.dry_run else "UNSEEN_READY_FOR_INFERENCE",
    )
    manifest_df = pd.DataFrame(rows)

    if args.dry_run:
        dry_manifest_path = manifest_path.with_name(manifest_path.stem + "_DRY_RUN.csv")
        manifest_df.to_csv(dry_manifest_path, index=False)
        log(f"Dry-run manifest written: {dry_manifest_path}")
    else:
        manifest_df.to_csv(manifest_path, index=False)
        log(f"Manifest written: {manifest_path}")

    log("\nBatch summary")
    log("-" * 72)
    for key, value in stats.items():
        log(f"{key}: {value}")
    log(f"manifest_rows: {len(manifest_df)}")

    if args.dry_run:
        log("\nDry run complete. No videos were copied or moved into the final unseen folder.")
        log("Remove --dry-run when the counts look correct.")
    else:
        if not args.keep_extracted:
            cleanup_path(extract_root)
            log(f"Cleaned extracted folder: {extract_root}")
        if not args.keep_zip:
            cleanup_path(zip_path)
            log(f"Cleaned zip file: {zip_path}")
        log("\nDone. Batch is ready for model inference/manual audit.")

    time.sleep(0.5)


if __name__ == "__main__":
    main()
