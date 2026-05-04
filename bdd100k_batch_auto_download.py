#!/usr/bin/env python3
"""
Automated BDD100K batch downloader + filter + tracker

What it does:
1) Scrapes the BDD100K video_parts index page
2) Starts at bdd100k_videos_train_16.zip
3) Downloads each zip to DATA_DIR
4) If the zip is already there locally, skips download and goes straight to extraction
5) Extracts it
6) Keeps only .mov files whose filename stem matches a target BDD_ID
7) Moves matching videos into SAVED_FOLDER
8) Rebuilds downloaded_videos_meta.csv from the source metadata CSV
9) Stops automatically once all target videos have been found

Important changes from the old version:
- If a batch zip already exists locally, it skips download and reuses it
- No Trash logic
- Cleanup happens only after successful batch processing
- Includes zip validation before extraction
- Batch 16 will NOT be re-downloaded if the zip is missing; the script stops instead
"""

import os
import re
import shutil
import time
import zipfile
from pathlib import Path
from urllib.parse import urljoin

import pandas as pd
import requests


# =========================
# CONFIG
# =========================
BASE_URL = "http://128.32.162.150/bdd100k/video_parts/"
START_BATCH = 16

DATA_DIR = Path("/Users/mattn/Documents/VT_Undergrad/Spring_25:26/CMDA_4864_CAPSTONE/team_repo/data")
SAVED_FOLDER = DATA_DIR / "annotated_videos_only"

# This is the annotations CSV you explicitly mentioned.
ORIGINAL_ANNOTATIONS_CSV = DATA_DIR / "bdd_sce.csv"

# Optional richer metadata CSV.
# If this exists, the script uses it as the source table for downloaded_videos_meta.csv
# so you keep extra columns beyond the basic bdd_sce fields.
OPTIONAL_SOURCE_META_CSV = DATA_DIR / "meta.csv"

# Output CSV that tracks annotation rows with matching downloaded videos
OUTPUT_META_CSV = DATA_DIR / "downloaded_videos_meta.csv"

REQUEST_TIMEOUT = 60
CHUNK_SIZE = 1024 * 1024 * 16  # 16 MB
SLEEP_BETWEEN_BATCHES_SEC = 2

# Do not re-download the start batch if it is missing locally.
REQUIRE_EXISTING_ZIP_FOR_START_BATCH = False


# =========================
# HELPERS
# =========================
def log(msg: str) -> None:
    print(msg, flush=True)


def safe_mkdir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def load_source_metadata() -> pd.DataFrame:
    """
    Prefer meta.csv if it exists, because your current script appears to expect it.
    Otherwise fall back to bdd_sce.csv.
    """
    if OPTIONAL_SOURCE_META_CSV.exists():
        log(f"Using source metadata: {OPTIONAL_SOURCE_META_CSV}")
        df = pd.read_csv(OPTIONAL_SOURCE_META_CSV)
    else:
        log(f"meta.csv not found. Falling back to: {ORIGINAL_ANNOTATIONS_CSV}")
        df = pd.read_csv(ORIGINAL_ANNOTATIONS_CSV)

    if "BDD_ID" not in df.columns:
        raise ValueError("Source metadata CSV must contain a 'BDD_ID' column.")

    df["BDD_ID"] = df["BDD_ID"].astype(str)
    return df


def load_original_annotations() -> pd.DataFrame:
    df = pd.read_csv(ORIGINAL_ANNOTATIONS_CSV)
    if "BDD_ID" not in df.columns:
        raise ValueError("bdd_sce.csv must contain a 'BDD_ID' column.")
    df["BDD_ID"] = df["BDD_ID"].astype(str)
    return df


def get_saved_video_ids(saved_folder: Path) -> set[str]:
    if not saved_folder.exists():
        return set()

    ids = set()
    for p in saved_folder.iterdir():
        if p.is_file() and not p.name.startswith(".") and p.suffix.lower() == ".mov":
            ids.add(p.stem)
    return ids


def scrape_available_train_parts(base_url: str) -> list[tuple[int, str]]:
    log(f"Reading index page: {base_url}")
    resp = requests.get(base_url, timeout=REQUEST_TIMEOUT)
    resp.raise_for_status()

    # Apache-style index page; regex is enough here
    matches = re.findall(r'href="(bdd100k_videos_train_(\d+)\.zip)"', resp.text)
    parts = []
    for href, idx_str in matches:
        parts.append((int(idx_str), urljoin(base_url, href)))

    parts = sorted(set(parts), key=lambda x: x[0])
    return parts


def download_file(url: str, dest_path: Path) -> None:
    log(f"Downloading: {url}")
    with requests.get(url, stream=True, timeout=REQUEST_TIMEOUT) as r:
        r.raise_for_status()
        total = 0
        with open(dest_path, "wb") as f:
            for chunk in r.iter_content(chunk_size=CHUNK_SIZE):
                if chunk:
                    f.write(chunk)
                    total += len(chunk)
        log(f"Saved zip: {dest_path} ({total / (1024**3):.2f} GB)")


def validate_zip(zip_path: Path) -> None:
    if not zip_path.exists():
        raise FileNotFoundError(f"Zip does not exist: {zip_path}")

    with zipfile.ZipFile(zip_path, "r") as zf:
        bad_member = zf.testzip()
        if bad_member is not None:
            raise zipfile.BadZipFile(f"Corrupt member inside zip: {bad_member}")


def extract_zip(zip_path: Path, extract_root: Path) -> Path:
    log(f"Validating zip before extraction: {zip_path}")
    validate_zip(zip_path)

    log(f"Extracting: {zip_path}")
    with zipfile.ZipFile(zip_path, "r") as zf:
        zf.extractall(extract_root)
    log(f"Extracted into: {extract_root}")
    return extract_root


def find_train_folder(root: Path) -> Path:
    """
    Tries common BDD100K extraction layouts and falls back to walking the tree.
    """
    common = [
        root / "bdd100k" / "videos" / "train",
        root / "videos" / "train",
        root / "train",
    ]
    for candidate in common:
        if candidate.exists() and candidate.is_dir():
            return candidate

    # Fallback: find a folder named train containing .mov files
    for dirpath, _, filenames in os.walk(root):
        if Path(dirpath).name == "train":
            if any(name.lower().endswith(".mov") for name in filenames):
                return Path(dirpath)

    raise FileNotFoundError(f"Could not locate extracted train folder under: {root}")


def cleanup_path(path: Path) -> None:
    if not path.exists():
        return

    if path.is_dir():
        shutil.rmtree(path)
    else:
        path.unlink()


def rebuild_downloaded_meta_csv(
    source_meta_df: pd.DataFrame,
    original_annotations_df: pd.DataFrame,
    saved_ids: set[str],
    output_csv: Path,
) -> pd.DataFrame:
    """
    Rebuilds downloaded_videos_meta.csv from scratch based on what is actually in SAVED_FOLDER.
    This is safer than incrementally appending because it stays consistent if the process is restarted.
    """
    df_downloaded = source_meta_df[source_meta_df["BDD_ID"].isin(saved_ids)].copy()

    # If CONFLICT_TYPE is missing or partially missing, merge it in from bdd_sce.csv
    if {"EVENT_ID", "BDD_ID"}.issubset(df_downloaded.columns) and {"EVENT_ID", "BDD_ID", "CONFLICT_TYPE"}.issubset(original_annotations_df.columns):
        merge_cols = ["EVENT_ID", "BDD_ID", "CONFLICT_TYPE"]
        merged = df_downloaded.merge(
            original_annotations_df[merge_cols],
            on=["EVENT_ID", "BDD_ID"],
            how="left",
            suffixes=("", "_from_bdd_sce"),
        )

        if "CONFLICT_TYPE" not in merged.columns and "CONFLICT_TYPE_from_bdd_sce" in merged.columns:
            merged = merged.rename(columns={"CONFLICT_TYPE_from_bdd_sce": "CONFLICT_TYPE"})
        elif "CONFLICT_TYPE_from_bdd_sce" in merged.columns:
            merged["CONFLICT_TYPE"] = merged["CONFLICT_TYPE"].combine_first(merged["CONFLICT_TYPE_from_bdd_sce"])
            merged = merged.drop(columns=["CONFLICT_TYPE_from_bdd_sce"])

        df_downloaded = merged

    df_downloaded.to_csv(output_csv, index=False)
    return df_downloaded


def process_batch_folder(batch_folder: Path, remaining_target_ids: set[str], saved_folder: Path) -> dict:
    safe_mkdir(saved_folder)

    found_count = 0
    deleted_count = 0
    duplicate_count = 0
    non_mov_count = 0
    total_seen = 0

    log(f"Scanning batch folder: {batch_folder}")

    for entry in batch_folder.iterdir():
        if entry.name.startswith("."):
            continue

        if not entry.is_file():
            continue

        total_seen += 1

        if entry.suffix.lower() != ".mov":
            try:
                entry.unlink()
                non_mov_count += 1
            except Exception:
                pass
            continue

        video_id = entry.stem
        destination = saved_folder / entry.name

        if video_id in remaining_target_ids:
            if destination.exists():
                entry.unlink()
                duplicate_count += 1
            else:
                shutil.move(str(entry), str(destination))
                found_count += 1
        else:
            entry.unlink()
            deleted_count += 1

    return {
        "total_seen": total_seen,
        "found_count": found_count,
        "deleted_count": deleted_count,
        "duplicate_count": duplicate_count,
        "non_mov_count": non_mov_count,
    }


def main() -> None:
    safe_mkdir(DATA_DIR)
    safe_mkdir(SAVED_FOLDER)

    source_meta_df = load_source_metadata()
    original_annotations_df = load_original_annotations()

    target_ids = set(source_meta_df["BDD_ID"].dropna().astype(str))
    total_target_unique = len(target_ids)

    saved_ids = get_saved_video_ids(SAVED_FOLDER)

    # Rebuild output CSV immediately so it matches current folder contents
    df_downloaded = rebuild_downloaded_meta_csv(
        source_meta_df=source_meta_df,
        original_annotations_df=original_annotations_df,
        saved_ids=saved_ids,
        output_csv=OUTPUT_META_CSV,
    )

    log(f"Current progress: {len(saved_ids)} / {total_target_unique} unique target videos already saved.")
    log(f"Current downloaded_videos_meta rows: {len(df_downloaded)}")

    if saved_ids >= target_ids:
        log("All corresponding .mov files have already been found. Nothing to do.")
        return

    parts = scrape_available_train_parts(BASE_URL)
    parts = [(idx, url) for idx, url in parts if idx >= START_BATCH]

    if not parts:
        log(f"No train zip files found at or after batch {START_BATCH}.")
        return

    for batch_idx, zip_url in parts:
        saved_ids = get_saved_video_ids(SAVED_FOLDER)
        remaining_target_ids = target_ids - saved_ids

        if not remaining_target_ids:
            log("All corresponding .mov files have been found and stored properly. Stopping.")
            break

        zip_name = f"bdd100k_videos_train_{batch_idx:02d}.zip"
        zip_path = DATA_DIR / zip_name
        extract_root = DATA_DIR / f"extracted_{batch_idx:02d}"

        log("")
        log("=" * 70)
        log(f"Processing batch {batch_idx:02d}")
        log(f"Remaining target videos: {len(remaining_target_ids)}")
        log("=" * 70)

        if batch_idx == START_BATCH and REQUIRE_EXISTING_ZIP_FOR_START_BATCH and not zip_path.exists():
            raise FileNotFoundError(
                f"Required existing zip for start batch is missing:\n{zip_path}\n"
                f"Restore batch {batch_idx:02d} into DATA_DIR first. "
                f"This script is configured to NOT re-download the start batch."
            )

        # Added check: if the zip is already there, skip download and go straight to extraction
        if zip_path.exists():
            log(f"Zip already exists locally. Skipping download and reusing: {zip_path}")
        else:
            download_file(zip_url, zip_path)

        if extract_root.exists():
            log(f"Extraction folder already exists. Reusing: {extract_root}")
        else:
            extract_zip(zip_path, extract_root)

        batch_folder = find_train_folder(extract_root)
        stats = process_batch_folder(
            batch_folder=batch_folder,
            remaining_target_ids=remaining_target_ids,
            saved_folder=SAVED_FOLDER,
        )

        saved_ids = get_saved_video_ids(SAVED_FOLDER)
        df_downloaded = rebuild_downloaded_meta_csv(
            source_meta_df=source_meta_df,
            original_annotations_df=original_annotations_df,
            saved_ids=saved_ids,
            output_csv=OUTPUT_META_CSV,
        )

        log("Batch processing complete.")
        log(f"Total files scanned: {stats['total_seen']}")
        log(f"Saved {stats['found_count']} new videos.")
        log(f"Skipped {stats['duplicate_count']} duplicate videos.")
        log(f"Deleted {stats['deleted_count']} irrelevant videos.")
        if stats["non_mov_count"]:
            log(f"Deleted {stats['non_mov_count']} non-.mov files.")

        log(f"Tracking update: {len(saved_ids)} / {total_target_unique} unique target videos found.")
        log(f"Wrote metadata CSV: {OUTPUT_META_CSV} ({len(df_downloaded)} rows)")

        # Clean up only after successful processing
        cleanup_path(extract_root)
        cleanup_path(zip_path)
        log(f"Cleaned up local batch artifacts for batch {batch_idx:02d}")

        time.sleep(SLEEP_BETWEEN_BATCHES_SEC)

    # Final check
    saved_ids = get_saved_video_ids(SAVED_FOLDER)
    df_downloaded = rebuild_downloaded_meta_csv(
        source_meta_df=source_meta_df,
        original_annotations_df=original_annotations_df,
        saved_ids=saved_ids,
        output_csv=OUTPUT_META_CSV,
    )

    if saved_ids >= target_ids:
        log("")
        log("DONE: All corresponding .mov files have been found and stored properly.")
    else:
        missing = len(target_ids - saved_ids)
        log("")
        log(f"Finished available batches, but {missing} target videos are still missing.")
        log("This likely means they were not present in the remaining train zip files on the site.")


if __name__ == "__main__":
    main()
