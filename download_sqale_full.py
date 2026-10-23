"""Download the full SQaLe dataset (511,630 examples across 33 Parquet shards).

Dataset: trl-lab/SQaLe-text-to-SQL-dataset
Target:  data/raw/sqale_full/
Files:   data/train-00000-of-00033.parquet ... data/train-00032-of-00033.parquet
Size:    ~3.6 GB total

Features per record:
  - schema (string DDL/schema definition)
  - question (NL query)
  - query (SELECT SQL)
  - token_count (schema, question, query, total)
  - num_joins, num_tables, number_of_columns
"""

from __future__ import annotations

import shutil
import subprocess
import sys
import time
from pathlib import Path

REPO_ID = "trl-lab/SQaLe-text-to-SQL-dataset"
BASE_URL = f"https://huggingface.co/datasets/{REPO_ID}/resolve/main/data"
TOTAL_SHARDS = 33
DEST_DIR = Path("data/raw/sqale_full")


def check_disk_space(required_gb: float = 6.0) -> bool:
    """Ensure there is enough disk space before starting download."""
    total, used, free = shutil.disk_usage(DEST_DIR.parent if DEST_DIR.parent.exists() else ".")
    free_gb = free / (1024 ** 3)
    print(f"Disk space check: {free_gb:.2f} GB free (requires >= {required_gb:.1f} GB)", flush=True)
    if free_gb < required_gb:
        print(f"ERROR: Insufficient disk space! Only {free_gb:.2f} GB free.", file=sys.stderr, flush=True)
        return False
    return True


def is_valid_parquet(path: Path) -> bool:
    """Quick check for Parquet magic bytes: starts with PAR1."""
    if not path.exists() or path.stat().st_size < 1_000_000:
        return False
    try:
        with open(path, "rb") as f:
            header = f.read(4)
            f.seek(-4, 2)
            footer = f.read(4)
            return header == b"PAR1" and footer == b"PAR1"
    except Exception:
        return False


def download_shard(shard_idx: int) -> bool:
    filename = f"train-{shard_idx:05d}-of-{TOTAL_SHARDS:05d}.parquet"
    target_path = DEST_DIR / filename
    temp_path = DEST_DIR / f"{filename}.tmp"
    url = f"{BASE_URL}/{filename}"

    if is_valid_parquet(target_path):
        size_mb = target_path.stat().st_size / (1024 * 1024)
        print(f"[{shard_idx + 1:02d}/{TOTAL_SHARDS:02d}] [SKIP] {filename} exists and is valid ({size_mb:.1f} MB)", flush=True)
        return True

    print(f"[{shard_idx + 1:02d}/{TOTAL_SHARDS:02d}] [FETCH] {filename} ...", flush=True)
    t0 = time.perf_counter()

    cmd = [
        "curl.exe",
        "-L",
        "-C", "-",
        "--retry", "3",
        "--retry-delay", "5",
        "--max-time", "600",
        "-o", str(temp_path),
        url,
    ]

    try:
        proc = subprocess.run(cmd, timeout=660)
        if proc.returncode == 0 and temp_path.exists():
            if is_valid_parquet(temp_path):
                if target_path.exists():
                    target_path.unlink()
                temp_path.rename(target_path)
                elapsed = time.perf_counter() - t0
                size_mb = target_path.stat().st_size / (1024 * 1024)
                rate = size_mb / max(elapsed, 0.1)
                print(f"[{shard_idx + 1:02d}/{TOTAL_SHARDS:02d}] [OK]   {filename} ({size_mb:.1f} MB in {elapsed:.1f}s, {rate:.1f} MB/s)", flush=True)
                return True
            else:
                print(f"[{shard_idx + 1:02d}/{TOTAL_SHARDS:02d}] [FAIL] {filename} downloaded but corrupted magic bytes", flush=True)
                if temp_path.exists():
                    temp_path.unlink()
                return False
        else:
            print(f"[{shard_idx + 1:02d}/{TOTAL_SHARDS:02d}] [FAIL] curl returned code {proc.returncode}", flush=True)
            return False
    except subprocess.TimeoutExpired:
        print(f"[{shard_idx + 1:02d}/{TOTAL_SHARDS:02d}] [FAIL] Timeout during download", flush=True)
        return False
    except Exception as exc:
        print(f"[{shard_idx + 1:02d}/{TOTAL_SHARDS:02d}] [FAIL] Error: {exc}", flush=True)
        return False


def main():
    DEST_DIR.mkdir(parents=True, exist_ok=True)

    print("=" * 70, flush=True)
    print("  SQaLe Full Dataset Downloader (33 Parquet Shards, ~3.6 GB)", flush=True)
    print("=" * 70, flush=True)

    if not check_disk_space(required_gb=6.0):
        sys.exit(1)

    t_start = time.perf_counter()
    success_count = 0
    fail_count = 0

    for i in range(TOTAL_SHARDS):
        if download_shard(i):
            success_count += 1
        else:
            fail_count += 1

    total_time = time.perf_counter() - t_start
    total_bytes = sum(f.stat().st_size for f in DEST_DIR.glob("*.parquet"))

    print("\n" + "=" * 70, flush=True)
    print(f"SQaLe Full Download Complete: {success_count}/{TOTAL_SHARDS} shards downloaded", flush=True)
    print(f"Total size: {total_bytes / (1024 * 1024):.1f} MB ({total_bytes / (1024 * 1024 * 1024):.2f} GB)", flush=True)
    print(f"Elapsed time: {total_time:.1f}s ({total_time / 60:.1f} minutes)", flush=True)
    print("=" * 70, flush=True)

    if fail_count > 0:
        print(f"WARNING: {fail_count} shards failed to download. Re-run this script to resume.", flush=True)
        sys.exit(1)


if __name__ == "__main__":
    main()
