"""Download WikiText-103 train shards via direct HTTP.

The train split has 2 shards (~150 MB each). Validation and test are
already downloaded from the previous run.
"""

from __future__ import annotations

import json
import time
import urllib.request
from pathlib import Path

RAW_DIR = Path("data/raw/wikitext103_raw")
RAW_DIR.mkdir(parents=True, exist_ok=True)

BASE = "https://huggingface.co/datasets/Salesforce/wikitext/resolve/main/wikitext-103-raw-v1"

# Correct file listing from the HF API
FILES = {
    "train-00000-of-00002": f"{BASE}/train-00000-of-00002.parquet",   # ~150 MB
    "train-00001-of-00002": f"{BASE}/train-00001-of-00002.parquet",   # ~150 MB
    "validation":           f"{BASE}/validation-00000-of-00001.parquet",
    "test":                 f"{BASE}/test-00000-of-00001.parquet",
}

TIMEOUT = 600  # 10 minutes per file (train shards are ~150 MB each)
MAX_RETRIES = 3


def download_file(name: str, url: str, dest: Path) -> bool:
    if dest.exists():
        size_mb = dest.stat().st_size / (1024 * 1024)
        print(f"  [SKIP] {name} already exists ({size_mb:.1f} MB)")
        return True

    for attempt in range(1, MAX_RETRIES + 1):
        print(f"  [{name}] Attempt {attempt}/{MAX_RETRIES} ...")
        t0 = time.perf_counter()
        try:
            req = urllib.request.Request(url, headers={"User-Agent": "sql-llm-dataset-downloader/1.0"})
            with urllib.request.urlopen(req, timeout=TIMEOUT) as resp:
                data = resp.read()
            dest.write_bytes(data)
            elapsed = time.perf_counter() - t0
            size_mb = len(data) / (1024 * 1024)
            print(f"  [{name}] OK - {size_mb:.1f} MB in {elapsed:.1f}s")
            return True
        except Exception as exc:
            elapsed = time.perf_counter() - t0
            print(f"  [{name}] Failed after {elapsed:.1f}s - {type(exc).__name__}: {exc}")
            if attempt < MAX_RETRIES:
                wait = attempt * 10
                print(f"  [{name}] Retrying in {wait}s ...")
                time.sleep(wait)
    return False


print("=" * 60)
print("  WikiText-103 Direct Download (corrected shards)")
print("=" * 60)

all_ok = True
for split_name, url in FILES.items():
    dest_file = RAW_DIR / f"{split_name}.parquet"
    if not download_file(split_name, url, dest_file):
        all_ok = False

if all_ok:
    manifest = {
        "dataset": "Salesforce/wikitext",
        "config": "wikitext-103-raw-v1",
        "format": "parquet",
        "splits": {
            "train": ["train-00000-of-00002.parquet", "train-00001-of-00002.parquet"],
            "validation": ["validation.parquet"],
            "test": ["test.parquet"],
        },
    }
    (RAW_DIR / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(f"\n  All WikiText-103 splits downloaded to {RAW_DIR}")
    print("  manifest.json written.")
else:
    print("\n  Some splits failed. Check your network and retry.")
    exit(1)
