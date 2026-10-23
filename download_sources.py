"""Retry failed dataset downloads: WikiText-103 and SQaLe.

Fixes applied:
  - WikiText-103: increased HTTP timeout via environment variable
  - SQaLe: corrected dataset name to trl-lab/SQaLe-text-to-SQL-dataset
  - Unicode: replaced symbols with ASCII for Windows cp1252 console
"""

from __future__ import annotations

import os
import sys
import time
import traceback
from pathlib import Path

# Increase default HTTP timeout for slow connections
os.environ["HF_HUB_DOWNLOAD_TIMEOUT"] = "120"

try:
    from datasets import load_dataset, Dataset
except ImportError:
    print("ERROR: 'datasets' library is not installed.")
    print("Run:  pip install datasets")
    sys.exit(1)

RAW_DIR = Path("data/raw")
RAW_DIR.mkdir(parents=True, exist_ok=True)

results: list[tuple[str, str, str]] = []


def _download(name: str, call, save_path: str) -> None:
    dest = RAW_DIR / save_path
    if dest.exists():
        print(f"\n{'='*60}")
        print(f"  [{name}] Already exists at {dest}, skipping.")
        print(f"{'='*60}")
        results.append((name, "SKIPPED", str(dest)))
        return

    print(f"\n{'='*60}")
    print(f"  [{name}] Downloading ...")
    print(f"{'='*60}")
    t0 = time.perf_counter()
    try:
        ds = call()
        ds.save_to_disk(str(dest))
        elapsed = time.perf_counter() - t0
        detail = f"Saved to {dest} in {elapsed:.1f}s"
        print(f"  [{name}] {detail}")
        results.append((name, "OK", detail))
    except Exception as exc:
        elapsed = time.perf_counter() - t0
        detail = f"{type(exc).__name__}: {exc}"
        print(f"  [{name}] FAILED after {elapsed:.1f}s -- {detail}")
        traceback.print_exc()
        results.append((name, "FAILED", detail))


# -------------------------------------------------------------------------
# A. WikiText-103 (retry with longer timeout)
# -------------------------------------------------------------------------
_download(
    "WikiText-103",
    lambda: load_dataset("Salesforce/wikitext", "wikitext-103-raw-v1"),
    "wikitext103",
)

# -------------------------------------------------------------------------
# B-D. Already downloaded — will be skipped automatically
# -------------------------------------------------------------------------
_download(
    "SQL Create Context",
    lambda: load_dataset("b-mc2/sql-create-context"),
    "sql_create_context",
)

_download(
    "Spider",
    lambda: load_dataset("xlangai/spider"),
    "spider",
)

_download(
    "CLINC OOS",
    lambda: load_dataset("clinc/clinc_oos", "plus"),
    "clinc_oos",
)

# -------------------------------------------------------------------------
# E. SQaLe — corrected dataset name: trl-lab/SQaLe-text-to-SQL-dataset
#    Stream a 2,000-row sample to keep disk use manageable.
# -------------------------------------------------------------------------
def _download_sqale_sample():
    ds = load_dataset("trl-lab/SQaLe-text-to-SQL-dataset", split="train", streaming=True)
    rows = []
    for i, row in enumerate(ds):
        rows.append(row)
        if (i + 1) % 500 == 0:
            print(f"    ... streamed {i + 1} rows")
        if i >= 1999:
            break
    return Dataset.from_list(rows)


_download(
    "SQaLe (sample)",
    _download_sqale_sample,
    "sqale_sample",
)

# -------------------------------------------------------------------------
# Summary (ASCII-safe for Windows console)
# -------------------------------------------------------------------------
print(f"\n\n{'='*60}")
print("  DOWNLOAD SUMMARY")
print(f"{'='*60}")
for name, status, detail in results:
    icon = {"OK": "[OK]", "SKIPPED": "[--]", "FAILED": "[!!]"}.get(status, "[??]")
    print(f"  {icon} {name:25s} {status:8s}  {detail}")
print(f"{'='*60}\n")

failed = [r for r in results if r[1] == "FAILED"]
if failed:
    print(f"  {len(failed)} dataset(s) failed. See errors above.")
    print("  Common fixes:")
    print("    - Run: huggingface-cli login   (for gated datasets)")
    print("    - Check internet connectivity")
    print("    - Ensure enough free disk space (~2-5 GB total)")
    sys.exit(1)
else:
    print("  All datasets downloaded successfully!")
    sys.exit(0)
