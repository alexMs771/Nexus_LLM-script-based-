"""Build the final Colab-ready mixture from source-aware validated corpora.

The supervised files contain only prompt/answer pairs.  DBML documentation and
English language text remain separate auxiliary corpora because they have
different training targets and must not be mixed into SQL supervision.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
SPLITS = ("train", "val", "test")
SQL_SOURCES = (
    "spider",
    "sql_create_context",
    "sqale_sample",
    "sqale_full",
    "sample_databases",
    "bird_mini_dev",
)


def digest(prompt: str, answer: str) -> str:
    value = f"{prompt.strip().lower()}\n{answer.strip().lower()}"
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def copy_text_corpus(source: Path, target: Path, field: str) -> dict[str, int]:
    target.mkdir(parents=True, exist_ok=True)
    counts: dict[str, int] = {}
    documents_file = source / "documents.jsonl"
    if documents_file.exists():
        target_file = target / "documents.jsonl"
        count = 0
        with documents_file.open(encoding="utf-8") as src, target_file.open("w", encoding="utf-8") as dst:
            for line in src:
                if not line.strip():
                    continue
                row = json.loads(line)
                value = str(row.get(field, "")).strip()
                if not value:
                    continue
                dst.write(json.dumps({field: value}, ensure_ascii=False, separators=(",", ":")) + "\n")
                count += 1
        counts["documents"] = count
        return counts
    for split in SPLITS:
        source_file = source / f"{split}.jsonl"
        if not source_file.exists():
            continue
        target_file = target / f"{split}.jsonl"
        count = 0
        with source_file.open(encoding="utf-8") as src, target_file.open("w", encoding="utf-8") as dst:
            for line in src:
                if not line.strip():
                    continue
                row = json.loads(line)
                value = str(row.get(field, "")).strip()
                if not value:
                    continue
                dst.write(json.dumps({field: value}, ensure_ascii=False, separators=(",", ":")) + "\n")
                count += 1
        counts[split] = count
    return counts


def build(dataset_root: Path) -> dict[str, Any]:
    external = dataset_root / "external"
    supervised = dataset_root / "supervised"
    auxiliary = dataset_root / "auxiliary"
    audit = dataset_root / "audit"
    supervised.mkdir(parents=True, exist_ok=True)
    auxiliary.mkdir(parents=True, exist_ok=True)
    audit.mkdir(parents=True, exist_ok=True)

    seen: set[str] = set()
    counts: dict[str, dict[str, int]] = {split: {"accepted": 0, "duplicates": 0, "invalid": 0} for split in SPLITS}
    source_counts: dict[str, dict[str, int]] = {}
    rejected_path = audit / "supervised_rejected.jsonl"

    with rejected_path.open("w", encoding="utf-8") as rejected:
        handles = {split: (supervised / f"{split}.jsonl").open("w", encoding="utf-8") for split in SPLITS}
        try:
            for source_name in SQL_SOURCES:
                source_dir = external / source_name
                if not source_dir.exists():
                    continue
                source_counts[source_name] = {}
                source_splits = ("test",) if source_name == "bird_mini_dev" else SPLITS
                for split in source_splits:
                    source_file = source_dir / f"{split}.jsonl"
                    if not source_file.exists():
                        continue
                    accepted = 0
                    with source_file.open(encoding="utf-8") as source:
                        for line_number, line in enumerate(source, 1):
                            if not line.strip():
                                continue
                            try:
                                row = json.loads(line)
                            except json.JSONDecodeError:
                                counts[split]["invalid"] += 1
                                rejected.write(json.dumps({"source": source_name, "split": split, "line": line_number, "reason": "invalid_json"}) + "\n")
                                continue
                            prompt = str(row.get("prompt", "")).strip()
                            answer = str(row.get("answer", "")).strip()
                            if not prompt or not answer:
                                counts[split]["invalid"] += 1
                                rejected.write(json.dumps({"source": source_name, "split": split, "line": line_number, "reason": "missing_prompt_or_answer"}) + "\n")
                                continue
                            key = digest(prompt, answer)
                            if key in seen:
                                counts[split]["duplicates"] += 1
                                continue
                            seen.add(key)
                            handles[split].write(json.dumps({"prompt": prompt, "answer": answer}, ensure_ascii=False, separators=(",", ":")) + "\n")
                            counts[split]["accepted"] += 1
                            accepted += 1
                    source_counts[source_name][split] = accepted
        finally:
            for handle in handles.values():
                handle.close()

    dbml_counts = copy_text_corpus(external / "database_docs", auxiliary / "dbml", "text")
    english_counts = copy_text_corpus(external / "wikitext103_raw", auxiliary / "english", "text")
    manifest = {
        "dataset_version": "colab-ready-v1",
        "purpose": "Domain-specific English-to-database SQL/DDL training with DBML and English auxiliary corpora.",
        "raw_data_untouched": True,
        "active_sources": list(SQL_SOURCES) + ["database_docs", "wikitext103_raw"],
        "supervised": {
            "fields": ["prompt", "answer"],
            "deduplication": "Global prompt+answer SHA-256 deduplication; malformed rows rejected.",
            "splits": counts,
            "source_counts": source_counts,
        },
        "auxiliary": {
            "dbml": {"field": "text", "splits": dbml_counts},
            "english": {"field": "text", "splits": english_counts},
        },
        "excluded_from_active_training": ["clinc_oos", "legacy synthetic/hardcoded processed corpora"],
        "legacy_archive": "data/old_data/processed_legacy_2026-10-26.zip",
        "training_note": "Train the supervised prompt/answer files for SQL/DDL generation. Use DBML and English as separate language-model or continued-pretraining inputs.",
    }
    (dataset_root / "manifest.json").write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-root", default="data/processed/colab_ready_v1")
    args = parser.parse_args()
    dataset_root = Path(args.dataset_root)
    if not dataset_root.is_absolute():
        dataset_root = ROOT / dataset_root
    result = build(dataset_root.resolve())
    print(json.dumps(result, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
