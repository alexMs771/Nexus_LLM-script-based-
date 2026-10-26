"""Prepare the validated project DDL instruction corpus for model training.

Only accepted DDL examples with a real expected output are included. Scope
rejections and clarification examples stay in the policy/evaluation corpus;
they are not turned into fake SQL targets.
"""

from __future__ import annotations

from collections import Counter
import hashlib
import json

from app.config import ROOT_DIR
from training.dataset import read_jsonl


SOURCE_DIR = ROOT_DIR / "data" / "processed"
OUTPUT_DIR = SOURCE_DIR / "ddl"


def _record(row: dict, split: str, index: int) -> dict | None:
    target = str(row.get("expected_ddl") or "").strip()
    request = str(row.get("request") or row.get("instruction") or "").strip()
    intent = str(row.get("intent") or "")
    task = str(row.get("task") or "generate_ddl")
    if not request or not target or intent != "IN_SCOPE" or task == "explain_ddl":
        return None
    return {
        "record_id": hashlib.sha1(f"ddl-v1:{split}:{index}:{request}".encode()).hexdigest(),
        "source_id": "project_ddl_v2",
        "source_split": split,
        "task_type": "ddl_generation",
        "dialect": str(row.get("dialect") or "postgresql"),
        "operation": str(row.get("operation") or task),
        "intent": intent,
        "question": request,
        "evidence": "",
        "schema_context": str(row.get("schema_context") or ""),
        "target_sql": target,
        "split_group": str(row.get("schema_group") or "legacy"),
        "source": str(row.get("source") or "project_authored"),
    }


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    counts: Counter[str] = Counter()
    records: Counter[str] = Counter()
    seen: set[str] = set()
    for split in ("train", "val", "test"):
        source = SOURCE_DIR / f"{split}.jsonl"
        output = OUTPUT_DIR / f"{split}.jsonl"
        rows = []
        if source.exists():
            for index, row in enumerate(read_jsonl(source)):
                item = _record(row, split, index)
                if item is None:
                    continue
                key = " ".join(" ".join((item["question"], item["target_sql"], item["dialect"])).lower().split())
                if key in seen:
                    counts["duplicates_removed"] += 1
                    continue
                seen.add(key)
                rows.append(item)
                records[split] += 1
        output.write_text("".join(json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n" for row in rows), encoding="utf-8")
    manifest = {
        "dataset_version": "ddl-instruction-v1",
        "source_manifest": str(SOURCE_DIR / "manifest.json"),
        "records": dict(records),
        "total_records": sum(records.values()),
        "duplicates_removed": counts["duplicates_removed"],
        "policy": "Only project records with an accepted DDL intent and explicit expected DDL are trained; no SQL targets are fabricated.",
    }
    (OUTPUT_DIR / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
