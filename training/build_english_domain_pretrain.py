"""Build a compact encoder-decoder English/domain pretraining corpus.

The output remains a single prompt/answer JSONL contract, but every record has
an explicit task in its prompt.  This prevents English reconstruction from
being confused with DDL generation while allowing one encoder-decoder trainer
to learn clean English, grammar repair, SQL/DBML vocabulary, and identifier
preservation before DDL fine-tuning.
"""

from __future__ import annotations

from collections import Counter
import hashlib
import json
from pathlib import Path
import re
from typing import Iterable, TextIO


ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "data" / "processed" / "high_quality_ddl_v1"
TARGET = ROOT / "data" / "processed" / "english_domain_pretrain_v1"
SPLITS = ("train", "val", "test")


def one_line(value: object) -> str:
    return re.sub(r"\s+", " ", str(value or "")).strip()


def digest(prompt: str, answer: str) -> str:
    return hashlib.sha256(f"{prompt.strip().lower()}\n{answer.strip().lower()}".encode()).hexdigest()


def chunks(text: str, maximum: int) -> Iterable[str]:
    words = text.split()
    current: list[str] = []
    length = 0
    for word in words:
        proposed = length + len(word) + (1 if current else 0)
        if current and proposed > maximum:
            yield " ".join(current)
            current = []
            length = 0
        current.append(word)
        length += len(word) + (1 if length else 0)
    if current:
        yield " ".join(current)


def stable_seed(text: str) -> int:
    return int(hashlib.sha256(text.encode("utf-8")).hexdigest()[:12], 16)


def mask_text(text: str, seed: int) -> str:
    """Create deterministic, light corruption for denoising pretraining."""
    words = text.split()
    if len(words) < 8:
        return text.lower()
    span = 1 + (seed % min(4, max(1, len(words) // 10)))
    start = (seed // 17) % max(1, len(words) - span + 1)
    masked = words[:start] + ["<mask>"] + words[start + span :]
    return " ".join(masked)


def english_prompt(noisy: str) -> str:
    return (
        "<|system|>\nRestore clean, grammatical English text. Preserve the meaning.\n"
        "<|task|>\nenglish_denoising\n<|input|>\n"
        f"{noisy}\n<|answer|>\n"
    )


def grammar_prompt(noisy: str) -> str:
    return (
        "<|system|>\nCorrect grammar and spelling only. Preserve database names, table names, "
        "column names, types, constraints, and dialect terms exactly. Do not generate SQL.\n"
        "<|task|>\ngrammar_correction\n<|input|>\n"
        f"{noisy}\n<|answer|>\n"
    )


def technical_prompt(noisy: str) -> str:
    return (
        "<|system|>\nRestore the clean technical database text. Preserve SQL, DBML syntax, "
        "identifiers, names, and punctuation exactly.\n<|task|>\n"
        "technical_text_reconstruction\n<|input|>\n"
        f"{noisy}\n<|answer|>\n"
    )


def request_from_prompt(prompt: str) -> str:
    match = re.search(r"<\|request\|>\s*\n(.+?)(?:\n<\|(?:ddl|dbml)\|>|$)", prompt, re.I | re.S)
    if match:
        return one_line(match.group(1))
    match = re.search(r"User request:\s*(.+)$", prompt, re.I | re.S)
    return one_line(match.group(1)) if match else one_line(prompt)


def load_rows(path: Path) -> Iterable[dict]:
    if not path.exists():
        return
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                yield json.loads(line)


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(description="Build merged English and SQL/DBML encoder-decoder pretraining data.")
    parser.add_argument("--english-train-limit", type=int, default=350000)
    parser.add_argument("--max-text-chars", type=int, default=700)
    args = parser.parse_args()
    if TARGET.exists():
        raise RuntimeError(f"Refusing to overwrite existing target: {TARGET}")
    if not SOURCE.exists():
        raise FileNotFoundError(SOURCE)
    TARGET.mkdir(parents=True)
    counts: Counter[str] = Counter()
    seen: set[str] = set()
    handles: dict[str, TextIO] = {}
    try:
        for split in SPLITS:
            path = TARGET / f"{split}.jsonl"
            handles[split] = path.open("w", encoding="utf-8")

        def add(split: str, prompt: str, answer: str, source: str) -> None:
            prompt, answer = prompt.strip(), answer.strip()
            if not prompt or not answer:
                counts["rejected_empty"] += 1
                return
            key = digest(prompt, answer)
            if key in seen:
                counts["duplicates_removed"] += 1
                return
            seen.add(key)
            handles[split].write(json.dumps({"prompt": prompt, "answer": answer}, ensure_ascii=False, separators=(",", ":")) + "\n")
            counts[f"{source}_{split}"] += 1

        # Evaluation English is retained completely. Training English is
        # deterministically reduced to keep the merged package Colab-sized.
        for split in ("test", "val", "train"):
            source_file = SOURCE / "english" / f"{split}.jsonl"
            if not source_file.exists():
                continue
            source_rows = sum(1 for _ in load_rows(source_file))
            stride = max(1, (source_rows + args.english_train_limit - 1) // args.english_train_limit) if split == "train" else 1
            for row_index, row in enumerate(load_rows(source_file)):
                if split == "train" and row_index % stride != 0:
                    continue
                text = one_line(row.get("text"))
                for chunk_index, clean in enumerate(chunks(text, args.max_text_chars)):
                    noisy = mask_text(clean, stable_seed(f"english:{split}:{row_index}:{chunk_index}:{clean}"))
                    add(split, english_prompt(noisy), clean, "english_denoising")

        # Grammar correction examples use real DDL user requests as targets,
        # preserving technical naming while exposing noisy natural language.
        for split in ("test", "val", "train"):
            for row in load_rows(SOURCE / "supervised" / "ddl_sql" / f"{split}.jsonl"):
                clean_request = request_from_prompt(str(row.get("prompt", "")))
                if clean_request:
                    noisy = mask_text(clean_request, stable_seed(f"grammar:{split}:{clean_request}"))
                    add(split, grammar_prompt(noisy), clean_request, "grammar_correction")

        # SQL, DBML, schemas, and database documentation are reconstructed as
        # technical text so identifiers and syntax are learned before DDL SFT.
        technical_sources: list[tuple[Path, str]] = []
        for split in SPLITS:
            technical_sources.extend([
                (SOURCE / "supervised" / "ddl_sql" / f"{split}.jsonl", "sql_technical"),
                (SOURCE / "supervised" / "dbml" / f"{split}.jsonl", "dbml_technical"),
            ])
        technical_sources.append((SOURCE / "dbml" / "documents.jsonl", "dbml_documentation"))
        for file_path in sorted((SOURCE / "sql_support").glob("*.jsonl")):
            technical_sources.append((file_path, "sql_support"))

        for file_path, source_name in technical_sources:
            filename_match = re.search(r"_(train|val|test)\.jsonl$", file_path.name)
            split = filename_match.group(1) if filename_match else "train"
            if file_path.name == "documents.jsonl":
                for row_index, row in enumerate(load_rows(file_path)):
                    clean = one_line(row.get("text"))
                    for chunk_index, piece in enumerate(chunks(clean, args.max_text_chars)):
                        noisy = mask_text(piece, stable_seed(f"docs:{row_index}:{chunk_index}:{piece}"))
                        add(split, technical_prompt(noisy), piece, source_name)
                continue
            for row_index, row in enumerate(load_rows(file_path)):
                prompt = one_line(row.get("prompt"))
                answer = one_line(row.get("answer"))
                clean = f"{prompt}\nAnswer:\n{answer}" if prompt and answer else (prompt or answer)
                for chunk_index, piece in enumerate(chunks(clean, args.max_text_chars)):
                    noisy = mask_text(piece, stable_seed(f"technical:{file_path.name}:{row_index}:{chunk_index}:{piece}"))
                    add(split, technical_prompt(noisy), piece, source_name)
    finally:
        for handle in handles.values():
            handle.close()

    manifest = {
        "dataset_version": "english-domain-pretrain-v1",
        "purpose": "Merged encoder-decoder pretraining corpus for clean English, grammar correction, SQL/DBML terminology, identifier preservation, and technical database text.",
        "format": {"fields": ["prompt", "answer"], "objective": "denoising_reconstruction"},
        "recommended_architecture": "custom_encoder_decoder_transformer",
        "source_dataset": "data/processed/high_quality_ddl_v1",
        "english_train_policy": f"Deterministic stride selection capped at {args.english_train_limit} source rows; validation and test English retained completely.",
        "max_text_chars_before_tokenization": args.max_text_chars,
        "tasks": ["english_denoising", "grammar_correction", "sql_technical", "dbml_technical", "dbml_documentation", "sql_support"],
        "counts": dict(counts),
        "split_integrity": "Global prompt+answer deduplication; evaluation splits processed before training to preserve held-out records.",
        "training_order": "Pretrain this corpus, then fine-tune a copy of the encoder-decoder checkpoint on supervised/ddl_sql and separately on supervised/dbml.",
    }
    (TARGET / "manifest.json").write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(manifest, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
