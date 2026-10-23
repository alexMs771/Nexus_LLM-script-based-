from __future__ import annotations

import json
from pathlib import Path

from app.prompts import build_prompt
from training.dataset import read_jsonl
from training.prepare_data import PROCESSED_DIR, main as prepare_data
from training.tokenizer import train_tokenizer


def main() -> None:
    train_path = PROCESSED_DIR / "train.jsonl"
    if not train_path.exists():
        prepare_data()
    rows = read_jsonl(train_path)
    texts = [build_prompt(row["request"], row.get("dialect", "postgresql"), row.get("task", "generate_ddl"), intent=row.get("intent", "IN_SCOPE")) + (row.get("expected_ddl") or "") for row in rows]
    unseen = "Create a table named unfamiliar_customer_2049 with e_mail_address VARCHAR(255);"
    report = {}
    for kind in ("byte", "byte_bpe"):
        tokenizer = train_tokenizer(texts, kind)
        lengths = [len(tokenizer.encode(text)) for text in texts]
        encoded = tokenizer.encode(unseen)
        report[kind] = {
            "vocabulary_size": tokenizer.vocab_size,
            "mean_sequence_length": sum(lengths) / max(len(lengths), 1),
            "max_sequence_length": max(lengths),
            "unseen_round_trip": tokenizer.decode(encoded) == unseen,
            "unseen_token_count": len(encoded),
            "oov_behavior": "lossless byte fallback",
        }
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
