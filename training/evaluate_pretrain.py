"""Evaluate English/domain denoising reconstruction on a held-out JSONL split."""

from __future__ import annotations

import argparse
from collections import defaultdict
import difflib
import json
from pathlib import Path
import re
import time

from app.models.encoder_decoder_model import EncoderDecoderDDLModel


def task_from_prompt(prompt: str) -> str:
    match = re.search(r"<\|task\|>\s*\n([^\n]+)", prompt, flags=re.I)
    return match.group(1).strip() if match else "unknown"


def normalized(value: str) -> str:
    return re.sub(r"\s+", " ", value.strip().lower())


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate encoder-decoder English/domain pretraining.")
    parser.add_argument("--model-dir", type=Path, required=True)
    parser.add_argument("--test-path", type=Path, required=True)
    parser.add_argument("--limit", type=int, default=500)
    parser.add_argument("--max-new-tokens", type=int, default=320)
    parser.add_argument("--out", type=Path, default=None)
    args = parser.parse_args()
    model = EncoderDecoderDDLModel.from_dir(args.model_dir)
    total = 0
    exact = 0
    similarity = 0.0
    by_task: dict[str, dict[str, float]] = defaultdict(lambda: {"total": 0, "exact": 0, "similarity": 0.0})
    samples: list[dict[str, str]] = []
    started = time.perf_counter()
    with args.test_path.open(encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            row = json.loads(line)
            prompt, expected = str(row.get("prompt", "")), str(row.get("answer", ""))
            if not prompt or not expected:
                continue
            generated = model.generate(prompt, max_new_tokens=args.max_new_tokens)
            score = difflib.SequenceMatcher(None, normalized(generated), normalized(expected)).ratio()
            task = task_from_prompt(prompt)
            is_exact = normalized(generated) == normalized(expected)
            total += 1
            exact += int(is_exact)
            similarity += score
            by_task[task]["total"] += 1
            by_task[task]["exact"] += int(is_exact)
            by_task[task]["similarity"] += score
            if len(samples) < 10 and score < 1.0:
                samples.append({"task": task, "expected": expected, "generated": generated})
            if args.limit and total >= args.limit:
                break
    for values in by_task.values():
        values["exact_accuracy"] = values["exact"] / max(values["total"], 1)
        values["average_similarity"] = values["similarity"] / max(values["total"], 1)
        del values["exact"]
        del values["similarity"]
    report = {
        "model_dir": str(args.model_dir),
        "model_version": model.version,
        "total": total,
        "exact_accuracy": exact / max(total, 1),
        "average_similarity": similarity / max(total, 1),
        "latency_ms_per_example": (time.perf_counter() - started) * 1000 / max(total, 1),
        "by_task": dict(by_task),
        "samples": samples,
    }
    output = args.out or args.model_dir / "evaluation_pretrain.json"
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(report, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
