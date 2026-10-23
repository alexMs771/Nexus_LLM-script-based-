from __future__ import annotations

import argparse
import json
from pathlib import Path

from app.config import settings
from app.models.ollama_model import OllamaDDLModel, OllamaUnavailable
from app.prompts import build_prompt
from app.services.validator import validate_sql
from training.dataset import read_jsonl
from training.prepare_data import PROCESSED_DIR, main as prepare_data


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate Ollama without mixing its weights or outputs into scratch training.")
    parser.add_argument("--model", default=settings.ollama_model)
    parser.add_argument("--variant", choices=["minimal", "common"], default="common")
    parser.add_argument("--out", type=Path, default=PROCESSED_DIR / "ollama_evaluation.json")
    args = parser.parse_args()
    if not (PROCESSED_DIR / "test.jsonl").exists():
        prepare_data()
    rows = read_jsonl(PROCESSED_DIR / "test.jsonl")
    model = OllamaDDLModel(settings.ollama_url, args.model, settings.ollama_timeout_seconds, args.variant)
    results = {"model": args.model, "prompt_variant": args.variant, "cases": [], "status": "completed"}
    try:
        for row in rows:
            if row["intent"] != "IN_SCOPE" or not row.get("expected_ddl"):
                continue
            prompt = build_prompt(row["request"], row.get("dialect", "postgresql"), row.get("task", "generate_ddl"), policy_variant=args.variant, intent=row.get("intent", "IN_SCOPE"))
            output = model.generate(prompt)
            validation = validate_sql(output, row.get("dialect", "postgresql"))
            results["cases"].append({"id": row["id"], "syntax_valid": validation.valid_syntax, "warnings": validation.warnings})
    except OllamaUnavailable as exc:
        results.update({"status": "blocked", "reason": str(exc)})
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(results, indent=2), encoding="utf-8")
    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
