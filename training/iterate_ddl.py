"""Run a repeatable train -> evaluate -> gate cycle for the own DDL SLM."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import shutil
import subprocess
import sys

from app.config import ROOT_DIR, settings


def _run(command: list[str]) -> None:
    print("$ " + " ".join(command), flush=True)
    completed = subprocess.run(command, cwd=ROOT_DIR, check=False)
    if completed.returncode:
        raise SystemExit(completed.returncode)


def main() -> None:
    parser = argparse.ArgumentParser(description="Train and evaluate one DDL model candidate.")
    parser.add_argument("--config", type=Path, default=ROOT_DIR / "training" / "configs" / "ddl_balanced.json")
    parser.add_argument("--train-path", type=Path, default=ROOT_DIR / "data" / "processed" / "ddl_balanced_v1" / "train.jsonl")
    parser.add_argument("--val-path", type=Path, default=ROOT_DIR / "data" / "processed" / "ddl_balanced_v1" / "val.jsonl")
    parser.add_argument("--test-path", type=Path, default=ROOT_DIR / "data" / "processed" / "ddl_eval_v2" / "test.jsonl")
    parser.add_argument("--out-dir", type=Path, default=ROOT_DIR / "artifacts" / "models" / "own-ddl-balanced-v1")
    parser.add_argument("--max-steps", type=int, default=8000)
    parser.add_argument("--candidates", type=int, default=3)
    parser.add_argument("--promote", action="store_true", help="Promote only if the quality gate passes.")
    parser.add_argument("--min-syntax", type=float, default=0.80)
    parser.add_argument("--min-exact", type=float, default=0.25)
    args = parser.parse_args()
    config = args.config if args.config.is_absolute() else ROOT_DIR / args.config
    train_path = args.train_path if args.train_path.is_absolute() else ROOT_DIR / args.train_path
    val_path = args.val_path if args.val_path.is_absolute() else ROOT_DIR / args.val_path
    test_path = args.test_path if args.test_path.is_absolute() else ROOT_DIR / args.test_path
    out_dir = args.out_dir if args.out_dir.is_absolute() else ROOT_DIR / args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    python = sys.executable
    _run([
        python, "-m", "training.train_lm", "--task", "ddl", "--config", str(config),
        "--train-path", str(train_path), "--val-path", str(val_path), "--out-dir", str(out_dir),
        "--max-steps", str(args.max_steps),
    ])
    report_path = out_dir / "evaluation_quality.json"
    _run([
        python, "-m", "training.evaluate_ddl", "--model-dir", str(out_dir),
        "--test-path", str(test_path), "--out", str(report_path),
        "--max-new-tokens", "320", "--candidates", str(args.candidates),
    ])
    report = json.loads(report_path.read_text(encoding="utf-8"))
    gate = {
        "syntax_accuracy": float(report.get("syntax_accuracy", 0.0)),
        "exact_match_accuracy": float(report.get("exact_match_accuracy", 0.0)),
        "min_syntax": args.min_syntax,
        "min_exact": args.min_exact,
        "passed": float(report.get("syntax_accuracy", 0.0)) >= args.min_syntax and float(report.get("exact_match_accuracy", 0.0)) >= args.min_exact,
    }
    (out_dir / "quality_gate.json").write_text(json.dumps(gate, indent=2), encoding="utf-8")
    print(json.dumps(gate, indent=2), flush=True)
    if args.promote and gate["passed"]:
        target = settings.own_ddl_model_dir
        staging = target.with_name(target.name + ".candidate")
        if staging.exists():
            shutil.rmtree(staging)
        shutil.copytree(out_dir, staging)
        if target.exists():
            backup = target.with_name(target.name + ".previous")
            if backup.exists():
                shutil.rmtree(backup)
            target.rename(backup)
        staging.rename(target)
        print(json.dumps({"promoted": True, "target": str(target)}, indent=2))
    elif args.promote:
        print(json.dumps({"promoted": False, "reason": "quality gate failed; current API model was preserved"}, indent=2))


if __name__ == "__main__":
    main()
