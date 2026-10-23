from __future__ import annotations

import argparse
from pathlib import Path

from app.config import ROOT_DIR
from training.dataset import read_jsonl
from training.intent_classifier import IntentClassifier
from training.prepare_data import PROCESSED_DIR, main as prepare_data


def main() -> None:
    parser = argparse.ArgumentParser(description="Train the independent lightweight intent classifier.")
    parser.add_argument("--train", type=Path, default=PROCESSED_DIR / "train.jsonl")
    parser.add_argument("--out", type=Path, default=ROOT_DIR / "artifacts" / "intent_classifier.json")
    args = parser.parse_args()
    if not args.train.exists():
        prepare_data()
    classifier = IntentClassifier.train(read_jsonl(args.train))
    classifier.save(args.out)
    print(f"saved {args.out} with {classifier.total} examples and {len(classifier.vocabulary)} features")


if __name__ == "__main__":
    main()
