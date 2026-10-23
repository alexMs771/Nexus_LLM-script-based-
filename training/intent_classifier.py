from __future__ import annotations

"""Dependency-free lightweight intent classifier.

This is intentionally separate from the generative Transformer. Character
n-grams make the classifier tolerant of misspellings and unseen subject names;
the model learns the requested action instead of memorising a topic blacklist.
"""

from collections import Counter, defaultdict
import json
import math
from pathlib import Path
import re


INTENT_LABELS = (
    "IN_SCOPE",
    "NEEDS_CLARIFICATION",
    "UNSUPPORTED_SQL_OPERATION",
    "OUT_OF_SCOPE",
    "RESTRICTED_OR_DESTRUCTIVE",
)


def features(text: str) -> list[str]:
    normalized = " ".join(text.lower().split())
    padded = f"  {normalized}  "
    chars = [f"c:{padded[index:index + width]}" for width in (3, 4, 5) for index in range(len(padded) - width + 1)]
    words = re.findall(r"[a-z0-9_]+", normalized)
    word_features = [f"w:{word}" for word in words]
    word_features.extend(f"b:{words[index]}_{words[index + 1]}" for index in range(len(words) - 1))
    structural: list[str] = []
    if re.search(r"\b(create|generate|make|build|set\s+up|define|add|alter|modify|rename|change)\b", normalized):
        structural.append("shape:ddl_action")
    if re.search(r"\b(table|tables|schema|schemas|column|columns|index|indexes|indices|view|views|constraint|constraints|primary\s+key|foreign\s+key)\b", normalized):
        structural.append("shape:ddl_object")
    if re.search(r"\b(select|insert|update|delete|merge|join|grant|revoke|upsert|query|fetch|find|retrieve|list|show)\b", normalized):
        structural.append("shape:unsupported_operation")
    if re.search(r"\b(drop|truncate|erase|destroy|wipe)\b|\b(ignore|bypass|override)\b", normalized):
        structural.append("shape:restricted_operation")
    if re.search(r"\b(what|why|how|explain|summarize|translate|recommend|write|tell)\b", normalized):
        structural.append("shape:question_or_general_task")
    if re.fullmatch(r"(?:please\s+)?(?:create|make|build|generate|set\s+up|define|add|alter|modify|rename|remove)\s+(?:a|an|the|new)?\s*(?:database\s+)?(?:table|schema|index|view|column|constraint)s?\.?", normalized):
        structural.append("shape:generic_ddl")
    elif "shape:ddl_action" in structural and "shape:ddl_object" in structural:
        structural.append("shape:complete_ddl")
    return chars + word_features + structural


class IntentClassifier:
    def __init__(self, class_counts: dict[str, int], feature_counts: dict[str, dict[str, int]], vocabulary: set[str]) -> None:
        self.class_counts = class_counts
        self.feature_counts = feature_counts
        self.vocabulary = vocabulary
        self.total = sum(class_counts.values())

    @classmethod
    def train(cls, rows: list[dict]) -> "IntentClassifier":
        class_counts: Counter[str] = Counter()
        feature_counts: dict[str, Counter[str]] = defaultdict(Counter)
        vocabulary: set[str] = set()
        for row in rows:
            label = row.get("intent")
            if label not in INTENT_LABELS:
                continue
            class_counts[label] += 1
            row_features = set(features(str(row.get("request", ""))))
            feature_counts[label].update(row_features)
            vocabulary.update(row_features)
        return cls(dict(class_counts), {label: dict(counts) for label, counts in feature_counts.items()}, vocabulary)

    def predict(self, text: str) -> tuple[str, float, dict[str, float]]:
        if not self.total:
            return "OUT_OF_SCOPE", 0.0, {}
        row_features = set(features(text))
        scores: dict[str, float] = {}
        vocabulary_size = max(len(self.vocabulary), 1)
        for label in INTENT_LABELS:
            class_count = self.class_counts.get(label, 0)
            log_probability = math.log((class_count + 1) / (self.total + len(INTENT_LABELS)))
            denominator = sum(self.feature_counts.get(label, {}).values()) + vocabulary_size
            counts = self.feature_counts.get(label, {})
            for feature in row_features:
                weight = 12.0 if feature.startswith("shape:") else 1.0
                log_probability += weight * math.log((counts.get(feature, 0) + 1) / denominator)
            scores[label] = log_probability
        ordered = sorted(scores.items(), key=lambda item: item[1], reverse=True)
        maximum = ordered[0][1]
        probabilities = {label: math.exp(score - maximum) for label, score in scores.items()}
        normalizer = sum(probabilities.values())
        probabilities = {label: value / normalizer for label, value in probabilities.items()}
        label = ordered[0][0]
        return label, probabilities[label], probabilities

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps({"class_counts": self.class_counts, "feature_counts": self.feature_counts, "vocabulary": sorted(self.vocabulary)}, indent=2), encoding="utf-8")

    @classmethod
    def load(cls, path: Path) -> "IntentClassifier":
        payload = json.loads(path.read_text(encoding="utf-8"))
        return cls(payload["class_counts"], payload["feature_counts"], set(payload["vocabulary"]))
