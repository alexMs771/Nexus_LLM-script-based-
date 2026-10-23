from __future__ import annotations

import json
from collections import Counter
from pathlib import Path


PAD = "<pad>"
BOS = "<bos>"
EOS = "<eos>"
UNK = "<unk>"
SPECIAL_TOKENS = [PAD, BOS, EOS, UNK]
BYTE_OFFSET = len(SPECIAL_TOKENS)


class ByteTokenizer:
    """Lossless byte-level baseline for unfamiliar SQL identifiers."""

    kind = "byte"

    def __init__(self) -> None:
        self.token_to_id = {token: index for index, token in enumerate(SPECIAL_TOKENS)}
        self.id_to_token = {index: token for token, index in self.token_to_id.items()}
        for value in range(256):
            token_id = BYTE_OFFSET + value
            self.id_to_token[token_id] = value

    @property
    def vocab_size(self) -> int:
        return BYTE_OFFSET + 256

    @property
    def pad_id(self) -> int: return 0
    @property
    def bos_id(self) -> int: return 1
    @property
    def eos_id(self) -> int: return 2
    @property
    def unk_id(self) -> int: return 3

    def encode(self, text: str, add_special: bool = True) -> list[int]:
        ids = [BYTE_OFFSET + value for value in text.encode("utf-8")]
        return [self.bos_id, *ids, self.eos_id] if add_special else ids

    def decode(self, ids: list[int], skip_special: bool = True) -> str:
        raw = bytearray()
        for token_id in ids:
            if token_id in (self.pad_id, self.bos_id, self.eos_id, self.unk_id) and skip_special:
                continue
            if BYTE_OFFSET <= token_id < BYTE_OFFSET + 256:
                raw.append(token_id - BYTE_OFFSET)
        return raw.decode("utf-8", errors="replace")

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps({"kind": self.kind}, indent=2), encoding="utf-8")

    @classmethod
    def load(cls, path: Path) -> "ByteTokenizer":
        return cls()


class SubwordTokenizer:
    """Small byte-pair tokenizer trained only on the permitted corpus.

    Byte symbols guarantee unseen identifiers remain representable. Learned
    merges reduce sequence length for recurring English and SQL substrings.
    """

    kind = "byte_bpe"

    def __init__(self, merges: list[tuple[int, int]] | None = None) -> None:
        self.merges = list(merges or [])
        self._merge_to_id = {
            pair: BYTE_OFFSET + 256 + index for index, pair in enumerate(self.merges)
        }
        self._id_to_bytes: dict[int, bytes] = {
            BYTE_OFFSET + value: bytes([value]) for value in range(256)
        }
        for pair, token_id in self._merge_to_id.items():
            self._id_to_bytes[token_id] = self._id_to_bytes[pair[0]] + self._id_to_bytes[pair[1]]

    @property
    def vocab_size(self) -> int:
        return BYTE_OFFSET + 256 + len(self.merges)

    @property
    def pad_id(self) -> int: return 0
    @property
    def bos_id(self) -> int: return 1
    @property
    def eos_id(self) -> int: return 2
    @property
    def unk_id(self) -> int: return 3

    def _apply_merges(self, symbols: list[int]) -> list[int]:
        for pair in self.merges:
            merged_id = self._merge_to_id[pair]
            output: list[int] = []
            index = 0
            while index < len(symbols):
                if index + 1 < len(symbols) and (symbols[index], symbols[index + 1]) == pair:
                    output.append(merged_id)
                    index += 2
                else:
                    output.append(symbols[index])
                    index += 1
            symbols = output
        return symbols

    def encode(self, text: str, add_special: bool = True) -> list[int]:
        symbols = [BYTE_OFFSET + value for value in text.encode("utf-8")]
        symbols = self._apply_merges(symbols)
        return [self.bos_id, *symbols, self.eos_id] if add_special else symbols

    def decode(self, ids: list[int], skip_special: bool = True) -> str:
        raw = b"".join(
            self._id_to_bytes[token_id]
            for token_id in ids
            if token_id in self._id_to_bytes
        )
        return raw.decode("utf-8", errors="replace")

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(
            json.dumps({"kind": self.kind, "merges": self.merges}, indent=2),
            encoding="utf-8",
        )

    @classmethod
    def load(cls, path: Path) -> "SubwordTokenizer":
        payload = json.loads(path.read_text(encoding="utf-8"))
        return cls(merges=[tuple(pair) for pair in payload.get("merges", [])])

    @classmethod
    def train(cls, texts: list[str], vocab_size: int = 512, min_frequency: int = 2) -> "SubwordTokenizer":
        sequences = [[BYTE_OFFSET + value for value in text.encode("utf-8")] for text in texts]
        merges: list[tuple[int, int]] = []
        max_merges = max(0, vocab_size - BYTE_OFFSET - 256)
        for _ in range(max_merges):
            counts: Counter[tuple[int, int]] = Counter()
            for sequence in sequences:
                counts.update(zip(sequence, sequence[1:]))
            if not counts:
                break
            pair, frequency = counts.most_common(1)[0]
            if frequency < min_frequency:
                break
            merges.append(pair)
            merged_id = BYTE_OFFSET + 256 + len(merges) - 1
            for index, sequence in enumerate(sequences):
                reduced: list[int] = []
                cursor = 0
                while cursor < len(sequence):
                    if cursor + 1 < len(sequence) and (sequence[cursor], sequence[cursor + 1]) == pair:
                        reduced.append(merged_id)
                        cursor += 2
                    else:
                        reduced.append(sequence[cursor])
                        cursor += 1
                sequences[index] = reduced
        return cls(merges=merges)


def train_tokenizer(texts: list[str], kind: str = "byte_bpe", vocab_size: int = 512):
    if kind == "byte":
        return ByteTokenizer()
    if kind == "byte_bpe":
        return SubwordTokenizer.train(texts, vocab_size=vocab_size)
    raise ValueError(f"Unknown tokenizer kind: {kind}")


def load_tokenizer(path: Path):
    payload = json.loads(path.read_text(encoding="utf-8"))
    if payload.get("kind") == "byte":
        return ByteTokenizer()
    return SubwordTokenizer.load(path)


# Compatibility name used by the original phase-1 code.
SQLTokenizer = SubwordTokenizer
