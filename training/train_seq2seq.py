"""Train the scratch encoder-decoder DDL model on compact prompt/answer JSONL."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import random
import time
from typing import Iterator

import torch
from torch.nn import CrossEntropyLoss
from torch.utils.data import DataLoader, IterableDataset

from app.config import ROOT_DIR
from app.models.encoder_decoder import EncoderDecoderTransformer
from training.tokenizer import load_tokenizer, train_tokenizer


def set_seed(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def count_jsonl(path: Path, limit: int = 0) -> int:
    count = 0
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                count += 1
                if limit and count >= limit:
                    break
    return count


def read_tokenizer_texts(path: Path, text_limit: int, byte_limit: int) -> list[str]:
    texts: list[str] = []
    used_bytes = 0
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            row = json.loads(line)
            text = f"{str(row.get('prompt', '')).strip()}\n{str(row.get('answer', '')).strip()}"
            if not text:
                continue
            remaining = byte_limit - used_bytes
            if remaining <= 0:
                break
            encoded = text.encode("utf-8")
            if len(encoded) > remaining:
                text = encoded[:remaining].decode("utf-8", errors="ignore")
                encoded = text.encode("utf-8")
            texts.append(text)
            used_bytes += len(encoded)
            if len(texts) >= text_limit:
                break
    return texts


def _bounded(ids: list[int], maximum: int, eos_id: int) -> list[int]:
    ids = ids[:maximum]
    if not ids:
        return [eos_id]
    ids[-1] = eos_id
    return ids


class Seq2SeqDataset(IterableDataset):
    def __init__(
        self,
        path: Path,
        tokenizer,
        max_source_len: int,
        max_target_len: int,
        limit: int = 0,
        shuffle: bool = False,
        shuffle_buffer: int = 1024,
    ) -> None:
        self.path = path
        self.tokenizer = tokenizer
        self.max_source_len = max_source_len
        self.max_target_len = max_target_len
        self.limit = limit
        self.shuffle = shuffle
        self.shuffle_buffer = max(1, shuffle_buffer)

    def _encode(self, row: dict) -> tuple[list[int], list[int], list[int]] | None:
        prompt = str(row.get("prompt", "")).strip()
        answer = str(row.get("answer", "")).strip()
        if not prompt or not answer:
            return None
        source = _bounded(self.tokenizer.encode(prompt, add_special=True), self.max_source_len, self.tokenizer.eos_id)
        target = _bounded(self.tokenizer.encode(answer, add_special=True), self.max_target_len, self.tokenizer.eos_id)
        if len(target) < 2:
            return None
        return source, target[:-1], target[1:]

    def __iter__(self) -> Iterator[tuple[list[int], list[int], list[int]]]:
        if not self.shuffle:
            emitted = 0
            with self.path.open(encoding="utf-8") as handle:
                for line in handle:
                    if not line.strip():
                        continue
                    item = self._encode(json.loads(line))
                    if item is None:
                        continue
                    yield item
                    emitted += 1
                    if self.limit and emitted >= self.limit:
                        return
            return

        buffered: list[tuple[list[int], list[int], list[int]]] = []
        yielded = 0
        with self.path.open(encoding="utf-8") as handle:
            for line in handle:
                if not line.strip():
                    continue
                item = self._encode(json.loads(line))
                if item is None:
                    continue
                if len(buffered) < self.shuffle_buffer:
                    buffered.append(item)
                    continue
                index = random.randrange(len(buffered))
                yield buffered.pop(index)
                yielded += 1
                buffered.append(item)
                if self.limit and yielded >= self.limit:
                    return
        while buffered:
            index = random.randrange(len(buffered))
            yield buffered.pop(index)
            yielded += 1
            if self.limit and yielded >= self.limit:
                return


def collate(batch: list[tuple[list[int], list[int], list[int]]], pad_id: int):
    source_width = max(len(item[0]) for item in batch)
    target_width = max(len(item[1]) for item in batch)
    source = torch.full((len(batch), source_width), pad_id, dtype=torch.long)
    target_input = torch.full((len(batch), target_width), pad_id, dtype=torch.long)
    labels = torch.full((len(batch), target_width), -100, dtype=torch.long)
    for index, (source_ids, target_ids, target_labels) in enumerate(batch):
        source[index, : len(source_ids)] = torch.tensor(source_ids, dtype=torch.long)
        target_input[index, : len(target_ids)] = torch.tensor(target_ids, dtype=torch.long)
        labels[index, : len(target_labels)] = torch.tensor(target_labels, dtype=torch.long)
    source_padding = source.eq(pad_id)
    target_padding = target_input.eq(pad_id)
    return source, target_input, labels, source_padding, target_padding


def evaluate(model, loader, device, limit: int = 0) -> tuple[float, int]:
    model.eval()
    loss_fn = CrossEntropyLoss(ignore_index=-100)
    total_loss = 0.0
    tokens = 0
    batches = 0
    with torch.no_grad():
        for source, target_input, labels, source_padding, target_padding in loader:
            source = source.to(device)
            target_input = target_input.to(device)
            labels = labels.to(device)
            source_padding = source_padding.to(device)
            target_padding = target_padding.to(device)
            logits = model(source, target_input, source_padding, target_padding)
            loss = loss_fn(logits.reshape(-1, logits.size(-1)), labels.reshape(-1))
            valid = int((labels != -100).sum().item())
            total_loss += float(loss.item()) * max(valid, 1)
            tokens += valid
            batches += 1
            if limit and batches >= limit:
                break
    return total_loss / max(tokens, 1), tokens


def atomic_save(value, path: Path) -> None:
    temporary = path.with_name(path.name + ".tmp")
    torch.save(value, temporary)
    temporary.replace(path)


def main() -> None:
    parser = argparse.ArgumentParser(description="Train the own encoder-decoder DDL SLM from scratch.")
    parser.add_argument("--config", type=Path, default=ROOT_DIR / "training" / "configs" / "seq2seq.json")
    parser.add_argument("--train-path", type=Path, required=True)
    parser.add_argument("--val-path", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--max-steps", type=int, default=None)
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--val-limit", type=int, default=0)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--init-from", type=Path, default=None, help="Initialize from another encoder-decoder artifact.")
    args = parser.parse_args()

    config = json.loads(args.config.read_text(encoding="utf-8"))
    model_config = config.get("model", {})
    tokenizer_config = config.get("tokenizer", {})
    train_config = config.get("training", {})
    seed = int(train_config.get("seed", 42))
    set_seed(seed)
    train_path = args.train_path if args.train_path.is_absolute() else ROOT_DIR / args.train_path
    val_path = args.val_path if args.val_path.is_absolute() else ROOT_DIR / args.val_path
    out_dir = args.out_dir if args.out_dir.is_absolute() else ROOT_DIR / args.out_dir
    if not train_path.exists() or not val_path.exists():
        raise SystemExit(f"Missing training or validation file: {train_path} / {val_path}")

    tokenizer_path = out_dir / "tokenizer.json"
    init_dir = None
    if args.init_from:
        init_dir = args.init_from if args.init_from.is_absolute() else ROOT_DIR / args.init_from
        init_tokenizer = init_dir / "tokenizer.json"
        if not init_tokenizer.exists():
            raise SystemExit(f"Initial tokenizer not found: {init_tokenizer}")
        tokenizer = load_tokenizer(init_tokenizer)
    elif tokenizer_path.exists():
        tokenizer = load_tokenizer(tokenizer_path)
    else:
        kind = str(tokenizer_config.get("kind", "byte"))
        texts = read_tokenizer_texts(
            train_path,
            int(tokenizer_config.get("text_limit", 10000)),
            int(tokenizer_config.get("byte_limit", 750000)),
        )
        tokenizer = train_tokenizer(texts, kind=kind, vocab_size=int(tokenizer_config.get("vocab_size", 512)))

    max_source_len = int(model_config.get("max_source_len", 768))
    max_target_len = int(model_config.get("max_target_len", 320))
    model = EncoderDecoderTransformer(
        vocab_size=tokenizer.vocab_size,
        d_model=int(model_config.get("d_model", 192)),
        n_heads=int(model_config.get("n_heads", 6)),
        n_encoder_layers=int(model_config.get("n_encoder_layers", 4)),
        n_decoder_layers=int(model_config.get("n_decoder_layers", 4)),
        d_ff=int(model_config.get("d_ff", 768)),
        max_source_len=max_source_len,
        max_target_len=max_target_len,
        dropout=float(model_config.get("dropout", 0.1)),
    )
    device = torch.device(("cuda" if torch.cuda.is_available() else "cpu") if args.device == "auto" else args.device)
    model.to(device)
    initialized_from = False
    if init_dir:
        init_config_path = init_dir / "config.json"
        init_weights_path = init_dir / "best.pt"
        if not init_weights_path.exists():
            init_weights_path = init_dir / "model.pt"
        if not init_config_path.exists() or not init_weights_path.exists():
            raise SystemExit(f"Initial encoder-decoder checkpoint is incomplete: {init_dir}")
        init_config = json.loads(init_config_path.read_text(encoding="utf-8"))
        if init_config.get("architecture") != EncoderDecoderTransformer.architecture:
            raise SystemExit("--init-from must point to a custom_encoder_decoder_transformer artifact.")
        source_state = torch.load(init_weights_path, map_location=device, weights_only=True)
        model.load_state_dict(source_state, strict=True)
        initialized_from = True
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(train_config.get("learning_rate", 3e-4)),
        weight_decay=float(train_config.get("weight_decay", 0.01)),
    )
    train_count = count_jsonl(train_path, args.limit)
    val_count = count_jsonl(val_path, args.val_limit)
    if not train_count or not val_count:
        raise SystemExit("Training and validation data must both contain usable rows.")
    batch_size = int(train_config.get("batch_size", 8))
    grad_accumulation = max(1, int(train_config.get("grad_accumulation", 4)))
    epochs = int(train_config.get("epochs", 3))
    configured_steps = int(train_config.get("max_steps", 8000))
    max_steps = args.max_steps if args.max_steps is not None else configured_steps
    estimated_steps = max(1, math.ceil(train_count / batch_size / grad_accumulation))
    total_steps = max(1, min(max_steps, epochs * estimated_steps))
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lambda step: max(0.0, 1.0 - step / total_steps))
    out_dir.mkdir(parents=True, exist_ok=True)
    tokenizer.save(tokenizer_path)
    artifact_config = {
        "architecture": EncoderDecoderTransformer.architecture,
        "model_id": "own-ddl-encoder-decoder",
        "model_version": "own-ddl-encoder-decoder-v1",
        "task": "ddl",
        "d_model": int(model_config.get("d_model", 192)),
        "n_heads": int(model_config.get("n_heads", 6)),
        "n_encoder_layers": int(model_config.get("n_encoder_layers", 4)),
        "n_decoder_layers": int(model_config.get("n_decoder_layers", 4)),
        "d_ff": int(model_config.get("d_ff", 768)),
        "max_source_len": max_source_len,
        "max_target_len": max_target_len,
        "dropout": float(model_config.get("dropout", 0.1)),
        "vocab_size": tokenizer.vocab_size,
        "tokenizer_kind": tokenizer.kind,
        "seed": seed,
        "train_path": str(train_path),
        "validation_path": str(val_path),
        "initialized_from": str(init_dir) if init_dir else "",
    }
    (out_dir / "config.json").write_text(json.dumps(artifact_config, indent=2), encoding="utf-8")

    train_dataset = Seq2SeqDataset(
        train_path,
        tokenizer,
        max_source_len,
        max_target_len,
        args.limit,
        shuffle=True,
        shuffle_buffer=int(train_config.get("shuffle_buffer", 1024)),
    )
    val_dataset = Seq2SeqDataset(val_path, tokenizer, max_source_len, max_target_len, args.val_limit, shuffle=False)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, collate_fn=lambda batch: collate(batch, tokenizer.pad_id))
    val_loader = DataLoader(val_dataset, batch_size=batch_size, collate_fn=lambda batch: collate(batch, tokenizer.pad_id))
    loss_fn = CrossEntropyLoss(ignore_index=-100)
    eval_every = max(1, int(train_config.get("eval_every", 250)))
    save_every = max(1, int(train_config.get("save_every", 250)))
    best_val = float("inf")
    global_step = 0
    history: list[dict] = []
    started = time.perf_counter()
    for epoch in range(1, epochs + 1):
        model.train()
        optimizer.zero_grad(set_to_none=True)
        train_loss = 0.0
        train_tokens = 0
        pending = 0
        for source, target_input, labels, source_padding, target_padding in train_loader:
            if global_step >= max_steps:
                break
            source = source.to(device)
            target_input = target_input.to(device)
            labels = labels.to(device)
            source_padding = source_padding.to(device)
            target_padding = target_padding.to(device)
            logits = model(source, target_input, source_padding, target_padding)
            loss = loss_fn(logits.reshape(-1, logits.size(-1)), labels.reshape(-1))
            (loss / grad_accumulation).backward()
            valid = int((labels != -100).sum().item())
            train_loss += float(loss.item()) * max(valid, 1)
            train_tokens += valid
            pending += 1
            if pending >= grad_accumulation:
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad(set_to_none=True)
                pending = 0
                global_step += 1
                if global_step % eval_every == 0:
                    val_loss, val_tokens = evaluate(model, val_loader, device, args.val_limit)
                    if val_loss < best_val:
                        best_val = val_loss
                        atomic_save(model.state_dict(), out_dir / "best.pt")
                if global_step % save_every == 0:
                    atomic_save(model.state_dict(), out_dir / "checkpoint.pt")
        if pending and global_step < max_steps:
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad(set_to_none=True)
            global_step += 1
        val_loss, val_tokens = evaluate(model, val_loader, device, args.val_limit)
        if val_loss < best_val:
            best_val = val_loss
            atomic_save(model.state_dict(), out_dir / "best.pt")
        atomic_save(model.state_dict(), out_dir / "final.pt")
        atomic_save(model.state_dict(), out_dir / "checkpoint.pt")
        record = {
            "epoch": epoch,
            "step": global_step,
            "train_loss": train_loss / max(train_tokens, 1),
            "val_loss": val_loss,
            "train_tokens": train_tokens,
            "val_tokens": val_tokens,
            "learning_rate": optimizer.param_groups[0]["lr"],
        }
        history.append(record)
        print(json.dumps(record), flush=True)
        if global_step >= max_steps:
            break
    atomic_save(model.state_dict(), out_dir / "model.pt")
    metadata = {
        **artifact_config,
        "device": str(device),
        "initialized_from": str(init_dir) if init_dir else "",
        "train_examples_available": train_count,
        "validation_examples_available": val_count,
        "training_seconds": round(time.perf_counter() - started, 3),
        "parameter_count": sum(parameter.numel() for parameter in model.parameters()),
        "checkpoint_files": ["best.pt", "final.pt", "model.pt", "checkpoint.pt", "config.json", "tokenizer.json"],
    }
    (out_dir / "training_history.json").write_text(
        json.dumps({"metadata": metadata, "history": history}, indent=2), encoding="utf-8"
    )
    print(json.dumps(metadata, indent=2))


if __name__ == "__main__":
    main()
