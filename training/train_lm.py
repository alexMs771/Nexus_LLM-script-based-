"""Train the local decoder-only Transformer from random initialization.

The default task is supervised NL-to-SQL fine-tuning over records produced by
``prepare_nl2sql``.  ``--task causal`` trains on the general-text corpus.  The
trainer is streaming and checkpointed so a large corpus is not loaded into RAM.
"""

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

from app.config import ROOT_DIR, settings
from app.models.transformer import SQLTransformer
from training.sql_format import build_ddl_prompt, build_sql_prompt
from training.tokenizer import ByteTokenizer, load_tokenizer, train_tokenizer


def set_seed(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def count_jsonl(path: Path, limit: int = 0) -> int:
    count = 0
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                count += 1
                if limit and count >= limit:
                    break
    return count


def _encode_row(row: dict, tokenizer, max_len: int, task: str) -> tuple[list[int], int] | None:
    if task in {"nl2sql", "ddl"}:
        question = str(row.get("question", "")).strip()
        target = str(row.get("target_sql", "")).strip()
        if not question or not target:
            return None
        if task == "ddl":
            prompt = build_ddl_prompt(
                requirement=question,
                dialect=str(row.get("dialect", "postgresql")),
                task=str(row.get("operation", "generate_ddl")),
                schema=str(row.get("schema_context", "")),
            )
        else:
            prompt = build_sql_prompt(
                question=question,
                schema=str(row.get("schema_context", "")),
                dialect=str(row.get("dialect", "postgresql")),
                evidence=str(row.get("evidence", "")),
            )
        prompt_ids = tokenizer.encode(prompt, add_special=False)
        target_ids = tokenizer.encode(target, add_special=False) + [tokenizer.eos_id]
        target_ids = target_ids[: max(1, max_len - 2)]
        if target_ids[-1] != tokenizer.eos_id:
            target_ids[-1] = tokenizer.eos_id
        prompt_budget = max(0, max_len - 1 - len(target_ids))
        prompt_ids = prompt_ids[:prompt_budget]
        ids = [tokenizer.bos_id, *prompt_ids, *target_ids]
        # Labels are shifted by one position, so this is the number to mask
        # from the beginning of the label tensor.
        prompt_mask = max(0, len(prompt_ids))
        return ids[:max_len], prompt_mask

    text = str(row.get("text", "")).strip()
    if not text:
        return None
    ids = tokenizer.encode(text, add_special=True)[:max_len]
    if len(ids) < 2:
        return None
    if ids[-1] != tokenizer.eos_id:
        ids[-1] = tokenizer.eos_id
    return ids, 0


class JsonlDataset(IterableDataset):
    def __init__(
        self,
        path: Path,
        tokenizer,
        max_len: int,
        task: str,
        limit: int = 0,
        shuffle: bool = False,
        shuffle_buffer: int = 1024,
    ) -> None:
        self.path = path
        self.tokenizer = tokenizer
        self.max_len = max_len
        self.task = task
        self.limit = limit
        self.shuffle = shuffle
        self.shuffle_buffer = max(1, shuffle_buffer)

    def __iter__(self) -> Iterator[tuple[list[int], int]]:
        if self.shuffle:
            buffered: list[tuple[list[int], int]] = []
            yielded = 0
            with self.path.open("r", encoding="utf-8") as handle:
                for line in handle:
                    if not line.strip():
                        continue
                    item = _encode_row(json.loads(line), self.tokenizer, self.max_len, self.task)
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
            return
        emitted = 0
        with self.path.open("r", encoding="utf-8") as handle:
            for line in handle:
                if not line.strip():
                    continue
                item = _encode_row(json.loads(line), self.tokenizer, self.max_len, self.task)
                if item is None:
                    continue
                yield item
                emitted += 1
                if self.limit and emitted >= self.limit:
                    break


def collate(batch: list[tuple[list[int], int]], pad_id: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    longest = max(len(ids) for ids, _ in batch)
    width = longest - 1
    inputs = torch.full((len(batch), width), pad_id, dtype=torch.long)
    targets = torch.full((len(batch), width), -100, dtype=torch.long)
    attention = torch.zeros((len(batch), width), dtype=torch.bool)
    for row_index, (ids, prompt_mask) in enumerate(batch):
        input_ids = torch.tensor(ids[:-1], dtype=torch.long)
        label_ids = torch.tensor(ids[1:], dtype=torch.long)
        inputs[row_index, : len(input_ids)] = input_ids
        targets[row_index, : len(label_ids)] = label_ids
        attention[row_index, : len(input_ids)] = True
        if prompt_mask:
            targets[row_index, : min(prompt_mask, width)] = -100
    return inputs, targets, attention


def _run_eval(model, loader, device) -> tuple[float, int]:
    model.eval()
    loss_fn = CrossEntropyLoss(ignore_index=-100)
    total_loss = 0.0
    tokens = 0
    with torch.no_grad():
        for inputs, targets, attention in loader:
            inputs, targets, attention = inputs.to(device), targets.to(device), attention.to(device)
            logits = model(inputs, attention)
            loss = loss_fn(logits.reshape(-1, logits.size(-1)), targets.reshape(-1))
            valid = int((targets != -100).sum().item())
            total_loss += float(loss.item()) * max(valid, 1)
            tokens += valid
    return total_loss / max(tokens, 1), tokens


def _atomic_torch_save(value, path: Path) -> None:
    temporary = path.with_name(path.name + ".tmp")
    torch.save(value, temporary)
    temporary.replace(path)


def parameter_count(model: torch.nn.Module) -> int:
    return sum(parameter.numel() for parameter in model.parameters())


def _read_config(path: Path) -> dict:
    payload = json.loads(path.read_text(encoding="utf-8"))
    model = payload.get("model", {})
    tokenizer = payload.get("tokenizer", {})
    training = payload.get("training", {})
    tokenizer_kind = tokenizer.get("kind", "byte")
    if tokenizer_kind not in {"byte", "byte_bpe"}:
        raise ValueError(f"Unsupported tokenizer kind: {tokenizer_kind}")
    return {
        **model,
        **training,
        "tokenizer_kind": tokenizer_kind,
        "tokenizer_vocab_size": int(tokenizer.get("vocab_size", 512)),
        "tokenizer_text_limit": int(tokenizer.get("text_limit", 10000)),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Train the own SQL Transformer from random initialization.")
    parser.add_argument("--config", type=Path, default=ROOT_DIR / "training" / "configs" / "tiny.json")
    parser.add_argument("--task", choices=["nl2sql", "ddl", "causal"], default="nl2sql")
    parser.add_argument("--train-path", type=Path, default=None)
    parser.add_argument("--val-path", type=Path, default=None)
    parser.add_argument("--out-dir", type=Path, default=settings.own_model_dir)
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--max-steps", type=int, default=None, help="Use a positive value for a bounded smoke run.")
    parser.add_argument("--limit", type=int, default=0, help="Limit training examples per epoch.")
    parser.add_argument("--val-limit", type=int, default=0)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--init-from", type=Path, default=None, help="Load model weights from another artifact directory and reset optimizer state.")
    parser.add_argument("--tokenizer-from", type=Path, default=None, help="Reuse a tokenizer.json from another artifact, useful for causal pretraining followed by task fine-tuning.")
    parser.add_argument("--device", default="auto")
    args = parser.parse_args()

    runtime = _read_config(args.config)
    epochs = args.epochs if args.epochs is not None else int(runtime.get("epochs", 1))
    batch_size = args.batch_size if args.batch_size is not None else int(runtime.get("batch_size", 4))
    max_steps = args.max_steps if args.max_steps is not None else int(runtime.get("max_steps", 0))
    seed = int(runtime.get("seed", 42))
    set_seed(seed)
    train_path = args.train_path
    if train_path is None:
        default_paths = {
            "nl2sql": ROOT_DIR / "data" / "processed" / "nl2sql" / "train.jsonl",
            "ddl": ROOT_DIR / "data" / "processed" / "ddl" / "train.jsonl",
            "causal": ROOT_DIR / "data" / "intermediate" / "general_text.jsonl",
        }
        train_path = default_paths[args.task]
    if not train_path.exists():
        raise SystemExit(f"Training data not found: {train_path}. Prepare the requested corpus first.")
    if args.task == "causal" and not train_path.exists():
        raise SystemExit("The causal task requires a prepared general-text JSONL file.")
    val_path = args.val_path
    if val_path is None and args.task in {"nl2sql", "ddl"}:
        val_path = ROOT_DIR / "data" / "processed" / args.task / "val.jsonl"

    tokenizer_source = args.tokenizer_from
    if tokenizer_source is None and args.init_from:
        candidate = args.init_from / "tokenizer.json"
        if candidate.exists():
            source_config_path = args.init_from / "config.json"
            source_config = json.loads(source_config_path.read_text(encoding="utf-8")) if source_config_path.exists() else {}
            if source_config.get("tokenizer_kind") == runtime.get("tokenizer_kind") == "byte_bpe":
                tokenizer_source = candidate
    if tokenizer_source is not None:
        tokenizer_path = tokenizer_source / "tokenizer.json" if tokenizer_source.is_dir() else tokenizer_source
        if not tokenizer_path.exists():
            raise SystemExit(f"Tokenizer artifact not found: {tokenizer_path}")
        tokenizer = load_tokenizer(tokenizer_path)
    elif runtime.get("tokenizer_kind") == "byte_bpe":
        tokenizer_texts: list[str] = []
        tokenizer_text_limit = max(1, int(runtime.get("tokenizer_text_limit", 10000)))
        tokenizer_byte_limit = max(1, int(runtime.get("tokenizer_byte_limit", 250000)))
        tokenizer_bytes = 0
        with train_path.open("r", encoding="utf-8") as handle:
            for line in handle:
                if not line.strip():
                    continue
                row = json.loads(line)
                if args.task == "ddl":
                    text_for_tokenizer = build_ddl_prompt(str(row.get("question", "")), str(row.get("dialect", "postgresql")), str(row.get("operation", "generate_ddl")), str(row.get("schema_context", ""))) + str(row.get("target_sql", ""))
                elif args.task == "nl2sql":
                    text_for_tokenizer = build_sql_prompt(str(row.get("question", "")), str(row.get("schema_context", "")), str(row.get("dialect", "postgresql")), str(row.get("evidence", ""))) + str(row.get("target_sql", ""))
                else:
                    text_for_tokenizer = str(row.get("text", ""))
                encoded_length = len(text_for_tokenizer.encode("utf-8"))
                remaining = tokenizer_byte_limit - tokenizer_bytes
                if remaining <= 0:
                    break
                if encoded_length > remaining:
                    text_for_tokenizer = text_for_tokenizer.encode("utf-8")[:remaining].decode("utf-8", errors="ignore")
                    encoded_length = len(text_for_tokenizer.encode("utf-8"))
                tokenizer_texts.append(text_for_tokenizer)
                tokenizer_bytes += encoded_length
                if len(tokenizer_texts) >= tokenizer_text_limit or tokenizer_bytes >= tokenizer_byte_limit:
                    break
        tokenizer = train_tokenizer(tokenizer_texts, kind="byte_bpe", vocab_size=int(runtime.get("tokenizer_vocab_size", 512)))
    else:
        tokenizer = ByteTokenizer()
    max_len = int(runtime.get("max_len", 512))
    train_count = count_jsonl(train_path, args.limit)
    val_count = count_jsonl(val_path, args.val_limit) if val_path and val_path.exists() else 0
    if not train_count:
        raise SystemExit("Training data contains no usable rows.")
    device = torch.device(
        ("cuda" if torch.cuda.is_available() else "cpu") if args.device == "auto" else args.device
    )
    model = SQLTransformer(
        vocab_size=tokenizer.vocab_size,
        d_model=int(runtime.get("d_model", 128)),
        n_heads=int(runtime.get("n_heads", 4)),
        n_layers=int(runtime.get("n_layers", 4)),
        d_ff=int(runtime.get("d_ff", 512)),
        max_len=max_len,
        dropout=float(runtime.get("dropout", 0.1)),
    ).to(device)
    initialized_from = False
    if args.init_from:
        source_weights = args.init_from / "best.pt"
        if not source_weights.exists():
            source_weights = args.init_from / "model.pt"
        if not source_weights.exists():
            raise SystemExit(f"Initial model weights not found in {args.init_from}.")
        source_state = torch.load(source_weights, map_location=device, weights_only=True)
        current_state = model.state_dict()
        compatible_state = {key: value for key, value in source_state.items() if key in current_state and current_state[key].shape == value.shape}
        model.load_state_dict(compatible_state, strict=False)
        initialized_from = True
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(runtime.get("learning_rate", 3e-4)),
        weight_decay=float(runtime.get("weight_decay", 0.1)),
    )
    steps_per_epoch = max(1, math.ceil(train_count / max(batch_size, 1) / max(int(runtime.get("grad_accumulation", 1)), 1)))
    total_updates = max(1, epochs * steps_per_epoch)
    if max_steps > 0:
        total_updates = min(total_updates, max_steps)
    scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer, lambda step: max(0.0, 1.0 - step / max(total_updates, 1))
    )
    args.out_dir.mkdir(parents=True, exist_ok=True)
    config = {
        "architecture": "custom_decoder_only_transformer",
        "model_id": "own-ddl" if args.task == "ddl" else "own-sql",
        "model_version": "own-ddl-v1" if args.task == "ddl" else "own-sql-v1",
        "task": args.task,
        "d_model": int(runtime.get("d_model", 128)),
        "n_heads": int(runtime.get("n_heads", 4)),
        "n_layers": int(runtime.get("n_layers", 4)),
        "d_ff": int(runtime.get("d_ff", 512)),
        "max_len": max_len,
        "dropout": float(runtime.get("dropout", 0.1)),
        "vocab_size": tokenizer.vocab_size,
        "tokenizer_kind": runtime.get("tokenizer_kind", "byte"),
        "seed": seed,
        "random_initialization": not initialized_from,
        "train_path": str(train_path),
        "validation_path": str(val_path) if val_path else "",
    }
    task_transition = False
    if args.resume and (args.out_dir / "config.json").exists():
        previous = json.loads((args.out_dir / "config.json").read_text(encoding="utf-8"))
        task_transition = bool(previous.get("task") and previous.get("task") != args.task)
        for key in ("d_model", "n_heads", "n_layers", "d_ff", "max_len", "vocab_size", "tokenizer_kind"):
            if previous.get(key) != config.get(key):
                raise SystemExit(f"Cannot resume: artifact field `{key}` changed from {previous.get(key)!r} to {config.get(key)!r}.")
    (args.out_dir / "config.json").write_text(json.dumps(config, indent=2), encoding="utf-8")
    tokenizer.save(args.out_dir / "tokenizer.json")

    history: list[dict] = []
    best_val = float("inf")
    start_epoch = 1
    global_step = 0
    checkpoint_path = args.out_dir / "checkpoint.pt"
    if args.resume and checkpoint_path.exists():
        checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
        model.load_state_dict(checkpoint["model"])
        task_transition = task_transition or bool(checkpoint.get("task") and checkpoint.get("task") != args.task)
        if not task_transition:
            optimizer.load_state_dict(checkpoint["optimizer"])
            scheduler.load_state_dict(checkpoint["scheduler"])
            start_epoch = int(checkpoint.get("epoch", 0)) + 1
            global_step = int(checkpoint.get("global_step", 0))
            history = json.loads((args.out_dir / "training_history.json").read_text(encoding="utf-8")).get("history", []) if (args.out_dir / "training_history.json").exists() else []
            best_val = min((float(item.get("val_loss", float("inf"))) for item in history if item.get("val_loss") is not None), default=float("inf"))

    shuffle_buffer = max(1, int(runtime.get("shuffle_buffer", 1024)))
    train_dataset = JsonlDataset(
        train_path,
        tokenizer,
        max_len,
        args.task,
        args.limit,
        shuffle=args.task in {"ddl"},
        shuffle_buffer=shuffle_buffer,
    )
    val_dataset = JsonlDataset(val_path, tokenizer, max_len, args.task, args.val_limit, shuffle=False, shuffle_buffer=shuffle_buffer) if val_count and val_path else None
    train_loader = DataLoader(train_dataset, batch_size=batch_size, collate_fn=lambda batch: collate(batch, tokenizer.pad_id))
    val_loader = DataLoader(val_dataset, batch_size=batch_size, collate_fn=lambda batch: collate(batch, tokenizer.pad_id)) if val_dataset else None
    grad_accumulation = max(1, int(runtime.get("grad_accumulation", 1)))
    save_every = max(1, int(runtime.get("save_every", 250)))
    eval_every = max(1, int(runtime.get("eval_every", 1)))
    loss_fn = CrossEntropyLoss(ignore_index=-100)
    started = time.perf_counter()
    stopped = False
    for epoch in range(start_epoch, start_epoch + epochs):
        model.train()
        optimizer.zero_grad(set_to_none=True)
        train_loss = 0.0
        train_tokens = 0
        pending = 0
        val_loss = None
        val_tokens = 0
        evaluated_step = None
        for batch_index, (inputs, targets, attention) in enumerate(train_loader, start=1):
            if max_steps and global_step >= max_steps:
                stopped = True
                break
            inputs, targets, attention = inputs.to(device), targets.to(device), attention.to(device)
            logits = model(inputs, attention)
            loss = loss_fn(logits.reshape(-1, logits.size(-1)), targets.reshape(-1))
            (loss / grad_accumulation).backward()
            valid = int((targets != -100).sum().item())
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
                # ``eval_every`` is measured in optimizer updates, not epochs.
                # This matters for bounded runs where a large dataset may not
                # complete enough epochs to reach the configured interval.
                if val_loader is not None and global_step % eval_every == 0:
                    val_loss, val_tokens = _run_eval(model, val_loader, device)
                    evaluated_step = global_step
                    if val_loss < best_val:
                        best_val = val_loss
                        _atomic_torch_save(model.state_dict(), args.out_dir / "best.pt")
                if global_step % save_every == 0:
                    _atomic_torch_save({"model": model.state_dict(), "optimizer": optimizer.state_dict(), "scheduler": scheduler.state_dict(), "epoch": epoch, "global_step": global_step, "task": args.task}, checkpoint_path)
        if pending and not stopped:
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad(set_to_none=True)
            global_step += 1
        # Always evaluate at an epoch boundary when the last optimizer update
        # was not already evaluated. This guarantees every completed run has
        # an honest validation measurement and a meaningful best checkpoint.
        if val_loader is not None and evaluated_step != global_step:
            val_loss, val_tokens = _run_eval(model, val_loader, device)
            evaluated_step = global_step
            if val_loss < best_val:
                best_val = val_loss
                _atomic_torch_save(model.state_dict(), args.out_dir / "best.pt")
        record = {"epoch": epoch, "step": global_step, "train_loss": train_loss / max(train_tokens, 1), "val_loss": val_loss, "train_tokens": train_tokens, "val_tokens": val_tokens, "learning_rate": optimizer.param_groups[0]["lr"]}
        history.append(record)
        _atomic_torch_save(model.state_dict(), args.out_dir / "final.pt")
        _atomic_torch_save({"model": model.state_dict(), "optimizer": optimizer.state_dict(), "scheduler": scheduler.state_dict(), "epoch": epoch, "global_step": global_step, "task": args.task}, checkpoint_path)
        print(json.dumps(record))
        if stopped or (max_steps and global_step >= max_steps):
            break

    if not (args.out_dir / "best.pt").exists():
        _atomic_torch_save(model.state_dict(), args.out_dir / "best.pt")
    _atomic_torch_save(model.state_dict(), args.out_dir / "model.pt")
    metadata = {
        **config,
        "parameter_count": parameter_count(model),
        "device": str(device),
        "train_examples_seen_per_epoch": train_count,
        "validation_examples_available": val_count,
        "training_seconds": round(time.perf_counter() - started, 3),
        "pretrained_weights_loaded": bool(task_transition or initialized_from),
        "checkpoint_files": ["best.pt", "final.pt", "model.pt", "checkpoint.pt", "config.json", "tokenizer.json"],
    }
    (args.out_dir / "training_history.json").write_text(json.dumps({"metadata": metadata, "history": history}, indent=2), encoding="utf-8")
    print(json.dumps(metadata, indent=2))


if __name__ == "__main__":
    main()
