from __future__ import annotations

import argparse
import json
import math
import random
import time
from pathlib import Path

import torch
from torch.nn import CrossEntropyLoss
from torch.utils.data import DataLoader, Dataset

from app.config import ROOT_DIR, settings
from app.models.transformer import SQLTransformer
from app.prompts import build_prompt
from training.dataset import read_jsonl
from training.prepare_data import PROCESSED_DIR, main as prepare_data
from training.tokenizer import train_tokenizer


def set_seed(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)


def format_row(row: dict) -> str:
    response = row.get("expected_ddl") or row.get("output") or ""
    return build_prompt(row["request"], row.get("dialect", "postgresql"), row.get("task", "generate_ddl"), intent=row.get("intent", "IN_SCOPE")) + response


class SQLTextDataset(Dataset):
    def __init__(self, rows: list[dict], tokenizer, max_len: int) -> None:
        self.examples = []
        for row in rows:
            ids = tokenizer.encode(format_row(row), add_special=True)[:max_len]
            if len(ids) >= 3:
                self.examples.append(ids)

    def __len__(self) -> int: return len(self.examples)
    def __getitem__(self, index: int) -> list[int]: return self.examples[index]


def collate(batch: list[list[int]], pad_id: int) -> tuple[torch.Tensor, torch.Tensor]:
    longest = max(len(item) for item in batch)
    inputs = torch.full((len(batch), longest - 1), pad_id, dtype=torch.long)
    targets = torch.full((len(batch), longest - 1), -100, dtype=torch.long)
    for row_index, ids in enumerate(batch):
        inputs[row_index, : len(ids) - 1] = torch.tensor(ids[:-1], dtype=torch.long)
        targets[row_index, : len(ids) - 1] = torch.tensor(ids[1:], dtype=torch.long)
    return inputs, targets


def run_epoch(model, loader, optimizer, device, grad_accumulation: int = 1) -> tuple[float, int]:
    training = optimizer is not None
    model.train(training)
    loss_fn = CrossEntropyLoss(ignore_index=-100)
    total_loss, token_count = 0.0, 0
    if training:
        optimizer.zero_grad(set_to_none=True)
    for batch_index, (inputs, targets) in enumerate(loader):
        inputs, targets = inputs.to(device), targets.to(device)
        logits = model(inputs)
        loss = loss_fn(logits.reshape(-1, logits.size(-1)), targets.reshape(-1))
        if training:
            (loss / max(grad_accumulation, 1)).backward()
            if (batch_index + 1) % max(grad_accumulation, 1) == 0 or batch_index + 1 == len(loader):
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)
        valid_tokens = int((targets != -100).sum().item())
        total_loss += float(loss.item()) * max(valid_tokens, 1)
        token_count += valid_tokens
    return total_loss / max(token_count, 1), token_count


def parameter_count(model: torch.nn.Module) -> int:
    return sum(parameter.numel() for parameter in model.parameters())


def main() -> None:
    parser = argparse.ArgumentParser(description="Train the custom decoder-only Transformer from random initialization.")
    parser.add_argument("--epochs", type=int, default=8)
    parser.add_argument("--d-model", type=int, default=64)
    parser.add_argument("--n-heads", type=int, default=4)
    parser.add_argument("--n-layers", type=int, default=2)
    parser.add_argument("--d-ff", type=int, default=256)
    parser.add_argument("--max-len", type=int, default=192)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--grad-accumulation", type=int, default=1)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--tokenizer", choices=["byte_bpe", "byte"], default="byte_bpe")
    parser.add_argument("--vocab-size", type=int, default=512)
    parser.add_argument("--out-dir", type=Path, default=settings.scratch_dir)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    set_seed(args.seed)

    if not (PROCESSED_DIR / "train.jsonl").exists():
        prepare_data()
    train_rows = read_jsonl(PROCESSED_DIR / "train.jsonl")
    val_rows = read_jsonl(PROCESSED_DIR / "val.jsonl") if (PROCESSED_DIR / "val.jsonl").exists() else []
    if not train_rows:
        raise SystemExit("Training dataset is empty.")
    tokenizer = train_tokenizer([format_row(row) for row in train_rows], args.tokenizer, args.vocab_size)
    train_dataset = SQLTextDataset(train_rows, tokenizer, args.max_len)
    val_dataset = SQLTextDataset(val_rows, tokenizer, args.max_len)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = SQLTransformer(tokenizer.vocab_size, args.d_model, args.n_heads, args.n_layers, args.d_ff, args.max_len).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)
    train_loader = DataLoader(train_dataset, batch_size=min(args.batch_size, len(train_dataset)), shuffle=True, collate_fn=lambda b: collate(b, tokenizer.pad_id))
    val_loader = DataLoader(val_dataset, batch_size=min(args.batch_size, len(val_dataset)), shuffle=False, collate_fn=lambda b: collate(b, tokenizer.pad_id)) if val_dataset else None
    args.out_dir.mkdir(parents=True, exist_ok=True)
    config = {
        "architecture": "custom_decoder_only_transformer",
        "d_model": args.d_model, "n_heads": args.n_heads, "n_layers": args.n_layers,
        "d_ff": args.d_ff, "max_len": args.max_len, "vocab_size": tokenizer.vocab_size,
        "tokenizer_kind": args.tokenizer, "seed": args.seed, "random_initialization": True,
    }
    manifest_path = PROCESSED_DIR / "manifest.json"
    if manifest_path.exists():
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        config["dataset_version"] = manifest.get("dataset_version")
        config["dataset_counts"] = manifest.get("counts")
    (args.out_dir / "config.json").write_text(json.dumps(config, indent=2), encoding="utf-8")
    tokenizer.save(args.out_dir / "tokenizer.json")
    best_val = float("inf")
    history: list[dict] = []
    start_epoch = 1
    if args.resume and (args.out_dir / "checkpoint.pt").exists():
        checkpoint = torch.load(args.out_dir / "checkpoint.pt", map_location=device, weights_only=True)
        model.load_state_dict(checkpoint["model"])
        optimizer.load_state_dict(checkpoint["optimizer"])
        history = json.loads((args.out_dir / "training_history.json").read_text(encoding="utf-8")).get("history", []) if (args.out_dir / "training_history.json").exists() else []
        start_epoch = int(checkpoint.get("epoch", 0)) + 1
        best_val = min((float(row.get("val_loss", float("inf"))) for row in history if row.get("val_loss") is not None), default=float("inf"))
    started = time.perf_counter()
    global_step = 0
    for epoch in range(start_epoch, start_epoch + args.epochs):
        train_loss, train_tokens = run_epoch(model, train_loader, optimizer, device, args.grad_accumulation)
        val_loss, val_tokens = run_epoch(model, val_loader, None, device) if val_loader else (float("nan"), 0)
        global_step += math.ceil(len(train_loader) / max(args.grad_accumulation, 1))
        record = {
            "epoch": epoch, "step": global_step, "train_loss": train_loss, "val_loss": val_loss,
            "train_perplexity": math.exp(min(train_loss, 50)),
            "val_perplexity": math.exp(min(val_loss, 50)) if val_loader else None,
            "train_tokens": train_tokens, "val_tokens": val_tokens,
        }
        history.append(record)
        torch.save(model.state_dict(), args.out_dir / "final.pt")
        torch.save({"model": model.state_dict(), "optimizer": optimizer.state_dict(), "epoch": epoch}, args.out_dir / "checkpoint.pt")
        if val_loader and val_loss < best_val:
            best_val = val_loss
            torch.save(model.state_dict(), args.out_dir / "best.pt")
        print(json.dumps(record))
    if not (args.out_dir / "best.pt").exists():
        torch.save(model.state_dict(), args.out_dir / "best.pt")
    elapsed = time.perf_counter() - started
    metadata = {
        **config,
        "parameter_count": parameter_count(model),
        "device": str(device),
        "train_examples": len(train_dataset), "validation_examples": len(val_dataset),
        "training_seconds": elapsed,
        "tokens_per_second": sum(item["train_tokens"] for item in history) / max(elapsed, 1e-9),
        "checkpoint_files": ["best.pt", "final.pt", "config.json", "tokenizer.json", "training_history.json"],
        "pretrained_weights_loaded": False,
    }
    (args.out_dir / "training_history.json").write_text(json.dumps({"metadata": metadata, "history": history}, indent=2), encoding="utf-8")
    # Legacy filename keeps old callers functional while metadata identifies the run.
    torch.save(model.state_dict(), args.out_dir / "model.pt")
    print(json.dumps(metadata, indent=2))


if __name__ == "__main__":
    main()
