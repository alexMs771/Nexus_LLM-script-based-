from __future__ import annotations

from pathlib import Path

import torch

from app.models.transformer import SQLTransformer
from training.tokenizer import load_tokenizer


class ScratchSQLModel:
    name = "scratch"

    def __init__(self, model: SQLTransformer, tokenizer, device: torch.device, version: str = "") -> None:
        self.model = model.to(device)
        self.tokenizer = tokenizer
        self.device = device
        self.version = version
        self.model.eval()

    @classmethod
    def from_dir(cls, artifact_dir: Path) -> "ScratchSQLModel":
        config_path = artifact_dir / "config.json"
        weights_path = artifact_dir / "best.pt"
        if not weights_path.exists():
            weights_path = artifact_dir / "model.pt"
        tokenizer_path = artifact_dir / "tokenizer.json"
        if not config_path.exists() or not weights_path.exists() or not tokenizer_path.exists():
            raise FileNotFoundError(
                f"Scratch model artifacts not found in {artifact_dir}. "
                "Prepare data and run `python -m training.train_lm` first."
            )
        import json

        config = json.loads(config_path.read_text(encoding="utf-8"))
        tokenizer = load_tokenizer(tokenizer_path)
        model = SQLTransformer(
            vocab_size=tokenizer.vocab_size,
            d_model=config["d_model"],
            n_heads=config["n_heads"],
            n_layers=config["n_layers"],
            d_ff=config["d_ff"],
            max_len=config["max_len"],
            dropout=0.0,
        )
        state = torch.load(weights_path, map_location="cpu", weights_only=True)
        model.load_state_dict(state)
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        return cls(model=model, tokenizer=tokenizer, device=device, version=config.get("model_version", ""))

    def generate(self, prompt: str, max_new_tokens: int = 256, temperature: float = 0.0) -> str:
        encoded = self.tokenizer.encode(prompt, add_special=True)
        if encoded and encoded[-1] == self.tokenizer.eos_id:
            encoded = encoded[:-1]
        input_ids = torch.tensor([encoded], dtype=torch.long, device=self.device)
        output_ids = self.model.generate(
            input_ids=input_ids,
            max_new_tokens=min(max_new_tokens, self.model.max_len),
            eos_id=self.tokenizer.eos_id,
            pad_id=self.tokenizer.pad_id,
            temperature=temperature,
        )
        new_tokens = output_ids[0, input_ids.size(1) :].tolist()
        return self.tokenizer.decode(new_tokens)
