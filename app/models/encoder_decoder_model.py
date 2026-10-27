from __future__ import annotations

import json
from pathlib import Path

import torch

from app.models.encoder_decoder import EncoderDecoderTransformer
from training.tokenizer import load_tokenizer


class EncoderDecoderDDLModel:
    """Serving wrapper for a trained scratch encoder-decoder checkpoint."""

    name = "own-encoder-decoder"

    def __init__(self, model: EncoderDecoderTransformer, tokenizer, device: torch.device, version: str = "") -> None:
        self.model = model.to(device)
        self.tokenizer = tokenizer
        self.device = device
        self.version = version
        self.model.eval()

    @classmethod
    def from_dir(cls, artifact_dir: Path) -> "EncoderDecoderDDLModel":
        config_path = artifact_dir / "config.json"
        weights_path = artifact_dir / "best.pt"
        if not weights_path.exists():
            weights_path = artifact_dir / "model.pt"
        tokenizer_path = artifact_dir / "tokenizer.json"
        if not config_path.exists() or not weights_path.exists() or not tokenizer_path.exists():
            raise FileNotFoundError(
                f"Encoder-decoder artifacts not found in {artifact_dir}. "
                "Run training.train_seq2seq first."
            )
        config = json.loads(config_path.read_text(encoding="utf-8"))
        if config.get("architecture") != EncoderDecoderTransformer.architecture:
            raise ValueError(f"Artifact is not encoder-decoder: {config.get('architecture')!r}")
        tokenizer = load_tokenizer(tokenizer_path)
        model = EncoderDecoderTransformer(
            vocab_size=int(config["vocab_size"]),
            d_model=int(config["d_model"]),
            n_heads=int(config["n_heads"]),
            n_encoder_layers=int(config["n_encoder_layers"]),
            n_decoder_layers=int(config["n_decoder_layers"]),
            d_ff=int(config["d_ff"]),
            max_source_len=int(config["max_source_len"]),
            max_target_len=int(config["max_target_len"]),
            dropout=0.0,
        )
        state = torch.load(weights_path, map_location="cpu", weights_only=True)
        model.load_state_dict(state)
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        return cls(model=model, tokenizer=tokenizer, device=device, version=config.get("model_version", ""))

    def generate(
        self,
        prompt: str,
        max_new_tokens: int = 256,
        temperature: float = 0.0,
        repetition_penalty: float = 1.04,
        no_repeat_ngram_size: int = 6,
        top_k: int = 0,
    ) -> str:
        encoded = self.tokenizer.encode(prompt, add_special=True)
        if len(encoded) > self.model.max_source_len:
            encoded = encoded[: self.model.max_source_len]
            encoded[-1] = self.tokenizer.eos_id
        source_ids = torch.tensor([encoded], dtype=torch.long, device=self.device)
        output_ids = self.model.generate(
            source_ids=source_ids,
            max_new_tokens=max_new_tokens,
            bos_id=self.tokenizer.bos_id,
            eos_id=self.tokenizer.eos_id,
            pad_id=self.tokenizer.pad_id,
            temperature=temperature,
            repetition_penalty=repetition_penalty,
            no_repeat_ngram_size=no_repeat_ngram_size,
            top_k=top_k,
        )
        generated = output_ids[0, 1:].tolist()
        return self.tokenizer.decode(generated)
