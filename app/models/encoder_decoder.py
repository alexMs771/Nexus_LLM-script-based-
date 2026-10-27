"""Scratch-trained encoder-decoder Transformer for prompt-to-DDL generation.

The encoder reads the complete user instruction bidirectionally.  The decoder
then generates the answer autoregressively while attending to the encoded
instruction through cross-attention.  This is intentionally separate from the
existing decoder-only model so old checkpoints remain loadable.
"""

from __future__ import annotations

import math

import torch
from torch import nn


class FeedForward(nn.Module):
    def __init__(self, d_model: int, d_ff: int, dropout: float) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_ff, d_model),
            nn.Dropout(dropout),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class EncoderLayer(nn.Module):
    def __init__(self, d_model: int, n_heads: int, d_ff: int, dropout: float) -> None:
        super().__init__()
        self.norm_self = nn.LayerNorm(d_model)
        self.self_attention = nn.MultiheadAttention(
            d_model, n_heads, dropout=dropout, batch_first=True
        )
        self.norm_ff = nn.LayerNorm(d_model)
        self.feed_forward = FeedForward(d_model, d_ff, dropout)

    def forward(self, x: torch.Tensor, key_padding_mask: torch.Tensor | None) -> torch.Tensor:
        normalized = self.norm_self(x)
        attended, _ = self.self_attention(
            normalized,
            normalized,
            normalized,
            key_padding_mask=key_padding_mask,
            need_weights=False,
        )
        x = x + attended
        return x + self.feed_forward(self.norm_ff(x))


class DecoderLayer(nn.Module):
    def __init__(self, d_model: int, n_heads: int, d_ff: int, dropout: float) -> None:
        super().__init__()
        self.norm_self = nn.LayerNorm(d_model)
        self.self_attention = nn.MultiheadAttention(
            d_model, n_heads, dropout=dropout, batch_first=True
        )
        self.norm_cross = nn.LayerNorm(d_model)
        self.cross_attention = nn.MultiheadAttention(
            d_model, n_heads, dropout=dropout, batch_first=True
        )
        self.norm_ff = nn.LayerNorm(d_model)
        self.feed_forward = FeedForward(d_model, d_ff, dropout)

    def forward(
        self,
        x: torch.Tensor,
        memory: torch.Tensor,
        target_key_padding_mask: torch.Tensor | None,
        source_key_padding_mask: torch.Tensor | None,
        causal_mask: torch.Tensor,
    ) -> torch.Tensor:
        normalized = self.norm_self(x)
        attended, _ = self.self_attention(
            normalized,
            normalized,
            normalized,
            attn_mask=causal_mask,
            key_padding_mask=target_key_padding_mask,
            need_weights=False,
        )
        x = x + attended
        normalized = self.norm_cross(x)
        attended, _ = self.cross_attention(
            normalized,
            memory,
            memory,
            key_padding_mask=source_key_padding_mask,
            need_weights=False,
        )
        x = x + attended
        return x + self.feed_forward(self.norm_ff(x))


class EncoderDecoderTransformer(nn.Module):
    """Small train-from-scratch sequence-to-sequence Transformer."""

    architecture = "custom_encoder_decoder_transformer"

    def __init__(
        self,
        vocab_size: int,
        d_model: int = 192,
        n_heads: int = 6,
        n_encoder_layers: int = 4,
        n_decoder_layers: int = 4,
        d_ff: int = 768,
        max_source_len: int = 768,
        max_target_len: int = 320,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        if d_model % n_heads != 0:
            raise ValueError("d_model must be divisible by n_heads")
        self.max_source_len = max_source_len
        self.max_target_len = max_target_len
        self.token_embed = nn.Embedding(vocab_size, d_model)
        self.source_pos_embed = nn.Embedding(max_source_len, d_model)
        self.target_pos_embed = nn.Embedding(max_target_len, d_model)
        self.dropout = nn.Dropout(dropout)
        self.encoder_layers = nn.ModuleList(
            [EncoderLayer(d_model, n_heads, d_ff, dropout) for _ in range(n_encoder_layers)]
        )
        self.decoder_layers = nn.ModuleList(
            [DecoderLayer(d_model, n_heads, d_ff, dropout) for _ in range(n_decoder_layers)]
        )
        self.encoder_norm = nn.LayerNorm(d_model)
        self.decoder_norm = nn.LayerNorm(d_model)
        self.output_projection = nn.Linear(d_model, vocab_size, bias=False)
        self.output_projection.weight = self.token_embed.weight

    def _positions(self, length: int, device: torch.device, embedding: nn.Embedding) -> torch.Tensor:
        if length > embedding.num_embeddings:
            raise ValueError(
                f"Sequence length {length} exceeds configured maximum {embedding.num_embeddings}"
            )
        return embedding(torch.arange(length, device=device).unsqueeze(0))

    def encode(
        self,
        source_ids: torch.Tensor,
        source_key_padding_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        source_len = source_ids.size(1)
        x = self.token_embed(source_ids) + self._positions(
            source_len, source_ids.device, self.source_pos_embed
        )
        x = self.dropout(x)
        for layer in self.encoder_layers:
            x = layer(x, source_key_padding_mask)
        return self.encoder_norm(x)

    def decode(
        self,
        target_ids: torch.Tensor,
        memory: torch.Tensor,
        target_key_padding_mask: torch.Tensor | None = None,
        source_key_padding_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        target_len = target_ids.size(1)
        x = self.token_embed(target_ids) + self._positions(
            target_len, target_ids.device, self.target_pos_embed
        )
        x = self.dropout(x)
        causal_mask = torch.triu(
            torch.ones(target_len, target_len, device=target_ids.device, dtype=torch.bool),
            diagonal=1,
        )
        for layer in self.decoder_layers:
            x = layer(
                x,
                memory,
                target_key_padding_mask,
                source_key_padding_mask,
                causal_mask,
            )
        return self.output_projection(self.decoder_norm(x))

    def forward(
        self,
        source_ids: torch.Tensor,
        target_ids: torch.Tensor,
        source_key_padding_mask: torch.Tensor | None = None,
        target_key_padding_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        memory = self.encode(source_ids, source_key_padding_mask)
        return self.decode(target_ids, memory, target_key_padding_mask, source_key_padding_mask)

    @staticmethod
    def _apply_repetition_penalty(logits: torch.Tensor, tokens: torch.Tensor, penalty: float) -> None:
        if penalty <= 1.0:
            return
        for batch_index in range(logits.size(0)):
            seen = torch.unique(tokens[batch_index])
            values = logits[batch_index, seen]
            logits[batch_index, seen] = torch.where(values < 0, values * penalty, values / penalty)

    @staticmethod
    def _block_repeated_ngrams(logits: torch.Tensor, tokens: torch.Tensor, ngram_size: int) -> None:
        if ngram_size <= 1 or tokens.size(1) < ngram_size - 1:
            return
        prefix = tokens[:, -(ngram_size - 1) :].tolist()
        for batch_index, history in enumerate(tokens.tolist()):
            blocked: set[int] = set()
            for index in range(len(history) - ngram_size + 1):
                if history[index : index + ngram_size - 1] == prefix[batch_index]:
                    blocked.add(history[index + ngram_size - 1])
            if blocked:
                logits[batch_index, list(blocked)] = float("-inf")

    @torch.no_grad()
    def generate(
        self,
        source_ids: torch.Tensor,
        max_new_tokens: int,
        bos_id: int,
        eos_id: int,
        pad_id: int,
        source_key_padding_mask: torch.Tensor | None = None,
        temperature: float = 0.0,
        repetition_penalty: float = 1.04,
        no_repeat_ngram_size: int = 6,
        top_k: int = 0,
    ) -> torch.Tensor:
        self.eval()
        memory = self.encode(source_ids, source_key_padding_mask)
        target_ids = torch.full(
            (source_ids.size(0), 1), bos_id, dtype=torch.long, device=source_ids.device
        )
        for _ in range(min(max_new_tokens, self.max_target_len - 1)):
            logits = self.decode(target_ids, memory, None, source_key_padding_mask)[:, -1, :]
            self._apply_repetition_penalty(logits, target_ids, repetition_penalty)
            self._block_repeated_ngrams(logits, target_ids, no_repeat_ngram_size)
            if top_k > 0 and top_k < logits.size(-1):
                cutoff = torch.topk(logits, top_k, dim=-1).values[:, -1].unsqueeze(-1)
                logits = logits.masked_fill(logits < cutoff, float("-inf"))
            if temperature <= 0:
                next_id = torch.argmax(logits, dim=-1, keepdim=True)
            else:
                next_id = torch.multinomial(torch.softmax(logits / temperature, dim=-1), 1)
            target_ids = torch.cat([target_ids, next_id], dim=1)
            if bool(torch.all((next_id == eos_id) | (next_id == pad_id))):
                break
        return target_ids
