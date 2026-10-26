from __future__ import annotations

import math

import torch
from torch import nn


class CausalSelfAttention(nn.Module):
    def __init__(self, d_model: int, n_heads: int, dropout: float) -> None:
        super().__init__()
        if d_model % n_heads != 0:
            raise ValueError("d_model must be divisible by n_heads")
        self.n_heads = n_heads
        self.head_dim = d_model // n_heads
        self.qkv = nn.Linear(d_model, 3 * d_model)
        self.proj = nn.Linear(d_model, d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor, attention_mask: torch.Tensor | None = None) -> torch.Tensor:
        batch, seq_len, d_model = x.shape
        qkv = self.qkv(x).reshape(batch, seq_len, 3, self.n_heads, self.head_dim)
        q, k, v = qkv.unbind(dim=2)
        q = q.transpose(1, 2)
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)
        scores = (q @ k.transpose(-2, -1)) / math.sqrt(self.head_dim)
        mask = torch.triu(torch.ones(seq_len, seq_len, device=x.device, dtype=torch.bool), diagonal=1)
        scores = scores.masked_fill(mask, float("-inf"))
        if attention_mask is not None:
            # Mask padded key positions. Query positions are ignored by the
            # loss, so masking keys is sufficient and keeps generation simple.
            scores = scores.masked_fill(~attention_mask[:, None, None, :].bool(), float("-inf"))
        weights = torch.softmax(scores, dim=-1)
        weights = self.dropout(weights)
        attended = weights @ v
        attended = attended.transpose(1, 2).reshape(batch, seq_len, d_model)
        return self.proj(attended)


class FeedForward(nn.Module):
    def __init__(self, d_model: int, d_ff: int, dropout: float) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.GELU(),
            nn.Linear(d_ff, d_model),
            nn.Dropout(dropout),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class TransformerBlock(nn.Module):
    def __init__(self, d_model: int, n_heads: int, d_ff: int, dropout: float) -> None:
        super().__init__()
        self.ln1 = nn.LayerNorm(d_model)
        self.attn = CausalSelfAttention(d_model, n_heads, dropout)
        self.ln2 = nn.LayerNorm(d_model)
        self.ff = FeedForward(d_model, d_ff, dropout)

    def forward(self, x: torch.Tensor, attention_mask: torch.Tensor | None = None) -> torch.Tensor:
        x = x + self.attn(self.ln1(x), attention_mask)
        x = x + self.ff(self.ln2(x))
        return x


class SQLTransformer(nn.Module):
    def __init__(
        self,
        vocab_size: int,
        d_model: int = 128,
        n_heads: int = 4,
        n_layers: int = 4,
        d_ff: int = 512,
        max_len: int = 256,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.max_len = max_len
        self.token_embed = nn.Embedding(vocab_size, d_model)
        self.pos_embed = nn.Embedding(max_len, d_model)
        self.dropout = nn.Dropout(dropout)
        self.blocks = nn.ModuleList(
            [TransformerBlock(d_model, n_heads, d_ff, dropout) for _ in range(n_layers)]
        )
        self.ln_f = nn.LayerNorm(d_model)
        self.head = nn.Linear(d_model, vocab_size, bias=False)
        self.head.weight = self.token_embed.weight

    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor | None = None) -> torch.Tensor:
        seq_len = input_ids.size(1)
        if seq_len > self.max_len:
            raise ValueError(f"Sequence length {seq_len} exceeds max_len {self.max_len}")
        positions = torch.arange(seq_len, device=input_ids.device).unsqueeze(0)
        x = self.token_embed(input_ids) + self.pos_embed(positions)
        x = self.dropout(x)
        for block in self.blocks:
            x = block(x, attention_mask)
        x = self.ln_f(x)
        return self.head(x)

    @torch.no_grad()
    def generate(
        self,
        input_ids: torch.Tensor,
        max_new_tokens: int,
        eos_id: int,
        pad_id: int,
        temperature: float = 1.0,
        repetition_penalty: float = 1.0,
        no_repeat_ngram_size: int = 0,
        top_k: int = 0,
    ) -> torch.Tensor:
        self.eval()
        tokens = input_ids
        for _ in range(max_new_tokens):
            context = tokens[:, -self.max_len :]
            logits = self(context)[:, -1, :]
            if repetition_penalty > 1.0:
                # Penalize tokens already present in the current sequence.  This
                # is a model-agnostic decoding control; it does not synthesize
                # SQL or impose a hard-coded response template.
                for batch_index in range(logits.size(0)):
                    seen = torch.unique(tokens[batch_index])
                    values = logits[batch_index, seen]
                    logits[batch_index, seen] = torch.where(
                        values < 0,
                        values * repetition_penalty,
                        values / repetition_penalty,
                    )
            if no_repeat_ngram_size > 1 and tokens.size(1) >= no_repeat_ngram_size - 1:
                prefix = tokens[:, -(no_repeat_ngram_size - 1) :]
                for batch_index in range(tokens.size(0)):
                    history = tokens[batch_index].tolist()
                    prefix_values = prefix[batch_index].tolist()
                    blocked: set[int] = set()
                    for index in range(len(history) - no_repeat_ngram_size + 1):
                        if history[index : index + no_repeat_ngram_size - 1] == prefix_values:
                            blocked.add(history[index + no_repeat_ngram_size - 1])
                    if blocked:
                        logits[batch_index, list(blocked)] = float("-inf")
            if top_k > 0 and top_k < logits.size(-1):
                cutoff = torch.topk(logits, top_k, dim=-1).values[:, -1].unsqueeze(-1)
                logits = logits.masked_fill(logits < cutoff, float("-inf"))
            if temperature <= 0:
                next_id = torch.argmax(logits, dim=-1, keepdim=True)
            else:
                probs = torch.softmax(logits / temperature, dim=-1)
                next_id = torch.multinomial(probs, num_samples=1)
            tokens = torch.cat([tokens, next_id], dim=1)
            if int(next_id.item()) in {eos_id, pad_id}:
                break
        return tokens
