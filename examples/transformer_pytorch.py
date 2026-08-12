"""A small decoder-only Transformer language model implemented with PyTorch.

This is the reference implementation for ``transformer_ocean.oc``.  It uses
PyTorch's standard Transformer building blocks and keeps the model deliberately
small so the file can also serve as a readable executable example.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor, nn


@dataclass
class TransformerConfig:
    vocab_size: int = 32
    max_seq_len: int = 32
    d_model: int = 64
    nhead: int = 4
    num_layers: int = 2
    dim_feedforward: int = 128
    dropout: float = 0.0


class TransformerLanguageModel(nn.Module):
    """GPT-style causal language model built from PyTorch modules."""

    def __init__(self, config: TransformerConfig) -> None:
        super().__init__()
        self.config = config
        self.token_embedding = nn.Embedding(config.vocab_size, config.d_model)
        self.position_embedding = nn.Embedding(config.max_seq_len, config.d_model)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=config.d_model,
            nhead=config.nhead,
            dim_feedforward=config.dim_feedforward,
            dropout=config.dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.transformer = nn.TransformerEncoder(
            encoder_layer,
            num_layers=config.num_layers,
            norm=nn.LayerNorm(config.d_model),
        )
        self.lm_head = nn.Linear(config.d_model, config.vocab_size, bias=False)

    def causal_mask(self, sequence_length: int, device: torch.device) -> Tensor:
        """Return an additive upper-triangular causal attention mask."""
        return torch.triu(
            torch.full(
                (sequence_length, sequence_length),
                float("-inf"),
                device=device,
            ),
            diagonal=1,
        )

    def forward(self, tokens: Tensor) -> Tensor:
        """Return next-token logits with shape ``(batch, sequence, vocab)``."""
        _, sequence_length = tokens.shape
        if sequence_length > self.config.max_seq_len:
            raise ValueError("sequence is longer than max_seq_len")

        positions = torch.arange(
            sequence_length,
            device=tokens.device,
            dtype=torch.long,
        ).unsqueeze(0)
        hidden = self.token_embedding(tokens) + self.position_embedding(positions)
        hidden = self.transformer(
            hidden,
            mask=self.causal_mask(sequence_length, tokens.device),
        )
        return self.lm_head(hidden)

    def loss(self, tokens: Tensor, targets: Tensor) -> Tensor:
        """Compute teacher-forced autoregressive cross-entropy."""
        logits = self(tokens)
        return nn.functional.cross_entropy(
            logits.reshape(-1, self.config.vocab_size),
            targets.reshape(-1),
        )

    @torch.no_grad()
    def generate(self, tokens: Tensor, max_new_tokens: int) -> Tensor:
        """Greedily append tokens selected from the final position's logits."""
        result = tokens.clone()
        for _ in range(max_new_tokens):
            context = result[:, -self.config.max_seq_len :]
            next_token = self(context)[:, -1, :].argmax(dim=-1, keepdim=True)
            result = torch.cat((result, next_token), dim=1)
        return result


def main() -> None:
    torch.manual_seed(7)
    config = TransformerConfig(
        vocab_size=16,
        max_seq_len=16,
        d_model=32,
        nhead=4,
        num_layers=2,
        dim_feedforward=64,
    )
    model = TransformerLanguageModel(config)

    # A tiny next-token training example: targets are the input shifted left.
    tokens = torch.tensor([[1, 4, 6, 2, 5, 3]], dtype=torch.long)
    targets = torch.tensor([[4, 6, 2, 5, 3, 7]], dtype=torch.long)

    model.train()
    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4)
    for _ in range(3):
        optimizer.zero_grad(set_to_none=True)
        loss = model.loss(tokens, targets)
        loss.backward()
        optimizer.step()

    model.eval()
    generated = model.generate(tokens[:, :3], max_new_tokens=4)
    print("loss:", float(loss))
    print("generated token ids:", generated.tolist())


if __name__ == "__main__":
    main()
