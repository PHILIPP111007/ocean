"""A faithful PyTorch GPT-2 proof of concept.

This file implements the GPT-2 decoder instead of wrapping the Hugging Face
model. Hugging Face is used only as the checkpoint and tokenizer source, so
the forward pass can later be compared with Ocean's implementation.

The default model is canonical GPT-2 Small:
vocab=50257, context=1024, hidden=768, heads=12, ff=3072, layers=12.
"""

from __future__ import annotations

import argparse
import math
import time
from dataclasses import dataclass
from typing import Optional, Sequence

import torch
from torch import Tensor, nn
from torch.nn import functional as F


@dataclass(frozen=True)
class GPT2Architecture:
    vocab_size: int
    context_length: int
    hidden_size: int
    num_heads: int
    intermediate_size: int
    num_layers: int
    layer_norm_epsilon: float = 1.0e-5
    initializer_range: float = 0.02
    activation_function: str = "gelu_new"
    embedding_dropout: float = 0.1
    residual_dropout: float = 0.1
    attention_dropout: float = 0.1

    @classmethod
    def from_huggingface(cls, config: object) -> "GPT2Architecture":
        return cls(
            vocab_size=int(config.vocab_size),
            context_length=int(config.n_positions),
            hidden_size=int(config.n_embd),
            num_heads=int(config.n_head),
            intermediate_size=int(config.n_inner or 4 * config.n_embd),
            num_layers=int(config.n_layer),
            layer_norm_epsilon=float(config.layer_norm_epsilon),
            initializer_range=float(config.initializer_range),
            activation_function=str(config.activation_function),
            embedding_dropout=float(getattr(config, "embd_pdrop", 0.1)),
            residual_dropout=float(getattr(config, "resid_pdrop", 0.1)),
            attention_dropout=float(getattr(config, "attn_pdrop", 0.1)),
        )


def gelu_new(x: Tensor) -> Tensor:
    """GPT-2's original tanh GELU approximation."""

    return 0.5 * x * (
        1.0
        + torch.tanh(
            math.sqrt(2.0 / math.pi) * (x + 0.044715 * torch.pow(x, 3))
        )
    )


class GPT2Attention(nn.Module):
    """GPT-2 masked multi-head self-attention with an optional KV cache."""

    def __init__(self, config: GPT2Architecture) -> None:
        super().__init__()
        if config.hidden_size % config.num_heads != 0:
            raise ValueError("hidden_size must be divisible by num_heads")
        self.num_heads = config.num_heads
        self.head_dim = config.hidden_size // config.num_heads
        self.max_positions = config.context_length
        self.scale = self.head_dim**-0.5
        self.c_attn = nn.Linear(config.hidden_size, 3 * config.hidden_size)
        self.c_proj = nn.Linear(config.hidden_size, config.hidden_size)
        self.attn_dropout = nn.Dropout(config.attention_dropout)
        self.resid_dropout = nn.Dropout(config.residual_dropout)
        self.register_buffer(
            "causal_mask",
            torch.tril(
                torch.ones(
                    config.context_length,
                    config.context_length,
                    dtype=torch.bool,
                )
            ).view(1, 1, config.context_length, config.context_length),
            persistent=False,
        )

    def _split_heads(self, x: Tensor) -> Tensor:
        batch, sequence, _ = x.shape
        return x.view(batch, sequence, self.num_heads, self.head_dim).transpose(1, 2)

    def _merge_heads(self, x: Tensor) -> Tensor:
        batch, _, sequence, _ = x.shape
        return x.transpose(1, 2).contiguous().view(
            batch, sequence, self.num_heads * self.head_dim
        )

    def forward(
        self,
        hidden_states: Tensor,
        layer_past: Optional[tuple[Tensor, Tensor]] = None,
        use_cache: bool = True,
    ) -> tuple[Tensor, Optional[tuple[Tensor, Tensor]]]:
        query, key, value = self.c_attn(hidden_states).split(
            self.num_heads * self.head_dim, dim=2
        )
        query = self._split_heads(query)
        key = self._split_heads(key)
        value = self._split_heads(value)

        past_length = 0
        if layer_past is not None:
            past_key, past_value = layer_past
            past_length = past_key.size(-2)
            key = torch.cat((past_key, key), dim=-2)
            value = torch.cat((past_value, value), dim=-2)

        sequence_length = hidden_states.size(1)
        total_length = key.size(-2)
        if total_length > self.max_positions:
            raise ValueError(
                f"sequence length {total_length} exceeds GPT-2 context "
                f"window {self.max_positions}"
            )

        attention_scores = torch.matmul(query, key.transpose(-1, -2)) * self.scale
        mask = self.causal_mask[
            :, :, past_length : past_length + sequence_length, :total_length
        ]
        attention_scores = attention_scores.masked_fill(
            ~mask, torch.finfo(attention_scores.dtype).min
        )
        attention_weights = F.softmax(attention_scores, dim=-1)
        attention_weights = self.attn_dropout(attention_weights)
        attention_output = torch.matmul(attention_weights, value)
        attention_output = self._merge_heads(attention_output)
        attention_output = self.c_proj(attention_output)
        attention_output = self.resid_dropout(attention_output)

        present = (key, value) if use_cache else None
        return attention_output, present


class GPT2MLP(nn.Module):
    def __init__(self, config: GPT2Architecture) -> None:
        super().__init__()
        self.c_fc = nn.Linear(config.hidden_size, config.intermediate_size)
        self.c_proj = nn.Linear(config.intermediate_size, config.hidden_size)
        self.dropout = nn.Dropout(config.residual_dropout)

    def forward(self, hidden_states: Tensor) -> Tensor:
        hidden_states = self.c_fc(hidden_states)
        hidden_states = gelu_new(hidden_states)
        hidden_states = self.c_proj(hidden_states)
        return self.dropout(hidden_states)


class GPT2Block(nn.Module):
    """One pre-layer-normalized GPT-2 decoder block."""

    def __init__(self, config: GPT2Architecture) -> None:
        super().__init__()
        self.ln_1 = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_epsilon)
        self.attn = GPT2Attention(config)
        self.ln_2 = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_epsilon)
        self.mlp = GPT2MLP(config)

    def forward(
        self,
        hidden_states: Tensor,
        layer_past: Optional[tuple[Tensor, Tensor]] = None,
        use_cache: bool = True,
    ) -> tuple[Tensor, Optional[tuple[Tensor, Tensor]]]:
        attention_output, present = self.attn(
            self.ln_1(hidden_states), layer_past=layer_past, use_cache=use_cache
        )
        hidden_states = hidden_states + attention_output
        hidden_states = hidden_states + self.mlp(self.ln_2(hidden_states))
        return hidden_states, present


@dataclass
class GPT2Output:
    logits: Tensor
    past_key_values: Optional[tuple[tuple[Tensor, Tensor], ...]]
    loss: Optional[Tensor] = None


class GPT2LMHeadModel(nn.Module):
    """Standalone GPT-2 language model with a Hugging Face-compatible layout."""

    def __init__(self, config: GPT2Architecture) -> None:
        super().__init__()
        self.config = config
        self.wte = nn.Embedding(config.vocab_size, config.hidden_size)
        self.wpe = nn.Embedding(config.context_length, config.hidden_size)
        self.drop = nn.Dropout(config.embedding_dropout)
        self.h = nn.ModuleList(
            [GPT2Block(config) for _ in range(config.num_layers)]
        )
        self.ln_f = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_epsilon)
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)
        self._initialize_weights(config.initializer_range)
        self.lm_head.weight = self.wte.weight

    def _initialize_weights(self, initializer_range: float) -> None:
        for module in self.modules():
            if isinstance(module, (nn.Linear, nn.Embedding)):
                nn.init.normal_(module.weight, mean=0.0, std=initializer_range)
                if isinstance(module, nn.Linear) and module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, nn.LayerNorm):
                nn.init.ones_(module.weight)
                nn.init.zeros_(module.bias)

    def forward(
        self,
        input_ids: Tensor,
        past_key_values: Optional[tuple[tuple[Tensor, Tensor], ...]] = None,
        labels: Optional[Tensor] = None,
        use_cache: bool = True,
    ) -> GPT2Output:
        if input_ids.ndim != 2:
            raise ValueError("input_ids must have shape [batch, sequence]")

        batch, sequence_length = input_ids.shape
        if past_key_values is None:
            layer_past_values = tuple(None for _ in self.h)
            past_length = 0
        else:
            if len(past_key_values) != len(self.h):
                raise ValueError("past_key_values has the wrong layer count")
            layer_past_values = past_key_values
            past_length = past_key_values[0][0].size(-2)

        total_length = past_length + sequence_length
        if total_length > self.config.context_length:
            raise ValueError(
                f"sequence length {total_length} exceeds GPT-2 context "
                f"window {self.config.context_length}"
            )

        position_ids = torch.arange(
            past_length, total_length, dtype=torch.long, device=input_ids.device
        ).unsqueeze(0).expand(batch, -1)
        hidden_states = self.wte(input_ids) + self.wpe(position_ids)
        hidden_states = self.drop(hidden_states)

        presents: list[tuple[Tensor, Tensor]] = []
        for block, layer_past in zip(self.h, layer_past_values):
            hidden_states, present = block(
                hidden_states, layer_past=layer_past, use_cache=use_cache
            )
            if use_cache:
                assert present is not None
                presents.append(present)

        logits = self.lm_head(self.ln_f(hidden_states))
        loss = None
        if labels is not None:
            if labels.shape != input_ids.shape:
                raise ValueError("labels must have the same shape as input_ids")
            loss = F.cross_entropy(
                logits[..., :-1, :].contiguous().view(-1, logits.size(-1)),
                labels[..., 1:].contiguous().view(-1),
            )

        return GPT2Output(
            logits=logits,
            past_key_values=tuple(presents) if use_cache else None,
            loss=loss,
        )

    @torch.inference_mode()
    def generate_greedy(self, input_ids: Tensor, max_new_tokens: int) -> Tensor:
        if input_ids.ndim != 2 or input_ids.size(1) == 0:
            raise ValueError("input_ids must be a non-empty [batch, sequence] tensor")
        if max_new_tokens < 0:
            raise ValueError("max_new_tokens must be non-negative")
        if input_ids.size(1) + max_new_tokens > self.config.context_length:
            raise ValueError("generation would exceed the GPT-2 context window")

        generated = input_ids
        output = self(input_ids, use_cache=True)
        past = output.past_key_values
        next_token = output.logits[:, -1, :].argmax(dim=-1, keepdim=True)
        for index in range(max_new_tokens):
            generated = torch.cat((generated, next_token), dim=1)
            if index + 1 == max_new_tokens:
                break
            output = self(next_token, past_key_values=past, use_cache=True)
            past = output.past_key_values
            next_token = output.logits[:, -1, :].argmax(dim=-1, keepdim=True)
        return generated


def _copy_huggingface_weights(target: GPT2LMHeadModel, source: nn.Module) -> None:
    """Copy Hugging Face Conv1D weights into ordinary torch Linear layers."""

    with torch.no_grad():
        target.wte.weight.copy_(source.transformer.wte.weight)
        target.wpe.weight.copy_(source.transformer.wpe.weight)
        target.ln_f.weight.copy_(source.transformer.ln_f.weight)
        target.ln_f.bias.copy_(source.transformer.ln_f.bias)
        for target_block, source_block in zip(target.h, source.transformer.h):
            target_block.ln_1.weight.copy_(source_block.ln_1.weight)
            target_block.ln_1.bias.copy_(source_block.ln_1.bias)
            target_block.ln_2.weight.copy_(source_block.ln_2.weight)
            target_block.ln_2.bias.copy_(source_block.ln_2.bias)
            target_block.attn.c_attn.weight.copy_(source_block.attn.c_attn.weight.t())
            target_block.attn.c_attn.bias.copy_(source_block.attn.c_attn.bias)
            target_block.attn.c_proj.weight.copy_(source_block.attn.c_proj.weight.t())
            target_block.attn.c_proj.bias.copy_(source_block.attn.c_proj.bias)
            target_block.mlp.c_fc.weight.copy_(source_block.mlp.c_fc.weight.t())
            target_block.mlp.c_fc.bias.copy_(source_block.mlp.c_fc.bias)
            target_block.mlp.c_proj.weight.copy_(source_block.mlp.c_proj.weight.t())
            target_block.mlp.c_proj.bias.copy_(source_block.mlp.c_proj.bias)
        target.lm_head.weight = target.wte.weight


def load_pretrained_gpt2(
    model_id: str = "gpt2",
    device: torch.device | str = "cpu",
    dtype: torch.dtype = torch.float32,
    local_files_only: bool = False,
) -> tuple[GPT2LMHeadModel, object]:
    """Load official GPT-2 weights and tokenizer through Hugging Face."""

    try:
        from transformers import AutoTokenizer, GPT2LMHeadModel as HFGPT2LMHeadModel
    except ImportError as exc:
        raise RuntimeError(
            "This PoC needs transformers. Install it with "
            "`python -m pip install -r requirements-ml.txt`."
        ) from exc

    source = HFGPT2LMHeadModel.from_pretrained(
        model_id, local_files_only=local_files_only
    )
    tokenizer = AutoTokenizer.from_pretrained(
        model_id, local_files_only=local_files_only, use_fast=True
    )
    config = GPT2Architecture.from_huggingface(source.config)
    target = GPT2LMHeadModel(config)
    _copy_huggingface_weights(target, source)
    del source
    target.to(device=device, dtype=dtype)
    target.eval()
    return target, tokenizer


def _resolve_device(value: str) -> torch.device:
    if value == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    device = torch.device(value)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but torch.cuda.is_available() is false")
    return device


def _resolve_dtype(value: str, device: torch.device) -> torch.dtype:
    if value == "float32":
        return torch.float32
    if value == "float16":
        return torch.float16
    if value == "bfloat16":
        return torch.bfloat16
    if value == "auto":
        return torch.float16 if device.type == "cuda" else torch.float32
    raise ValueError(f"unsupported dtype: {value}")


def _synchronize(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    
    file = open("tinyshakespeare.txt", "r")
    prompt = file.read()[:1023]
    file.close()
    parser.add_argument("--model-id", default="gpt2")
    parser.add_argument("--prompt", default=prompt)
    parser.add_argument("--max-new-tokens", type=int, default=32)
    parser.add_argument("--device", default="auto", help="auto, cpu, cuda, or cuda:N")
    parser.add_argument(
        "--dtype",
        choices=("auto", "float32", "float16", "bfloat16"),
        default="float32",
    )
    parser.add_argument("--local-files-only", action="store_true")
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args(argv)

    torch.manual_seed(args.seed)
    device = _resolve_device(args.device)
    dtype = _resolve_dtype(args.dtype, device)
    model, tokenizer = load_pretrained_gpt2(
        args.model_id,
        device=device,
        dtype=dtype,
        local_files_only=args.local_files_only,
    )
    input_ids = tokenizer(args.prompt, return_tensors="pt")["input_ids"].to(device)

    _synchronize(device)
    start = time.perf_counter()
    with torch.inference_mode():
        output_ids = model.generate_greedy(input_ids, args.max_new_tokens)
    _synchronize(device)
    elapsed = time.perf_counter() - start

    generated_ids = output_ids[0, input_ids.size(1) :]
    generated_count = int(generated_ids.numel())
    print("mode = inference")
    print("model id =", args.model_id)
    print("device =", device)
    print("dtype =", dtype)
    print(
        "GPT2 config = vocab",
        model.config.vocab_size,
        "context",
        model.config.context_length,
        "hidden",
        model.config.hidden_size,
        "heads",
        model.config.num_heads,
        "ff",
        model.config.intermediate_size,
        "layers",
        model.config.num_layers,
    )
    print("parameters =", sum(parameter.numel() for parameter in model.parameters()))
    print("prompt tokens =", int(input_ids.size(1)))
    print("generated tokens =", generated_count)
    print("elapsed seconds =", f"{elapsed:.6f}")
    if generated_count:
        print("milliseconds per token =", f"{elapsed * 1000.0 / generated_count:.6f}")
        print("tokens per second =", f"{generated_count / elapsed:.6f}")
    print("generated text =", tokenizer.decode(output_ids[0], skip_special_tokens=True))
    print("[ok] PyTorch GPT2 checkpoint inference")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())





"""
python gpt2_pytorch.py \
    --device cpu \
    --dtype float16 \
    --max-new-tokens 100
"""
