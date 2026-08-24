"""GPT-2 Small with Ocean's sublinear-attention routing algorithm.

This is a research/reference variant of ``gpt2_pytorch.py``.  The GPT-2
weights, tokenizer, embeddings, MLPs, layer norms, residual connections and
LM head stay unchanged.  Only self-attention is replaced with the proposed
route:

1. keep an exact local window of the latest ``local_window`` tokens;
2. summarize that window with one vector per head;
3. summarize historical blocks and select ``top_k_blocks`` by cosine
   similarity to the latest-window summary;
4. always include the latest ``mandatory_recent_blocks`` blocks;
5. include one deterministic pseudo-random exploration block;
6. reuse this block route for ``refresh_interval`` query positions.

The exploration block is deterministic rather than stochastic so benchmark
results and generated text are reproducible.  For short prefixes the selected
tokens cover the complete causal prefix, so the result is exactly dense.
For long prefixes this is intentionally an approximation: it is a quality /
speed experiment, not a claim of mathematical equivalence to dense GPT-2.
"""

from __future__ import annotations

import argparse
import time
from typing import Optional, Sequence

import torch
from torch import Tensor, nn
from torch.nn import functional as F

from gpt2_pytorch import (
    GPT2Architecture,
    GPT2Block,
    GPT2LMHeadModel,
    GPT2Output,
    _copy_huggingface_weights,
    _resolve_device,
    _resolve_dtype,
    _synchronize,
)


class SublinearGPT2Attention(nn.Module):
    """GPT-2 attention with local-window plus routed historical blocks."""

    def __init__(
        self,
        config: GPT2Architecture,
        local_window: int = 100,
        block_size: int = 64,
        top_k_blocks: int = 5,
        refresh_interval: int = 50,
        mandatory_recent_blocks: int = 2,
        exploration_blocks: int = 1,
    ) -> None:
        super().__init__()
        if config.hidden_size % config.num_heads != 0:
            raise ValueError("hidden_size must be divisible by num_heads")
        if local_window <= 0 or block_size <= 0:
            raise ValueError("local_window and block_size must be positive")
        if top_k_blocks < 0 or refresh_interval <= 0:
            raise ValueError("top_k_blocks must be non-negative and refresh_interval positive")
        if mandatory_recent_blocks < 0 or exploration_blocks < 0:
            raise ValueError("block counts must be non-negative")

        self.num_heads = config.num_heads
        self.head_dim = config.hidden_size // config.num_heads
        self.max_positions = config.context_length
        self.scale = self.head_dim**-0.5
        self.local_window = local_window
        self.block_size = block_size
        self.top_k_blocks = top_k_blocks
        self.refresh_interval = refresh_interval
        self.mandatory_recent_blocks = mandatory_recent_blocks
        self.exploration_blocks = exploration_blocks

        # Keep the exact GPT-2 projection and dropout structure.
        self.c_attn = nn.Linear(config.hidden_size, 3 * config.hidden_size)
        self.c_proj = nn.Linear(config.hidden_size, config.hidden_size)
        self.attn_dropout = nn.Dropout(config.attention_dropout)
        self.resid_dropout = nn.Dropout(config.residual_dropout)

        # Route state is intentionally not part of the model checkpoint.  It
        # is reset at the beginning of every independent prompt and reused by
        # subsequent cached decode calls.
        self._route_epoch: Optional[int] = None
        self._route_blocks: Optional[list[list[int]]] = None

    def reset_route_state(self) -> None:
        self._route_epoch = None
        self._route_blocks = None

    def _split_heads(self, x: Tensor) -> Tensor:
        batch, sequence, _ = x.shape
        return x.view(batch, sequence, self.num_heads, self.head_dim).transpose(1, 2)

    def _merge_heads(self, x: Tensor) -> Tensor:
        batch, _, sequence, _ = x.shape
        return x.transpose(1, 2).contiguous().view(
            batch, sequence, self.num_heads * self.head_dim
        )

    def _build_route(
        self,
        key: Tensor,
        prefix_length: int,
        absolute_position: int,
    ) -> list[list[int]]:
        """Select historical blocks from a causal prefix.

        The route is shared across the batch and selected independently for
        each attention head.  Sharing it across a batch keeps the route compact
        and is appropriate for the single-stream autoregressive PoC.
        """

        recent_start = max(0, prefix_length - self.local_window)
        recent = key[:, :, recent_start:prefix_length, :]
        query_summary = recent.mean(dim=0).mean(dim=1)  # [heads, head_dim]

        block_count = (prefix_length + self.block_size - 1) // self.block_size
        summaries: list[Tensor] = []
        for block_index in range(block_count):
            start = block_index * self.block_size
            end = min(prefix_length, start + self.block_size)
            summaries.append(key[:, :, start:end, :].mean(dim=2).mean(dim=0))
        block_summary = torch.stack(summaries, dim=1)  # [heads, blocks, head_dim]
        query_summary = F.normalize(query_summary, dim=-1)
        block_summary = F.normalize(block_summary, dim=-1)
        cosine = (block_summary * query_summary.unsqueeze(1)).sum(dim=-1)

        top_count = min(self.top_k_blocks, block_count)
        top_blocks = cosine.topk(top_count, dim=1).indices.tolist()
        epoch = absolute_position // self.refresh_interval
        routes: list[list[int]] = []
        for head in range(self.num_heads):
            selected_blocks = set(int(index) for index in top_blocks[head])

            first_recent_block = max(
                0, block_count - self.mandatory_recent_blocks
            )
            for block_index in range(first_recent_block, block_count):
                selected_blocks.add(block_index)

            # Deterministic exploration gives the route the same effect as a
            # random block while keeping generation reproducible.
            for offset in range(self.exploration_blocks):
                if block_count:
                    exploration = (
                        epoch * 1103515245 + head * 12345 + offset * 2654435761
                    ) % block_count
                    selected_blocks.add(int(exploration))

            # Cache only historical block IDs.  The local token window is
            # rebuilt for every query because it moves forward while a route
            # is reused for refresh_interval positions.
            routes.append(sorted(selected_blocks))
        return routes

    def _route_for_query(
        self, key: Tensor, prefix_length: int, absolute_position: int
    ) -> list[list[int]]:
        epoch = absolute_position // self.refresh_interval
        if self._route_epoch != epoch or self._route_blocks is None:
            self._route_blocks = self._build_route(key, prefix_length, absolute_position)
            self._route_epoch = epoch

        recent_start = max(0, prefix_length - self.local_window)
        routes: list[list[int]] = []
        for selected_blocks in self._route_blocks:
            token_indices = set(range(recent_start, prefix_length))
            for block_index in selected_blocks:
                start = block_index * self.block_size
                end = min(prefix_length, start + self.block_size)
                token_indices.update(range(start, end))
            routes.append(sorted(token_indices))
        return routes

    def _sparse_attention(
        self,
        query: Tensor,
        key: Tensor,
        value: Tensor,
        past_length: int,
    ) -> Tensor:
        """Compute causal attention over the routed token subsets."""

        batch, _, query_length, _ = query.shape
        outputs: list[Tensor] = []
        for query_index in range(query_length):
            absolute_position = past_length + query_index
            prefix_length = absolute_position + 1
            routes = self._route_for_query(key, prefix_length, absolute_position)
            head_outputs: list[Tensor] = []
            for head in range(self.num_heads):
                indices = torch.tensor(
                    routes[head], dtype=torch.long, device=key.device
                )
                selected_key = key[:, head : head + 1].index_select(2, indices)
                selected_value = value[:, head : head + 1].index_select(2, indices)
                selected_query = query[:, head : head + 1, query_index : query_index + 1]
                scores = torch.matmul(
                    selected_query, selected_key.transpose(-1, -2)
                ) * self.scale
                weights = F.softmax(scores, dim=-1)
                weights = self.attn_dropout(weights)
                head_outputs.append(torch.matmul(weights, selected_value))
            outputs.append(torch.cat(head_outputs, dim=1))
        return torch.cat(outputs, dim=2).view(batch, self.num_heads, query_length, self.head_dim)

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

        total_length = key.size(-2)
        if total_length > self.max_positions:
            raise ValueError(
                f"sequence length {total_length} exceeds GPT-2 context "
                f"window {self.max_positions}"
            )

        attention_output = self._sparse_attention(query, key, value, past_length)
        attention_output = self._merge_heads(attention_output)
        attention_output = self.c_proj(attention_output)
        attention_output = self.resid_dropout(attention_output)
        present = (key, value) if use_cache else None
        return attention_output, present


class SublinearGPT2Block(GPT2Block):
    def __init__(
        self,
        config: GPT2Architecture,
        local_window: int,
        block_size: int,
        top_k_blocks: int,
        refresh_interval: int,
        mandatory_recent_blocks: int,
        exploration_blocks: int,
    ) -> None:
        super().__init__(config)
        self.attn = SublinearGPT2Attention(
            config,
            local_window=local_window,
            block_size=block_size,
            top_k_blocks=top_k_blocks,
            refresh_interval=refresh_interval,
            mandatory_recent_blocks=mandatory_recent_blocks,
            exploration_blocks=exploration_blocks,
        )


class SublinearGPT2LMHeadModel(GPT2LMHeadModel):
    """GPT-2 with the proposed sublinear attention mechanism."""

    def __init__(
        self,
        config: GPT2Architecture,
        local_window: int = 100,
        block_size: int = 64,
        top_k_blocks: int = 5,
        refresh_interval: int = 50,
        mandatory_recent_blocks: int = 2,
        exploration_blocks: int = 1,
    ) -> None:
        super().__init__(config)
        for index in range(config.num_layers):
            self.h[index] = SublinearGPT2Block(
                config,
                local_window=local_window,
                block_size=block_size,
                top_k_blocks=top_k_blocks,
                refresh_interval=refresh_interval,
                mandatory_recent_blocks=mandatory_recent_blocks,
                exploration_blocks=exploration_blocks,
            )

    def reset_route_state(self) -> None:
        for block in self.h:
            block.attn.reset_route_state()

    def forward(
        self,
        input_ids: Tensor,
        past_key_values: Optional[tuple[tuple[Tensor, Tensor], ...]] = None,
        labels: Optional[Tensor] = None,
        use_cache: bool = True,
    ) -> GPT2Output:
        if past_key_values is None:
            self.reset_route_state()
        return super().forward(
            input_ids,
            past_key_values=past_key_values,
            labels=labels,
            use_cache=use_cache,
        )


def load_pretrained_sublinear_gpt2(
    model_id: str,
    device: torch.device,
    dtype: torch.dtype,
    local_files_only: bool,
    local_window: int,
    block_size: int,
    top_k_blocks: int,
    refresh_interval: int,
    mandatory_recent_blocks: int,
    exploration_blocks: int,
) -> tuple[SublinearGPT2LMHeadModel, object]:
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
    model = SublinearGPT2LMHeadModel(
        config,
        local_window=local_window,
        block_size=block_size,
        top_k_blocks=top_k_blocks,
        refresh_interval=refresh_interval,
        mandatory_recent_blocks=mandatory_recent_blocks,
        exploration_blocks=exploration_blocks,
    )
    _copy_huggingface_weights(model, source)
    del source
    model.to(device=device, dtype=dtype)
    model.eval()
    return model, tokenizer


def main(argv: Optional[Sequence[str]] = None) -> int:
    file = open("tinyshakespeare.txt", "r")
    prompt = file.read()[:1023]
    file.close()
    parser = argparse.ArgumentParser(description=__doc__)
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
    parser.add_argument("--local-window", type=int, default=100)
    parser.add_argument("--block-size", type=int, default=64)
    parser.add_argument("--top-k-blocks", type=int, default=5)
    parser.add_argument("--refresh-interval", type=int, default=50)
    parser.add_argument("--mandatory-recent-blocks", type=int, default=2)
    parser.add_argument("--exploration-blocks", type=int, default=1)
    args = parser.parse_args(argv)

    device = _resolve_device(args.device)
    dtype = _resolve_dtype(args.dtype, device)
    model, tokenizer = load_pretrained_sublinear_gpt2(
        args.model_id,
        device,
        dtype,
        args.local_files_only,
        args.local_window,
        args.block_size,
        args.top_k_blocks,
        args.refresh_interval,
        args.mandatory_recent_blocks,
        args.exploration_blocks,
    )
    input_ids = tokenizer(args.prompt, return_tensors="pt")["input_ids"].to(device)

    _synchronize(device)
    start = time.perf_counter()
    with torch.inference_mode():
        output_ids = model.generate_greedy(input_ids, args.max_new_tokens)
    _synchronize(device)
    elapsed = time.perf_counter() - start

    generated_count = int(output_ids.size(1) - input_ids.size(1))
    print("mode = inference")
    print("model id =", args.model_id)
    print("attention = sublinear routed blocks")
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
    print(
        "attention config = local_window",
        args.local_window,
        "block_size",
        args.block_size,
        "top_k_blocks",
        args.top_k_blocks,
        "refresh_interval",
        args.refresh_interval,
        "mandatory_recent_blocks",
        args.mandatory_recent_blocks,
        "exploration_blocks",
        args.exploration_blocks,
    )
    print("parameters =", sum(parameter.numel() for parameter in model.parameters()))
    print("prompt tokens =", int(input_ids.size(1)))
    print("generated tokens =", generated_count)
    print("elapsed seconds =", f"{elapsed:.6f}")
    if generated_count:
        print("milliseconds per token =", f"{elapsed * 1000.0 / generated_count:.6f}")
        print("tokens per second =", f"{generated_count / elapsed:.6f}")
    print("generated text =", tokenizer.decode(output_ids[0], skip_special_tokens=True))
    print("[ok] PyTorch GPT2 sublinear-attention inference")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

"""
python gpt2_pytorch_sublinear_attention.py \
    --device cpu \
    --dtype float16 \
    --max-new-tokens 100 \
    --local-window 100 \
    --block-size 64 \
    --top-k-blocks 5 \
    --refresh-interval 50 \
    --mandatory-recent-blocks 2 \
    --exploration-blocks 1
"""