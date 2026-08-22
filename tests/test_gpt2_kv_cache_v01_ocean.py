from __future__ import annotations

import re
import shutil
import subprocess
from pathlib import Path

from main import compile_c, compile_pipeline


OCEAN_SOURCE = r'''
import "./gpt2_native_ternary_model.oc"


def main() -> int:
    var config: GPT2Config = GPT2Config(11, 8, 8, 2, 16, 1)
    var model: GPT2Ternary = GPT2Ternary(config)
    model.eval()

    var tokens: Tensor[int64] = Tensor.zeros(1, 3, "cpu")
    tokens[0, 0] = 1
    tokens[0, 1] = 4
    tokens[0, 2] = 7
    var positions: Tensor[int64] = Tensor.zeros(1, 3, "cpu")
    positions[0, 0] = 0
    positions[0, 1] = 1
    positions[0, 2] = 2

    var causal_bias: Tensor[float32] = Tensor.zeros(3, 3, "cpu")
    var row: int = 0
    while row < 3:
        var column: int = row + 1
        while column < 3:
            causal_bias[row, column] = -1000000000.0
            column = column + 1
        row = row + 1

    var previous_grad_enabled: bool = Tensor.grad_enabled()
    Tensor.set_grad_enabled(False)
    var full_logits: Tensor[float32] = model.forward(tokens, positions, causal_bias)
    var prefill_cache: GPT2KVCache = model.new_kv_cache("cpu")
    var prefill_hidden: Tensor[float32] = model.forward_prefill_hidden(tokens, positions, prefill_cache, causal_bias)
    var prefill_last: Tensor[float32] = prefill_hidden.slice(1, 2, 3, 1)
    var prefill_lm_weight: Tensor[float32] = model.tied_lm_weight()
    var prefill_logits: Tensor[float32] = prefill_last.matmul(prefill_lm_weight)
    var cache: GPT2KVCache = model.new_kv_cache("cpu")
    var index: int = 0
    var cached_logits: Tensor[float32] = Tensor.zeros(1, 1, config.vocab_size, "cpu")
    while index < 3:
        var token: Tensor[int64] = tokens.slice(1, index, index + 1, 1)
        var position: Tensor[int64] = positions.slice(1, index, index + 1, 1)
        cached_logits = model.forward_cached(token, position, cache, index)
        index = index + 1
    Tensor.set_grad_enabled(previous_grad_enabled)

    var full_last: Tensor[float32] = full_logits.slice(1, 2, 3, 1)
    var full_last_2d: Tensor[float32] = full_last.reshape([1, config.vocab_size])
    var prefill_logits_2d: Tensor[float32] = prefill_logits.reshape([1, config.vocab_size])
    var cached_logits_2d: Tensor[float32] = cached_logits.reshape([1, config.vocab_size])
    var max_prefill_difference: float64 = 0.0
    var max_difference: float64 = 0.0
    var token_index: int = 0
    while token_index < config.vocab_size:
        var prefill_difference: float64 = full_last_2d.get(0, token_index) - prefill_logits_2d.get(0, token_index)
        if prefill_difference < 0.0:
            prefill_difference = -prefill_difference
        if prefill_difference > max_prefill_difference:
            max_prefill_difference = prefill_difference
        var difference: float64 = full_last_2d.get(0, token_index) - cached_logits_2d.get(0, token_index)
        if difference < 0.0:
            difference = -difference
        if difference > max_difference:
            max_difference = difference
        token_index = token_index + 1

    print("max prefill logits difference =", max_prefill_difference)
    print("max kv logits difference =", max_difference)
    print("[ok] Ocean GPT2 KV cache equivalence")
    return 0
'''


def test_gpt2_kv_cache_matches_full_forward_cpu(tmp_path):
    root = Path(__file__).resolve().parents[1]
    shutil.copy2(
        root / "examples/ML/gpt2_native_ternary_model.oc",
        tmp_path / "gpt2_native_ternary_model.oc",
    )
    source = tmp_path / "gpt2_kv_cache_v01.oc"
    source.write_text(OCEAN_SOURCE, encoding="utf-8")
    c_path = tmp_path / "gpt2_kv_cache_v01.generated.c"
    binary = tmp_path / "gpt2_kv_cache_v01"

    compile_pipeline(root, source, c_path, quiet=True)
    compile_c(c_path, binary)
    result = subprocess.run(
        [str(binary)], check=True, capture_output=True, text=True, timeout=60
    )
    match = re.search(
        r"max kv logits difference\s*=\s*([0-9eE+.\-]+)", result.stdout
    )
    assert match is not None, result.stdout
    assert float(match.group(1)) < 1.0e-3, result.stdout
    prefill_match = re.search(
        r"max prefill logits difference\s*=\s*([0-9eE+.\-]+)", result.stdout
    )
    assert prefill_match is not None, result.stdout
    assert float(prefill_match.group(1)) < 1.0e-3, result.stdout
    assert "[ok] ocean gpt2 kv cache equivalence" in result.stdout.lower()
