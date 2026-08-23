from __future__ import annotations

import subprocess
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


C_SOURCE = r'''
#include <stdio.h>
#include "std/tensor/tensor_runtime.h"

int main(void) {
    const size_t cache_shape[4] = {1, 2, 4, 3};
    const size_t value_shape[4] = {1, 2, 2, 3};
    const size_t value_strides[4] = {12, 6, 3, 1};
    const float values[12] = {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12};

    ocean_tensor_handle_t cache = ocean_tensor_zeros_nd(
        cache_shape, 4, "float32", "cpu"
    );
    ocean_tensor_handle_t value = ocean_tensor_from_cpu_strided(
        values, value_shape, value_strides, 4, "float32", "cpu"
    );

    ocean_tensor_cache_write(cache, value, 1);
    const size_t expected_indices[12] = {
        3, 4, 5, 6, 7, 8, 15, 16, 17, 18, 19, 20
    };
    for (size_t index = 0; index < 12; ++index) {
        if (ocean_tensor_get_flat_f32(cache, expected_indices[index]) != values[index]) {
            return 1;
        }
    }

    const size_t logits_shape[2] = {1, 5};
    const size_t logits_strides[2] = {5, 1};
    const float logits_values[5] = {0.5f, 2.0f, -1.0f, 4.0f, 4.0f};
    ocean_tensor_handle_t logits = ocean_tensor_from_cpu_strided(
        logits_values, logits_shape, logits_strides, 2, "float32", "cpu"
    );
    if (ocean_tensor_argmax(logits) != 3) return 1;

    ocean_tensor_release(logits);
    ocean_tensor_release(value);
    ocean_tensor_release(cache);
    puts("KV cache write v0.1 CPU: OK");
    return 0;
}
'''


def test_kv_cache_write_cpu_runtime(tmp_path):
    source = tmp_path / "kv_cache_v01.c"
    binary = tmp_path / "kv_cache_v01"
    source.write_text(C_SOURCE, encoding="utf-8")
    subprocess.run(
        [
            "gcc", "-std=c11", "-O2", "-Wall", "-Wextra", "-Wpedantic",
            "-Werror", "-I", str(ROOT), str(source),
            str(ROOT / "std/tensor/tensor_runtime.c"),
            "-lm", "-o", str(binary),
        ],
        check=True,
    )
    result = subprocess.run(
        [str(binary)], check=True, capture_output=True, text=True
    )
    assert "KV cache write v0.1 CPU: OK" in result.stdout
