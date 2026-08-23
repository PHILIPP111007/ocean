from pathlib import Path
import subprocess
import tempfile


ROOT = Path(__file__).resolve().parents[1]


C_SOURCE = r'''
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include "std/tensor/tensor_runtime.h"

static void check_close(float actual, float expected, const char *name) {
    if (fabsf(actual - expected) > 1e-5f) {
        fprintf(stderr, "%s: %.8f != %.8f\n", name, actual, expected);
        exit(1);
    }
}

int main(void) {
    size_t query_shape[4] = {1, 1, 2, 2};
    size_t key_shape[4] = {1, 1, 4, 2};
    ocean_tensor_handle_t query = ocean_tensor_zeros_nd(
        query_shape, 4, "float32", "cpu"
    );
    ocean_tensor_handle_t key = ocean_tensor_zeros_nd(
        key_shape, 4, "float32", "cpu"
    );
    ocean_tensor_handle_t value = ocean_tensor_zeros_nd(
        key_shape, 4, "float32", "cpu"
    );
    const float query_data[4] = {1.0f, 0.0f, 0.0f, 1.0f};
    const float key_data[8] = {
        1.0f, 0.0f, 0.0f, 1.0f,
        -1.0f, 0.0f, 0.0f, -1.0f,
    };
    const float value_data[8] = {
        10.0f, 100.0f, 20.0f, 200.0f,
        30.0f, 300.0f, 40.0f, 400.0f,
    };
    for (size_t i = 0; i < 4; ++i) {
        ocean_tensor_set_flat_f32(query, i, query_data[i]);
    }
    for (size_t i = 0; i < 8; ++i) {
        ocean_tensor_set_flat_f32(key, i, key_data[i]);
        ocean_tensor_set_flat_f32(value, i, value_data[i]);
    }

    ocean_tensor_handle_t output = ocean_tensor_sparse_attention(
        query, key, value, 2, 1.0, 0, false
    );
    float weight_high = expf(1.0f) / (expf(1.0f) + 1.0f);
    float weight_low = 1.0f / (expf(1.0f) + 1.0f);
    check_close(
        ocean_tensor_get_flat_f32(output, 0),
        10.0f * weight_high + 20.0f * weight_low,
        "top-k q0 dim0"
    );
    check_close(
        ocean_tensor_get_flat_f32(output, 1),
        100.0f * weight_high + 200.0f * weight_low,
        "top-k q0 dim1"
    );
    check_close(
        ocean_tensor_get_flat_f32(output, 2),
        20.0f * weight_high + 10.0f * weight_low,
        "top-k q1 dim0"
    );
    check_close(
        ocean_tensor_get_flat_f32(output, 3),
        200.0f * weight_high + 100.0f * weight_low,
        "top-k q1 dim1"
    );

    ocean_tensor_handle_t causal = ocean_tensor_sparse_attention(
        query, key, value, 4, 1.0, 0, true
    );
    check_close(ocean_tensor_get_flat_f32(causal, 0), 10.0f, "causal q0 dim0");
    check_close(ocean_tensor_get_flat_f32(causal, 1), 100.0f, "causal q0 dim1");
    check_close(
        ocean_tensor_get_flat_f32(causal, 2),
        10.0f * weight_low + 20.0f * weight_high,
        "causal q1 dim0"
    );
    check_close(
        ocean_tensor_get_flat_f32(causal, 3),
        100.0f * weight_low + 200.0f * weight_high,
        "causal q1 dim1"
    );

    ocean_tensor_release(causal);
    ocean_tensor_release(output);
    ocean_tensor_release(value);
    ocean_tensor_release(key);
    ocean_tensor_release(query);
    puts("SparseAttention CPU reference: OK");
    return 0;
}
'''


def test_sparse_attention_cpu_reference():
    with tempfile.TemporaryDirectory(prefix="ocean_sparse_attention_") as td:
        td = Path(td)
        source = td / "test.c"
        binary = td / "test"
        source.write_text(C_SOURCE, encoding="utf-8")
        subprocess.run([
            "gcc", "-std=c11", "-O2", "-Wall", "-Wextra", "-Wpedantic", "-Werror",
            "-I", str(ROOT), str(source),
            str(ROOT / "std/tensor/tensor_runtime.c"),
            "-lm", "-o", str(binary),
        ], check=True)
        completed = subprocess.run(
            [str(binary)], check=True, capture_output=True, text=True
        )
        assert "SparseAttention CPU reference: OK" in completed.stdout
