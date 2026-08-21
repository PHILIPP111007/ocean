from pathlib import Path
import subprocess


ROOT = Path(__file__).resolve().parents[1]


EMBEDDING_SOURCE = r'''
#include <math.h>
#include <stdio.h>
#include "std/tensor/tensor_runtime.h"

static void expect_close(float actual, float expected, const char *name) {
    if (fabsf(actual - expected) > 1e-6f) {
        fprintf(stderr, "%s: %.8f != %.8f\n", name, actual, expected);
        _Exit(1);
    }
}

int main(void) {
    size_t weight_shape[2] = {4, 3};
    ocean_tensor_handle_t weight = ocean_tensor_zeros_nd(
        weight_shape, 2, "float32", "cpu"
    );
    for (size_t index = 0; index < 12; ++index) {
        ocean_tensor_set_flat_f32(weight, index, (float)index + 0.5f);
    }

    size_t index_shape[1] = {4};
    ocean_tensor_handle_t indices = ocean_tensor_zeros_nd(
        index_shape, 1, "int64", "cpu"
    );
    const int64_t tokens[4] = {0, 2, 2, 3};
    for (size_t index = 0; index < 4; ++index) {
        ocean_tensor_set_flat_i64(indices, index, tokens[index]);
    }

    ocean_tensor_handle_t output = ocean_tensor_embedding_forward(
        weight, indices
    );
    const float expected_output[12] = {
        0.5f, 1.5f, 2.5f,
        6.5f, 7.5f, 8.5f,
        6.5f, 7.5f, 8.5f,
        9.5f, 10.5f, 11.5f,
    };
    for (size_t index = 0; index < 12; ++index) {
        expect_close(
            ocean_tensor_get_flat_f32(output, index),
            expected_output[index],
            "embedding forward"
        );
    }

    ocean_tensor_handle_t upstream = ocean_tensor_zeros_nd(
        weight_shape, 2, "float32", "cpu"
    );
    for (size_t index = 0; index < 12; ++index) {
        ocean_tensor_set_flat_f32(upstream, index, (float)index + 1.0f);
    }
    ocean_tensor_handle_t gradient = ocean_tensor_embedding_backward(
        upstream, indices, 4, 3
    );
    const float expected_gradient[12] = {
        1.0f, 2.0f, 3.0f,
        0.0f, 0.0f, 0.0f,
        11.0f, 13.0f, 15.0f,
        10.0f, 11.0f, 12.0f,
    };
    for (size_t index = 0; index < 12; ++index) {
        expect_close(
            ocean_tensor_get_flat_f32(gradient, index),
            expected_gradient[index],
            "embedding backward"
        );
    }

    ocean_tensor_release(gradient);
    ocean_tensor_release(upstream);
    ocean_tensor_release(output);
    ocean_tensor_release(indices);
    ocean_tensor_release(weight);
    puts("Embedding v0.1 CPU: OK");
    return 0;
}
'''


def test_embedding_v01_cpu_runtime(tmp_path):
    source = tmp_path / "embedding_v01.c"
    binary = tmp_path / "embedding_v01"
    source.write_text(EMBEDDING_SOURCE, encoding="utf-8")
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
    assert "Embedding v0.1 CPU: OK" in result.stdout
