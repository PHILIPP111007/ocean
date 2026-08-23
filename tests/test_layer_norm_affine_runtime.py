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
    size_t input_shape[2] = {2, 3};
    size_t affine_shape[2] = {1, 3};
    ocean_tensor_handle_t input = ocean_tensor_zeros_nd(
        input_shape, 2, "float32", "cpu"
    );
    ocean_tensor_handle_t gamma = ocean_tensor_zeros_nd(
        affine_shape, 2, "float32", "cpu"
    );
    ocean_tensor_handle_t beta = ocean_tensor_zeros_nd(
        affine_shape, 2, "float32", "cpu"
    );
    const float values[6] = {-1.0f, 0.0f, 2.0f, 3.0f, 4.0f, 8.0f};
    const float gamma_values[3] = {2.0f, 3.0f, 4.0f};
    const float beta_values[3] = {-1.0f, 0.5f, 2.0f};
    for (size_t i = 0; i < 6; ++i) {
        ocean_tensor_set_flat_f32(input, i, values[i]);
    }
    for (size_t i = 0; i < 3; ++i) {
        ocean_tensor_set_flat_f32(gamma, i, gamma_values[i]);
        ocean_tensor_set_flat_f32(beta, i, beta_values[i]);
    }

    ocean_tensor_handle_t output = ocean_tensor_layer_norm_affine(
        input, gamma, beta, -1, 1e-5
    );
    for (size_t row = 0; row < 2; ++row) {
        float mean = 0.0f;
        for (size_t column = 0; column < 3; ++column) {
            mean += values[row * 3 + column];
        }
        mean /= 3.0f;
        float variance = 0.0f;
        for (size_t column = 0; column < 3; ++column) {
            float delta = values[row * 3 + column] - mean;
            variance += delta * delta;
        }
        variance /= 3.0f;
        for (size_t column = 0; column < 3; ++column) {
            float normalized = (values[row * 3 + column] - mean)
                / sqrtf(variance + 1e-5f);
            float expected = normalized * gamma_values[column]
                + beta_values[column];
            char name[64];
            snprintf(name, sizeof(name), "layer_norm_affine[%zu]", row * 3 + column);
            check_close(
                ocean_tensor_get_flat_f32(output, row * 3 + column),
                expected,
                name
            );
        }
    }

    ocean_tensor_release(output);
    ocean_tensor_release(beta);
    ocean_tensor_release(gamma);
    ocean_tensor_release(input);
    puts("LayerNorm affine runtime: OK");
    return 0;
}
'''


def test_layer_norm_affine_runtime_cpu():
    with tempfile.TemporaryDirectory(prefix="ocean_layer_norm_affine_") as td:
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
        assert "LayerNorm affine runtime: OK" in completed.stdout
