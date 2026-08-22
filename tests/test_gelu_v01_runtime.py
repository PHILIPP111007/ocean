from pathlib import Path
import subprocess


ROOT = Path(__file__).resolve().parents[1]


GELU_SOURCE = r'''
#include <math.h>
#include <stdio.h>
#include "std/tensor/tensor_runtime.h"

int main(void) {
    size_t shape[1] = {3};
    size_t strides[1] = {1};
    const float values[3] = {-1.0f, 0.0f, 1.0f};
    const float upstream_values[3] = {1.0f, 1.0f, 1.0f};
    ocean_tensor_handle_t input = ocean_tensor_from_cpu_strided(
        values, shape, strides, 1, "float32", "cpu"
    );
    ocean_tensor_handle_t upstream = ocean_tensor_from_cpu_strided(
        upstream_values, shape, strides, 1, "float32", "cpu"
    );
    ocean_tensor_handle_t output = ocean_tensor_gelu(input);
    ocean_tensor_handle_t gradient = ocean_tensor_gelu_backward(upstream, input);

    const float expected_output[3] = {
        -0.158808f, 0.0f, 0.841192f
    };
    const float expected_gradient[3] = {
        -0.082964f, 0.5f, 1.082964f
    };
    for (size_t index = 0; index < 3; ++index) {
        if (fabsf(ocean_tensor_get_flat_f32(output, index) - expected_output[index]) > 1e-5f ||
            fabsf(ocean_tensor_get_flat_f32(gradient, index) - expected_gradient[index]) > 1e-5f) {
            return 1;
        }
    }

    ocean_tensor_release(gradient);
    ocean_tensor_release(output);
    ocean_tensor_release(upstream);
    ocean_tensor_release(input);
    puts("GELU v0.1 CPU: OK");
    return 0;
}
'''


def test_gelu_v01_cpu_runtime(tmp_path):
    source = tmp_path / "gelu_v01.c"
    binary = tmp_path / "gelu_v01"
    source.write_text(GELU_SOURCE, encoding="utf-8")
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
    assert "GELU v0.1 CPU: OK" in result.stdout
