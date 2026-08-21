from pathlib import Path
import subprocess


ROOT = Path(__file__).resolve().parents[1]


TERNARY_SOURCE = r'''
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include "std/tensor/tensor_runtime.h"

int main(void) {
    size_t shape[2] = {2, 4};
    ocean_tensor_handle_t input = ocean_tensor_zeros_nd(
        shape, 2, "float32", "cpu"
    );
    const float values[8] = {
        -2.0f, -0.1f, 0.1f, 0.6f,
        1.2f, -0.4f, 0.0f, 0.3f,
    };
    for (size_t index = 0; index < 8; ++index) {
        ocean_tensor_set_flat_f32(input, index, values[index]);
    }

    ocean_tensor_handle_t output = ocean_tensor_ternary_quantize(input);
    const float scale = 4.7f / 8.0f;
    const float expected[8] = {
        -scale, 0.0f, 0.0f, scale,
        scale, -scale, 0.0f, scale,
    };
    for (size_t index = 0; index < 8; ++index) {
        float actual = ocean_tensor_get_flat_f32(output, index);
        if (fabsf(actual - expected[index]) > 1e-6f) {
            fprintf(stderr, "ternary mismatch at %zu: %.8f != %.8f\n",
                    index, actual, expected[index]);
            return 1;
        }
    }

    ocean_tensor_release(output);
    ocean_tensor_release(input);
    puts("Ternary quantize v0.1 CPU: OK");
    return 0;
}
'''


def test_ternary_quantize_v01_cpu_runtime(tmp_path):
    source = tmp_path / "ternary_quantize_v01.c"
    binary = tmp_path / "ternary_quantize_v01"
    source.write_text(TERNARY_SOURCE, encoding="utf-8")
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
    assert "Ternary quantize v0.1 CPU: OK" in result.stdout
