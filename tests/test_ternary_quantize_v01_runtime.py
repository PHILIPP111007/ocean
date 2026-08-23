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

    double packed_scale = ocean_tensor_ternary_scale(input);
    if (fabs(packed_scale - (double)scale) > 1e-6) return 1;
    ocean_tensor_handle_t packed = ocean_tensor_ternary_pack(
        input, packed_scale, false
    );
    if (ocean_tensor_shape(packed, 0) != 2 ||
        ocean_tensor_shape(packed, 1) != 1) return 1;
    if (ocean_tensor_get_flat_i32(packed, 0) != 66 ||
        ocean_tensor_get_flat_i32(packed, 1) != 73) return 1;

    size_t vector_shape[2] = {1, 2};
    ocean_tensor_handle_t vector = ocean_tensor_zeros_nd(
        vector_shape, 2, "float32", "cpu"
    );
    ocean_tensor_set_flat_f32(vector, 0, 1.0f);
    ocean_tensor_set_flat_f32(vector, 1, 2.0f);
    ocean_tensor_handle_t packed_result = ocean_tensor_packed_matmul_inference(
        vector, packed, packed_scale, 4
    );
    const float packed_expected[4] = {
        scale, -2.0f * scale, 0.0f, 3.0f * scale,
    };
    for (size_t index = 0; index < 4; ++index) {
        if (fabsf(ocean_tensor_get_flat_f32(packed_result, index) - packed_expected[index]) > 1e-6f) {
            return 1;
        }
    }

    size_t bias_shape[2] = {1, 4};
    ocean_tensor_handle_t bias = ocean_tensor_zeros_nd(
        bias_shape, 2, "float32", "cpu"
    );
    for (size_t index = 0; index < 4; ++index) {
        ocean_tensor_set_flat_f32(bias, index, (float)index);
    }
    ocean_tensor_handle_t fused_qkv = ocean_tensor_packed_qkv_inference(
        vector, packed, packed_scale, bias,
        packed, packed_scale, bias,
        packed, packed_scale, bias, 4
    );
    if (ocean_tensor_shape(fused_qkv, 0) != 1 ||
        ocean_tensor_shape(fused_qkv, 1) != 12) return 1;
    for (size_t projection = 0; projection < 3; ++projection) {
        for (size_t index = 0; index < 4; ++index) {
            float expected = packed_expected[index] + (float)index;
            size_t fused_index = projection * 4 + index;
            if (fabsf(ocean_tensor_get_flat_f32(fused_qkv, fused_index) - expected) > 1e-6f) {
                return 1;
            }
        }
    }

    ocean_tensor_release(fused_qkv);
    ocean_tensor_release(bias);
    ocean_tensor_release(packed_result);
    ocean_tensor_release(vector);
    ocean_tensor_release(packed);
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
