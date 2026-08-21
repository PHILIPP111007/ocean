from pathlib import Path
import subprocess


ROOT = Path(__file__).resolve().parents[1]


CROSS_ENTROPY_SOURCE = r'''
#include <math.h>
#include <stdio.h>
#include "std/tensor/tensor_runtime.h"

static void expect_close(float actual, float expected, const char *name) {
    if (fabsf(actual - expected) > 2e-5f) {
        fprintf(stderr, "%s: %.8f != %.8f\n", name, actual, expected);
        _Exit(1);
    }
}

int main(void) {
    size_t logits_shape[2] = {2, 3};
    ocean_tensor_handle_t logits = ocean_tensor_zeros_nd(
        logits_shape, 2, "float32", "cpu"
    );
    const float values[6] = {1.0f, 2.0f, 3.0f, 1.0f, 0.0f, -1.0f};
    for (size_t i = 0; i < 6; ++i) {
        ocean_tensor_set_flat_f32(logits, i, values[i]);
    }

    size_t target_shape[1] = {2};
    ocean_tensor_handle_t targets = ocean_tensor_zeros_nd(
        target_shape, 1, "int64", "cpu"
    );
    ocean_tensor_set_flat_i64(targets, 0, 2);
    ocean_tensor_set_flat_i64(targets, 1, 0);

    ocean_tensor_handle_t probabilities = NULL;
    ocean_tensor_handle_t loss = ocean_tensor_cross_entropy_forward(
        logits, targets, &probabilities
    );
    float row0_denominator = expf(1.0f) + expf(2.0f) + expf(3.0f);
    float row1_denominator = expf(1.0f) + 1.0f + expf(-1.0f);
    const float expected_probabilities[6] = {
        expf(1.0f) / row0_denominator,
        expf(2.0f) / row0_denominator,
        expf(3.0f) / row0_denominator,
        expf(1.0f) / row1_denominator,
        1.0f / row1_denominator,
        expf(-1.0f) / row1_denominator,
    };
    for (size_t i = 0; i < 6; ++i) {
        expect_close(
            ocean_tensor_get_flat_f32(probabilities, i),
            expected_probabilities[i],
            "CrossEntropy probabilities"
        );
    }
    float expected_loss = (
        logf(row0_denominator) - 3.0f
        + logf(row1_denominator) - 1.0f
    ) * 0.5f;
    expect_close(
        ocean_tensor_get_flat_f32(loss, 0), expected_loss,
        "CrossEntropy loss"
    );

    ocean_tensor_handle_t upstream = ocean_tensor_zeros(1, 1, "cpu");
    ocean_tensor_fill(upstream, 1.0);
    ocean_tensor_handle_t gradient = ocean_tensor_cross_entropy_backward(
        upstream, probabilities, targets
    );
    for (size_t row = 0; row < 2; ++row) {
        for (size_t cls = 0; cls < 3; ++cls) {
            float expected = expected_probabilities[row * 3 + cls] * 0.5f;
            if ((row == 0 && cls == 2) || (row == 1 && cls == 0)) {
                expected -= 0.5f;
            }
            expect_close(
                ocean_tensor_get_flat_f32(gradient, row * 3 + cls),
                expected,
                "CrossEntropy gradient"
            );
        }
    }

    ocean_tensor_release(gradient);
    ocean_tensor_release(upstream);
    ocean_tensor_release(loss);
    ocean_tensor_release(probabilities);
    ocean_tensor_release(targets);
    ocean_tensor_release(logits);
    puts("CrossEntropy v0.1 CPU: OK");
    return 0;
}
'''


def test_cross_entropy_v01_cpu_runtime(tmp_path):
    source = tmp_path / "cross_entropy_v01.c"
    binary = tmp_path / "cross_entropy_v01"
    source.write_text(CROSS_ENTROPY_SOURCE, encoding="utf-8")
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
    assert "CrossEntropy v0.1 CPU: OK" in result.stdout
