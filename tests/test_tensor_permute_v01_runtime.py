from __future__ import annotations

import subprocess
from pathlib import Path


def test_tensor_permute_v01_forward_backward(tmp_path):
    root = Path(__file__).resolve().parents[1]
    source = tmp_path / "tensor_permute_v01.c"
    binary = tmp_path / "tensor_permute_v01"

    source.write_text(
        r"""
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "std/tensor/tensor_runtime.h"
#include "std/tensor/autograd_runtime.h"

#define N 24

static void fail(const char *message) {
    fprintf(stderr, "permute v0.1 failed: %s\n", message);
    exit(1);
}

static ocean_tensor_handle_t input_tensor(const float *data) {
    const size_t shape[4] = {2, 3, 2, 2};
    const size_t strides[4] = {12, 4, 2, 1};
    return ocean_tensor_from_cpu_strided(
        data, shape, strides, 4, "float32", "cpu"
    );
}

static ocean_tensor_handle_t target_tensor(const float *data) {
    const size_t shape[4] = {2, 2, 3, 2};
    const size_t strides[4] = {12, 6, 2, 1};
    return ocean_tensor_from_cpu_strided(
        data, shape, strides, 4, "float32", "cpu"
    );
}

static double loss_value(
    const float *input_data,
    const float *target_data
) {
    const int axes[4] = {0, 2, 1, 3};

    ocean_tensor_handle_t x = input_tensor(input_data);
    ocean_tensor_handle_t target = target_tensor(target_data);
    ocean_tensor_handle_t y =
        ocean_autograd_permute(x, axes, 4);
    ocean_tensor_handle_t loss =
        ocean_autograd_mse_loss(y, target);

    double value = ocean_tensor_item(loss);

    ocean_tensor_release(loss);
    ocean_tensor_release(y);
    ocean_tensor_release(target);
    ocean_tensor_release(x);
    return value;
}

int main(void) {
    float data[N];
    float target_data[N];

    for (int i = 0; i < N; ++i) {
        data[i] = (float)(0.03 * (double)(i + 1));
        target_data[i] = (float)(
            0.02 * sin((double)(i + 3) * 0.31)
        );
    }

    const int axes[4] = {0, 2, 1, 3};
    const int negative_axes[4] = {0, -2, 1, -1};

    ocean_tensor_handle_t x = input_tensor(data);
    ocean_tensor_handle_t y =
        ocean_tensor_permute(x, axes, 4);
    ocean_tensor_handle_t yn =
        ocean_tensor_permute(x, negative_axes, 4);

    if (
        ocean_tensor_shape(y, 0) != 2
        || ocean_tensor_shape(y, 1) != 2
        || ocean_tensor_shape(y, 2) != 3
        || ocean_tensor_shape(y, 3) != 2
    ) {
        fail("forward shape mismatch");
    }

    for (size_t i = 0; i < (size_t)N; ++i) {
        double a = ocean_tensor_get_flat_f32(y, i);
        double b = ocean_tensor_get_flat_f32(yn, i);
        if (fabs(a - b) > 1e-7) {
            fail("negative-axis permutation mismatch");
        }
    }

    /* Check exact coordinate mapping:
       output[b, d2, d1, d3] = input[b, d1, d2, d3]. */
    for (int b = 0; b < 2; ++b) {
        for (int d2 = 0; d2 < 2; ++d2) {
            for (int d1 = 0; d1 < 3; ++d1) {
                for (int d3 = 0; d3 < 2; ++d3) {
                    size_t input_index = (size_t)(
                        ((b * 3 + d1) * 2 + d2) * 2 + d3
                    );
                    size_t output_index = (size_t)(
                        ((b * 2 + d2) * 3 + d1) * 2 + d3
                    );
                    double actual =
                        ocean_tensor_get_flat_f32(y, output_index);
                    double expected = data[input_index];
                    if (fabs(actual - expected) > 1e-7) {
                        fail("coordinate mapping mismatch");
                    }
                }
            }
        }
    }

    ocean_tensor_release(yn);
    ocean_tensor_release(y);
    ocean_tensor_release(x);

    /* Autograd + finite difference. */
    x = input_tensor(data);
    ocean_tensor_handle_t target = target_tensor(target_data);
    ocean_autograd_set_requires_grad(x, true);

    y = ocean_autograd_permute(x, axes, 4);
    ocean_tensor_handle_t loss =
        ocean_autograd_mse_loss(y, target);
    ocean_autograd_backward(loss);

    if (!ocean_autograd_has_grad(x)) {
        fail("input gradient missing");
    }

    ocean_tensor_handle_t grad =
        ocean_autograd_grad_copy(x);

    const double step = 1e-3;
    double max_error = 0.0;

    for (int index = 0; index < N; ++index) {
        float plus_data[N];
        float minus_data[N];
        memcpy(plus_data, data, sizeof(data));
        memcpy(minus_data, data, sizeof(data));

        plus_data[index] += (float)step;
        minus_data[index] -= (float)step;

        double plus = loss_value(plus_data, target_data);
        double minus = loss_value(minus_data, target_data);
        double numeric = (plus - minus) / (2.0 * step);
        double analytic =
            ocean_tensor_get_flat_f32(grad, (size_t)index);
        double error = fabs(numeric - analytic);

        if (error > max_error) max_error = error;

        if (error > 2.5e-3) {
            fprintf(
                stderr,
                "grad[%d] numeric=% .9f analytic=% .9f error=%.9g\n",
                index,
                numeric,
                analytic,
                error
            );
            fail("finite-difference mismatch");
        }
    }

    printf("permute max gradient error = %.9g\n", max_error);
    puts("Tensor permute v0.1: OK");

    ocean_tensor_release(grad);
    ocean_tensor_release(loss);
    ocean_tensor_release(y);
    ocean_tensor_release(target);
    ocean_tensor_release(x);
    return 0;
}
""",
        encoding="utf-8",
    )

    subprocess.run(
        [
            "gcc",
            "-std=c11",
            "-O2",
            "-Wall",
            "-Wextra",
            "-Wpedantic",
            "-Werror",
            f"-I{root}",
            str(source),
            str(root / "std/tensor/autograd_runtime.c"),
            str(root / "std/tensor/tensor_runtime.c"),
            "-lm",
            "-o",
            str(binary),
        ],
        check=True,
    )

    result = subprocess.run(
        [str(binary)],
        check=True,
        capture_output=True,
        text=True,
    )

    assert "Tensor permute v0.1: OK" in result.stdout
