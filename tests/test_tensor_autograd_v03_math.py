from __future__ import annotations

import subprocess
from pathlib import Path


def _compile(root: Path, source: Path, binary: Path) -> None:
    subprocess.run(
        [
            "gcc", "-std=c11", "-O2",
            "-Wall", "-Wextra", "-Wpedantic", "-Werror",
            f"-I{root}",
            str(source),
            str(root / "std/tensor/autograd_runtime.c"),
            str(root / "std/tensor/tensor_runtime.c"),
            "-lm",
            "-o", str(binary),
        ],
        check=True,
    )


def test_v03_math_forward_and_gradients(tmp_path):
    root = Path(__file__).resolve().parents[1]
    source = tmp_path / "v03_math.c"
    binary = tmp_path / "v03_math"

    source.write_text(
        r"""
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include "std/tensor/tensor_runtime.h"
#include "std/tensor/autograd_runtime.h"

static void check(int condition, const char *message) {
    if (!condition) {
        fprintf(stderr, "v0.3 math test failed: %s\n", message);
        exit(1);
    }
}

static ocean_tensor_handle_t tensor_from(const float *data) {
    size_t shape[2] = {2, 3};
    size_t strides[2] = {3, 1};
    return ocean_tensor_from_cpu_strided(
        data, shape, strides, 2, "float32", "cpu"
    );
}

static double mse_softmax(const float *data, const float *target_data) {
    ocean_tensor_handle_t x = tensor_from(data);
    ocean_tensor_handle_t target = tensor_from(target_data);
    ocean_tensor_handle_t y = ocean_autograd_softmax(x, -1);
    ocean_tensor_handle_t loss = ocean_autograd_mse_loss(y, target);
    double value = ocean_tensor_item(loss);
    ocean_tensor_release(loss);
    ocean_tensor_release(y);
    ocean_tensor_release(target);
    ocean_tensor_release(x);
    return value;
}

static double mse_layer_norm(const float *data, const float *target_data) {
    ocean_tensor_handle_t x = tensor_from(data);
    ocean_tensor_handle_t target = tensor_from(target_data);
    ocean_tensor_handle_t y = ocean_autograd_layer_norm(x, -1, 1e-5);
    ocean_tensor_handle_t loss = ocean_autograd_mse_loss(y, target);
    double value = ocean_tensor_item(loss);
    ocean_tensor_release(loss);
    ocean_tensor_release(y);
    ocean_tensor_release(target);
    ocean_tensor_release(x);
    return value;
}

int main(void) {
    const float input_data[6] = {
        0.7f, 1.2f, 2.0f,
        1.5f, 0.4f, 2.3f
    };
    const float target_data[6] = {
        0.1f, 0.4f, 0.5f,
        0.2f, 0.3f, 0.5f
    };

    ocean_tensor_handle_t x = tensor_from(input_data);

    ocean_tensor_handle_t e = ocean_autograd_exp(x);
    ocean_tensor_handle_t l = ocean_autograd_log(x);
    ocean_tensor_handle_t s = ocean_autograd_sqrt(x);
    ocean_tensor_handle_t p = ocean_autograd_pow(x, 2.0);
    ocean_tensor_handle_t sm = ocean_autograd_softmax(x, -1);
    ocean_tensor_handle_t ln = ocean_autograd_layer_norm(x, -1, 1e-5);

    check(fabs(ocean_tensor_get_flat_f32(e, 0) - expf(0.7f)) < 1e-5, "exp");
    check(fabs(ocean_tensor_get_flat_f32(l, 1) - logf(1.2f)) < 1e-5, "log");
    check(fabs(ocean_tensor_get_flat_f32(s, 2) - sqrtf(2.0f)) < 1e-5, "sqrt");
    check(fabs(ocean_tensor_get_flat_f32(p, 3) - 2.25f) < 1e-5, "pow");

    for (int row = 0; row < 2; ++row) {
        double total = 0.0;
        for (int col = 0; col < 3; ++col) {
            total += ocean_tensor_get_2d(sm, row, col);
        }
        check(fabs(total - 1.0) < 1e-5, "softmax normalization");
    }

    for (int row = 0; row < 2; ++row) {
        double mean = 0.0;
        for (int col = 0; col < 3; ++col) {
            mean += ocean_tensor_get_2d(ln, row, col);
        }
        mean /= 3.0;
        check(fabs(mean) < 1e-5, "LayerNorm zero mean");
    }

    ocean_tensor_release(ln);
    ocean_tensor_release(sm);
    ocean_tensor_release(p);
    ocean_tensor_release(s);
    ocean_tensor_release(l);
    ocean_tensor_release(e);
    ocean_tensor_release(x);

    x = tensor_from(input_data);
    ocean_autograd_set_requires_grad(x, true);
    ocean_tensor_handle_t target = tensor_from(target_data);
    sm = ocean_autograd_softmax(x, -1);
    ocean_tensor_handle_t loss = ocean_autograd_mse_loss(sm, target);
    ocean_autograd_backward(loss);
    ocean_tensor_handle_t grad = ocean_autograd_grad_copy(x);

    const double h = 1e-3;
    for (int i = 0; i < 6; ++i) {
        float plus[6];
        float minus[6];
        for (int j = 0; j < 6; ++j) {
            plus[j] = input_data[j];
            minus[j] = input_data[j];
        }
        plus[i] += (float)h;
        minus[i] -= (float)h;

        double numeric =
            (mse_softmax(plus, target_data) - mse_softmax(minus, target_data))
            / (2.0 * h);
        double analytic = ocean_tensor_get_flat_f32(grad, (size_t)i);
        check(
            fabs(numeric - analytic) < 3e-3,
            "softmax finite-difference gradient"
        );
    }

    ocean_tensor_release(grad);
    ocean_tensor_release(loss);
    ocean_tensor_release(sm);
    ocean_tensor_release(target);
    ocean_tensor_release(x);

    x = tensor_from(input_data);
    ocean_autograd_set_requires_grad(x, true);
    target = tensor_from(target_data);
    ln = ocean_autograd_layer_norm(x, -1, 1e-5);
    loss = ocean_autograd_mse_loss(ln, target);
    ocean_autograd_backward(loss);
    grad = ocean_autograd_grad_copy(x);

    for (int i = 0; i < 6; ++i) {
        float plus[6];
        float minus[6];
        for (int j = 0; j < 6; ++j) {
            plus[j] = input_data[j];
            minus[j] = input_data[j];
        }
        plus[i] += (float)h;
        minus[i] -= (float)h;

        double numeric =
            (mse_layer_norm(plus, target_data)
                - mse_layer_norm(minus, target_data))
            / (2.0 * h);
        double analytic = ocean_tensor_get_flat_f32(grad, (size_t)i);
        check(
            fabs(numeric - analytic) < 5e-3,
            "LayerNorm finite-difference gradient"
        );
    }

    ocean_tensor_release(grad);
    ocean_tensor_release(loss);
    ocean_tensor_release(ln);
    ocean_tensor_release(target);
    ocean_tensor_release(x);

    puts("Tensor/autograd v0.3 math: OK");
    return 0;
}
""",
        encoding="utf-8",
    )

    _compile(root, source, binary)
    result = subprocess.run(
        [str(binary)], check=True, capture_output=True, text=True
    )
    assert result.stdout.strip() == "Tensor/autograd v0.3 math: OK"
