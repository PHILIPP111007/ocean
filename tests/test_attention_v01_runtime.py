from __future__ import annotations

import subprocess
from pathlib import Path


def test_scaled_dot_product_attention_4d_forward_backward(tmp_path):
    root = Path(__file__).resolve().parents[1]
    source = tmp_path / "attention_v01.c"
    binary = tmp_path / "attention_v01"

    source.write_text(
        r"""
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "std/tensor/tensor_runtime.h"
#include "std/tensor/autograd_runtime.h"

#define B 2
#define H 2
#define T 3
#define D 4
#define N (B * H * T * D)

static void fail(const char *message) {
    fprintf(stderr, "attention v0.1 failed: %s\n", message);
    exit(1);
}

static ocean_tensor_handle_t tensor4(const float *data) {
    const size_t shape[4] = {B, H, T, D};
    const size_t strides[4] = {H * T * D, T * D, D, 1};
    return ocean_tensor_from_cpu_strided(
        data, shape, strides, 4, "float32", "cpu"
    );
}

static ocean_tensor_handle_t attention(
    ocean_tensor_handle_t q,
    ocean_tensor_handle_t k,
    ocean_tensor_handle_t v
) {
    ocean_tensor_handle_t kt =
        ocean_autograd_transpose_dims(k, -2, -1);
    ocean_tensor_handle_t scores =
        ocean_autograd_matmul(q, kt);
    ocean_tensor_handle_t scaled =
        ocean_autograd_scalar(scores, sqrt((double)D), 3);
    ocean_tensor_handle_t weights =
        ocean_autograd_softmax(scaled, -1);
    ocean_tensor_handle_t output =
        ocean_autograd_matmul(weights, v);

    ocean_tensor_release(weights);
    ocean_tensor_release(scaled);
    ocean_tensor_release(scores);
    ocean_tensor_release(kt);
    return output;
}

static double attention_loss(
    const float *q_data,
    const float *k_data,
    const float *v_data,
    const float *target_data
) {
    ocean_tensor_handle_t q = tensor4(q_data);
    ocean_tensor_handle_t k = tensor4(k_data);
    ocean_tensor_handle_t v = tensor4(v_data);
    ocean_tensor_handle_t target = tensor4(target_data);

    ocean_tensor_handle_t output = attention(q, k, v);
    ocean_tensor_handle_t loss =
        ocean_autograd_mse_loss(output, target);
    double value = ocean_tensor_item(loss);

    ocean_tensor_release(loss);
    ocean_tensor_release(output);
    ocean_tensor_release(target);
    ocean_tensor_release(v);
    ocean_tensor_release(k);
    ocean_tensor_release(q);
    return value;
}

static void fill_data(
    float *q,
    float *k,
    float *v,
    float *target
) {
    for (int i = 0; i < N; ++i) {
        q[i] = (float)(
            0.15 * sin((double)(i + 1) * 0.37)
            + 0.03 * (double)(i % 5)
        );
        k[i] = (float)(
            0.17 * cos((double)(i + 2) * 0.29)
            - 0.02 * (double)(i % 7)
        );
        v[i] = (float)(
            0.11 * sin((double)(i + 3) * 0.19)
            + 0.04 * (double)(i % 3)
        );
        target[i] = (float)(
            0.05 * cos((double)(i + 4) * 0.23)
        );
    }
}

static void check_shape(
    ocean_tensor_handle_t tensor,
    int a,
    int b,
    int c,
    int d,
    const char *name
) {
    if (ocean_tensor_ndim(tensor) != 4) fail(name);
    if (
        ocean_tensor_shape(tensor, 0) != a
        || ocean_tensor_shape(tensor, 1) != b
        || ocean_tensor_shape(tensor, 2) != c
        || ocean_tensor_shape(tensor, 3) != d
    ) {
        fail(name);
    }
}

static void finite_difference(
    const char *name,
    int which,
    const float *q0,
    const float *k0,
    const float *v0,
    const float *target,
    ocean_tensor_handle_t grad
) {
    const double h = 1e-3;
    double max_error = 0.0;

    for (int i = 0; i < N; ++i) {
        float qp[N], qm[N];
        float kp[N], km[N];
        float vp[N], vm[N];

        memcpy(qp, q0, sizeof(qp));
        memcpy(qm, q0, sizeof(qm));
        memcpy(kp, k0, sizeof(kp));
        memcpy(km, k0, sizeof(km));
        memcpy(vp, v0, sizeof(vp));
        memcpy(vm, v0, sizeof(vm));

        if (which == 0) {
            qp[i] += (float)h;
            qm[i] -= (float)h;
        } else if (which == 1) {
            kp[i] += (float)h;
            km[i] -= (float)h;
        } else {
            vp[i] += (float)h;
            vm[i] -= (float)h;
        }

        double plus = attention_loss(qp, kp, vp, target);
        double minus = attention_loss(qm, km, vm, target);
        double numeric = (plus - minus) / (2.0 * h);
        double analytic =
            (double)ocean_tensor_get_flat_f32(grad, (size_t)i);
        double error = fabs(numeric - analytic);

        if (error > max_error) max_error = error;

        if (error > 2.5e-3) {
            fprintf(
                stderr,
                "%s grad[%d]: numeric=% .9f analytic=% .9f error=%.9g\n",
                name, i, numeric, analytic, error
            );
            fail("finite-difference mismatch");
        }
    }

    printf("%s max gradient error = %.9g\n", name, max_error);
}

int main(void) {
    float q_data[N], k_data[N], v_data[N], target_data[N];
    fill_data(q_data, k_data, v_data, target_data);

    ocean_tensor_handle_t q = tensor4(q_data);
    ocean_tensor_handle_t k = tensor4(k_data);
    ocean_tensor_handle_t v = tensor4(v_data);

    ocean_tensor_handle_t kt =
        ocean_autograd_transpose_dims(k, -2, -1);
    check_shape(kt, B, H, D, T, "K^T shape");

    ocean_tensor_handle_t scores =
        ocean_autograd_matmul(q, kt);
    check_shape(scores, B, H, T, T, "scores shape");

    ocean_tensor_handle_t scaled =
        ocean_autograd_scalar(scores, sqrt((double)D), 3);
    ocean_tensor_handle_t weights =
        ocean_autograd_softmax(scaled, -1);
    check_shape(weights, B, H, T, T, "weights shape");

    for (int b = 0; b < B; ++b) {
        for (int h = 0; h < H; ++h) {
            for (int row = 0; row < T; ++row) {
                double total = 0.0;
                for (int col = 0; col < T; ++col) {
                    size_t index = (size_t)(
                        ((b * H + h) * T + row) * T + col
                    );
                    total += ocean_tensor_get_flat_f32(weights, index);
                }
                if (fabs(total - 1.0) > 2e-6) {
                    fail("softmax row sum");
                }
            }
        }
    }

    ocean_tensor_handle_t output =
        ocean_autograd_matmul(weights, v);
    check_shape(output, B, H, T, D, "output shape");

    for (size_t i = 0; i < ocean_tensor_size(output); ++i) {
        if (!isfinite(ocean_tensor_get_flat_f32(output, i))) {
            fail("NaN/Inf in output");
        }
    }

    ocean_tensor_release(output);
    ocean_tensor_release(weights);
    ocean_tensor_release(scaled);
    ocean_tensor_release(scores);
    ocean_tensor_release(kt);
    ocean_tensor_release(v);
    ocean_tensor_release(k);
    ocean_tensor_release(q);

    q = tensor4(q_data);
    k = tensor4(k_data);
    v = tensor4(v_data);
    ocean_tensor_handle_t target = tensor4(target_data);

    ocean_autograd_set_requires_grad(q, true);
    ocean_autograd_set_requires_grad(k, true);
    ocean_autograd_set_requires_grad(v, true);

    output = attention(q, k, v);
    ocean_tensor_handle_t loss =
        ocean_autograd_mse_loss(output, target);

    double loss_value = ocean_tensor_item(loss);
    if (!isfinite(loss_value) || loss_value <= 0.0) {
        fail("invalid loss");
    }

    ocean_autograd_backward(loss);

    if (
        !ocean_autograd_has_grad(q)
        || !ocean_autograd_has_grad(k)
        || !ocean_autograd_has_grad(v)
    ) {
        fail("missing Q/K/V gradient");
    }

    ocean_tensor_handle_t qg = ocean_autograd_grad_copy(q);
    ocean_tensor_handle_t kg = ocean_autograd_grad_copy(k);
    ocean_tensor_handle_t vg = ocean_autograd_grad_copy(v);

    finite_difference("Q", 0, q_data, k_data, v_data, target_data, qg);
    finite_difference("K", 1, q_data, k_data, v_data, target_data, kg);
    finite_difference("V", 2, q_data, k_data, v_data, target_data, vg);

    printf("attention loss = %.9f\n", loss_value);
    puts("4D scaled dot-product attention v0.1: OK");

    ocean_tensor_release(vg);
    ocean_tensor_release(kg);
    ocean_tensor_release(qg);
    ocean_tensor_release(loss);
    ocean_tensor_release(output);
    ocean_tensor_release(target);
    ocean_tensor_release(v);
    ocean_tensor_release(k);
    ocean_tensor_release(q);
    return 0;
}
""",
        encoding="utf-8",
    )

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

    result = subprocess.run(
        [str(binary)],
        check=True,
        capture_output=True,
        text=True,
    )
    assert "4D scaled dot-product attention v0.1: OK" in result.stdout
