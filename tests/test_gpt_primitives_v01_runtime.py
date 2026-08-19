from __future__ import annotations

import subprocess
from pathlib import Path


def test_gpt_primitives_v01_runtime(tmp_path):
    root = Path(__file__).resolve().parents[1]
    source = tmp_path / "gpt_primitives_v01.c"
    binary = tmp_path / "gpt_primitives_v01"

    source.write_text(
        r"""
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "std/tensor/tensor_runtime.h"
#include "std/tensor/autograd_runtime.h"

static void fail(const char *message) {
    fprintf(stderr, "GPT primitives v0.1 failed: %s\n", message);
    exit(1);
}

static ocean_tensor_handle_t float_tensor(
    const float *data,
    const size_t *shape,
    const size_t *strides,
    size_t ndim
) {
    return ocean_tensor_from_cpu_strided(
        data, shape, strides, ndim, "float32", "cpu"
    );
}

static ocean_tensor_handle_t int64_tensor(
    const int64_t *data,
    const size_t *shape,
    const size_t *strides,
    size_t ndim
) {
    return ocean_tensor_from_cpu_strided(
        data, shape, strides, ndim, "int64", "cpu"
    );
}

static double embedding_loss(
    const float *weight_data,
    const int64_t *indices_data,
    const float *target_data
) {
    const size_t ws[2] = {5, 3};
    const size_t wst[2] = {3, 1};
    const size_t is[2] = {2, 3};
    const size_t ist[2] = {3, 1};
    const size_t os[3] = {2, 3, 3};
    const size_t ost[3] = {9, 3, 1};

    ocean_tensor_handle_t w = float_tensor(weight_data, ws, wst, 2);
    ocean_tensor_handle_t i = int64_tensor(indices_data, is, ist, 2);
    ocean_tensor_handle_t t = float_tensor(target_data, os, ost, 3);
    ocean_tensor_handle_t y = ocean_autograd_embedding(w, i);
    ocean_tensor_handle_t loss = ocean_autograd_mse_loss(y, t);
    double value = ocean_tensor_item(loss);

    ocean_tensor_release(loss);
    ocean_tensor_release(y);
    ocean_tensor_release(t);
    ocean_tensor_release(i);
    ocean_tensor_release(w);
    return value;
}

static double ce_loss(
    const float *logits_data,
    const int64_t *targets_data
) {
    const size_t ls[3] = {2, 2, 4};
    const size_t lst[3] = {8, 4, 1};
    const size_t ts[2] = {2, 2};
    const size_t tst[2] = {2, 1};

    ocean_tensor_handle_t l = float_tensor(logits_data, ls, lst, 3);
    ocean_tensor_handle_t t = int64_tensor(targets_data, ts, tst, 2);
    ocean_tensor_handle_t loss =
        ocean_autograd_cross_entropy(l, t);
    double value = ocean_tensor_item(loss);

    ocean_tensor_release(loss);
    ocean_tensor_release(t);
    ocean_tensor_release(l);
    return value;
}

int main(void) {
    float weight_data[15];
    float embed_target[18];
    const int64_t indices[6] = {0, 2, 2, 4, 1, 2};

    for (int i = 0; i < 15; ++i) {
        weight_data[i] = (float)(0.03 * (double)(i - 4));
    }
    for (int i = 0; i < 18; ++i) {
        embed_target[i] =
            (float)(0.02 * sin((double)(i + 1) * 0.4));
    }

    const size_t ws[2] = {5, 3};
    const size_t wst[2] = {3, 1};
    const size_t is[2] = {2, 3};
    const size_t ist[2] = {3, 1};
    const size_t os[3] = {2, 3, 3};
    const size_t ost[3] = {9, 3, 1};

    ocean_tensor_handle_t w = float_tensor(weight_data, ws, wst, 2);
    ocean_tensor_handle_t i = int64_tensor(indices, is, ist, 2);
    ocean_tensor_handle_t t = float_tensor(embed_target, os, ost, 3);
    ocean_autograd_set_requires_grad(w, true);

    ocean_tensor_handle_t y = ocean_autograd_embedding(w, i);
    if (
        ocean_tensor_ndim(y) != 3
        || ocean_tensor_shape(y, 0) != 2
        || ocean_tensor_shape(y, 1) != 3
        || ocean_tensor_shape(y, 2) != 3
    ) fail("Embedding shape");

    ocean_tensor_handle_t mse = ocean_autograd_mse_loss(y, t);
    ocean_autograd_backward(mse);
    ocean_tensor_handle_t wg = ocean_autograd_grad_copy(w);
    if (!wg) fail("Embedding gradient");

    const double step = 1e-3;
    double max_embed_error = 0.0;
    for (int index = 0; index < 15; ++index) {
        float plus[15];
        float minus[15];
        memcpy(plus, weight_data, sizeof(plus));
        memcpy(minus, weight_data, sizeof(minus));
        plus[index] += (float)step;
        minus[index] -= (float)step;

        double numeric = (
            embedding_loss(plus, indices, embed_target)
            - embedding_loss(minus, indices, embed_target)
        ) / (2.0 * step);
        double analytic =
            ocean_tensor_get_flat_f32(wg, (size_t)index);
        double error = fabs(numeric - analytic);
        if (error > max_embed_error) max_embed_error = error;
        if (error > 2.5e-3) fail("Embedding finite difference");
    }
    printf("Embedding max gradient error = %.9g\n", max_embed_error);

    ocean_tensor_release(wg);
    ocean_tensor_release(mse);
    ocean_tensor_release(y);
    ocean_tensor_release(t);
    ocean_tensor_release(i);
    ocean_tensor_release(w);

    float logits_data[16] = {
         0.2f, -0.1f,  0.4f,  0.0f,
        -0.2f,  0.3f,  0.1f,  0.5f,
         0.7f, -0.4f,  0.2f,  0.1f,
         0.0f,  0.6f, -0.3f,  0.2f
    };
    const int64_t targets_data[4] = {2, 3, 0, 1};
    const size_t ls[3] = {2, 2, 4};
    const size_t lst[3] = {8, 4, 1};
    const size_t ts[2] = {2, 2};
    const size_t tst[2] = {2, 1};

    ocean_tensor_handle_t logits =
        float_tensor(logits_data, ls, lst, 3);
    ocean_tensor_handle_t targets =
        int64_tensor(targets_data, ts, tst, 2);
    ocean_autograd_set_requires_grad(logits, true);

    ocean_tensor_handle_t ce =
        ocean_autograd_cross_entropy(logits, targets);
    double ce_value = ocean_tensor_item(ce);
    if (!isfinite(ce_value) || ce_value <= 0.0) {
        fail("CrossEntropy forward");
    }

    ocean_autograd_backward(ce);
    ocean_tensor_handle_t lg =
        ocean_autograd_grad_copy(logits);
    if (!lg) fail("CrossEntropy gradient");

    double max_ce_error = 0.0;
    for (int index = 0; index < 16; ++index) {
        float plus[16];
        float minus[16];
        memcpy(plus, logits_data, sizeof(plus));
        memcpy(minus, logits_data, sizeof(minus));
        plus[index] += (float)step;
        minus[index] -= (float)step;

        double numeric = (
            ce_loss(plus, targets_data)
            - ce_loss(minus, targets_data)
        ) / (2.0 * step);
        double analytic =
            ocean_tensor_get_flat_f32(lg, (size_t)index);
        double error = fabs(numeric - analytic);
        if (error > max_ce_error) max_ce_error = error;
        if (error > 2.5e-3) fail("CrossEntropy finite difference");
    }

    printf("CrossEntropy max gradient error = %.9g\n", max_ce_error);
    puts("GPT primitives v0.1: OK");

    ocean_tensor_release(lg);
    ocean_tensor_release(ce);
    ocean_tensor_release(targets);
    ocean_tensor_release(logits);
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
    assert "GPT primitives v0.1: OK" in result.stdout
