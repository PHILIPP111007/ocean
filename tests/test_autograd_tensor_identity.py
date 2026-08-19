from __future__ import annotations

import subprocess
from pathlib import Path


def test_autograd_identity_survives_pointer_reuse(tmp_path):
    root = Path(__file__).resolve().parents[1]
    source = tmp_path / "autograd_identity.c"
    binary = tmp_path / "autograd_identity"

    source.write_text(
        r"""
#include <stdio.h>
#include <stdlib.h>

#include "std/tensor/tensor_runtime.h"
#include "std/tensor/autograd_runtime.h"

static ocean_tensor_handle_t make_tensor(void) {
    const float data[6] = {
        0.7f, 1.2f, 2.0f,
        1.5f, 0.4f, 2.3f
    };
    const size_t shape[2] = {2, 3};
    const size_t strides[2] = {3, 1};

    return ocean_tensor_from_cpu_strided(
        data, shape, strides, 2, "float32", "cpu"
    );
}

int main(void) {
    ocean_tensor_handle_t first = make_tensor();
    uint64_t first_id = ocean_tensor_identity(first);

    ocean_autograd_set_requires_grad(first, true);

    if (!ocean_autograd_requires_grad(first)) {
        fprintf(stderr, "first Tensor missing requires_grad\n");
        return 1;
    }

    void *old_address = (void *)first;
    ocean_tensor_release(first);

    int reused = 0;

    for (int attempt = 0; attempt < 10000; ++attempt) {
        ocean_tensor_handle_t next = make_tensor();

        if ((void *)next == old_address) {
            reused = 1;

            if (ocean_tensor_identity(next) == first_id) {
                fprintf(stderr, "Tensor identity was reused\n");
                return 2;
            }

            if (ocean_autograd_requires_grad(next)) {
                fprintf(stderr, "new Tensor inherited stale requires_grad\n");
                return 3;
            }

            if (ocean_autograd_has_grad(next)) {
                fprintf(stderr, "new Tensor inherited stale grad\n");
                return 4;
            }

            ocean_tensor_release(next);
            break;
        }

        if (ocean_autograd_requires_grad(next) || ocean_autograd_has_grad(next)) {
            fprintf(stderr, "new Tensor inherited stale autograd state\n");
            return 5;
        }

        ocean_tensor_release(next);
    }

    printf("autograd tensor identity: OK reuse=%d\n", reused);
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
    assert result.stdout.startswith("autograd tensor identity: OK")
