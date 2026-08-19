from __future__ import annotations

import subprocess
from pathlib import Path


def test_autograd_metadata_dies_with_tensor_handle(tmp_path):
    root = Path(__file__).resolve().parents[1]
    source = tmp_path / "autograd_lifetime.c"
    binary = tmp_path / "autograd_lifetime"

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
        data,
        shape,
        strides,
        2,
        "float32",
        "cpu"
    );
}

int main(void) {
    ocean_tensor_handle_t first = make_tensor();
    ocean_autograd_set_requires_grad(first, true);

    if (!ocean_autograd_requires_grad(first)) {
        fprintf(stderr, "first Tensor did not enter autograd\n");
        return 1;
    }

    void *old_address = (void *)first;
    ocean_tensor_release(first);

    int reused = 0;

    for (int attempt = 0; attempt < 10000; ++attempt) {
        ocean_tensor_handle_t next = make_tensor();

        if (ocean_autograd_requires_grad(next)) {
            fprintf(
                stderr,
                "new Tensor inherited stale requires_grad metadata\n"
            );
            return 2;
        }

        if (ocean_autograd_has_grad(next)) {
            fprintf(
                stderr,
                "new Tensor inherited stale gradient metadata\n"
            );
            return 3;
        }

        if ((void *)next == old_address) {
            reused = 1;
            ocean_tensor_release(next);
            break;
        }

        ocean_tensor_release(next);
    }

    printf("autograd tensor lifetime: OK reuse=%d\n", reused);
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

    assert result.stdout.startswith("autograd tensor lifetime: OK")
