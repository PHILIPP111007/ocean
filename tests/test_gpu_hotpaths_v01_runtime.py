from pathlib import Path
import shlex
import shutil
import subprocess

import pytest


ROOT = Path(__file__).resolve().parents[1]


def opencl_available():
    if shutil.which("pkg-config") is None:
        return False
    probe = subprocess.run(["pkg-config", "--exists", "OpenCL"], check=False)
    if probe.returncode != 0:
        return False
    if shutil.which("clinfo"):
        info = subprocess.run(
            ["clinfo", "-l"], check=False, capture_output=True, text=True
        )
        if info.returncode != 0 or "device" not in (
            info.stdout + info.stderr
        ).lower():
            return False
    return True


GPU_HOTPATH_SOURCE = r'''
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include "std/tensor/tensor_runtime.h"

static void check_close(
    ocean_tensor_handle_t left,
    ocean_tensor_handle_t right,
    const char *name
) {
    if (ocean_tensor_size(left) != ocean_tensor_size(right)) {
        fprintf(stderr, "%s size mismatch\n", name);
        exit(1);
    }
    for (size_t i = 0; i < ocean_tensor_size(left); ++i) {
        double delta = fabs(
            ocean_tensor_get_flat_f32(left, i)
            - ocean_tensor_get_flat_f32(right, i)
        );
        if (delta > 2e-4) {
            fprintf(stderr, "%s mismatch at %zu: %.8f %.8f\n", name, i,
                    ocean_tensor_get_flat_f32(left, i),
                    ocean_tensor_get_flat_f32(right, i));
            exit(1);
        }
    }
}

int main(void) {
    size_t shape[2] = {2, 4};
    ocean_tensor_handle_t cpu = ocean_tensor_zeros_nd(
        shape, 2, "float32", "cpu"
    );
    for (size_t i = 0; i < 8; ++i) {
        ocean_tensor_set_flat_f32(cpu, i, (float)i * 0.25f - 0.5f);
    }
    ocean_tensor_handle_t gpu = ocean_tensor_to(cpu, "gpu");

    ocean_tensor_handle_t cpu_softmax = ocean_tensor_softmax(cpu, -1);
    ocean_tensor_handle_t gpu_softmax = ocean_tensor_softmax(gpu, -1);
    check_close(cpu_softmax, gpu_softmax, "softmax");

    ocean_tensor_handle_t cpu_norm = ocean_tensor_layer_norm(cpu, -1, 1e-5);
    ocean_tensor_handle_t gpu_norm = ocean_tensor_layer_norm(gpu, -1, 1e-5);
    check_close(cpu_norm, gpu_norm, "layer_norm");

    ocean_tensor_handle_t cpu_sum = ocean_tensor_sum_dim(cpu, -1, true);
    ocean_tensor_handle_t gpu_sum = ocean_tensor_sum_dim(gpu, -1, true);
    check_close(cpu_sum, gpu_sum, "sum_dim");

    ocean_tensor_handle_t cpu_mean = ocean_tensor_mean_dim(cpu, -1, true);
    ocean_tensor_handle_t gpu_mean = ocean_tensor_mean_dim(gpu, -1, true);
    check_close(cpu_mean, gpu_mean, "mean_dim");

    ocean_tensor_handle_t cpu_parameter = ocean_tensor_copy(cpu);
    ocean_tensor_handle_t gpu_parameter = ocean_tensor_copy(gpu);
    ocean_tensor_handle_t cpu_gradient = ocean_tensor_zeros_nd(
        shape, 2, "float32", "cpu"
    );
    for (size_t i = 0; i < 8; ++i) {
        ocean_tensor_set_flat_f32(cpu_gradient, i, 0.1f + (float)i * 0.01f);
    }
    ocean_tensor_handle_t gpu_gradient = ocean_tensor_to(cpu_gradient, "gpu");
    ocean_tensor_sgd_update(cpu_parameter, cpu_gradient, 0.05);
    ocean_tensor_sgd_update(gpu_parameter, gpu_gradient, 0.05);
    check_close(cpu_parameter, gpu_parameter, "sgd");

    ocean_tensor_handle_t cpu_first = ocean_tensor_copy(cpu);
    ocean_tensor_handle_t cpu_second = ocean_tensor_copy(cpu);
    ocean_tensor_handle_t gpu_first = ocean_tensor_copy(gpu);
    ocean_tensor_handle_t gpu_second = ocean_tensor_copy(gpu);
    ocean_tensor_fill(cpu_first, 0.0);
    ocean_tensor_fill(cpu_second, 0.0);
    ocean_tensor_fill(gpu_first, 0.0);
    ocean_tensor_fill(gpu_second, 0.0);
    ocean_tensor_adamw_update(
        cpu_parameter, cpu_gradient, cpu_first, cpu_second,
        0.001, 0.9, 0.999, 1e-8, 0.01, 0.1, 0.001
    );
    ocean_tensor_adamw_update(
        gpu_parameter, gpu_gradient, gpu_first, gpu_second,
        0.001, 0.9, 0.999, 1e-8, 0.01, 0.1, 0.001
    );
    check_close(cpu_parameter, gpu_parameter, "adamw parameter");
    check_close(cpu_first, gpu_first, "adamw first moment");
    check_close(cpu_second, gpu_second, "adamw second moment");

    ocean_tensor_release(gpu_second);
    ocean_tensor_release(gpu_first);
    ocean_tensor_release(cpu_second);
    ocean_tensor_release(cpu_first);
    ocean_tensor_release(gpu_gradient);
    ocean_tensor_release(cpu_gradient);
    ocean_tensor_release(gpu_parameter);
    ocean_tensor_release(cpu_parameter);
    ocean_tensor_release(gpu_mean);
    ocean_tensor_release(cpu_mean);
    ocean_tensor_release(gpu_sum);
    ocean_tensor_release(cpu_sum);
    ocean_tensor_release(gpu_norm);
    ocean_tensor_release(cpu_norm);
    ocean_tensor_release(gpu_softmax);
    ocean_tensor_release(cpu_softmax);
    ocean_tensor_release(gpu);
    ocean_tensor_release(cpu);
    puts("GPU hotpaths v0.1: OK");
    return 0;
}
'''


@pytest.mark.skipif(
    not opencl_available(),
    reason="OpenCL development/runtime device is unavailable",
)
def test_gpu_hotpaths_v01_runtime(tmp_path):
    source = tmp_path / "gpu_hotpaths_v01.c"
    binary = tmp_path / "gpu_hotpaths_v01"
    source.write_text(GPU_HOTPATH_SOURCE, encoding="utf-8")

    cflags = shlex.split(
        subprocess.check_output(["pkg-config", "--cflags", "OpenCL"], text=True)
    )
    libs = shlex.split(
        subprocess.check_output(["pkg-config", "--libs", "OpenCL"], text=True)
    )
    subprocess.run(
        [
            "gcc", "-std=c11", "-O2", "-Wall", "-Wextra", "-Wpedantic",
            "-Werror", "-DOCEAN_TENSOR_ENABLE_OPENCL", "-I", str(ROOT),
            *cflags, str(source), str(ROOT / "std/tensor/tensor_runtime.c"),
            "-lm", *libs, "-o", str(binary),
        ],
        check=True,
    )
    result = subprocess.run(
        [str(binary)], check=True, capture_output=True, text=True
    )
    assert "GPU hotpaths v0.1: OK" in result.stdout
