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
#include "std/tensor/autograd_runtime.h"

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

    size_t cross_entropy_target_shape[1] = {2};
    ocean_tensor_handle_t cross_entropy_targets_cpu = ocean_tensor_zeros_nd(
        cross_entropy_target_shape, 1, "int64", "cpu"
    );
    ocean_tensor_set_flat_i64(cross_entropy_targets_cpu, 0, 1);
    ocean_tensor_set_flat_i64(cross_entropy_targets_cpu, 1, 3);
    ocean_tensor_handle_t cross_entropy_targets_gpu = ocean_tensor_to(
        cross_entropy_targets_cpu, "gpu"
    );
    ocean_tensor_handle_t cross_entropy_probabilities_cpu = NULL;
    ocean_tensor_handle_t cross_entropy_probabilities_gpu = NULL;
    ocean_tensor_handle_t cross_entropy_loss_cpu =
        ocean_tensor_cross_entropy_forward(
            cpu,
            cross_entropy_targets_cpu,
            &cross_entropy_probabilities_cpu
        );
    ocean_tensor_handle_t cross_entropy_loss_gpu =
        ocean_tensor_cross_entropy_forward(
            gpu,
            cross_entropy_targets_gpu,
            &cross_entropy_probabilities_gpu
        );
    check_close(
        cross_entropy_probabilities_cpu,
        cross_entropy_probabilities_gpu,
        "cross entropy probabilities"
    );
    check_close(cross_entropy_loss_cpu, cross_entropy_loss_gpu,
                "cross entropy loss");
    ocean_tensor_handle_t cross_entropy_upstream_cpu =
        ocean_tensor_zeros(1, 1, "cpu");
    ocean_tensor_fill(cross_entropy_upstream_cpu, 1.0);
    ocean_tensor_handle_t cross_entropy_upstream_gpu = ocean_tensor_to(
        cross_entropy_upstream_cpu, "gpu"
    );
    ocean_tensor_handle_t cross_entropy_gradient_cpu =
        ocean_tensor_cross_entropy_backward(
            cross_entropy_upstream_cpu,
            cross_entropy_probabilities_cpu,
            cross_entropy_targets_cpu
        );
    ocean_tensor_handle_t cross_entropy_gradient_gpu =
        ocean_tensor_cross_entropy_backward(
            cross_entropy_upstream_gpu,
            cross_entropy_probabilities_gpu,
            cross_entropy_targets_gpu
        );
    check_close(
        cross_entropy_gradient_cpu,
        cross_entropy_gradient_gpu,
        "cross entropy backward"
    );

    size_t embedding_weight_shape[2] = {4, 3};
    ocean_tensor_handle_t embedding_weight_cpu = ocean_tensor_zeros_nd(
        embedding_weight_shape, 2, "float32", "cpu"
    );
    for (size_t i = 0; i < 12; ++i) {
        ocean_tensor_set_flat_f32(
            embedding_weight_cpu, i, (float)i + 0.5f
        );
    }
    ocean_tensor_handle_t embedding_weight_gpu = ocean_tensor_to(
        embedding_weight_cpu, "gpu"
    );
    size_t embedding_index_shape[1] = {4};
    ocean_tensor_handle_t embedding_indices_cpu = ocean_tensor_zeros_nd(
        embedding_index_shape, 1, "int64", "cpu"
    );
    const int64_t embedding_tokens[4] = {0, 2, 2, 3};
    for (size_t i = 0; i < 4; ++i) {
        ocean_tensor_set_flat_i64(
            embedding_indices_cpu, i, embedding_tokens[i]
        );
    }
    ocean_tensor_handle_t embedding_indices_gpu = ocean_tensor_to(
        embedding_indices_cpu, "gpu"
    );
    ocean_tensor_handle_t embedding_cpu = ocean_tensor_embedding_forward(
        embedding_weight_cpu, embedding_indices_cpu
    );
    ocean_tensor_handle_t embedding_gpu = ocean_tensor_embedding_forward(
        embedding_weight_gpu, embedding_indices_gpu
    );
    check_close(embedding_cpu, embedding_gpu, "embedding forward");

    ocean_tensor_handle_t embedding_upstream_cpu = ocean_tensor_zeros_nd(
        embedding_weight_shape, 2, "float32", "cpu"
    );
    for (size_t i = 0; i < 12; ++i) {
        ocean_tensor_set_flat_f32(
            embedding_upstream_cpu, i, (float)i + 1.0f
        );
    }
    ocean_tensor_handle_t embedding_upstream_gpu = ocean_tensor_to(
        embedding_upstream_cpu, "gpu"
    );
    ocean_tensor_handle_t embedding_gradient_cpu =
        ocean_tensor_embedding_backward(
            embedding_upstream_cpu, embedding_indices_cpu, 4, 3
        );
    ocean_tensor_handle_t embedding_gradient_gpu =
        ocean_tensor_embedding_backward(
            embedding_upstream_gpu, embedding_indices_gpu, 4, 3
        );
    check_close(
        embedding_gradient_cpu, embedding_gradient_gpu,
        "embedding backward"
    );

    ocean_tensor_handle_t autograd_weight_cpu = ocean_tensor_copy(
        embedding_weight_cpu
    );
    ocean_tensor_handle_t autograd_weight_gpu = ocean_tensor_copy(
        embedding_weight_gpu
    );
    ocean_autograd_set_requires_grad(autograd_weight_cpu, true);
    ocean_autograd_set_requires_grad(autograd_weight_gpu, true);
    ocean_tensor_handle_t autograd_embedding_cpu = ocean_autograd_embedding(
        autograd_weight_cpu, embedding_indices_cpu
    );
    ocean_tensor_handle_t autograd_embedding_gpu = ocean_autograd_embedding(
        autograd_weight_gpu, embedding_indices_gpu
    );
    ocean_tensor_handle_t autograd_target_cpu = ocean_tensor_zeros_nd(
        embedding_weight_shape, 2, "float32", "cpu"
    );
    ocean_tensor_handle_t autograd_target_gpu = ocean_tensor_to(
        autograd_target_cpu, "gpu"
    );
    ocean_tensor_handle_t autograd_loss_cpu = ocean_autograd_mse_loss(
        autograd_embedding_cpu, autograd_target_cpu
    );
    ocean_tensor_handle_t autograd_loss_gpu = ocean_autograd_mse_loss(
        autograd_embedding_gpu, autograd_target_gpu
    );
    ocean_autograd_backward(autograd_loss_cpu);
    ocean_autograd_backward(autograd_loss_gpu);
    ocean_tensor_handle_t autograd_gradient_cpu =
        ocean_autograd_grad_copy(autograd_weight_cpu);
    ocean_tensor_handle_t autograd_gradient_gpu =
        ocean_autograd_grad_copy(autograd_weight_gpu);
    check_close(
        autograd_gradient_cpu, autograd_gradient_gpu,
        "embedding autograd backward"
    );

    ocean_tensor_handle_t cpu_ternary = ocean_tensor_ternary_quantize(cpu);
    ocean_tensor_handle_t gpu_ternary = ocean_tensor_ternary_quantize(gpu);
    check_close(cpu_ternary, gpu_ternary, "ternary quantize");

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

    ocean_tensor_handle_t cpu_softmax_input = ocean_tensor_copy(cpu);
    ocean_tensor_handle_t gpu_softmax_input = ocean_tensor_copy(gpu);
    ocean_autograd_set_requires_grad(cpu_softmax_input, true);
    ocean_autograd_set_requires_grad(gpu_softmax_input, true);
    ocean_tensor_handle_t cpu_softmax_output =
        ocean_autograd_softmax(cpu_softmax_input, -1);
    ocean_tensor_handle_t gpu_softmax_output =
        ocean_autograd_softmax(gpu_softmax_input, -1);
    ocean_tensor_handle_t cpu_softmax_target = ocean_tensor_zeros_nd(
        shape, 2, "float32", "cpu"
    );
    ocean_tensor_handle_t gpu_softmax_target = ocean_tensor_to(
        cpu_softmax_target, "gpu"
    );
    ocean_tensor_handle_t cpu_softmax_loss = ocean_autograd_mse_loss(
        cpu_softmax_output, cpu_softmax_target
    );
    ocean_tensor_handle_t gpu_softmax_loss = ocean_autograd_mse_loss(
        gpu_softmax_output, gpu_softmax_target
    );
    ocean_autograd_backward(cpu_softmax_loss);
    ocean_autograd_backward(gpu_softmax_loss);
    ocean_tensor_handle_t cpu_softmax_grad =
        ocean_autograd_grad_copy(cpu_softmax_input);
    ocean_tensor_handle_t gpu_softmax_grad =
        ocean_autograd_grad_copy(gpu_softmax_input);
    check_close(cpu_softmax_grad, gpu_softmax_grad, "softmax backward");

    ocean_tensor_handle_t cpu_norm_input = ocean_tensor_copy(cpu);
    ocean_tensor_handle_t gpu_norm_input = ocean_tensor_copy(gpu);
    ocean_autograd_set_requires_grad(cpu_norm_input, true);
    ocean_autograd_set_requires_grad(gpu_norm_input, true);
    ocean_tensor_handle_t cpu_norm_output =
        ocean_autograd_layer_norm(cpu_norm_input, -1, 1e-5);
    ocean_tensor_handle_t gpu_norm_output =
        ocean_autograd_layer_norm(gpu_norm_input, -1, 1e-5);
    ocean_tensor_handle_t cpu_norm_target = ocean_tensor_zeros_nd(
        shape, 2, "float32", "cpu"
    );
    ocean_tensor_handle_t gpu_norm_target = ocean_tensor_to(
        cpu_norm_target, "gpu"
    );
    ocean_tensor_handle_t cpu_norm_loss = ocean_autograd_mse_loss(
        cpu_norm_output, cpu_norm_target
    );
    ocean_tensor_handle_t gpu_norm_loss = ocean_autograd_mse_loss(
        gpu_norm_output, gpu_norm_target
    );
    ocean_autograd_backward(cpu_norm_loss);
    ocean_autograd_backward(gpu_norm_loss);
    ocean_tensor_handle_t cpu_norm_grad =
        ocean_autograd_grad_copy(cpu_norm_input);
    ocean_tensor_handle_t gpu_norm_grad =
        ocean_autograd_grad_copy(gpu_norm_input);
    check_close(cpu_norm_grad, gpu_norm_grad, "layer_norm backward");

    ocean_tensor_release(gpu_norm_grad);
    ocean_tensor_release(cpu_norm_grad);
    ocean_tensor_release(gpu_norm_loss);
    ocean_tensor_release(cpu_norm_loss);
    ocean_tensor_release(gpu_norm_target);
    ocean_tensor_release(cpu_norm_target);
    ocean_tensor_release(gpu_norm_output);
    ocean_tensor_release(cpu_norm_output);
    ocean_tensor_release(gpu_norm_input);
    ocean_tensor_release(cpu_norm_input);
    ocean_tensor_release(gpu_softmax_grad);
    ocean_tensor_release(cpu_softmax_grad);
    ocean_tensor_release(gpu_softmax_loss);
    ocean_tensor_release(cpu_softmax_loss);
    ocean_tensor_release(gpu_softmax_target);
    ocean_tensor_release(cpu_softmax_target);
    ocean_tensor_release(gpu_softmax_output);
    ocean_tensor_release(cpu_softmax_output);
    ocean_tensor_release(gpu_softmax_input);
    ocean_tensor_release(cpu_softmax_input);

    ocean_tensor_release(gpu_ternary);
    ocean_tensor_release(cpu_ternary);

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
    ocean_tensor_release(embedding_gradient_gpu);
    ocean_tensor_release(embedding_gradient_cpu);
    ocean_tensor_release(cross_entropy_gradient_gpu);
    ocean_tensor_release(cross_entropy_gradient_cpu);
    ocean_tensor_release(cross_entropy_upstream_gpu);
    ocean_tensor_release(cross_entropy_upstream_cpu);
    ocean_tensor_release(cross_entropy_loss_gpu);
    ocean_tensor_release(cross_entropy_loss_cpu);
    ocean_tensor_release(cross_entropy_probabilities_gpu);
    ocean_tensor_release(cross_entropy_probabilities_cpu);
    ocean_tensor_release(cross_entropy_targets_gpu);
    ocean_tensor_release(cross_entropy_targets_cpu);
    ocean_tensor_release(autograd_gradient_gpu);
    ocean_tensor_release(autograd_gradient_cpu);
    ocean_tensor_release(autograd_loss_gpu);
    ocean_tensor_release(autograd_loss_cpu);
    ocean_tensor_release(autograd_target_gpu);
    ocean_tensor_release(autograd_target_cpu);
    ocean_tensor_release(autograd_embedding_gpu);
    ocean_tensor_release(autograd_embedding_cpu);
    ocean_tensor_release(autograd_weight_gpu);
    ocean_tensor_release(autograd_weight_cpu);
    ocean_tensor_release(embedding_upstream_gpu);
    ocean_tensor_release(embedding_upstream_cpu);
    ocean_tensor_release(embedding_gpu);
    ocean_tensor_release(embedding_cpu);
    ocean_tensor_release(embedding_indices_gpu);
    ocean_tensor_release(embedding_indices_cpu);
    ocean_tensor_release(embedding_weight_gpu);
    ocean_tensor_release(embedding_weight_cpu);
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
            *cflags, str(source),
            str(ROOT / "std/tensor/autograd_runtime.c"),
            str(ROOT / "std/tensor/tensor_runtime.c"),
            "-lm", *libs, "-o", str(binary),
        ],
        check=True,
    )
    result = subprocess.run(
        [str(binary)], check=False, capture_output=True, text=True
    )
    if result.returncode != 0 and (
        "clGetPlatformIDs failed" in result.stderr
        or "no OpenCL platform is available" in result.stderr
    ):
        pytest.skip("OpenCL headers are installed but no runtime platform is available")
    if result.returncode != 0:
        raise subprocess.CalledProcessError(
            result.returncode, result.args,
            output=result.stdout, stderr=result.stderr
        )
    assert "GPU hotpaths v0.1: OK" in result.stdout
