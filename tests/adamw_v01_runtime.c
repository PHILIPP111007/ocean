
#include <math.h>
#include <stdio.h>
#include <stdlib.h>

#include "std/tensor/autograd_runtime.h"
#include "std/tensor/tensor_runtime.h"

static void require_close(double actual, double expected, double tolerance) {
    if (fabs(actual - expected) > tolerance) {
        fprintf(
            stderr,
            "AdamW mismatch: actual=%.12f expected=%.12f\n",
            actual,
            expected
        );
        exit(1);
    }
}

static void backward_quadratic(ocean_tensor_handle_t parameter) {
    ocean_tensor_handle_t target = ocean_tensor_zeros(1, 2, "cpu");
    ocean_tensor_handle_t loss =
        ocean_autograd_mse_loss(parameter, target);

    ocean_autograd_backward(loss);

    ocean_tensor_release(loss);
    ocean_tensor_release(target);
}

int main(void) {
    ocean_tensor_handle_t parameter =
        ocean_tensor_zeros(1, 2, "cpu");

    ocean_tensor_set_2d(parameter, 0, 0, 1.0);
    ocean_tensor_set_2d(parameter, 0, 1, -2.0);
    ocean_autograd_set_requires_grad(parameter, true);

    int state_id = ocean_autograd_adamw_create();

    backward_quadratic(parameter);
    int step1 = ocean_autograd_adamw_begin_step(state_id);
    ocean_autograd_adamw_step(
        state_id,
        step1,
        parameter,
        0.1,
        0.9,
        0.999,
        1e-8,
        0.01
    );

    double p10 = ocean_tensor_get_2d(parameter, 0, 0);
    double p11 = ocean_tensor_get_2d(parameter, 0, 1);

    require_close(p10, 0.899000001, 2e-6);
    require_close(p11, -1.8980000005, 2e-6);

    ocean_autograd_zero_grad(parameter);

    backward_quadratic(parameter);
    int step2 = ocean_autograd_adamw_begin_step(state_id);
    ocean_autograd_adamw_step(
        state_id,
        step2,
        parameter,
        0.1,
        0.9,
        0.999,
        1e-8,
        0.01
    );

    double p20 = ocean_tensor_get_2d(parameter, 0, 0);
    double p21 = ocean_tensor_get_2d(parameter, 0, 1);

    require_close(p20, 0.7985190281887788, 3e-6);
    require_close(p21, -1.7962725891500528, 3e-6);

    printf("step1 = %.9f %.9f\n", p10, p11);
    printf("step2 = %.9f %.9f\n", p20, p21);
    printf("[ok] Ocean AdamW runtime v0.1\n");

    ocean_tensor_release(parameter);
    return 0;
}
