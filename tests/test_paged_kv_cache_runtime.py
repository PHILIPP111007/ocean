from pathlib import Path
import subprocess
import tempfile


ROOT = Path(__file__).resolve().parents[1]


C_SOURCE = r'''
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include "std/tensor/tensor_runtime.h"

static void close_to(float actual, float expected, const char *name) {
    if (fabsf(actual - expected) > 1e-4f) {
        fprintf(stderr, "%s: %.8f != %.8f\n", name, actual, expected);
        exit(1);
    }
}

int main(void) {
    size_t shape[4] = {1, 1, 4, 2};
    ocean_tensor_handle_t key = ocean_tensor_zeros_nd(
        shape, 4, "float32", "cpu"
    );
    ocean_tensor_handle_t value = ocean_tensor_zeros_nd(
        shape, 4, "float32", "cpu"
    );
    const float key_data[8] = {
        1.0f, 0.0f, 0.0f, 1.0f,
        -1.0f, 0.0f, 0.0f, -1.0f,
    };
    const float value_data[8] = {
        10.0f, 100.0f, 20.0f, 200.0f,
        30.0f, 300.0f, 40.0f, 400.0f,
    };
    for (size_t i = 0; i < 8; ++i) {
        ocean_tensor_set_flat_f32(key, i, key_data[i]);
        ocean_tensor_set_flat_f32(value, i, value_data[i]);
    }

    ocean_paged_kv_cache_handle_t cache =
        ocean_paged_kv_cache_create(1, 1, 2, 2, "cpu");
    ocean_paged_kv_cache_write(cache, key, value, 0);
    if (ocean_paged_kv_cache_length(cache) != 4 ||
        ocean_paged_kv_cache_page_size(cache) != 2) {
        fprintf(stderr, "unexpected PagedKVCache metadata\n");
        return 1;
    }

    ocean_tensor_handle_t materialized_key =
        ocean_paged_kv_cache_materialize_key(cache);
    ocean_tensor_handle_t materialized_value =
        ocean_paged_kv_cache_materialize_value(cache);
    for (size_t i = 0; i < 8; ++i) {
        close_to(ocean_tensor_get_flat_f32(materialized_key, i), key_data[i], "key");
        close_to(ocean_tensor_get_flat_f32(materialized_value, i), value_data[i], "value");
    }

    ocean_tensor_handle_t summaries = ocean_paged_kv_cache_build_summaries(
        cache, 4, 2
    );
    ocean_tensor_handle_t dense_summaries =
        ocean_tensor_sparse_attention_build_summaries_active(
            materialized_key, 4, 2
        );
    for (size_t i = 0; i < 4; ++i) {
        close_to(ocean_tensor_get_flat_f32(summaries, i),
            ocean_tensor_get_flat_f32(dense_summaries, i), "summary");
    }
    ocean_tensor_handle_t hierarchy = ocean_paged_kv_cache_build_hierarchy(
        summaries, 4, 2
    );
    ocean_tensor_handle_t dense_hierarchy =
        ocean_tensor_sparse_attention_build_hierarchy_active(
            dense_summaries, 4, 2
        );
    for (size_t i = 0; i < 8; ++i) {
        close_to(ocean_tensor_get_flat_f32(hierarchy, i),
            ocean_tensor_get_flat_f32(dense_hierarchy, i), "hierarchy");
    }

    size_t query_shape[4] = {1, 1, 2, 2};
    ocean_tensor_handle_t query = ocean_tensor_zeros_nd(
        query_shape, 4, "float32", "cpu"
    );
    ocean_tensor_set_flat_f32(query, 0, 1.0f);
    ocean_tensor_set_flat_f32(query, 3, 1.0f);
    size_t route_shape[3] = {1, 1, 2};
    ocean_tensor_handle_t route = ocean_tensor_zeros_nd(
        route_shape, 3, "int32", "cpu"
    );
    ocean_tensor_set_flat_i32(route, 0, 0);
    ocean_tensor_set_flat_i32(route, 1, 1);
    ocean_tensor_handle_t paged = ocean_paged_kv_cache_sparse_attention_routed(
        cache, query, route, 4, 2, 1.0, 0, false
    );
    ocean_tensor_handle_t dense = ocean_tensor_sparse_attention(
        query, materialized_key, materialized_value, 4, 1.0, 0, false
    );
    for (size_t i = 0; i < 4; ++i) {
        close_to(ocean_tensor_get_flat_f32(paged, i),
            ocean_tensor_get_flat_f32(dense, i), "paged attention");
    }

    ocean_tensor_release(dense_hierarchy);
    ocean_tensor_release(hierarchy);
    ocean_tensor_release(dense_summaries);
    ocean_tensor_release(summaries);
    ocean_tensor_release(dense);
    ocean_tensor_release(paged);
    ocean_tensor_release(route);
    ocean_tensor_release(query);
    ocean_tensor_release(materialized_value);
    ocean_tensor_release(materialized_key);
    ocean_paged_kv_cache_release(cache);
    ocean_tensor_release(value);
    ocean_tensor_release(key);
    puts("PagedKVCache CPU reference: OK");
    return 0;
}
'''


def test_paged_kv_cache_cpu_reference():
    with tempfile.TemporaryDirectory(prefix="ocean_paged_kv_cache_") as td:
        td = Path(td)
        source = td / "test.c"
        binary = td / "test"
        source.write_text(C_SOURCE, encoding="utf-8")
        subprocess.run([
            "gcc", "-std=c11", "-O2", "-Wall", "-Wextra", "-Wpedantic", "-Werror",
            "-I", str(ROOT), str(source), str(ROOT / "std/tensor/tensor_runtime.c"),
            "-lm", "-o", str(binary),
        ], check=True)
        completed = subprocess.run(
            [str(binary)], check=False, capture_output=True, text=True
        )
        assert completed.returncode == 0, completed.stderr
        assert "PagedKVCache CPU reference: OK" in completed.stdout
