#ifndef OCEAN_STD_TENSOR_CUDA_BACKEND_H
#define OCEAN_STD_TENSOR_CUDA_BACKEND_H

#include <stddef.h>
#include <stdint.h>

#define OCEAN_CUDA_MAX_BROADCAST_RANK 16

typedef struct ocean_cuda_broadcast_desc {
    int ndim;
    size_t output_shape[OCEAN_CUDA_MAX_BROADCAST_RANK];
    size_t left_shape[OCEAN_CUDA_MAX_BROADCAST_RANK];
    size_t right_shape[OCEAN_CUDA_MAX_BROADCAST_RANK];
    size_t left_strides[OCEAN_CUDA_MAX_BROADCAST_RANK];
    size_t right_strides[OCEAN_CUDA_MAX_BROADCAST_RANK];
} ocean_cuda_broadcast_desc;

typedef struct ocean_cuda_strided_copy_desc {
    int ndim;
    size_t shape[OCEAN_CUDA_MAX_BROADCAST_RANK];
    size_t source_strides[OCEAN_CUDA_MAX_BROADCAST_RANK];
    size_t destination_strides[OCEAN_CUDA_MAX_BROADCAST_RANK];
    size_t item_size;
    size_t total;
} ocean_cuda_strided_copy_desc;

#ifdef __cplusplus
extern "C" {
#endif

void *ocean_cuda_malloc(size_t bytes);
void ocean_cuda_free(void *device_data);
void ocean_cuda_memcpy_h2d(void *device_data, const void *host_data, size_t bytes);
void ocean_cuda_memcpy_d2h(void *host_data, const void *device_data, size_t bytes);
void ocean_cuda_memcpy_d2d(void *destination, const void *source, size_t bytes);
void ocean_cuda_zero(void *device_data, size_t bytes);
void ocean_cuda_synchronize(void);
void ocean_cuda_copy_strided(
    const void *source,
    void *destination,
    const ocean_cuda_strided_copy_desc *descriptor
);
void ocean_cuda_fill_f32(void *device_data, float value, size_t size);
void ocean_cuda_fill_i32(void *device_data, int value, size_t size);
void ocean_cuda_set_f32(void *device_data, size_t index, float value);
void ocean_cuda_set_i32(void *device_data, size_t index, int value);
void ocean_cuda_set_i64(void *device_data, size_t index, int64_t value);

void ocean_cuda_binary_f32(
    const void *left,
    const void *right,
    void *output,
    size_t size,
    int operation
);
void ocean_cuda_binary_i32(
    const void *left,
    const void *right,
    void *output,
    size_t size,
    int operation
);
void ocean_cuda_binary_broadcast_f32(
    const void *left,
    const void *right,
    void *output,
    size_t size,
    int operation,
    const ocean_cuda_broadcast_desc *descriptor
);
void ocean_cuda_scalar_f32(
    const void *input,
    void *output,
    size_t size,
    float scalar,
    int operation
);
void ocean_cuda_scalar_i32(
    const void *input,
    void *output,
    size_t size,
    int scalar,
    int operation
);
void ocean_cuda_matmul_f32(
    const void *left,
    const void *right,
    void *output,
    int rows,
    int inner,
    int columns
);
void ocean_cuda_matmul_i32(
    const void *left,
    const void *right,
    void *output,
    int rows,
    int inner,
    int columns
);

void ocean_cuda_softmax_last_dim(
    const void *input, void *output, int rows, int width
);
void ocean_cuda_causal_softmax(
    const void *input, void *output, int rows, int width
);
void ocean_cuda_layer_norm_last_dim(
    const void *input, void *output, int rows, int width, float epsilon
);
void ocean_cuda_layer_norm_affine_last_dim(
    const void *input,
    const void *gamma,
    const void *beta,
    void *output,
    int rows,
    int width,
    float epsilon
);
void ocean_cuda_gelu(const void *input, void *output, size_t size);
void ocean_cuda_ternary_quantize(
    const void *input, void *output, size_t size
);
void ocean_cuda_ternary_pack(
    const void *input,
    void *output,
    int source_rows,
    int source_cols,
    int output_rows,
    int packed_cols,
    float scale,
    int transpose
);
void ocean_cuda_packed_linear(
    const void *input,
    const void *packed,
    const void *bias,
    void *output,
    int rows,
    int cols_a,
    int cols_b,
    int packed_cols,
    float scale
);
void ocean_cuda_packed_qkv(
    const void *input,
    const void *q_packed,
    const void *q_bias,
    const void *k_packed,
    const void *k_bias,
    const void *v_packed,
    const void *v_bias,
    void *output,
    int rows,
    int cols_a,
    int cols_b,
    int packed_cols,
    float q_scale,
    float k_scale,
    float v_scale
);
void ocean_cuda_packed_qkv_split(
    const void *input,
    const void *q_packed,
    const void *q_bias,
    const void *k_packed,
    const void *k_bias,
    const void *v_packed,
    const void *v_bias,
    void *q_output,
    void *k_output,
    void *v_output,
    int rows,
    int cols_a,
    int cols_b,
    int packed_cols,
    float q_scale,
    float k_scale,
    float v_scale
);
void ocean_cuda_packed_qkv_attention_decode(
    const void *input,
    const void *q_packed,
    const void *q_bias,
    const void *k_packed,
    const void *k_bias,
    const void *v_packed,
    const void *v_bias,
    void *cache_k,
    void *cache_v,
    void *output,
    int cols_a,
    int packed_cols,
    int max_seq,
    int position,
    int n_heads,
    int head_dim,
    float q_scale,
    float k_scale,
    float v_scale
);
void ocean_cuda_cache_write(
    void *cache,
    const void *value,
    int batches,
    int heads,
    int sequence,
    int value_sequence,
    int width,
    int position
);
void ocean_cuda_paged_kv_write(
    void *key_page,
    void *value_page,
    const void *key,
    const void *value,
    int batches,
    int heads,
    int page_size,
    int head_dim,
    int source_sequence,
    int source_start,
    int destination_start,
    int count
);
void ocean_cuda_packed_qkv_paged_append(
    const void *input,
    const void *q_packed,
    const void *q_bias,
    const void *k_packed,
    const void *k_bias,
    const void *v_packed,
    const void *v_bias,
    void *q_output,
    void *key_page,
    void *value_page,
    int batches,
    int cols_a,
    int cols_b,
    int packed_cols,
    int heads,
    int head_dim,
    int page_size,
    int destination,
    float q_scale,
    float k_scale,
    float v_scale
);
void ocean_cuda_permute_swap12_f32(
    const void *input,
    void *output,
    int batches,
    int first_dim,
    int second_dim,
    int head_dim
);
void ocean_cuda_cache_slice(
    const void *cache,
    void *output,
    int batches,
    int heads,
    int source_sequence,
    int output_sequence,
    int width,
    int start
);
void *ocean_cuda_page_table_create(int capacity);
void ocean_cuda_page_table_update(
    void *table,
    int index,
    const void *page
);
void ocean_cuda_page_table_release(void *table);
void ocean_cuda_sparse_attention_routed_paged(
    const void *query,
    const void *key_pages,
    const void *value_pages,
    const void *route,
    void *output,
    int batches,
    int heads,
    int query_length,
    int active_length,
    int head_dim,
    int route_blocks,
    int page_size,
    float scale,
    int query_start,
    int causal
);
void ocean_cuda_sparse_build_paged_summary(
    const void *key_pages,
    void *summaries,
    int batches,
    int heads,
    int page_count,
    int page_size,
    int active_length,
    int head_dim,
    int page_index
);
void ocean_cuda_sparse_build_summaries(
    const void *key,
    void *summaries,
    int batches,
    int heads,
    int key_length,
    int active_length,
    int head_dim,
    int block_size
);
void ocean_cuda_sparse_update_summary(
    const void *key,
    void *summaries,
    int batches,
    int heads,
    int key_length,
    int summary_blocks,
    int active_length,
    int head_dim,
    int block_size,
    int position
);
void ocean_cuda_sparse_build_hierarchy(
    const void *summaries,
    void *hierarchy,
    int batches,
    int heads,
    int summary_blocks,
    int valid_blocks,
    int head_dim,
    int leaf_count
);
void ocean_cuda_sparse_update_hierarchy(
    const void *summaries,
    void *hierarchy,
    int batches,
    int heads,
    int summary_blocks,
    int valid_blocks,
    int head_dim,
    int leaf_count,
    int block
);
void ocean_cuda_sparse_build_route(
    const void *key,
    const void *summaries,
    void *route,
    int batches,
    int heads,
    int key_length,
    int summary_blocks,
    int active_length,
    int head_dim,
    int summary_window,
    int semantic_blocks,
    int local_blocks,
    int block_size,
    unsigned int random_seed
);
void ocean_cuda_sparse_build_hierarchical_route(
    const void *key,
    const void *hierarchy,
    void *route,
    int batches,
    int heads,
    int key_length,
    int tree_nodes,
    int leaf_count,
    int active_length,
    int head_dim,
    int summary_window,
    int semantic_blocks,
    int local_blocks,
    int block_size,
    unsigned int random_seed
);
void ocean_cuda_sparse_build_paged_hierarchical_route(
    const void *key_pages,
    const void *hierarchy,
    void *route,
    int batches,
    int heads,
    int page_count,
    int page_size,
    int tree_nodes,
    int leaf_count,
    int active_length,
    int head_dim,
    int summary_window,
    int semantic_blocks,
    int local_blocks,
    int block_size,
    unsigned int random_seed
);
void ocean_cuda_sparse_attention_routed(
    const void *query,
    const void *key,
    const void *value,
    const void *route,
    void *output,
    int batches,
    int heads,
    int query_length,
    int key_length,
    int active_length,
    int head_dim,
    int route_blocks,
    int block_size,
    float scale,
    int query_start,
    int causal
);
void ocean_cuda_sparse_attention(
    const void *query,
    const void *key,
    const void *value,
    const void *summaries,
    void *output,
    int batches,
    int heads,
    int query_length,
    int key_length,
    int active_length,
    int head_dim,
    int summary_blocks,
    int top_k,
    int top_blocks,
    int block_size,
    float scale,
    int query_start,
    int causal
);
void ocean_cuda_embedding_forward(
    const void *weight,
    const void *indices,
    void *output,
    int index_count,
    int vocab,
    int dim
);
int ocean_cuda_argmax_f32(const void *input, size_t size);

#ifdef __cplusplus
}
#endif

#endif
