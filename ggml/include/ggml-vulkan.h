#pragma once

#include "ggml.h"
#include "ggml-backend.h"

#ifdef  __cplusplus
extern "C" {
#endif

#define GGML_VK_NAME "Vulkan"
#define GGML_VK_MAX_DEVICES 16

// backend API
GGML_BACKEND_API ggml_backend_t ggml_backend_vk_init(size_t dev_num);

GGML_BACKEND_API bool ggml_backend_is_vk(ggml_backend_t backend);
GGML_BACKEND_API int  ggml_backend_vk_get_device_count(void);
GGML_BACKEND_API void ggml_backend_vk_get_device_description(int device, char * description, size_t description_size);
GGML_BACKEND_API void ggml_backend_vk_get_device_memory(int device, size_t * free, size_t * total);

GGML_BACKEND_API ggml_backend_buffer_type_t ggml_backend_vk_buffer_type(size_t dev_num);
// pinned host buffer for use with the CPU backend for faster copies between CPU and GPU
GGML_BACKEND_API ggml_backend_buffer_type_t ggml_backend_vk_host_buffer_type(void);

GGML_BACKEND_API ggml_backend_reg_t ggml_backend_vk_reg(void);

// ---- TriAttention GPU scoring (Vulkan) ----
// Named triattention_vk_* (not triattention_gpu_*) so this coexists with the
// CUDA API in ggml-cuda.h without symbol clashes in combined-backend builds.
// Unlike CUDA's raw-device-pointer API, this one takes ggml_tensor* for the K
// cache since Vulkan tensor storage is a VkBuffer+offset, not a bare pointer.
typedef struct triattention_vk_state triattention_vk_state;

struct triattention_vk_head_calib {
    const float * q_mean_real;    // [freq_count]
    const float * q_mean_imag;
    const float * q_mean_abs;
    const float * extra_weight;
};

struct triattention_vk_config {
    uint32_t head_dim;
    uint32_t freq_count;
    uint32_t n_kv_heads;
    uint32_t n_sampled;
    uint32_t n_offsets;
    enum ggml_type k_type;
    bool     need_wht_inv;
    bool     disable_trig;
};

GGML_BACKEND_API triattention_vk_state * triattention_vk_init(
    ggml_backend_t backend,
    const struct triattention_vk_config * config,
    const struct triattention_vk_head_calib * head_calibs,
    const float * omega,
    const float * freq_scale_sq,
    const float * offsets);

// Batched scoring, mirroring the CUDA path's "upload once, enqueue N kernels,
// sync once" pattern instead of one submit+fence-wait per sampled head:
//   1. triattention_vk_batch_begin() uploads the (shared) candidate cell
//      indices/positions once and opens a single command buffer.
//   2. triattention_vk_batch_enqueue_head() records one dispatch per sampled
//      (layer, kv_head) into that same command buffer (no submit yet).
//   3. triattention_vk_batch_end() submits the whole batch, waits on a single
//      fence, and reads all n_sampled*n_cells scores back in one copy.
GGML_BACKEND_API void triattention_vk_batch_begin(
    triattention_vk_state * state,
    const uint32_t * cell_indices_host,
    const int32_t  * positions_host,
    uint32_t n_cells,
    uint32_t n_total_scores);

GGML_BACKEND_API void triattention_vk_batch_enqueue_head(
    triattention_vk_state * state,
    const struct ggml_tensor * k_tensor,
    uint32_t kv_head_idx,
    uint32_t head_calib_idx,
    int64_t round_start,
    int agg_mode,
    uint32_t score_offset_elems);

GGML_BACKEND_API void triattention_vk_batch_end(
    triattention_vk_state * state,
    float * scores_host_out,
    uint32_t n_total_scores);

GGML_BACKEND_API void triattention_vk_free(triattention_vk_state * state);

#ifdef  __cplusplus
}
#endif
