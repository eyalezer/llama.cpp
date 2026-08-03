// Note: this description is outdated
//
// An interface allowing to compute ggml_cgraph with Metal
//
// This is a fully functional interface that extends ggml with GPU support for Apple devices.
// A similar interface can be created for other GPU backends (e.g. Vulkan, CUDA, etc.)
//
// How it works?
//
// As long as your program can create and evaluate a ggml_cgraph on the CPU, you can use this
// interface to evaluate the same graph on the GPU. Instead of using ggml_graph_compute(), you
// use ggml_metal_graph_compute() (or ggml_vulkan_graph_compute(), etc.)
//
// You only need to make sure that all memory buffers that you used during the graph creation
// are mapped to the device memory with the ggml_metal_add_buffer() function. This mapping is
// used during the graph evaluation to determine the arguments of the compute kernels.
//
// Synchronization between device and host memory (for example for input and output tensors)
// is done with the ggml_metal_set_tensor() and ggml_metal_get_tensor() functions.
//

#pragma once

#include "ggml.h"
#include "ggml-backend.h"

#include <stddef.h>
#include <stdbool.h>

struct ggml_tensor;
struct ggml_cgraph;

#ifdef __cplusplus
extern "C" {
#endif

//
// backend API
// user-code should use only these functions
//

// TODO: remove in the future
GGML_BACKEND_API ggml_backend_t ggml_backend_metal_init(void);

GGML_BACKEND_API bool ggml_backend_is_metal(ggml_backend_t backend);

GGML_BACKEND_API void ggml_backend_metal_set_abort_callback(ggml_backend_t backend, ggml_abort_callback abort_callback, void * user_data);

// helper to check if the device supports a specific family
// ideally, the user code should be doing these checks
// ref: https://developer.apple.com/metal/Metal-Feature-Set-Tables.pdf
GGML_BACKEND_API bool ggml_backend_metal_supports_family(ggml_backend_t backend, int family);

// capture all command buffers committed the next time `ggml_backend_graph_compute` is called
GGML_BACKEND_API void ggml_backend_metal_capture_next_compute(ggml_backend_t backend);

GGML_BACKEND_API ggml_backend_reg_t ggml_backend_metal_reg(void);

// ---- TriAttention GPU scoring (Metal) ----
// Named triattention_mtl_* (not triattention_gpu_*) so this coexists with the
// CUDA/Vulkan APIs in ggml-cuda.h/ggml-vulkan.h without symbol clashes in
// combined-backend builds. Mirrors triattention_vk_* field-for-field; unlike
// Vulkan, the Metal backend is a process-wide singleton, so there is no
// device-index parameter.
typedef struct triattention_mtl_state triattention_mtl_state;

struct triattention_mtl_head_calib {
    const float * q_mean_real;    // [freq_count]
    const float * q_mean_imag;
    const float * q_mean_abs;
    const float * extra_weight;
};

struct triattention_mtl_config {
    uint32_t head_dim;
    uint32_t freq_count;
    uint32_t n_kv_heads;
    uint32_t n_sampled;
    uint32_t n_offsets;
    enum ggml_type k_type;
    bool     need_wht_inv;
    bool     disable_trig;
};

GGML_BACKEND_API triattention_mtl_state * triattention_mtl_init(
    ggml_backend_t backend,
    const struct triattention_mtl_config * config,
    const struct triattention_mtl_head_calib * head_calibs,
    const float * omega,
    const float * freq_scale_sq,
    const float * offsets);

// Batched scoring, mirroring the CUDA/Vulkan "upload once, enqueue N kernels,
// sync once" pattern:
//   1. triattention_mtl_batch_begin() uploads the (shared) candidate cell
//      indices/positions once and prepares a single command buffer.
//   2. triattention_mtl_batch_enqueue_head() records one dispatch per sampled
//      (layer, kv_head) into that same command buffer (no commit yet).
//   3. triattention_mtl_batch_end() commits the whole batch, waits for
//      completion, and reads all n_sampled*n_cells scores back in one copy.
GGML_BACKEND_API void triattention_mtl_batch_begin(
    triattention_mtl_state * state,
    const uint32_t * cell_indices_host,
    const int32_t  * positions_host,
    uint32_t n_cells,
    uint32_t n_total_scores);

GGML_BACKEND_API void triattention_mtl_batch_enqueue_head(
    triattention_mtl_state * state,
    const struct ggml_tensor * k_tensor,
    uint32_t kv_head_idx,
    uint32_t head_calib_idx,
    int64_t round_start,
    int agg_mode,
    uint32_t score_offset_elems);

GGML_BACKEND_API void triattention_mtl_batch_end(
    triattention_mtl_state * state,
    float * scores_host_out,
    uint32_t n_total_scores);

GGML_BACKEND_API void triattention_mtl_free(triattention_mtl_state * state);

#ifdef __cplusplus
}
#endif
