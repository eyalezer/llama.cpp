// TriAttention GPU scoring (Metal host-side API), mirroring the Vulkan
// implementation in ggml-vulkan.cpp function-for-function. See ggml-metal.h
// for the public triattention_mtl_* declarations.

#include "ggml-metal.h"

#include "ggml.h"
#include "ggml-impl.h"
#include "ggml-backend-impl.h"

#include "ggml-metal-context.h"
#include "ggml-metal-device.h"
#include "ggml-metal-impl.h"

#include <algorithm>
#include <cstring>
#include <vector>

struct triattention_mtl_state {
    ggml_metal_t          ctx;
    ggml_metal_device_t   dev;
    ggml_metal_library_t  lib;
    triattention_mtl_config cfg;

    ggml_metal_buffer_t d_omega;
    ggml_metal_buffer_t d_freq_scale_sq;
    ggml_metal_buffer_t d_offsets;
    ggml_metal_buffer_t d_q_mean_real;
    ggml_metal_buffer_t d_q_mean_imag;
    ggml_metal_buffer_t d_q_mean_abs;
    ggml_metal_buffer_t d_extra_weight;

    // Batch scratch buffers, grown on demand and reused across prune() calls
    // instead of being created/destroyed for every sampled head.
    ggml_metal_buffer_t d_cells        = nullptr;
    ggml_metal_buffer_t d_pos          = nullptr;
    ggml_metal_buffer_t d_scores       = nullptr;
    size_t              cells_capacity  = 0; // bytes
    size_t              pos_capacity    = 0; // bytes
    size_t              scores_capacity = 0; // bytes

    // Open batch state, live between batch_begin() and batch_end()
    ggml_metal_cmd_buf_t batch_cmd_buf = nullptr;
    ggml_metal_encoder_t batch_encoder = nullptr;
    uint32_t              batch_n_cells = 0;
};

static int32_t triattention_mtl_k_type_id(ggml_type t) {
    switch (t) {
        case GGML_TYPE_F32:      return 0;
        case GGML_TYPE_F16:      return 1;
        case GGML_TYPE_Q8_0:     return 2;
        case GGML_TYPE_TURBO2_0: return 3;
        case GGML_TYPE_TURBO3_0: return 4;
        case GGML_TYPE_TURBO4_0: return 5;
        default:
            GGML_ABORT("triattention_mtl: unsupported K cache type");
    }
}

// Mirrors the static helper of the same name in ggml-metal-ops.cpp: resolves
// the ggml_metal_buffer_t (and the tensor's byte offset within it) backing a
// ggml_tensor, so the K cache tensor can be bound without a full graph-compute.
static struct ggml_metal_buffer_id triattention_mtl_get_buffer_id(const struct ggml_tensor * t) {
    if (!t) {
        return { nullptr, 0 };
    }

    ggml_backend_buffer_t buffer = t->view_src ? t->view_src->buffer : t->buffer;
    ggml_metal_buffer_t   ctx    = (ggml_metal_buffer_t) buffer->context;

    return ggml_metal_buffer_get_id(ctx, t);
}

triattention_mtl_state * triattention_mtl_init(
        ggml_backend_t backend,
        const struct triattention_mtl_config * config,
        const struct triattention_mtl_head_calib * head_calibs,
        const float * omega,
        const float * freq_scale_sq,
        const float * offsets) {
    GGML_ASSERT(ggml_backend_is_metal(backend));

    ggml_metal_t ctx = (ggml_metal_t) backend->context;
    ggml_metal_device_t dev = ggml_metal_get_device(ctx);

    triattention_mtl_state * state = new triattention_mtl_state();
    state->ctx = ctx;
    state->dev = dev;
    state->lib = ggml_metal_device_get_library(dev);
    state->cfg = *config;

    const size_t fc_bytes    = sizeof(float) * config->freq_count;
    const size_t calib_bytes = sizeof(float) * config->freq_count * config->n_sampled;

    state->d_omega         = ggml_metal_buffer_init(dev, fc_bytes, true);
    state->d_freq_scale_sq = ggml_metal_buffer_init(dev, fc_bytes, true);
    state->d_offsets       = ggml_metal_buffer_init(dev, sizeof(float) * config->n_offsets, true);
    state->d_q_mean_real   = ggml_metal_buffer_init(dev, calib_bytes, true);
    state->d_q_mean_imag   = ggml_metal_buffer_init(dev, calib_bytes, true);
    state->d_q_mean_abs    = ggml_metal_buffer_init(dev, calib_bytes, true);
    state->d_extra_weight  = ggml_metal_buffer_init(dev, calib_bytes, true);

    std::memcpy(ggml_metal_buffer_get_base(state->d_omega), omega, fc_bytes);
    std::memcpy(ggml_metal_buffer_get_base(state->d_freq_scale_sq), freq_scale_sq, fc_bytes);
    std::memcpy(ggml_metal_buffer_get_base(state->d_offsets), offsets, sizeof(float) * config->n_offsets);

    // Calibration arrays are per-head; pack them contiguously so batch_enqueue_head
    // can bind a sub-range [head_calib_idx * freq_count, ...] via a byte offset.
    float * real_dst = (float *) ggml_metal_buffer_get_base(state->d_q_mean_real);
    float * imag_dst = (float *) ggml_metal_buffer_get_base(state->d_q_mean_imag);
    float * abs_dst  = (float *) ggml_metal_buffer_get_base(state->d_q_mean_abs);
    float * ew_dst   = (float *) ggml_metal_buffer_get_base(state->d_extra_weight);
    for (uint32_t h = 0; h < config->n_sampled; h++) {
        std::memcpy(real_dst + h * config->freq_count, head_calibs[h].q_mean_real,   fc_bytes);
        std::memcpy(imag_dst + h * config->freq_count, head_calibs[h].q_mean_imag,   fc_bytes);
        std::memcpy(abs_dst  + h * config->freq_count, head_calibs[h].q_mean_abs,    fc_bytes);
        std::memcpy(ew_dst   + h * config->freq_count, head_calibs[h].extra_weight,  fc_bytes);
    }

    return state;
}

// Grows `buf` in-place (destroy + recreate) if its current capacity is
// smaller than `needed_bytes`; no-op otherwise. Lets batch scratch buffers
// be reused across prune() calls instead of churned per sampled head.
static void triattention_mtl_ensure_capacity(ggml_metal_device_t dev, ggml_metal_buffer_t & buf, size_t & capacity, size_t needed_bytes) {
    if (buf && capacity >= needed_bytes) {
        return;
    }
    if (buf) {
        ggml_metal_buffer_free(buf);
    }
    buf = ggml_metal_buffer_init(dev, needed_bytes, true);
    capacity = needed_bytes;
}

// Uploads the (shared, per-prune-call) candidate cell indices/positions once
// and prepares a single command buffer that subsequent batch_enqueue_head()
// calls record dispatches into — mirrors the CUDA/Vulkan path's single upload
// followed by N enqueued kernel dispatches.
void triattention_mtl_batch_begin(
        triattention_mtl_state * state,
        const uint32_t * cell_indices_host,
        const int32_t  * positions_host,
        uint32_t n_cells,
        uint32_t n_total_scores) {
    ggml_metal_device_t dev = state->dev;

    triattention_mtl_ensure_capacity(dev, state->d_cells,  state->cells_capacity,  sizeof(uint32_t) * n_cells);
    triattention_mtl_ensure_capacity(dev, state->d_pos,    state->pos_capacity,    sizeof(int32_t)  * n_cells);
    triattention_mtl_ensure_capacity(dev, state->d_scores, state->scores_capacity, sizeof(float)    * n_total_scores);

    std::memcpy(ggml_metal_buffer_get_base(state->d_cells), cell_indices_host, sizeof(uint32_t) * n_cells);
    std::memcpy(ggml_metal_buffer_get_base(state->d_pos),   positions_host,    sizeof(int32_t)  * n_cells);

    state->batch_n_cells = n_cells;
    state->batch_cmd_buf = ggml_metal_device_new_command_buffer(dev);
    state->batch_encoder = ggml_metal_encoder_init(state->batch_cmd_buf, false);
}

// Records one head's dispatch into the currently open batch (no commit).
void triattention_mtl_batch_enqueue_head(
        triattention_mtl_state * state,
        const struct ggml_tensor * k_tensor,
        uint32_t kv_head_idx,
        uint32_t head_calib_idx,
        int64_t round_start,
        int agg_mode,
        uint32_t score_offset_elems) {
    const uint32_t n_cells = state->batch_n_cells;
    if (n_cells == 0) {
        return;
    }

    const triattention_mtl_config & cfg = state->cfg;

    const size_t row_bytes          = ggml_row_size(cfg.k_type, cfg.head_dim);
    const size_t head_offset_bytes  = row_bytes * kv_head_idx;
    const size_t calib_offset_bytes = sizeof(float) * cfg.freq_count * head_calib_idx;
    const size_t score_offset_bytes = sizeof(float) * score_offset_elems;

    ggml_metal_kargs_triattention_score args = {
        /* .k_type_id         = */ triattention_mtl_k_type_id(cfg.k_type),
        /* .need_wht_inv      = */ cfg.need_wht_inv ? 1 : 0,
        /* .row_bytes         = */ (uint32_t) row_bytes,
        /* .head_offset_bytes = */ (uint32_t) head_offset_bytes,
        /* .padded_hd         = */ cfg.head_dim,
        /* .n_cells           = */ n_cells,
        /* .round_start       = */ (int32_t) round_start,
        /* .freq_count        = */ cfg.freq_count,
        /* .n_offsets         = */ cfg.n_offsets,
        /* .agg_mode          = */ (uint32_t) agg_mode,
        /* .disable_trig      = */ cfg.disable_trig ? 1u : 0u,
    };

    struct ggml_metal_buffer_id bid_k        = triattention_mtl_get_buffer_id(k_tensor);
    struct ggml_metal_buffer_id bid_cells    = ggml_metal_buffer_get_id_raw(state->d_cells);
    struct ggml_metal_buffer_id bid_pos      = ggml_metal_buffer_get_id_raw(state->d_pos);
    struct ggml_metal_buffer_id bid_omega    = ggml_metal_buffer_get_id_raw(state->d_omega);
    struct ggml_metal_buffer_id bid_fscale   = ggml_metal_buffer_get_id_raw(state->d_freq_scale_sq);
    struct ggml_metal_buffer_id bid_offsets  = ggml_metal_buffer_get_id_raw(state->d_offsets);
    struct ggml_metal_buffer_id bid_qreal    = ggml_metal_buffer_get_id_raw(state->d_q_mean_real);
    struct ggml_metal_buffer_id bid_qimag    = ggml_metal_buffer_get_id_raw(state->d_q_mean_imag);
    struct ggml_metal_buffer_id bid_qabs     = ggml_metal_buffer_get_id_raw(state->d_q_mean_abs);
    struct ggml_metal_buffer_id bid_ew       = ggml_metal_buffer_get_id_raw(state->d_extra_weight);
    struct ggml_metal_buffer_id bid_scores   = ggml_metal_buffer_get_id_raw(state->d_scores);

    bid_qreal.offs  += calib_offset_bytes;
    bid_qimag.offs  += calib_offset_bytes;
    bid_qabs.offs   += calib_offset_bytes;
    bid_ew.offs     += calib_offset_bytes;
    bid_scores.offs += score_offset_bytes;

    ggml_metal_pipeline_with_params pipeline = ggml_metal_library_get_pipeline_triattention_score(state->lib);

    ggml_metal_encoder_t enc = state->batch_encoder;

    ggml_metal_encoder_set_pipeline(enc, pipeline);
    ggml_metal_encoder_set_bytes(enc, &args, sizeof(args), 0);
    ggml_metal_encoder_set_buffer(enc, bid_k,       1);
    ggml_metal_encoder_set_buffer(enc, bid_cells,   2);
    ggml_metal_encoder_set_buffer(enc, bid_pos,     3);
    ggml_metal_encoder_set_buffer(enc, bid_omega,   4);
    ggml_metal_encoder_set_buffer(enc, bid_fscale,  5);
    ggml_metal_encoder_set_buffer(enc, bid_offsets, 6);
    ggml_metal_encoder_set_buffer(enc, bid_qreal,   7);
    ggml_metal_encoder_set_buffer(enc, bid_qimag,   8);
    ggml_metal_encoder_set_buffer(enc, bid_qabs,    9);
    ggml_metal_encoder_set_buffer(enc, bid_ew,      10);
    ggml_metal_encoder_set_buffer(enc, bid_scores,  11);

    ggml_metal_encoder_dispatch_threadgroups(enc, (int) n_cells, 1, 1, (int) cfg.freq_count, 1, 1);
}

// Commits the whole batch, waits for completion, and reads all scores back
// to host in one copy.
void triattention_mtl_batch_end(
        triattention_mtl_state * state,
        float * scores_host_out,
        uint32_t n_total_scores) {
    if (state->batch_n_cells == 0) {
        return;
    }

    ggml_metal_encoder_end_encoding(state->batch_encoder);
    ggml_metal_encoder_free(state->batch_encoder);
    ggml_metal_cmd_buf_commit_and_wait(state->batch_cmd_buf);

    std::memcpy(scores_host_out, ggml_metal_buffer_get_base(state->d_scores), sizeof(float) * n_total_scores);

    state->batch_encoder = nullptr;
    state->batch_cmd_buf = nullptr;
    state->batch_n_cells = 0;
}

void triattention_mtl_free(triattention_mtl_state * state) {
    if (state == nullptr) {
        return;
    }
    ggml_metal_buffer_free(state->d_omega);
    ggml_metal_buffer_free(state->d_freq_scale_sq);
    ggml_metal_buffer_free(state->d_offsets);
    ggml_metal_buffer_free(state->d_q_mean_real);
    ggml_metal_buffer_free(state->d_q_mean_imag);
    ggml_metal_buffer_free(state->d_q_mean_abs);
    ggml_metal_buffer_free(state->d_extra_weight);
    if (state->d_cells)  ggml_metal_buffer_free(state->d_cells);
    if (state->d_pos)    ggml_metal_buffer_free(state->d_pos);
    if (state->d_scores) ggml_metal_buffer_free(state->d_scores);
    delete state;
}
