// TriAttention: Trigonometric KV Cache Eviction for llama.cpp
// Based on arXiv 2604.04921 (MIT/NVIDIA/ZJU)
//
// This file implements the complete TriAttention scoring and pruning pipeline:
//   1. Binary calibration file loader (.triattention format)
//   2. RoPE inversion (post-RoPE K → pre-RoPE K)
//   3. Trigonometric key importance scoring (Eqs. 6-10 from paper)
//   4. Three pruning modes: global union, per-KV-head, per-layer-per-head
//   5. Position tracking hooks for correct RoPE inversion after pruning
//   6. KV cache integration hooks
//
// All math references cite equation numbers from: "TriAttention: Decoding-Time
// Trigonometric Key Cache Eviction for Long-Context LLM Inference" (2604.04921)

#include "llama-triattention.h"
#include "llama-kv-cache.h"
#include "llama-hparams.h"
#include "ggml.h"
#include "ggml-backend.h"
#ifdef GGML_USE_CUDA
#include "ggml-cuda.h"   // GPU scoring: triattention_gpu_init, _score_head, etc.
#endif
#ifdef GGML_USE_VULKAN
#include "ggml-vulkan.h" // GPU scoring (Vulkan): triattention_vk_init, _score_head, etc.
#endif
#ifdef GGML_USE_METAL
#include "ggml-metal.h"  // GPU scoring (Metal): triattention_mtl_init, _score_head, etc.
#endif

// Block types and dequant declarations are in ggml-common.h (ggml/src/)
// which is not on the include path for src/. We declare the dequant
// functions with void* parameters and cast at call sites.
// Block sizes (bytes per 128 elements): turbo2=10, turbo3=14, turbo4=68, q8_0=34

#ifdef _MSC_VER
#define _USE_MATH_DEFINES
#endif
#include <math.h>

#include <algorithm>
#include <cassert>
#include <cstdlib>
#include <cstring>
#include <numeric>
#include <random>
#include <vector>

// For timing
#ifdef _WIN32
#define WIN32_LEAN_AND_MEAN
#ifndef NOMINMAX
#   define NOMINMAX
#endif
#include <windows.h>
#else
#include <sys/time.h>
#endif

// Pre-computed WHT inverse rotation matrix R^T (128x128)
// Used to convert turbo2/turbo3 dequant output from WHT-rotated space
// back to the original post-RoPE embedding space.
// turbo4 dequant already applies R^T internally, so this is only needed
// for turbo2_0 and turbo3_0 types.
#include "turbo-rotation-data.h"

// TurboQuant dequant function declarations (from ggml-turbo-quant.c)
// Using void* since block type definitions live in ggml-common.h (not on include path)
extern "C" {
    void dequantize_row_turbo2_0(const void * GGML_RESTRICT x, float * GGML_RESTRICT y, int64_t k);
    void dequantize_row_turbo3_0(const void * GGML_RESTRICT x, float * GGML_RESTRICT y, int64_t k);
    void dequantize_row_turbo4_0(const void * GGML_RESTRICT x, float * GGML_RESTRICT y, int64_t k);
}

// Standard ggml dequant for Q8_0, F16, etc.
extern "C" {
    void dequantize_row_q8_0(const void * GGML_RESTRICT x, float * GGML_RESTRICT y, int64_t k);
}

// ============================================================================
// Internal helpers
// ============================================================================

static double triattention_time_ms(void) {
#ifdef _WIN32
    LARGE_INTEGER freq, cnt;
    QueryPerformanceFrequency(&freq);
    QueryPerformanceCounter(&cnt);
    return (double)cnt.QuadPart / (double)freq.QuadPart * 1000.0;
#else
    struct timeval tv;
    gettimeofday(&tv, NULL);
    return (double)tv.tv_sec * 1000.0 + (double)tv.tv_usec / 1000.0;
#endif
}

// Matrix-vector multiply: out[i] = sum_j mat[i*d + j] * vec[j]
// Used for inverse WHT rotation on turbo2/turbo3 dequant output
static void matvec_128(const float * mat, const float * vec, float * out) {
    for (int i = 0; i < 128; i++) {
        float sum = 0.0f;
        const float * row = mat + i * 128;
        for (int j = 0; j < 128; j++) {
            sum += row[j] * vec[j];
        }
        out[i] = sum;
    }
}

// ============================================================================
// Binary calibration file I/O
// ============================================================================

// Load .triattention calibration file (TRIA v1/v2 dense format, see
// llama-triattention.h for the layout). Returns nullptr on any error, with
// diagnostic printed to stderr.
static triattention_calibration * triattention_load_calibration(const char * path) {
    FILE * f = fopen(path, "rb");
    if (!f) {
        fprintf(stderr, "[TriAttention] ERROR: cannot open calibration file: %s\n", path);
        return nullptr;
    }

    uint32_t magic;
    if (fread(&magic, sizeof(uint32_t), 1, f) != 1 || magic != TRIATTENTION_MAGIC) {
        fprintf(stderr, "[TriAttention] ERROR: invalid magic in %s (got 0x%08x, expected 0x%08x)\n",
                path, magic, TRIATTENTION_MAGIC);
        fclose(f);
        return nullptr;
    }

    uint32_t version;
    if (fread(&version, sizeof(uint32_t), 1, f) != 1 ||
        version < TRIATTENTION_VERSION_MIN || version > TRIATTENTION_VERSION_MAX) {
        fprintf(stderr, "[TriAttention] ERROR: unsupported version %u in %s (expected %u-%u)\n",
                version, path, TRIATTENTION_VERSION_MIN, TRIATTENTION_VERSION_MAX);
        fclose(f);
        return nullptr;
    }

    auto * cal = new triattention_calibration();
    memset(cal, 0, sizeof(triattention_calibration));

    uint32_t num_heads = 0;
    float rope_theta_f32 = 0.0f;
    float attn_scale = 0.0f;

    bool ok = true;
    ok = ok && fread(&cal->num_layers,   sizeof(uint32_t), 1, f) == 1;
    ok = ok && fread(&num_heads,         sizeof(uint32_t), 1, f) == 1;
    ok = ok && fread(&cal->num_kv_heads, sizeof(uint32_t), 1, f) == 1;
    ok = ok && fread(&cal->head_dim,     sizeof(uint32_t), 1, f) == 1;
    ok = ok && fread(&cal->freq_count,   sizeof(uint32_t), 1, f) == 1;
    ok = ok && fread(&rope_theta_f32,    sizeof(float),    1, f) == 1;
    ok = ok && fread(&attn_scale,        sizeof(float),    1, f) == 1;

    if (!ok) {
        fprintf(stderr, "[TriAttention] ERROR: truncated header in %s\n", path);
        delete cal;
        fclose(f);
        return nullptr;
    }
    (void) attn_scale; // not applied yet, see docs/TRIATTENTION.md

    cal->num_attn_heads = num_heads;
    cal->rope_theta     = (double) rope_theta_f32;

    // Skip reserved bytes up to the fixed 64-byte header.
    if (fseek(f, TRIATTENTION_HEADER_SIZE, SEEK_SET) != 0) {
        fprintf(stderr, "[TriAttention] ERROR: cannot seek past header in %s\n", path);
        delete cal;
        fclose(f);
        return nullptr;
    }

    if (cal->freq_count != cal->head_dim / 2) {
        fprintf(stderr, "[TriAttention] ERROR: freq_count (%u) != head_dim/2 (%u) in %s\n",
                cal->freq_count, cal->head_dim / 2, path);
        delete cal;
        fclose(f);
        return nullptr;
    }

    if (cal->num_attn_heads == 0 || cal->num_kv_heads == 0 ||
        cal->num_attn_heads % cal->num_kv_heads != 0) {
        fprintf(stderr, "[TriAttention] ERROR: invalid head counts (attn=%u, kv=%u) in %s\n",
                cal->num_attn_heads, cal->num_kv_heads, path);
        delete cal;
        fclose(f);
        return nullptr;
    }

    cal->num_kv_groups = cal->num_attn_heads / cal->num_kv_heads;

    // v2+: per-layer budget scales. v1 files don't have this block — default to 1.0.
    cal->layer_budget_scales = new float[cal->num_layers];
    if (version >= 2) {
        if (fread(cal->layer_budget_scales, sizeof(float), cal->num_layers, f) != cal->num_layers) {
            fprintf(stderr, "[TriAttention] ERROR: truncated layer_budget_scales in %s\n", path);
            delete[] cal->layer_budget_scales;
            delete cal;
            fclose(f);
            return nullptr;
        }
    } else {
        for (uint32_t li = 0; li < cal->num_layers; li++) {
            cal->layer_budget_scales[li] = 1.0f;
        }
    }

    const uint32_t fc = cal->freq_count;

    // Dense layer-major grid: for li in [0,num_layers) for hi in [0,num_heads).
    // Entries with all-zero q_abs_mean (SSM layers / uncaptured heads) are skipped.
    std::vector<uint32_t> sampled_layer;
    std::vector<uint32_t> sampled_head;
    std::vector<triattention_head_stats> head_stats;

    std::vector<float> q_mean_real_tmp(fc);
    std::vector<float> q_mean_imag_tmp(fc);
    std::vector<float> q_abs_mean_tmp(fc);
    std::vector<float> mrl_tmp(fc);

    for (uint32_t li = 0; li < cal->num_layers && ok; li++) {
        for (uint32_t hi = 0; hi < cal->num_attn_heads && ok; hi++) {
            ok = ok && fread(q_mean_real_tmp.data(), sizeof(float), fc, f) == fc;
            ok = ok && fread(q_mean_imag_tmp.data(), sizeof(float), fc, f) == fc;
            ok = ok && fread(q_abs_mean_tmp.data(),  sizeof(float), fc, f) == fc;
            ok = ok && fread(mrl_tmp.data(),          sizeof(float), fc, f) == fc; // discarded

            if (!ok) {
                break;
            }

            bool all_zero = true;
            for (uint32_t k = 0; k < fc; k++) {
                if (q_abs_mean_tmp[k] != 0.0f) {
                    all_zero = false;
                    break;
                }
            }
            if (all_zero) {
                continue;
            }

            triattention_head_stats hs;
            hs.q_mean_real  = new float[fc];
            hs.q_mean_imag  = new float[fc];
            hs.q_abs_mean   = new float[fc];
            hs.q_mean_abs   = nullptr;  // computed at init time
            hs.extra_weight = nullptr;  // computed at init time
            memcpy(hs.q_mean_real, q_mean_real_tmp.data(), fc * sizeof(float));
            memcpy(hs.q_mean_imag, q_mean_imag_tmp.data(), fc * sizeof(float));
            memcpy(hs.q_abs_mean,  q_abs_mean_tmp.data(),  fc * sizeof(float));

            sampled_layer.push_back(li);
            sampled_head.push_back(hi);
            head_stats.push_back(hs);
        }
    }

    if (!ok) {
        fprintf(stderr, "[TriAttention] ERROR: truncated stats grid in %s\n", path);
        for (auto & hs : head_stats) {
            delete[] hs.q_mean_real;
            delete[] hs.q_mean_imag;
            delete[] hs.q_abs_mean;
        }
        delete[] cal->layer_budget_scales;
        delete cal;
        fclose(f);
        return nullptr;
    }

    fclose(f);

    cal->n_sampled     = (uint32_t) sampled_layer.size();
    cal->sampled_layer = new uint32_t[cal->n_sampled];
    cal->sampled_head  = new uint32_t[cal->n_sampled];
    cal->head_stats    = new triattention_head_stats[cal->n_sampled];
    memcpy(cal->sampled_layer, sampled_layer.data(), cal->n_sampled * sizeof(uint32_t));
    memcpy(cal->sampled_head,  sampled_head.data(),  cal->n_sampled * sizeof(uint32_t));
    for (uint32_t h = 0; h < cal->n_sampled; h++) {
        cal->head_stats[h] = head_stats[h];
    }

    cal->model_name[0] = '\0'; // not carried by this file format

    fprintf(stderr, "[TriAttention] Loaded calibration: layers=%u, attn_heads=%u, kv_heads=%u, "
            "head_dim=%u, sampled=%u, rope_theta=%.1f, version=%u\n",
            cal->num_layers, cal->num_attn_heads,
            cal->num_kv_heads, cal->head_dim, cal->n_sampled, cal->rope_theta, version);

    return cal;
}

static void triattention_free_calibration(triattention_calibration * cal) {
    if (!cal) return;

    for (uint32_t h = 0; h < cal->n_sampled; h++) {
        delete[] cal->head_stats[h].q_mean_real;
        delete[] cal->head_stats[h].q_mean_imag;
        delete[] cal->head_stats[h].q_abs_mean;
        delete[] cal->head_stats[h].q_mean_abs;
        delete[] cal->head_stats[h].extra_weight;
    }
    delete[] cal->sampled_layer;
    delete[] cal->sampled_head;
    delete[] cal->head_stats;
    delete[] cal->layer_budget_scales;
    delete cal;
}

// ============================================================================
// Precomputation at init time
// ============================================================================

// Build RoPE frequency array: omega[f] = rope_theta^(-2f/head_dim)
// Paper Eq. 1: theta_f = base^{-2f/d}
static void triattention_build_omega(float * omega, uint32_t freq_count, uint32_t head_dim, double rope_theta) {
    for (uint32_t f = 0; f < freq_count; f++) {
        double exponent = -2.0 * (double)f / (double)head_dim;
        omega[f] = (float)pow(rope_theta, exponent);
    }
}

// Build frequency scaling squared: freq_scale_sq[f] = cos^2(omega[f]*0) + sin^2(omega[f]*0)
// For standard RoPE this is always 1.0, but for scaled RoPE (YaRN etc.)
// the scaling factors at position 0 capture any frequency-dependent scaling.
// Paper Section 3.2: "frequency scaling factor"
static void triattention_build_freq_scale_sq(float * freq_scale_sq, const float * omega, uint32_t freq_count) {
    for (uint32_t f = 0; f < freq_count; f++) {
        // At position 0: cos(omega*0)=1, sin(omega*0)=0
        // So freq_scale_sq = 1.0 for all standard RoPE variants.
        // If we later support YaRN scaling, this would use the actual scaling factors.
        float c = cosf(omega[f] * 0.0f);
        float s = sinf(omega[f] * 0.0f);
        freq_scale_sq[f] = c * c + s * s;
    }
}

// Build geometric offset array: {1, 2, 4, 8, ..., offset_max}
// Paper Eq. 9: D = {2^0, 2^1, ..., 2^{log2(max_length)}}
static uint32_t triattention_build_offsets(float * offsets, uint32_t offset_max) {
    uint32_t n = 0;
    for (uint32_t d = 1; d <= offset_max; d *= 2) {
        offsets[n++] = (float)d;
    }
    return n;
}

// Precompute derived quantities per head from calibration stats:
//   q_mean_abs[f] = sqrt(q_mean_real[f]^2 + q_mean_imag[f]^2)  = ||E[q_f]||
//   extra_weight[f] = q_abs_mean[f] - q_mean_abs[f]              = E[||q_f||] - ||E[q_f]||
// Paper Eq. 8: the "norm excess" term weighted by (1 - R_f)
static void triattention_precompute_head_derived(triattention_head_stats * hs, uint32_t freq_count, bool disable_mlr) {
    hs->q_mean_abs   = new float[freq_count];
    hs->extra_weight = new float[freq_count];

    for (uint32_t f = 0; f < freq_count; f++) {
        float re = hs->q_mean_real[f];
        float im = hs->q_mean_imag[f];
        hs->q_mean_abs[f] = sqrtf(re * re + im * im);

        if (disable_mlr) {
            // Ablation: use q_abs_mean directly as the norm contribution
            hs->extra_weight[f] = hs->q_abs_mean[f];
        } else {
            // Standard: MLR-weighted norm excess = E[||q_f||] - ||E[q_f]||
            // This is >= 0 because ||E[x]|| <= E[||x||] (Jensen's inequality)
            hs->extra_weight[f] = hs->q_abs_mean[f] - hs->q_mean_abs[f];
            if (hs->extra_weight[f] < 0.0f) {
                hs->extra_weight[f] = 0.0f;  // Numerical safety
            }
        }
    }
}

// ============================================================================
// Core scoring functions: CPU implementations
// ============================================================================

// Invert RoPE rotation on post-RoPE key vectors.
// Paper Eq. 4: recover pre-RoPE K from post-RoPE K using known positions.
//
// For "half" style (Llama/Qwen): dimensions split as [real | imag]
//   k_pre[f]    = k_post[f]*cos(omega[f]*pos) + k_post[f+fc]*sin(omega[f]*pos)
//   k_pre[f+fc] = k_post[f+fc]*cos(omega[f]*pos) - k_post[f]*sin(omega[f]*pos)
//
// For "interleaved" style: pairs are (2f, 2f+1)
//   k_pre[2f]   = k_post[2f]*cos(omega[f]*pos) + k_post[2f+1]*sin(omega[f]*pos)
//   k_pre[2f+1] = k_post[2f+1]*cos(omega[f]*pos) - k_post[2f]*sin(omega[f]*pos)
void triattention_invert_rope(
    float       * out,
    const float * post_rope_k,
    const int32_t * positions,
    const float * omega,
    uint32_t n_keys,
    uint32_t head_dim,
    uint32_t freq_count,
    uint32_t rope_style)
{
    for (uint32_t i = 0; i < n_keys; i++) {
        const float * src = post_rope_k + (size_t)i * head_dim;
        float       * dst = out         + (size_t)i * head_dim;
        const float   pos = (float)positions[i];

        if (rope_style == 0) {
            // Half style: [real_0..real_{fc-1} | imag_0..imag_{fc-1}]
            for (uint32_t f = 0; f < freq_count; f++) {
                float angle = omega[f] * pos;
                float c = cosf(angle);
                float s = sinf(angle);
                float re = src[f];
                float im = src[f + freq_count];
                // Invert rotation: multiply by conjugate rotation matrix
                dst[f]              = re * c + im * s;
                dst[f + freq_count] = im * c - re * s;
            }
        } else {
            // Interleaved style: [re_0, im_0, re_1, im_1, ...]
            for (uint32_t f = 0; f < freq_count; f++) {
                float angle = omega[f] * pos;
                float c = cosf(angle);
                float s = sinf(angle);
                float re = src[2 * f];
                float im = src[2 * f + 1];
                dst[2 * f]     = re * c + im * s;
                dst[2 * f + 1] = im * c - re * s;
            }
        }
    }
}

// Score cached keys for a single (layer, attention_head) pair.
// Paper Eqs. 6-10: trigonometric importance scoring with MLR norm term.
//
// For each key at position p_k with base distance Delta = round_start - p_k:
//   1. Convert pre-RoPE K to complex representation
//   2. Compute amplitude: amp_f = ||E[q_f]|| * |k_f|
//   3. Compute phase: phi_f = angle(E[q_f] * conj(k_f))
//   4. Compute trig score: S_trig(Delta+delta) = sum_f amp_f * fscale_sq_f * cos(omega_f*(Delta+delta) + phi_f)
//   5. Compute norm score: S_norm = sum_f extra_f * fscale_sq_f * |k_f|
//   6. Aggregate over geometric offsets
void triattention_score_keys(
    float       * out_scores,
    const float * pre_rope_k,
    const triattention_head_stats * stats,
    const float * omega,
    const float * freq_scale_sq,
    const float * offsets,
    const int32_t * key_positions,
    int64_t  round_start,
    uint32_t n_keys,
    uint32_t head_dim,
    uint32_t freq_count,
    uint32_t n_offsets,
    enum triattention_agg agg,
    bool disable_trig)
{
    const float inv_n_offsets = 1.0f / (float)n_offsets;

    for (uint32_t i = 0; i < n_keys; i++) {
        const float * k = pre_rope_k + (size_t)i * head_dim;
        const float   base_delta = (float)(round_start - key_positions[i]);

        // Precompute per-frequency quantities for this key
        // Using "half" layout: k_re = k[f], k_im = k[f + freq_count]
        // (interleaved would be k[2f], k[2f+1] — handled at invert_rope stage,
        //  output from invert_rope is always in half layout for scoring)

        float total_score = 0.0f;

        if (!disable_trig) {
            // Full scoring: trigonometric + norm terms
            for (uint32_t d = 0; d < n_offsets; d++) {
                float delta = base_delta + offsets[d];
                float offset_score = 0.0f;

                for (uint32_t f = 0; f < freq_count; f++) {
                    float k_re = k[f];
                    float k_im = k[f + freq_count];
                    float k_mag = sqrtf(k_re * k_re + k_im * k_im);

                    // Amplitude: ||E[q_f]|| * |k_f|  (Paper Eq. 7)
                    float amp = stats->q_mean_abs[f] * k_mag;

                    // Phase from conj multiply: E[q_f] * conj(k_f)
                    // = (q_re + i*q_im) * (k_re - i*k_im)
                    // = (q_re*k_re + q_im*k_im) + i*(q_im*k_re - q_re*k_im)
                    float conj_re = stats->q_mean_real[f] * k_re + stats->q_mean_imag[f] * k_im;
                    float conj_im = stats->q_mean_imag[f] * k_re - stats->q_mean_real[f] * k_im;
                    float phi = atan2f(conj_im, conj_re);

                    // Trigonometric score (Paper Eq. 6):
                    // S_trig += amp * fscale^2 * cos(omega * delta + phi)
                    float phase = omega[f] * delta + phi;
                    offset_score += amp * freq_scale_sq[f] * cosf(phase);

                    // Norm excess term (Paper Eq. 8):
                    // S_norm += extra_weight * fscale^2 * |k_f|
                    offset_score += stats->extra_weight[f] * freq_scale_sq[f] * k_mag;
                }

                if (agg == TRIATTENTION_AGG_MAX) {
                    total_score = (d == 0) ? offset_score : fmaxf(total_score, offset_score);
                } else {
                    total_score += offset_score;
                }
            }

            if (agg == TRIATTENTION_AGG_MEAN) {
                total_score *= inv_n_offsets;
            }
        } else {
            // Ablation: norm-only scoring (disable_trig=true)
            // Only the position-independent norm term
            for (uint32_t f = 0; f < freq_count; f++) {
                float k_re = k[f];
                float k_im = k[f + freq_count];
                float k_mag = sqrtf(k_re * k_re + k_im * k_im);
                total_score += stats->extra_weight[f] * freq_scale_sq[f] * k_mag;
            }
        }

        out_scores[i] = total_score;
    }
}

// ============================================================================
// KV cache dequantization helper
// ============================================================================

// Dequantize K values for a specific KV head from the cache tensor.
// Handles all supported quantization types and applies inverse WHT
// rotation for turbo2/turbo3 types.
//
// Parameters:
//   out         — [n_cells, padded_head_dim] dequantized float output
//   k_tensor    — the raw K cache tensor for this layer
//   cell_indices— [n_cells] which cell slots to extract
//   kv_head_idx — which KV head (0..n_kv_heads-1)
//   n_cells     — number of cells to dequantize
//   padded_hd   — padded head dimension (128-aligned for turbo types)
//   need_wht_inv— whether to apply inverse WHT rotation (turbo2/turbo3)
//
// Note: This function copies data from potentially GPU-resident tensors
// to CPU memory, which involves a synchronous transfer. This is acceptable
// because pruning happens infrequently (every divide_length tokens).
static void triattention_dequant_kv_head(
    float              * out,
    const ggml_tensor  * k_tensor,
    const uint32_t     * cell_indices,
    uint32_t             kv_head_idx,
    uint32_t             n_cells,
    uint32_t             padded_hd,
    bool                 need_wht_inv)
{
    const ggml_type k_type = k_tensor->type;
    const uint64_t  n_embd_k_gqa = k_tensor->ne[0];  // total K embedding (all KV heads)
    const size_t    row_bytes = ggml_row_size(k_type, n_embd_k_gqa);

    // Byte offset to this KV head within a row
    const size_t head_offset_bytes = ggml_row_size(k_type, (uint64_t)kv_head_idx * padded_hd);
    const size_t head_bytes = ggml_row_size(k_type, padded_hd);

    // Temporary buffer for one quantized head block
    std::vector<uint8_t> quant_buf(head_bytes);

    // Temporary buffer for dequantized values (before WHT inverse)
    std::vector<float> dequant_tmp(padded_hd);

    for (uint32_t ci = 0; ci < n_cells; ci++) {
        const uint32_t cell_idx = cell_indices[ci];

        // Byte offset in the full tensor: row_bytes * cell_idx + head_offset_bytes
        // This addresses stream 0 (the common case for unified KV caches)
        const size_t tensor_offset = (size_t)cell_idx * row_bytes + head_offset_bytes;

        // Copy quantized data from backend (may be GPU) to CPU
        ggml_backend_tensor_get(k_tensor, quant_buf.data(), tensor_offset, head_bytes);

        // Dequantize based on type
        float * dst = need_wht_inv ? dequant_tmp.data() : (out + (size_t)ci * padded_hd);

        switch (k_type) {
            case GGML_TYPE_TURBO2_0:
                dequantize_row_turbo2_0(quant_buf.data(), dst, padded_hd);
                break;            
            case GGML_TYPE_TURBO3_0:
                dequantize_row_turbo3_0(quant_buf.data(), dst, padded_hd);
                break;
            case GGML_TYPE_TURBO4_0:
                dequantize_row_turbo4_0(quant_buf.data(), dst, padded_hd);
                break;
            case GGML_TYPE_Q8_0:
                dequantize_row_q8_0(quant_buf.data(), dst, padded_hd);
                break;
            case GGML_TYPE_F16: {
                const ggml_fp16_t * src16 = (const ggml_fp16_t *)quant_buf.data();
                for (uint32_t j = 0; j < padded_hd; j++) {
                    dst[j] = ggml_fp16_to_fp32(src16[j]);
                }
                break;
            }
            case GGML_TYPE_BF16: {
                const ggml_bf16_t * src16 = (const ggml_bf16_t *)quant_buf.data();
                for (uint32_t j = 0; j < padded_hd; j++) {
                    dst[j] = ggml_bf16_to_fp32(src16[j]);
                }
                break;
            }
            case GGML_TYPE_F32: {
                memcpy(dst, quant_buf.data(), padded_hd * sizeof(float));
                break;
            }
            default:
                fprintf(stderr, "[TriAttention] ERROR: unsupported K cache type %d\n", k_type);
                memset(out + (size_t)ci * padded_hd, 0, padded_hd * sizeof(float));
                continue;
        }

        // Apply inverse WHT rotation for turbo2/turbo3
        // turbo4 dequant already applies R^T internally
        if (need_wht_inv) {
            float * final_dst = out + (size_t)ci * padded_hd;
            // Process in 128-element blocks (WHT block size)
            for (uint32_t b = 0; b < padded_hd; b += 128) {
                matvec_128(TURBO_ROTATION_RT, dequant_tmp.data() + b, final_dst + b);
            }
        }
    }
}

// ============================================================================
// Public API: Init / Free
// ============================================================================

triattention_state * triattention_init(
    const char * stats_path,
    const triattention_config * cfg,
    uint32_t kv_size,
    double   rope_theta,
    uint32_t head_dim,
    uint32_t rot_dim,
    uint32_t n_kv_heads,
    uint32_t rope_style)
{
    // Load calibration file
    triattention_calibration * cal = triattention_load_calibration(stats_path);
    if (!cal) {
        return nullptr;
    }

    // The file doesn't carry rope_style — it's resolved by the caller from the model.
    cal->rope_style = rope_style;

    // Validate model compatibility. The file's head_dim is the *rotary* dim
    // (may be < the model's full per-head embedding dim for partial rotary
    // architectures, e.g. Ornith: rot_dim=64 of head_dim=256).
    if (cal->head_dim != rot_dim) {
        fprintf(stderr, "[TriAttention] ERROR: rotary dim mismatch (calibration=%u, model=%u)\n",
                cal->head_dim, rot_dim);
        triattention_free_calibration(cal);
        return nullptr;
    }
    if (cal->num_kv_heads != n_kv_heads) {
        fprintf(stderr, "[TriAttention] ERROR: n_kv_heads mismatch (calibration=%u, model=%u)\n",
                cal->num_kv_heads, n_kv_heads);
        triattention_free_calibration(cal);
        return nullptr;
    }
    // Warn if rope_theta differs significantly (>1% relative)
    if (fabs(cal->rope_theta - rope_theta) / fmax(cal->rope_theta, 1.0) > 0.01) {
        fprintf(stderr, "[TriAttention] WARNING: rope_theta mismatch (calibration=%.1f, model=%.1f)\n",
                cal->rope_theta, rope_theta);
    }

    // Allocate state (value-initialized: all scalars zeroed, protected_ranges empty)
    auto * state = new triattention_state();

    state->cal  = cal;
    state->cfg  = *cfg;
    state->kv_size = kv_size;
    state->head_dim = head_dim;
    state->absolute_position = 0;
    state->last_prune_position = 0;
    state->prefix_length     = 0;

    const uint32_t fc = cal->freq_count;
    const uint32_t padded_hd = ((head_dim + 127) / 128) * 128;

    // Build precomputed arrays. omega uses rot_dim (the rotary dim), not the
    // full head_dim, since RoPE's frequency base only spans the rotated part.
    state->omega = new float[fc];
    triattention_build_omega(state->omega, fc, rot_dim, rope_theta);

    state->freq_scale_sq = new float[fc];
    triattention_build_freq_scale_sq(state->freq_scale_sq, state->omega, fc);

    // Geometric offsets — max 17 elements for offset_max=65536
    state->offsets = new float[32];  // generous allocation
    state->n_offsets = triattention_build_offsets(state->offsets, cfg->offset_max);

    // Precompute derived head stats
    for (uint32_t h = 0; h < cal->n_sampled; h++) {
        triattention_precompute_head_derived(&cal->head_stats[h], fc, cfg->disable_mlr);
    }

    // Allocate cell position tracking
    state->cell_positions = new int32_t[kv_size];
    for (uint32_t i = 0; i < kv_size; i++) {
        state->cell_positions[i] = -1;
    }

    // Allocate scratch buffers (sized to padded_hd, matching triattention_try_prune()'s usage)
    // These are sized for worst-case (scoring all cells)
    state->dequant_buf  = new float[(size_t)kv_size * padded_hd];
    state->unrot_buf    = new float[(size_t)kv_size * padded_hd];
    state->score_buf    = new float[(size_t)cal->n_sampled * kv_size];
    state->combined_buf = new float[kv_size];
    state->keep_indices = new uint32_t[cfg->budget];

    // Init monitoring
    state->total_prune_calls   = 0;
    state->total_tokens_evicted = 0;
    state->total_prune_time_ms = 0.0;
    state->last_prune_time_ms  = 0.0;

    fprintf(stderr, "[TriAttention] Initialized: budget=%u, window=%u, mode=%d, offsets=%u, "
            "kv_size=%u, sampled_heads=%u\n",
            cfg->budget, cfg->divide_length, (int)cfg->mode,
            state->n_offsets, kv_size, cal->n_sampled);

    return state;
}

void triattention_free(triattention_state * state) {
    if (!state) return;

    triattention_free_calibration(state->cal);

    delete[] state->omega;
    delete[] state->freq_scale_sq;
    delete[] state->offsets;
    delete[] state->cell_positions;
    delete[] state->dequant_buf;
    delete[] state->unrot_buf;
    delete[] state->score_buf;
    delete[] state->combined_buf;
    delete[] state->keep_indices;

    // Free GPU scoring resources if initialized
#ifdef GGML_USE_CUDA
    if (state->d_scores) {
        triattention_gpu_free_dev(state->d_scores);
        state->d_scores = nullptr;
    }
    if (state->d_gpu_state) {
        triattention_gpu_free((triattention_gpu_state *)state->d_gpu_state);
        state->d_gpu_state = nullptr;
    }
#endif

#ifdef GGML_USE_VULKAN
    // Free Vulkan scoring resources if initialized (mutually exclusive with CUDA above)
    if (state->d_vk_state) {
        triattention_vk_free((triattention_vk_state *)state->d_vk_state);
        state->d_vk_state = nullptr;
    }
    if (state->vk_backend) {
        ggml_backend_free((ggml_backend_t)state->vk_backend);
        state->vk_backend = nullptr;
    }
#endif
#ifdef GGML_USE_METAL
    if (state->d_mtl_state) {
        triattention_mtl_free((triattention_mtl_state *)state->d_mtl_state);
        state->d_mtl_state = nullptr;
    }
    if (state->mtl_backend) {
        ggml_backend_free((ggml_backend_t)state->mtl_backend);
        state->mtl_backend = nullptr;
    }
#endif

    delete state;
}

// ============================================================================
// Position tracking hooks
// ============================================================================

void triattention_on_token_added(
    triattention_state * state,
    uint32_t cell_idx,
    int32_t  abs_pos)
{
    if (!state || cell_idx >= state->kv_size) return;
    state->cell_positions[cell_idx] = abs_pos;
    if (abs_pos + 1 > (int32_t)state->absolute_position) {
        state->absolute_position = abs_pos + 1;
    }
}

void triattention_on_cell_removed(
    triattention_state * state,
    uint32_t cell_idx)
{
    if (!state || cell_idx >= state->kv_size) return;
    state->cell_positions[cell_idx] = -1;
}

void triattention_on_position_shift(
    triattention_state * state,
    int32_t delta,
    int32_t p0,
    int32_t p1)
{
    if (!state || delta == 0) return;

    for (uint32_t i = 0; i < state->kv_size; i++) {
        int32_t pos = state->cell_positions[i];
        if (pos >= 0 && pos >= p0 && (p1 < 0 || pos < p1)) {
            state->cell_positions[i] = pos + delta;
            if (state->cell_positions[i] < 0) {
                state->cell_positions[i] = -1;
            }
        }
    }
}

void triattention_on_reset(triattention_state * state) {
    if (!state) return;
    state->absolute_position = 0;
    state->last_prune_position = 0;
    state->prefix_length     = 0;
    for (uint32_t i = 0; i < state->kv_size; i++) {
        state->cell_positions[i] = -1;
    }
}

// ============================================================================
// Trigger logic
// ============================================================================

bool triattention_should_prune(
    const triattention_state * state,
    uint32_t n_used)
{
    if (!state) return false;

    switch (state->cfg.trigger) {
        case TRIATTENTION_TRIGGER_INTERVAL:
            // Track elapsed tokens since the last *successful* prune rather than
            // requiring absolute_position to land exactly on a divide_length multiple.
            // A skipped/delayed prune call (e.g. due to a decode stall) used to permanently
            // desync the modulo check, stalling pruning until n_used grew far past budget.
            return n_used >= state->cfg.budget &&
                   state->absolute_position > 0 &&
                   (state->absolute_position - state->last_prune_position) >= (int64_t)state->cfg.divide_length;

        case TRIATTENTION_TRIGGER_SLACK:
            return n_used >= (state->cfg.budget + state->cfg.divide_length);

        default:
            return false;
    }
}

void triattention_protect_range(
    triattention_state * state,
    int64_t pos_start,
    int64_t pos_end)
{
    if (!state || pos_end <= pos_start) return;

    auto & ranges = state->protected_ranges;

    // Insert in sorted order by start, then merge overlapping/adjacent ranges.
    auto it = std::lower_bound(ranges.begin(), ranges.end(), std::make_pair(pos_start, pos_end));
    ranges.insert(it, {pos_start, pos_end});

    std::vector<std::pair<int64_t,int64_t>> merged;
    merged.reserve(ranges.size());
    for (const auto & r : ranges) {
        if (!merged.empty() && r.first <= merged.back().second) {
            merged.back().second = std::max(merged.back().second, r.second);
        } else {
            merged.push_back(r);
        }
    }
    ranges = std::move(merged);
}

// ============================================================================
// Main pruning implementation
// ============================================================================

// Helper: z-score normalize an array in-place
// After normalization: mean=0, std=1
static void zscore_normalize(float * scores, uint32_t n) {
    if (n <= 1) return;

    double sum = 0.0;
    for (uint32_t i = 0; i < n; i++) sum += scores[i];
    double mean = sum / n;

    double var_sum = 0.0;
    for (uint32_t i = 0; i < n; i++) {
        double d = scores[i] - mean;
        var_sum += d * d;
    }
    double std = sqrt(var_sum / n);
    if (std < 1e-10) std = 1e-10;

    for (uint32_t i = 0; i < n; i++) {
        scores[i] = (float)((scores[i] - mean) / std);
    }
}

// Helper: partial argsort — find top-K indices by score (descending)
// Returns indices of the K highest-scoring elements
static void top_k_indices(
    uint32_t       * out_indices,
    const float    * scores,
    uint32_t         n,
    uint32_t         k)
{
    if (k >= n) {
        // Keep all
        for (uint32_t i = 0; i < n; i++) out_indices[i] = i;
        return;
    }

    // Create index array and partial sort
    std::vector<uint32_t> idx(n);
    std::iota(idx.begin(), idx.end(), 0);

    std::partial_sort(idx.begin(), idx.begin() + k, idx.end(),
        [&scores](uint32_t a, uint32_t b) {
            return scores[a] > scores[b];  // descending
        });

    for (uint32_t i = 0; i < k; i++) {
        out_indices[i] = idx[i];
    }
}

// ============================================================================
// GPU scoring: lazy init
// ============================================================================

// Initializes GPU scoring state on first prune, using the K tensor type
// of the actual cache to select the right kernel variant.
// k_type must be a type supported by the GPU kernel (Q4_K, Q8_0, F16, F32,
// TURBO2_0, TURBO3_0, TURBO4_0). On failure, falls back silently to CPU.
#ifdef GGML_USE_CUDA
static void triattention_init_gpu(triattention_state * state, ggml_type k_type) {
    if (state->gpu_init_tried) return;
    state->gpu_init_tried = true;

    const triattention_calibration * cal = state->cal;
    const triattention_config & cfg = state->cfg;

    triattention_gpu_config gcfg = {};
    gcfg.head_dim     = state->head_dim;
    gcfg.freq_count   = cal->freq_count;
    gcfg.n_kv_heads   = cal->num_kv_heads;
    gcfg.n_sampled    = cal->n_sampled;
    gcfg.n_offsets    = state->n_offsets;
    gcfg.k_type       = k_type;
    gcfg.need_wht_inv = (k_type == GGML_TYPE_TURBO2_0 || k_type == GGML_TYPE_TURBO3_0);
    gcfg.disable_trig = cfg.disable_trig;

    std::vector<triattention_gpu_head_calib> gcalibs(cal->n_sampled);
    for (uint32_t h = 0; h < cal->n_sampled; h++) {
        gcalibs[h].q_mean_real  = cal->head_stats[h].q_mean_real;
        gcalibs[h].q_mean_imag  = cal->head_stats[h].q_mean_imag;
        gcalibs[h].q_mean_abs   = cal->head_stats[h].q_mean_abs;
        gcalibs[h].extra_weight = cal->head_stats[h].extra_weight;
    }

    auto * gpu_st = triattention_gpu_init(
        &gcfg, gcalibs.data(),
        state->omega, state->freq_scale_sq, state->offsets, nullptr);
    if (!gpu_st) {
        fprintf(stderr, "[TriAttention] GPU init failed, using CPU scoring\n");
        return;
    }

    // Pre-allocate one head's worth of score buffer on device
    float * d_s = triattention_gpu_alloc_scores(state->kv_size, nullptr);
    if (!d_s) {
        triattention_gpu_free(gpu_st);
        fprintf(stderr, "[TriAttention] GPU score buffer alloc failed, using CPU scoring\n");
        return;
    }

    state->d_gpu_state  = gpu_st;
    state->d_scores     = d_s;
    state->use_gpu      = true;

    fprintf(stderr, "[TriAttention] GPU scoring enabled (k_type=%d, heads=%u)\n",
            (int)k_type, cal->n_sampled);
}
#endif // GGML_USE_CUDA

#ifdef GGML_USE_VULKAN
// Initializes Vulkan GPU scoring state, mutually exclusive with the CUDA path
// above (only attempted if CUDA init did not already succeed). `kt` is a
// representative K tensor used only to confirm its buffer lives on a Vulkan
// device before allocating a Vulkan backend instance.
static void triattention_init_vk(triattention_state * state, ggml_type k_type, const ggml_tensor * kt) {
    if (state->vk_init_tried) return;
    state->vk_init_tried = true;

    if (kt == nullptr || kt->buffer == nullptr) {
        return;
    }
    const char * buft_name = ggml_backend_buft_name(ggml_backend_buffer_get_type(kt->buffer));
    if (strstr(buft_name, "Vulkan") == nullptr) {
        return; // K cache is not on a Vulkan device, nothing to do here
    }

    ggml_backend_t backend = ggml_backend_vk_init(0);
    if (!backend) {
        fprintf(stderr, "[TriAttention] Vulkan backend init failed, using CPU scoring\n");
        return;
    }

    const triattention_calibration * cal = state->cal;
    const triattention_config & cfg = state->cfg;

    triattention_vk_config vcfg = {};
    vcfg.head_dim     = state->head_dim;
    vcfg.freq_count   = cal->freq_count;
    vcfg.n_kv_heads   = cal->num_kv_heads;
    vcfg.n_sampled    = cal->n_sampled;
    vcfg.n_offsets    = state->n_offsets;
    vcfg.k_type       = k_type;
    vcfg.need_wht_inv = (k_type == GGML_TYPE_TURBO2_0 || k_type == GGML_TYPE_TURBO3_0);
    vcfg.disable_trig = cfg.disable_trig;

    std::vector<triattention_vk_head_calib> vcalibs(cal->n_sampled);
    for (uint32_t h = 0; h < cal->n_sampled; h++) {
        vcalibs[h].q_mean_real  = cal->head_stats[h].q_mean_real;
        vcalibs[h].q_mean_imag  = cal->head_stats[h].q_mean_imag;
        vcalibs[h].q_mean_abs   = cal->head_stats[h].q_mean_abs;
        vcalibs[h].extra_weight = cal->head_stats[h].extra_weight;
    }

    auto * vk_st = triattention_vk_init(
        backend, &vcfg, vcalibs.data(),
        state->omega, state->freq_scale_sq, state->offsets);
    if (!vk_st) {
        ggml_backend_free(backend);
        fprintf(stderr, "[TriAttention] Vulkan GPU init failed, using CPU scoring\n");
        return;
    }

    state->vk_backend  = backend;
    state->d_vk_state  = vk_st;
    state->use_vk      = true;

    fprintf(stderr, "[TriAttention] Vulkan scoring enabled (k_type=%d, heads=%u)\n",
            (int)k_type, cal->n_sampled);
}
#endif // GGML_USE_VULKAN

#ifdef GGML_USE_METAL
// Initializes Metal GPU scoring state, mutually exclusive with the CUDA/Vulkan
// paths above (only attempted if neither already succeeded). `kt` is a
// representative K tensor used only to confirm its buffer lives on a Metal
// device before allocating a Metal backend instance.
static void triattention_init_mtl(triattention_state * state, ggml_type k_type, const ggml_tensor * kt) {
    if (state->mtl_init_tried) return;
    state->mtl_init_tried = true;

    if (kt == nullptr || kt->buffer == nullptr) {
        return;
    }
    const char * buft_name = ggml_backend_buft_name(ggml_backend_buffer_get_type(kt->buffer));
    if (strstr(buft_name, "Metal") == nullptr) {
        return; // K cache is not on a Metal device, nothing to do here
    }

    ggml_backend_t backend = ggml_backend_metal_init();
    if (!backend) {
        fprintf(stderr, "[TriAttention] Metal backend init failed, using CPU scoring\n");
        return;
    }

    const triattention_calibration * cal = state->cal;
    const triattention_config & cfg = state->cfg;

    triattention_mtl_config mcfg = {};
    mcfg.head_dim     = state->head_dim;
    mcfg.freq_count   = cal->freq_count;
    mcfg.n_kv_heads   = cal->num_kv_heads;
    mcfg.n_sampled    = cal->n_sampled;
    mcfg.n_offsets    = state->n_offsets;
    mcfg.k_type       = k_type;
    mcfg.need_wht_inv = (k_type == GGML_TYPE_TURBO2_0 || k_type == GGML_TYPE_TURBO3_0);
    mcfg.disable_trig = cfg.disable_trig;

    std::vector<triattention_mtl_head_calib> mcalibs(cal->n_sampled);
    for (uint32_t h = 0; h < cal->n_sampled; h++) {
        mcalibs[h].q_mean_real  = cal->head_stats[h].q_mean_real;
        mcalibs[h].q_mean_imag  = cal->head_stats[h].q_mean_imag;
        mcalibs[h].q_mean_abs   = cal->head_stats[h].q_mean_abs;
        mcalibs[h].extra_weight = cal->head_stats[h].extra_weight;
    }

    auto * mtl_st = triattention_mtl_init(
        backend, &mcfg, mcalibs.data(),
        state->omega, state->freq_scale_sq, state->offsets);
    if (!mtl_st) {
        ggml_backend_free(backend);
        fprintf(stderr, "[TriAttention] Metal GPU init failed, using CPU scoring\n");
        return;
    }

    state->mtl_backend = backend;
    state->d_mtl_state = mtl_st;
    state->use_mtl     = true;

    fprintf(stderr, "[TriAttention] Metal scoring enabled (k_type=%d, heads=%u)\n",
            (int)k_type, cal->n_sampled);
}
#endif // GGML_USE_METAL

// ============================================================================
// Internal pruning implementation (called from KV cache integration)
// ============================================================================

// This is the real workhorse. Called from llama_kv_cache::triattention_try_prune()
// with direct access to the K tensors.
//
// Parameters:
//   state       — TriAttention runtime state
//   k_tensors   — array of K cache tensors, indexed by internal layer id
//   n_layers    — number of layers in k_tensors array
//   layer_map   — maps model layer index → internal layer id in k_tensors
//   v_cells     — reference to the cache's cell metadata (for rm operations)
//   v_heads     — reference to the cache's head pointer array (for updating after prune)
//   kv_size     — cache capacity
//
// Returns: number of cells evicted
int32_t triattention_prune_impl(
    triattention_state * state,
    ggml_tensor * const * k_tensors,
    uint32_t              n_layers,
    const int32_t       * layer_map,
    uint32_t              kv_size)
{
    if (!state) return -1;

    double t_start = triattention_time_ms();

    const auto & cfg = state->cfg;
    const auto * cal = state->cal;
    const uint32_t fc = cal->freq_count;
    const uint32_t hd = state->head_dim; // full per-head K dim (buffer stride), not cal->head_dim (rotary)
    const uint32_t padded_hd = ((hd + 127) / 128) * 128;
    const uint32_t budget = cfg.budget;

    // ---- Step 1: Enumerate occupied cells ----
    std::vector<uint32_t> occupied_indices;
    std::vector<int32_t>  occupied_positions;
    occupied_indices.reserve(kv_size);
    occupied_positions.reserve(kv_size);

    for (uint32_t i = 0; i < kv_size; i++) {
        if (state->cell_positions[i] >= 0) {
            occupied_indices.push_back(i);
            occupied_positions.push_back(state->cell_positions[i]);
        }
    }

    const uint32_t n_occupied = (uint32_t)occupied_indices.size();
    if (n_occupied <= budget) return 0;

    // ---- Step 2: Separate protected tokens from eviction candidates ----
    // Two protection classes:
    //   1. Prefix-protected: initial prompt tokens (if protect_prefill is set)
    //   2. Recent-protected: the most recent divide_length positions are never evicted.
    //      This ensures seq_pos_max remains unchanged after pruning, so the server's
    //      position counter (which expects Y = seq_pos_max + 1) stays consistent.
    //      Without this, evicting the highest-position token would cause
    //      "inconsistent sequence positions" errors.

    // Compute max position for recent-window protection
    int32_t max_pos = -1;
    for (uint32_t i = 0; i < n_occupied; i++) {
        if (occupied_positions[i] > max_pos) {
            max_pos = occupied_positions[i];
        }
    }

    const int32_t recent_threshold = max_pos - (int32_t)cfg.divide_length + 1;

    // Binary search: ranges are sorted, non-overlapping [start,end) intervals.
    const auto & protected_ranges = state->protected_ranges;
    auto in_protected_range = [&protected_ranges](int32_t pos) {
        auto it = std::upper_bound(
            protected_ranges.begin(), protected_ranges.end(), pos,
            [](int64_t p, const std::pair<int64_t,int64_t> & r) { return p < r.first; });
        if (it == protected_ranges.begin()) return false;
        --it;
        return (int64_t)pos < it->second;
    };

    std::vector<uint32_t> decode_local_idx;   // index into occupied_indices
    std::vector<uint32_t> decode_cell_idx;    // actual cell indices
    std::vector<int32_t>  decode_positions;
    uint32_t n_protected = 0;  // prefix + recent + explicitly-protected count

    for (uint32_t i = 0; i < n_occupied; i++) {
        const bool is_prefix = cfg.protect_prefill &&
                               occupied_positions[i] < (int32_t)state->prefix_length;
        const bool is_recent = occupied_positions[i] >= recent_threshold;
        const bool is_range_protected = !protected_ranges.empty() &&
                               in_protected_range(occupied_positions[i]);

        if (is_prefix || is_recent || is_range_protected) {
            n_protected++;
        } else {
            decode_local_idx.push_back(i);
            decode_cell_idx.push_back(occupied_indices[i]);
            decode_positions.push_back(occupied_positions[i]);
        }
    }

    const uint32_t n_decode = (uint32_t)decode_cell_idx.size();
    const uint32_t decode_budget = (budget > n_protected) ? (budget - n_protected) : 0;

    if (n_decode <= decode_budget) return 0;

    // ---- Step 3: Score all sampled (layer, head) pairs ----
    // score_buf layout: [n_sampled, n_decode] — row-major
    float * score_buf = state->score_buf;

    // Determine K tensor type (used for lazy GPU init)
    const ggml_type k_type = (n_layers > 0 && k_tensors[0]) ? k_tensors[0]->type : GGML_TYPE_F32;

    // Lazy GPU init: runs only once per state lifetime
#ifdef GGML_USE_CUDA
    if (!state->gpu_init_tried) {
        triattention_init_gpu(state, k_type);
    }
#endif

#ifdef GGML_USE_VULKAN
    // Only attempted if CUDA scoring did not already take over this state.
    if (!state->use_gpu && !state->vk_init_tried) {
        const ggml_tensor * kt0 = (n_layers > 0) ? k_tensors[0] : nullptr;
        triattention_init_vk(state, k_type, kt0);
    }
#endif

#ifdef GGML_USE_METAL
    // Only attempted if CUDA/Vulkan scoring did not already take over this state.
    if (!state->use_gpu && !state->use_vk && !state->mtl_init_tried) {
        const ggml_tensor * kt0 = (n_layers > 0) ? k_tensors[0] : nullptr;
        triattention_init_mtl(state, k_type, kt0);
    }
#endif

#ifdef GGML_USE_CUDA
    if (state->use_gpu) {
        // ---- GPU path ----
        // Upload the n_decode candidate cell indices + positions to device.
        // Kernels are enqueued into the default stream (nullptr), ordered after the upload.
        uint32_t * d_cell_indices = nullptr;
        int32_t  * d_positions    = nullptr;
        triattention_gpu_upload_cells(
            &d_cell_indices, &d_positions,
            decode_cell_idx.data(), decode_positions.data(),
            n_decode, nullptr);

        // Allocate a packed score buffer [n_sampled × n_decode] on device
        float * d_scores_all = triattention_gpu_alloc_scores(
            (uint32_t)((size_t)cal->n_sampled * n_decode), nullptr);

        // Launch one scoring kernel per sampled head (all async on default stream)
        for (uint32_t sh = 0; sh < cal->n_sampled; sh++) {
            const uint32_t layer_idx = cal->sampled_layer[sh];
            const uint32_t attn_head = cal->sampled_head[sh];
            const uint32_t kv_head   = attn_head / cal->num_kv_groups;

            // Find internal layer index
            int32_t ikv = -1;
            for (uint32_t l = 0; l < n_layers; l++) {
                if (layer_map[l] == (int32_t)layer_idx) { ikv = (int32_t)l; break; }
            }
            if (ikv < 0) {
                // Layer not in cache — will zero out after copy
                continue;
            }

            const ggml_tensor * kt = k_tensors[ikv];
            const size_t row_bytes = (size_t)(ggml_nbytes(kt) / (size_t)kt->ne[1]);
            const uint64_t n_embd  = (uint64_t)kt->ne[0];

            triattention_gpu_score_head(
                (triattention_gpu_state *)state->d_gpu_state,
                kt->data,                                   // device pointer (K on GPU)
                n_embd,
                row_bytes,
                kv_head,
                sh,                                         // head_calib_idx
                d_cell_indices,
                d_positions,
                n_decode,
                (int64_t)state->absolute_position,          // round_start
                (int)cfg.agg,
                d_scores_all + (size_t)sh * n_decode,       // output slice
                nullptr);                                    // default stream
        }

        // Copy all scores to host with one synchronization point
        triattention_gpu_scores_to_host(
            score_buf, d_scores_all,
            (uint32_t)((size_t)cal->n_sampled * n_decode), nullptr);

        // Zero out scores for heads whose layers were not in cache
        for (uint32_t sh = 0; sh < cal->n_sampled; sh++) {
            const uint32_t layer_idx = cal->sampled_layer[sh];
            int32_t ikv = -1;
            for (uint32_t l = 0; l < n_layers; l++) {
                if (layer_map[l] == (int32_t)layer_idx) { ikv = (int32_t)l; break; }
            }
            if (ikv < 0) {
                memset(score_buf + (size_t)sh * n_decode, 0, n_decode * sizeof(float));
            }
        }

        triattention_gpu_free_dev(d_scores_all);
        triattention_gpu_free_dev(d_cell_indices);
        triattention_gpu_free_dev(d_positions);

    }
#endif // GGML_USE_CUDA

#ifdef GGML_USE_VULKAN
    if (state->use_vk) {
        // ---- Vulkan GPU path (batched: upload once, enqueue N, sync once) ----
        // Mirrors the CUDA path above instead of doing one submit+fence-wait
        // per sampled head.
        auto * vk_state = (triattention_vk_state *)state->d_vk_state;
        triattention_vk_batch_begin(
            vk_state, decode_cell_idx.data(), decode_positions.data(),
            n_decode, (uint32_t)((size_t)cal->n_sampled * n_decode));

        for (uint32_t sh = 0; sh < cal->n_sampled; sh++) {
            const uint32_t layer_idx = cal->sampled_layer[sh];
            const uint32_t attn_head = cal->sampled_head[sh];
            const uint32_t kv_head   = attn_head / cal->num_kv_groups;

            int32_t ikv = -1;
            for (uint32_t l = 0; l < n_layers; l++) {
                if (layer_map[l] == (int32_t)layer_idx) { ikv = (int32_t)l; break; }
            }
            if (ikv < 0) {
                memset(score_buf + (size_t)sh * n_decode, 0, n_decode * sizeof(float));
                continue;
            }

            const ggml_tensor * kt = k_tensors[ikv];
            triattention_vk_batch_enqueue_head(
                vk_state, kt, kv_head, sh,                    // head_calib_idx
                (int64_t)state->absolute_position,            // round_start
                (int)cfg.agg,
                (uint32_t)((size_t)sh * n_decode));           // output offset
        }

        triattention_vk_batch_end(
            vk_state, score_buf, (uint32_t)((size_t)cal->n_sampled * n_decode));
    }

#endif // GGML_USE_VULKAN

#ifdef GGML_USE_METAL
    if (state->use_mtl) {
        // ---- Metal GPU path (batched: upload once, enqueue N, sync once) ----
        // Mirrors the CUDA/Vulkan paths above instead of doing one
        // commit+wait per sampled head.
        auto * mtl_state = (triattention_mtl_state *)state->d_mtl_state;
        triattention_mtl_batch_begin(
            mtl_state, decode_cell_idx.data(), decode_positions.data(),
            n_decode, (uint32_t)((size_t)cal->n_sampled * n_decode));

        for (uint32_t sh = 0; sh < cal->n_sampled; sh++) {
            const uint32_t layer_idx = cal->sampled_layer[sh];
            const uint32_t attn_head = cal->sampled_head[sh];
            const uint32_t kv_head   = attn_head / cal->num_kv_groups;

            int32_t ikv = -1;
            for (uint32_t l = 0; l < n_layers; l++) {
                if (layer_map[l] == (int32_t)layer_idx) { ikv = (int32_t)l; break; }
            }
            if (ikv < 0) {
                memset(score_buf + (size_t)sh * n_decode, 0, n_decode * sizeof(float));
                continue;
            }

            const ggml_tensor * kt = k_tensors[ikv];
            triattention_mtl_batch_enqueue_head(
                mtl_state, kt, kv_head, sh,                    // head_calib_idx
                (int64_t)state->absolute_position,            // round_start
                (int)cfg.agg,
                (uint32_t)((size_t)sh * n_decode));           // output offset
        }

        triattention_mtl_batch_end(
            mtl_state, score_buf, (uint32_t)((size_t)cal->n_sampled * n_decode));
    }
#endif // GGML_USE_METAL

    if(!state->use_gpu && !state->use_vk && !state->use_mtl) {
        // ---- CPU fallback path ----
        for (uint32_t sh = 0; sh < cal->n_sampled; sh++) {
            const uint32_t layer_idx = cal->sampled_layer[sh];
            const uint32_t attn_head = cal->sampled_head[sh];
            const uint32_t kv_head   = attn_head / cal->num_kv_groups;

            // Find internal layer id
            int32_t ikv = -1;
            for (uint32_t l = 0; l < n_layers; l++) {
                if (layer_map[l] == (int32_t)layer_idx) {
                    ikv = (int32_t)l;
                    break;
                }
            }
            if (ikv < 0) {
                // Layer not in cache (filtered out) — zero scores
                memset(score_buf + (size_t)sh * n_decode, 0, n_decode * sizeof(float));
                continue;
            }

            const ggml_tensor * k_tensor = k_tensors[ikv];
            const ggml_type k_type_l = k_tensor->type;
            const bool need_wht_inv = (k_type_l == GGML_TYPE_TURBO2_0 || k_type_l == GGML_TYPE_TURBO3_0);

            // 3a. Dequantize K for this KV head for all decode cells
            triattention_dequant_kv_head(
                state->dequant_buf,
                k_tensor,
                decode_cell_idx.data(),
                kv_head,
                n_decode,
                padded_hd,
                need_wht_inv);

            // 3b. Invert RoPE → pre-RoPE K
            triattention_invert_rope(
                state->unrot_buf,
                state->dequant_buf,
                decode_positions.data(),
                state->omega,
                n_decode,
                padded_hd,
                fc,
                cal->rope_style);

            // 3c. Score keys
            triattention_score_keys(
                score_buf + (size_t)sh * n_decode,
                state->unrot_buf,
                &cal->head_stats[sh],
                state->omega,
                state->freq_scale_sq,
                state->offsets,
                decode_positions.data(),
                state->absolute_position,
                n_decode,
                padded_hd,
                fc,
                state->n_offsets,
                cfg.agg,
                cfg.disable_trig);
        }
    }

    // ---- Step 3.5: Apply v2 per-layer budget scales as a score weight ----
    // Approximates the reference scorer's per-layer eviction budget within
    // llama.cpp's single, position-based (not per-layer) eviction decision.
    for (uint32_t sh = 0; sh < cal->n_sampled; sh++) {
        const float scale = cal->layer_budget_scales[cal->sampled_layer[sh]];
        if (scale == 1.0f) continue;
        float * s = score_buf + (size_t)sh * n_decode;
        for (uint32_t i = 0; i < n_decode; i++) {
            s[i] *= scale;
        }
    }

    // ---- Step 4: Combine scores across heads ----
    float * combined = state->combined_buf;

    if (cfg.mode == TRIATTENTION_MODE_GLOBAL) {
        // ---- Global union-based selection (Paper default) ----
        // 4a. Z-score normalize per head
        if (cfg.normalize_scores) {
            for (uint32_t sh = 0; sh < cal->n_sampled; sh++) {
                zscore_normalize(score_buf + (size_t)sh * n_decode, n_decode);
            }
        }

        // 4b. Add tie-breaking noise
        if (cfg.seed >= 0) {
            std::mt19937 rng((uint32_t)cfg.seed + (uint32_t)state->total_prune_calls);
            std::uniform_real_distribution<float> noise(-1e-6f, 1e-6f);
            for (uint32_t sh = 0; sh < cal->n_sampled; sh++) {
                float * s = score_buf + (size_t)sh * n_decode;
                for (uint32_t i = 0; i < n_decode; i++) {
                    s[i] += noise(rng);
                }
            }
        }

        // 4c. Union selection:
        //   - Each head independently picks top-B
        //   - Union all selected indices
        //   - Combined score = max over all heads for each index
        //   - Keep top-B from union by combined score

        // First: compute max-over-heads score for each decode token
        for (uint32_t i = 0; i < n_decode; i++) {
            float max_score = -1e30f;
            for (uint32_t sh = 0; sh < cal->n_sampled; sh++) {
                float s = score_buf[sh * n_decode + i];
                if (s > max_score) max_score = s;
            }
            combined[i] = max_score;
        }

        // 4d. Select top-B by combined score
        top_k_indices(state->keep_indices, combined, n_decode, decode_budget);

    } else if (cfg.mode == TRIATTENTION_MODE_PER_KV_HEAD) {
        // ---- Per-KV-head independent selection ----
        // Group sampled attention heads by KV head.
        // For each KV head: aggregate scores across layers → independent top-B

        const uint32_t n_kv = cal->num_kv_heads;
        std::vector<std::vector<uint32_t>> kv_head_to_sampled(n_kv);

        for (uint32_t sh = 0; sh < cal->n_sampled; sh++) {
            uint32_t kv_h = cal->sampled_head[sh] / cal->num_kv_groups;
            kv_head_to_sampled[kv_h].push_back(sh);
        }

        // For each decode token: max score across all KV heads
        // (simplified: we use the per-KV-head max as the combined score)
        for (uint32_t i = 0; i < n_decode; i++) combined[i] = -1e30f;

        for (uint32_t kv_h = 0; kv_h < n_kv; kv_h++) {
            const auto & heads = kv_head_to_sampled[kv_h];
            if (heads.empty()) continue;

            for (uint32_t i = 0; i < n_decode; i++) {
                float max_val = -1e30f;
                for (uint32_t sh_idx : heads) {
                    float s = score_buf[sh_idx * n_decode + i];
                    if (cfg.normalize_scores) {
                        // Normalization already applied above? No, only for global mode.
                        // Apply inline for per-KV-head mode.
                    }
                    if (s > max_val) max_val = s;
                }
                if (max_val > combined[i]) combined[i] = max_val;
            }
        }

        // Normalize if requested
        if (cfg.normalize_scores) {
            // Per-KV-head z-score normalization on the per-head scores
            for (uint32_t sh = 0; sh < cal->n_sampled; sh++) {
                zscore_normalize(score_buf + (size_t)sh * n_decode, n_decode);
            }
            // Recompute combined after normalization
            for (uint32_t i = 0; i < n_decode; i++) combined[i] = -1e30f;
            for (uint32_t sh = 0; sh < cal->n_sampled; sh++) {
                for (uint32_t i = 0; i < n_decode; i++) {
                    float s = score_buf[sh * n_decode + i];
                    if (s > combined[i]) combined[i] = s;
                }
            }
        }

        // Select top-B by combined score
        top_k_indices(state->keep_indices, combined, n_decode, decode_budget);

    } else if (cfg.mode == TRIATTENTION_MODE_PER_LAYER_HEAD) {
        // ---- Per-layer-per-head independent selection ----
        // Each (layer, KV head) selects independently.
        // Final combined score = mean of per-(layer,kv_head) scores

        if (cfg.normalize_scores) {
            for (uint32_t sh = 0; sh < cal->n_sampled; sh++) {
                zscore_normalize(score_buf + (size_t)sh * n_decode, n_decode);
            }
        }

        // Aggregate: mean score across all sampled heads for each token
        for (uint32_t i = 0; i < n_decode; i++) {
            float sum = 0.0f;
            for (uint32_t sh = 0; sh < cal->n_sampled; sh++) {
                sum += score_buf[sh * n_decode + i];
            }
            combined[i] = sum / (float)cal->n_sampled;
        }

        top_k_indices(state->keep_indices, combined, n_decode, decode_budget);
    }

    // ---- Step 5: Build keep set and evict ----
    // Convert keep_indices (which index into decode arrays) to actual cell indices
    std::vector<bool> keep_set(n_decode, false);
    for (uint32_t i = 0; i < decode_budget && i < n_decode; i++) {
        keep_set[state->keep_indices[i]] = true;
    }

    // Evict cells not in keep set
    uint32_t n_evicted = 0;
    for (uint32_t i = 0; i < n_decode; i++) {
        if (!keep_set[i]) {
            uint32_t cell_idx = decode_cell_idx[i];
            state->cell_positions[cell_idx] = -1;
            n_evicted++;
        }
    }

    // ---- Step 6: Update statistics ----
    double t_end = triattention_time_ms();
    state->total_prune_calls++;
    state->total_tokens_evicted += n_evicted;
    state->last_prune_time_ms = t_end - t_start;
    state->total_prune_time_ms += state->last_prune_time_ms;
    state->last_prune_position = state->absolute_position;

    if (cfg.enable_logging) {
        fprintf(stderr, "[TriAttention] Pruned: %u → %u tokens (%u evicted, %u protected [prefix=%lld, recent=%d]), "
                "%.2f ms [%s], pos=%lld\n",
                n_occupied, n_occupied - n_evicted, n_evicted, n_protected,
                (long long)state->prefix_length, (int)cfg.divide_length,
                state->last_prune_time_ms, (state->use_gpu || state->use_vk || state->use_mtl) ? "GPU" : "CPU",
                (long long)state->absolute_position);
    }

    return (int32_t)n_evicted;
}

// ============================================================================
// Monitoring
// ============================================================================

void triattention_print_stats(const triattention_state * state, FILE * stream) {
    if (!state) return;

    fprintf(stream, "\n=== TriAttention Statistics ===\n");
    fprintf(stream, "  Model:            %s\n", state->cal->model_name);
    fprintf(stream, "  Budget:           %u tokens\n", state->cfg.budget);
    fprintf(stream, "  Pruning interval: %u tokens\n", state->cfg.divide_length);
    fprintf(stream, "  Pruning mode:     %s\n",
            state->cfg.mode == TRIATTENTION_MODE_GLOBAL         ? "global (union)" :
            state->cfg.mode == TRIATTENTION_MODE_PER_KV_HEAD    ? "per-KV-head" :
            state->cfg.mode == TRIATTENTION_MODE_PER_LAYER_HEAD ? "per-layer-per-head" : "unknown");
    fprintf(stream, "  Score aggregation: %s\n",
            state->cfg.agg == TRIATTENTION_AGG_MEAN ? "mean" : "max");
    fprintf(stream, "  Sampled heads:    %u of %u\n", state->cal->n_sampled, state->cal->num_attn_heads);
    fprintf(stream, "  Geometric offsets: %u (max %u)\n", state->n_offsets, state->cfg.offset_max);
    fprintf(stream, "  ---\n");
    fprintf(stream, "  Total prune calls:    %llu\n", (unsigned long long)state->total_prune_calls);
    fprintf(stream, "  Total tokens evicted: %llu\n", (unsigned long long)state->total_tokens_evicted);
    fprintf(stream, "  Total prune time:     %.2f ms\n", state->total_prune_time_ms);
    if (state->total_prune_calls > 0) {
        fprintf(stream, "  Avg time per prune:   %.2f ms\n",
                state->total_prune_time_ms / state->total_prune_calls);
        fprintf(stream, "  Avg tokens per prune: %.1f\n",
                (double)state->total_tokens_evicted / state->total_prune_calls);
    }
    fprintf(stream, "  Last prune time:      %.2f ms\n", state->last_prune_time_ms);
    fprintf(stream, "  Current position:     %lld\n", (long long)state->absolute_position);
    fprintf(stream, "===============================\n\n");
}
