#pragma once

#include "common.cuh"

#define TQ_CHAIN_MAX_OPS 8

enum tq_chain_code {
    TQ_CHAIN_ADD = 0,
    TQ_CHAIN_MUL,
    TQ_CHAIN_DIV,
    TQ_CHAIN_SCALE,
    TQ_CHAIN_CLAMP,
    TQ_CHAIN_SILU,
    TQ_CHAIN_SIGMOID,
    TQ_CHAIN_SOFTPLUS,
    TQ_CHAIN_GELU,
    TQ_CHAIN_RELU,
    TQ_CHAIN_NEG,
    TQ_CHAIN_SQR,
};

struct tq_chain_desc {
    int           n_ops;
    int           code   [TQ_CHAIN_MAX_OPS];
    const float * other  [TQ_CHAIN_MAX_OPS];
    int           chain_is_lhs[TQ_CHAIN_MAX_OPS];
    int           bcast  [TQ_CHAIN_MAX_OPS];
    float         p0     [TQ_CHAIN_MAX_OPS];
    float         p1     [TQ_CHAIN_MAX_OPS];
};

void ggml_cuda_op_elem_chain(ggml_backend_cuda_context & ctx,
                             const float * src, float * dst, int64_t n,
                             const tq_chain_desc & desc);