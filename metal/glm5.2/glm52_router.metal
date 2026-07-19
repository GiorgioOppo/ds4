// GLM 5.2 (`glm-dsa`) kernels.
//
// Keep these kernels architecture-owned instead of adding GLM branches to the
// DeepSeek V4 source. The first correctness tranche contains the exact GLM
// router; compact DSA attention and MoE kernels are added beside it only after
// their CPU/reference fixtures are available.

#include <metal_stdlib>
using namespace metal;

struct ds4_metal_args_glm52_router_select {
    uint32_t n_expert;
    uint32_t n_expert_used;
    float    expert_weight_scale;
    uint32_t pad0;
};

static inline float ds4_glm52_router_sigmoid(float x) {
    // Stable form matching the reference implementation for large logits.
    if (x >= 0.0f) {
        const float e = exp(-x);
        return 1.0f / (1.0f + e);
    }
    const float e = exp(x);
    return e / (1.0f + e);
}

static inline bool ds4_glm52_router_better(
        threadgroup const float *scores,
        int32_t                  a,
        int32_t                  b) {
    const float sa = scores[(uint)a];
    const float sb = scores[(uint)b];
    // The lower expert id wins exact ties. This is part of deterministic greedy
    // inference and must stay identical across CPU, Metal and cache budgets.
    return sa > sb || (sa == sb && a < b);
}

// GLM 5.2 router for one or more token rows. Selection uses
// sigmoid(logit)+bias, while route weights use the unbiased sigmoid values.
// One 256-lane threadgroup owns one token row.
kernel void kernel_glm52_router_select(
        constant ds4_metal_args_glm52_router_select &args,
        device const float   *logits,
        device const float   *bias,
        device int32_t       *selected,
        device float         *weights,
        device float         *probs,
        threadgroup float    *scratch [[threadgroup(0)]],
        uint token [[threadgroup_position_in_grid]],
        uint tid [[thread_position_in_threadgroup]]) {
    threadgroup float *selection_scores = scratch;
    threadgroup int32_t *indices =
        (threadgroup int32_t *)(scratch + 256u);

    device const float *token_logits =
        logits + (uint64_t)token * args.n_expert;
    device int32_t *token_selected =
        selected + (uint64_t)token * args.n_expert_used;
    device float *token_weights =
        weights + (uint64_t)token * args.n_expert_used;
    device float *token_probs =
        probs + (uint64_t)token * args.n_expert;

    const uint n_expert = min(args.n_expert, 256u);
    const bool active = tid < n_expert;
    const float probability = active
        ? ds4_glm52_router_sigmoid(token_logits[tid])
        : 0.0f;
    if (active) token_probs[tid] = probability;
    selection_scores[tid] = active ? probability + bias[tid] : -INFINITY;
    indices[tid] = (int32_t)tid;
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // Power-of-two bitonic network. All 256 lanes participate in every barrier.
    for (uint width = 2u; width <= 256u; width <<= 1u) {
        for (uint stride = width >> 1u; stride > 0u; stride >>= 1u) {
            const uint other = tid ^ stride;
            if (other > tid) {
                const int32_t a = indices[tid];
                const int32_t b = indices[other];
                const bool descending = (tid & width) == 0u;
                const bool should_swap = descending
                    ? ds4_glm52_router_better(selection_scores, b, a)
                    : ds4_glm52_router_better(selection_scores, a, b);
                if (should_swap) {
                    indices[tid] = b;
                    indices[other] = a;
                }
            }
            threadgroup_barrier(mem_flags::mem_threadgroup);
        }
    }

    const uint used = min(args.n_expert_used, n_expert);
    if (tid < used) token_selected[tid] = indices[tid];
    threadgroup_barrier(mem_flags::mem_threadgroup);

    if (tid < used) {
        float sum = 0.0f;
        for (uint i = 0u; i < used; i++) {
            sum += token_probs[(uint)token_selected[i]];
        }
        sum = max(sum, 6.103515625e-5f);
        token_weights[tid] =
            token_probs[(uint)token_selected[tid]] / sum *
            args.expert_weight_scale;
    }
}

