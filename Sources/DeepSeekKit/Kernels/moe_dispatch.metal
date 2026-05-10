#include <metal_stdlib>
using namespace metal;

// MoE token dispatch: gather tokens by expert, scatter weighted outputs back.
//
// Mirrors the routing-then-execution pattern in
// Reference/inference/model.py:629-644:
//
//   counts = bincount(indices.flatten(), minlength=n_routed_experts)
//   for i in range(...):
//       idx, top = where(indices == i)
//       y[idx] += expert(x[idx], weights[idx, top, None])
//
// On Metal we replace the per-expert `where` with a host-prepared
// permutation: for each expert i, list the (token, slot) pairs whose
// indices[token, slot] == i, along with the corresponding weight.
// The host writes a flat `assignments` buffer of length `total_assignments`
// containing source token rows; `expertOffsets[i]` gives the starting
// offset for expert i's slice.
//
// Buffers:
//   x          : [N, D] f32          — input rows
//   assignTok  : [T]    i32          — source token row for assignment t
//                                       (T = sum of expert counts == N * topK)
//   gathered   : [T, D] f32          — output (one row per assignment)
//
// `assignTok[t]` is used for gather-by-expert: for each (b,m,k) routing
// decision, T = N * topK and the host has sorted the assignments so that
// all assignments going to expert i are contiguous.

kernel void moe_gather(
    device const float* x          [[buffer(0)]],
    device const int*   assignTok  [[buffer(1)]],
    device float*       gathered   [[buffer(2)]],
    constant uint2&     dims       [[buffer(3)]],   // (T, D)
    uint2 gid [[thread_position_in_grid]]
) {
    uint d = gid.x, t = gid.y;
    uint T = dims.x, D = dims.y;
    if (t >= T || d >= D) return;
    int srcTok = assignTok[t];
    if (srcTok < 0) {
        gathered[t * D + d] = 0.0f;
        return;
    }
    gathered[t * D + d] = x[(uint)srcTok * D + d];
}

// Inverse: weighted scatter of expert outputs into a [N, D] dense buffer.
//
//   y[N, D]  starts zero (caller's responsibility)
//   assignTok[T] : source token row for assignment t
//   weights[T]   : gating weight for assignment t
//   outs[T, D]   : expert output row t
//
//   for t in 0..T: y[assignTok[t]] += weights[t] * outs[t]
//
// Each thread handles one (n, d) cell and accumulates across all
// assignments that land on row n. To avoid race conditions we iterate over
// assignments in sequence per (n, d) — assignments come from the host with
// index information `tokAssignments[n]` listing which `t`s belong to token n.

kernel void moe_scatter(
    device float*       y               [[buffer(0)]],
    device const int*   tokSlotStart    [[buffer(1)]],   // [N+1] — prefix sum
    device const int*   tokSlotIdx      [[buffer(2)]],   // [N * topK] — which t's belong to this token
    device const float* weights         [[buffer(3)]],   // [T]
    device const float* outs            [[buffer(4)]],   // [T, D]
    constant uint2&     dims            [[buffer(5)]],   // (N, D)
    uint2 gid [[thread_position_in_grid]]
) {
    uint d = gid.x, n = gid.y;
    uint N = dims.x, D = dims.y;
    if (n >= N || d >= D) return;
    int begin = tokSlotStart[n];
    int end = tokSlotStart[n + 1];
    float acc = y[n * D + d];
    for (int s = begin; s < end; s++) {
        int t = tokSlotIdx[s];
        if (t < 0) continue;
        acc += weights[t] * outs[t * D + d];
    }
    y[n * D + d] = acc;
}
