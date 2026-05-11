#include <metal_stdlib>
using namespace metal;

// GPU-side construction of the [B, S, K] topk index table consumed by the
// sparse attention kernel. Replaces the host-side AttentionIndices path
// so MLA can build the table without committing the command buffer to
// read indexer output back to host.
//
// Layout: out[(b * S + s) * K + k]. The first kWin slots hold sliding-
// window indices; the next kComp slots hold either deterministic
// compressed indices (no-indexer layers) or Indexer-emitted topk
// (compress_ratio == 4 layers). Each kernel writes its own slice.

// Fill window indices into out[:, :, :kWin]. Mirrors
// AttentionIndices.slidingWindow (model.py:get_window_topk_idxs).
//
// startPos == 0           → prefill: matrix[s, k] = max(s-W+1, 0) + k,
//                           masked to -1 when k >= S or v > s
// 0 < startPos < W-1      → mid-ring decode: k if k <= startPos else -1
// startPos >= W-1         → full ring decode: rotated wrap
kernel void attn_window_indices_i32(
    device int*       out      [[buffer(0)]],
    constant uint4&   dims     [[buffer(1)]],   // (B, S, K, W=kWin)
    constant int&     startPos [[buffer(2)]],
    uint3 gid [[thread_position_in_grid]]
) {
    uint k = gid.x, s = gid.y, b = gid.z;
    uint B = dims.x, S = dims.y, K = dims.z, W = dims.w;
    if (b >= B || s >= S || k >= W) return;

    int idx;
    if (startPos >= int(W) - 1) {
        int sp = startPos % int(W);
        int split = int(W) - sp - 1;
        if (int(k) < split) idx = sp + 1 + int(k);
        else               idx = int(k) - split;
    } else if (startPos > 0) {
        idx = (int(k) <= startPos) ? int(k) : -1;
    } else {
        if (int(k) >= int(S)) {
            idx = -1;
        } else {
            int base = max(int(s) - int(W) + 1, 0);
            int v = base + int(k);
            idx = (v > int(s)) ? -1 : v;
        }
    }
    out[(b * S + s) * K + k] = idx;
}

// Fill deterministic compressed indices into out[:, :, kWin:kWin+kComp]
// for layers without an Indexer (ratio == 128). Mirrors
// AttentionIndices.compressed.
kernel void attn_compressed_indices_i32(
    device int*       out    [[buffer(0)]],
    constant uint4&   dims   [[buffer(1)]],   // (B, S, K, kComp)
    constant int4&    params [[buffer(2)]],   // (kWin, startPos, ratio, offset)
    uint3 gid [[thread_position_in_grid]]
) {
    uint k = gid.x, s = gid.y, b = gid.z;
    uint B = dims.x, S = dims.y, K = dims.z, kComp = dims.w;
    int kWin = params.x, startPos = params.y, ratio = params.z, offsetVal = params.w;
    if (b >= B || s >= S || k >= kComp) return;

    int idx;
    if (startPos > 0) {
        idx = int(k) + offsetVal;
    } else {
        int limit = (int(s) + 1) / ratio;
        idx = (int(k) >= limit) ? -1 : int(k) + offsetVal;
    }
    out[(b * S + s) * K + kWin + k] = idx;
}

// Copy Indexer-emitted topk ([B, S, kComp]) into out[:, :, kWin:kWin+kComp].
kernel void attn_copy_compressed_i32(
    device int*        out  [[buffer(0)]],
    device const int*  comp [[buffer(1)]],
    constant uint4&    dims [[buffer(2)]],    // (B, S, K, kComp)
    constant uint&     kWin [[buffer(3)]],
    uint3 gid [[thread_position_in_grid]]
) {
    uint k = gid.x, s = gid.y, b = gid.z;
    uint B = dims.x, S = dims.y, K = dims.z, kComp = dims.w;
    if (b >= B || s >= S || k >= kComp) return;
    out[(b * S + s) * K + kWin + k] = comp[(b * S + s) * kComp + k];
}
