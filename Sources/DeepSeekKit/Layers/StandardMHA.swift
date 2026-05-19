import Foundation
import Metal

/// Standard multi-head attention with Grouped-Query support (TODO
/// §10.2 / T2). Used by Llama-family architectures where DeepSeek
/// uses MLA. Mirrors the layer shape published by `llama.cpp` for
/// Llama 2 / 3 / 3.1 / Code-Llama / Mistral / Qwen:
///
///   Q = wQ(x)            [B, S, Hq * D]
///   K = wK(x)            [B, S, Hkv * D]
///   V = wV(x)            [B, S, Hkv * D]
///   RoPE on Q and K (last D of each head)
///   append K, V to per-layer KV cache at row `startPos`
///   attn(Q, K_cache, V_cache, causal, scale = 1/sqrt(D))
///   y = wO(attn)         [B, S, hidden]
///
/// GQA: `nKVHeads <= nHeads`; the SDPA kernel maps each query head
/// `h_q` to `h_kv = (h_q * Hkv) / Hq` — Hq must be a multiple of
/// Hkv (every released Llama/Mistral checkpoint satisfies this).
///
/// KV cache: separate `K_cache` and `V_cache` tensors, each
/// `[B, maxSeq, Hkv, D]`. Allocated lazily on first call so a
/// `releaseCache()` returns the pages to the system between
/// unrelated prompts. The cache is private; the kernel reads the
/// "valid prefix" of length `startPos + S` on each turn.
///
/// The current SDPA kernel is the naive streaming-softmax variant
/// in `Kernels/standard_sdpa.metal`. It correctly handles GQA +
/// causal masking + arbitrary S, but is not tiled — a
/// FlashAttention-style replacement is on the perf roadmap and
/// will drop in here without changing the Swift API.
public final class StandardMHA {
    public let nHeads: Int
    public let nKVHeads: Int
    public let headDim: Int
    public let maxSeq: Int
    public let softmaxScale: Float

    public let wQ: Linear
    public let wK: Linear
    public let wV: Linear
    public let wO: Linear
    public let rope: RoPE

    private var kCache: Tensor?
    private var vCache: Tensor?

    /// Per-batch cache shape. `B` is not part of `init` — we
    /// allocate on the first forward pass once we know the batch
    /// size. Cached for reuse + lazy realloc after
    /// `releaseCache()`.
    private var cachedB: Int = 0

    public init(nHeads: Int, nKVHeads: Int, headDim: Int,
                maxSeq: Int,
                wQ: Linear, wK: Linear, wV: Linear, wO: Linear,
                rope: RoPE)
    {
        precondition(nHeads % nKVHeads == 0,
                      "GQA: nHeads (\(nHeads)) must be a multiple of nKVHeads (\(nKVHeads))")
        self.nHeads = nHeads
        self.nKVHeads = nKVHeads
        self.headDim = headDim
        self.maxSeq = maxSeq
        self.softmaxScale = 1.0 / sqrt(Float(headDim))
        self.wQ = wQ
        self.wK = wK
        self.wV = wV
        self.wO = wO
        self.rope = rope
    }

    /// Drop the KV cache buffers. ARC frees the underlying
    /// `MTLBuffer`s, returning unified-memory pages to the system.
    /// The next `callAsFunction` re-allocates lazily.
    public func releaseCache() {
        kCache = nil
        vCache = nil
        cachedB = 0
    }

    /// Forward pass. `x` is `[B, S, hidden]` F32; output is the
    /// same shape. `startPos` is the absolute position of `x[:, 0]`
    /// in the model's context window (0 for prefill, growing by 1
    /// for each decoded token).
    public func callAsFunction(_ x: Tensor,
                                startPos: Int,
                                in cmd: inout MTLCommandBuffer) -> Tensor
    {
        precondition(x.dtype == .f32 && x.shape.count == 3,
                      "StandardMHA: expected f32 [B, S, hidden]")
        let B = x.shape[0]
        let S = x.shape[1]
        let hidden = x.shape[2]

        let (kCache, vCache) = ensureCache(B: B)

        // 1. Q / K / V projections. Linear preserves leading dims
        //    and replaces the last with outFeatures, so we get
        //    [B, S, Hq*D] / [B, S, Hkv*D] / [B, S, Hkv*D].
        var q = wQ(x, in: cmd)
        var k = wK(x, in: cmd)
        var v = wV(x, in: cmd)

        // 2. RoPE on Q and K. RoPE.apply needs a 3-D
        //    [tokens, heads, headDim] shape — reshape from [B, S, H*D].
        let qHeads = q.reshape([B * S, nHeads, headDim])
        let kHeads = k.reshape([B * S, nKVHeads, headDim])
        rope.apply(qHeads, startPos: startPos, inverse: false, in: cmd)
        rope.apply(kHeads, startPos: startPos, inverse: false, in: cmd)
        // qHeads / kHeads share the buffer of q / k; the in-place
        // rotation mutated the original tensors. Continue with q, k
        // (still [B, S, H*D]) — same underlying bytes.

        // 3. Append K, V into the per-layer cache at row `startPos`.
        appendToCache(src: k, dst: kCache, B: B, S: S,
                       startPos: startPos, in: cmd)
        appendToCache(src: v, dst: vCache, B: B, S: S,
                       startPos: startPos, in: cmd)

        // 4. SDPA. Kernel wants Q [B, S_q, Hq, D] and KV
        //    [B, S_kv, Hkv, D], where S_kv = startPos + S.
        let qReshaped = q.reshape([B, S, nHeads, headDim])
        let attn = runSDPA(q: qReshaped, kCache: kCache, vCache: vCache,
                            B: B, S_q: S, S_kv: startPos + S,
                            startPos: startPos, in: cmd)
        // attn: [B, S, Hq, D]

        // 5. Output projection.
        let attnFlat = attn.reshape([B * S, nHeads * headDim])
        let projFlat = wO(attnFlat, in: cmd)
        return projFlat.reshape([B, S, hidden])
    }

    // MARK: - Cache lifecycle

    private func ensureCache(B: Int) -> (k: Tensor, v: Tensor) {
        if let k = kCache, let v = vCache, cachedB == B {
            return (k, v)
        }
        let shape = [B, maxSeq, nKVHeads, headDim]
        let k = Tensor.empty(shape: shape, dtype: .f32)
        let v = Tensor.empty(shape: shape, dtype: .f32)
        kCache = k
        vCache = v
        cachedB = B
        return (k, v)
    }

    private func appendToCache(src: Tensor, dst: Tensor,
                                B: Int, S: Int, startPos: Int,
                                in cmd: MTLCommandBuffer)
    {
        let pipeline = Device.shared.makePipeline("kv_cache_append_f32")
        let enc = cmd.makeComputeCommandEncoder()!
        enc.setComputePipelineState(pipeline)
        enc.setBuffer(src.buffer, offset: src.offset, index: 0)
        enc.setBuffer(dst.buffer, offset: dst.offset, index: 1)
        let HKVxD = nKVHeads * headDim
        var dims = SIMD4<UInt32>(UInt32(B), UInt32(S),
                                  UInt32(maxSeq), UInt32(HKVxD))
        enc.setBytes(&dims, length: MemoryLayout<SIMD4<UInt32>>.size, index: 2)
        var sp = UInt32(startPos)
        enc.setBytes(&sp, length: 4, index: 3)
        // 3-D dispatch over (B, S, HKVxD). Per-axis grid is small for
        // B and S in typical decode (1×1), so we put the threadgroup
        // mass on the innermost axis.
        let tgDepth = min(256, pipeline.maxTotalThreadsPerThreadgroup)
        enc.dispatchThreads(
            MTLSize(width: B, height: S, depth: HKVxD),
            threadsPerThreadgroup: MTLSize(width: 1, height: 1,
                                            depth: tgDepth))
        enc.endEncoding()
    }

    private func runSDPA(q: Tensor, kCache: Tensor, vCache: Tensor,
                          B: Int, S_q: Int, S_kv: Int, startPos: Int,
                          in cmd: MTLCommandBuffer) -> Tensor
    {
        let out = Tensor.empty(shape: [B, S_q, nHeads, headDim],
                                dtype: .f32)
        let pipeline = Device.shared.makePipeline("sdpa_naive_causal_gqa_f32")
        let enc = cmd.makeComputeCommandEncoder()!
        enc.setComputePipelineState(pipeline)
        enc.setBuffer(q.buffer, offset: q.offset, index: 0)
        enc.setBuffer(kCache.buffer, offset: kCache.offset, index: 1)
        enc.setBuffer(vCache.buffer, offset: vCache.offset, index: 2)
        enc.setBuffer(out.buffer, offset: 0, index: 3)
        var dims = SIMD4<UInt32>(UInt32(B), UInt32(S_q),
                                  UInt32(S_kv), UInt32(headDim))
        enc.setBytes(&dims, length: MemoryLayout<SIMD4<UInt32>>.size, index: 4)
        var heads = SIMD2<UInt32>(UInt32(nHeads), UInt32(nKVHeads))
        enc.setBytes(&heads, length: MemoryLayout<SIMD2<UInt32>>.size, index: 5)
        var sp = UInt32(startPos)
        enc.setBytes(&sp, length: 4, index: 6)
        var scale = softmaxScale
        enc.setBytes(&scale, length: 4, index: 7)
        // Grid: (B, Hq, S_q). Each thread runs one query row's
        // streaming softmax over the cached S_kv keys.
        enc.dispatchThreads(
            MTLSize(width: B, height: nHeads, depth: S_q),
            threadsPerThreadgroup: MTLSize(width: 1, height: 32,
                                            depth: 1))
        enc.endEncoding()
        return out
    }
}
