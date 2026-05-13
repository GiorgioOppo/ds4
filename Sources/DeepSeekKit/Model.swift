import Foundation
import Metal

/// Embedding table + lookup. Stripped down from `ParallelEmbedding`
/// in model.py:83 — single-rank (no tensor parallel).
public final class ParallelEmbedding {
    public let vocabSize: Int
    public let dim: Int
    public let weight: Tensor      // [vocab, dim] f32 or bf16

    private static let pF32  = Device.shared.makePipeline("embed_lookup_f32")
    private static let pBF16 = Device.shared.makePipeline("embed_lookup_bf16_to_f32")

    public init(vocabSize: Int, dim: Int, weight: Tensor) {
        precondition(weight.dtype == .f32 || weight.dtype == .bf16,
                     "ParallelEmbedding: weight must be f32 or bf16, got \(weight.dtype)")
        self.vocabSize = vocabSize; self.dim = dim; self.weight = weight
    }

    /// `ids`: flat [N] Int32 array of token ids.
    public func lookup(_ ids: [Int32], in cmd: MTLCommandBuffer) -> Tensor {
        let N = ids.count
        let idsT = ids.withUnsafeBytes { Tensor.from(bytes: $0, shape: [N], dtype: .i32) }
        let out = Tensor.empty(shape: [N, dim], dtype: .f32)

        let pipeline: MTLComputePipelineState = (weight.dtype == .bf16) ? Self.pBF16 : Self.pF32
        let enc = cmd.makeComputeCommandEncoder()!
        enc.setComputePipelineState(pipeline)
        enc.setBuffer(weight.buffer, offset: weight.offset, index: 0)
        enc.setBuffer(idsT.buffer, offset: 0, index: 1)
        enc.setBuffer(out.buffer, offset: 0, index: 2)
        var dims = SIMD2<UInt32>(UInt32(N), UInt32(dim))
        enc.setBytes(&dims, length: MemoryLayout.size(ofValue: dims), index: 3)
        enc.dispatchThreads(MTLSize(width: dim, height: N, depth: 1),
                            threadsPerThreadgroup: MTLSize(width: 16, height: 16, depth: 1))
        enc.endEncoding()
        // Mirrors model.py:101 — F.embedding returns the weight's dtype
        // (BF16 in V4-Flash). We always allocate F32 storage; round-trip
        // so layer 0 sees the same numerical distribution the network was
        // trained on.
        Elementwise.bf16RoundTripInplace(out, in: cmd)
        return out
    }
}

/// LM head with HC mixing. Mirrors `ParallelHead` in model.py:703–735.
/// `hc_head` here is the simpler sigmoid-only collapse (no Sinkhorn) used
/// before the lm_head matmul. Logits are produced only for the LAST sequence
/// position via `get_logits(x[:, -1])`.
public final class ParallelHead {
    public let vocabSize: Int
    public let dim: Int
    public let normEps: Float
    public let hcEps: Float
    public let weight: Tensor              // [vocab, dim] f32

    private static let pRsqrt = Device.shared.makePipeline("rsqrt_mean_square_f32")
    private static let pCollapse = Device.shared.makePipeline("hc_collapse_f32")

    public init(vocabSize: Int, dim: Int, normEps: Float, hcEps: Float, weight: Tensor) {
        self.vocabSize = vocabSize; self.dim = dim
        self.normEps = normEps; self.hcEps = hcEps
        self.weight = weight
    }

    /// `x`: [B, S, hc, D] f32. Returns [B, vocab] f32 (logits on last token).
    public func callAsFunction(_ x: Tensor,
                                hcFn: Tensor, hcScale: Tensor, hcBase: Tensor,
                                norm: RMSNorm,
                                in cmd: MTLCommandBuffer) -> Tensor {
        precondition(x.dtype == .f32 && x.shape.count == 4)
        let B = x.shape[0], S = x.shape[1], HC = x.shape[2], D = x.shape[3]
        let N = B * S
        precondition(D == dim)
        let hcDim = HC * D
        let hcMult = HC

        let xFlat = x.reshape([N, hcDim])

        // 1. rsqrt(mean(x²) + eps)
        let rsqrt = Tensor.empty(shape: [N], dtype: .f32)
        do {
            let enc = cmd.makeComputeCommandEncoder()!
            enc.setComputePipelineState(Self.pRsqrt)
            enc.setBuffer(xFlat.buffer, offset: xFlat.offset, index: 0)
            enc.setBuffer(rsqrt.buffer, offset: 0, index: 1)
            var d = UInt32(hcDim); var e = normEps
            enc.setBytes(&d, length: 4, index: 2)
            enc.setBytes(&e, length: 4, index: 3)
            enc.setThreadgroupMemoryLength(256 * MemoryLayout<Float>.size, index: 0)
            enc.dispatchThreadgroups(MTLSize(width: N, height: 1, depth: 1),
                                     threadsPerThreadgroup: MTLSize(width: 256, height: 1, depth: 1))
            enc.endEncoding()
        }

        // 2. mixes = Linear(hcFn)(x_flat); mixes *= rsqrt; pre = sigmoid(mixes*scale + base) + eps
        // ParallelHead.hc_head runs in FP32 (model.py:730 `x = x.flatten(2).float()`)
        // — no BF16 quantisation on the gating mixes.
        let lin = Linear(inFeatures: hcDim, outFeatures: hcMult,
                         weight: hcFn, scale: nil,
                         castOutputToBF16: false)
        let mixes = lin(xFlat, in: cmd)        // [N, hcMult]

        // mixes *= rsqrt (broadcast)
        let bcast = Device.shared.makePipeline("broadcast_row_mul_f32")
        do {
            let enc = cmd.makeComputeCommandEncoder()!
            enc.setComputePipelineState(bcast)
            enc.setBuffer(mixes.buffer, offset: 0, index: 0)
            enc.setBuffer(rsqrt.buffer, offset: 0, index: 1)
            var dims = SIMD2<UInt32>(UInt32(N), UInt32(hcMult))
            enc.setBytes(&dims, length: MemoryLayout.size(ofValue: dims), index: 2)
            enc.dispatchThreads(MTLSize(width: hcMult, height: N, depth: 1),
                                threadsPerThreadgroup: MTLSize(width: 16, height: 16, depth: 1))
            enc.endEncoding()
        }
        // pre = sigmoid(mixes * hcScale + hcBase) + hcEps  — done host-side via scalar broadcast.
        cmd.commit(); cmd.waitUntilCompleted()
        let mixesPtr = mixes.buffer.contents().bindMemory(to: Float.self, capacity: N * hcMult)
        let scalePtr = hcScale.buffer.contents().bindMemory(to: Float.self, capacity: 1)
        let basePtr = hcBase.buffer.contents().bindMemory(to: Float.self, capacity: hcMult)
        let scaleVal = scalePtr[0]
        for i in 0..<(N * hcMult) {
            let h = i % hcMult
            let v = 1.0 / (1.0 + expf(-(mixesPtr[i] * scaleVal + basePtr[h])))
            mixesPtr[i] = v + hcEps
        }

        // 3. y[N, D] = Σ_h pre[N, h] * x[N, h, D]
        let y = Tensor.empty(shape: [N, D], dtype: .f32)
        let cmd2 = Device.shared.queue.makeCommandBuffer()!
        do {
            let enc = cmd2.makeComputeCommandEncoder()!
            enc.setComputePipelineState(Self.pCollapse)
            enc.setBuffer(x.buffer, offset: x.offset, index: 0)
            enc.setBuffer(mixes.buffer, offset: 0, index: 1)
            enc.setBuffer(y.buffer, offset: 0, index: 2)
            var dims = SIMD3<UInt32>(UInt32(N), UInt32(HC), UInt32(D))
            enc.setBytes(&dims, length: MemoryLayout.size(ofValue: dims), index: 3)
            enc.dispatchThreads(MTLSize(width: D, height: N, depth: 1),
                                threadsPerThreadgroup: MTLSize(width: 16, height: 16, depth: 1))
            enc.endEncoding()
        }

        // 4. norm(y); take last token of each batch; matmul against lm_head.
        let yNorm = norm(y, in: cmd2)
        // Slice last token per batch via blit copy.
        let lastTok = Tensor.empty(shape: [B, dim], dtype: .f32)
        let blit = cmd2.makeBlitCommandEncoder()!
        let bytesPerRow = dim * MemoryLayout<Float>.size
        for b in 0..<B {
            let src = (b * S + S - 1) * bytesPerRow
            let dst = b * bytesPerRow
            blit.copy(from: yNorm.buffer, sourceOffset: src,
                      to: lastTok.buffer, destinationOffset: dst,
                      size: bytesPerRow)
        }
        blit.endEncoding()

        // 5. logits = lastTok @ weight^T   (Linear with weight shape [vocab, dim])
        // castOutputToBF16=false: the logits feed straight into argmax /
        // softmax (sampling) — losing 16 mantissa bits here would
        // collapse ties and warp the temperature scaling. The reference
        // also keeps the LM-head output in F32 for the same reason.
        let lmHead = Linear(inFeatures: dim, outFeatures: vocabSize,
                            weight: weight, scale: nil,
                            castOutputToBF16: false)
        let logits = lmHead(lastTok, in: cmd2)
        cmd2.commit(); cmd2.waitUntilCompleted()
        return logits
    }
}

/// DeepSeek-V4 transformer. Mirrors `Transformer` in
/// `Reference/inference/model.py` lines 769–809.
public final class Transformer {
    public let config: ModelConfig
    public let embed: ParallelEmbedding
    public let layers: [Block]
    public let mtp: [MTPBlock]
    public let norm: RMSNorm
    public let head: ParallelHead

    public let hcHeadFn: Tensor
    public let hcHeadBase: Tensor
    public let hcHeadScale: Tensor

    private let pHcExpand: MTLComputePipelineState

    /// Optional reference to the WeightLoader that built this model.
    /// `Transformer.load` parks it here so the loader (and its
    /// `shardLayers` index) stays alive for the model's lifetime and
    /// `forward(...)` can call `prefetchLayer` / `releaseLayer`
    /// between blocks. Nil for non-streaming strategies — `forward`
    /// short-circuits on nil.
    internal var weightLoader: WeightLoader? = nil

    public init(config: ModelConfig,
                embed: ParallelEmbedding,
                layers: [Block],
                mtp: [MTPBlock],
                norm: RMSNorm,
                head: ParallelHead,
                hcHeadFn: Tensor, hcHeadBase: Tensor, hcHeadScale: Tensor) {
        self.config = config
        self.embed = embed
        self.layers = layers
        self.mtp = mtp
        self.norm = norm
        self.head = head
        self.hcHeadFn = hcHeadFn
        self.hcHeadBase = hcHeadBase
        self.hcHeadScale = hcHeadScale
        self.pHcExpand = Device.shared.makePipeline("hc_expand_f32")
    }

    /// `inputIds`: [[Int]] — outer is batch, inner is seqlen. All inner
    /// arrays must have the same length.
    public func forward(inputIds: [[Int]], startPos: Int) -> Tensor {
        let B = inputIds.count
        precondition(B > 0)
        let S = inputIds[0].count
        for row in inputIds { precondition(row.count == S, "ragged batch not supported") }
        MemoryLogger.snapshot("forward:start", force: true)

        let flatIds: [Int32] = inputIds.flatMap { $0.map(Int32.init) }
        let cmd = Device.shared.queue.makeCommandBuffer()!

        // 1. embed → [B*S, dim]
        let h = embed.lookup(flatIds, in: cmd)

        // 2. hc-expand [B*S, dim] → [B*S, hc, dim]
        let hc = config.hcMult
        let hExpanded = Tensor.empty(shape: [B * S, hc, config.dim], dtype: .f32)
        let enc = cmd.makeComputeCommandEncoder()!
        enc.setComputePipelineState(pHcExpand)
        enc.setBuffer(h.buffer, offset: 0, index: 0)
        enc.setBuffer(hExpanded.buffer, offset: 0, index: 1)
        var dims = SIMD3<UInt32>(UInt32(B * S), UInt32(hc), UInt32(config.dim))
        enc.setBytes(&dims, length: MemoryLayout.size(ofValue: dims), index: 2)
        enc.dispatchThreads(MTLSize(width: config.dim, height: hc, depth: B * S),
                            threadsPerThreadgroup: MTLSize(width: 16, height: 4, depth: 4))
        enc.endEncoding()
        cmd.commit(); cmd.waitUntilCompleted()
        MemoryLogger.snapshot("forward:embed+hc-expanded", force: true)
        traceTensorStats("embed", h)
        traceTensorStats("hc-expand", hExpanded)

        // 3. Run each block sequentially. Each commits its own buffers.
        //
        // Streaming hints: when `weightLoader.streamingEnabled` is
        // true, we release layer K's pages immediately after its
        // commit+wait completes. Working window is bounded to one
        // layer at a time (~max-shard bytes), with shared shards
        // (embed/head/norms, owner == -1) excluded from release.
        //
        // Previous revision added MADV_WILLNEED prefetch on K+1
        // before computing K. That backfired: the kernel would
        // start pulling K+1's pages while K-1's MADV_DONTNEED hint
        // hadn't been honoured yet → ~3 layers resident
        // simultaneously, OOMing 16 GB Macs. Letting the natural
        // page-fault path handle the next layer keeps residency
        // strictly to "the layer the GPU is currently reading
        // from".
        var x = hExpanded.reshape([B, S, hc, config.dim])
        let loader = self.weightLoader
        // Layers to dump stats for under --trace-norms. Densified through
        // the first few layers because the prefill-vs-decode residual
        // divergence appears between layer 0 and 10, exactly where the
        // V4 Compressor / Indexer modules activate (compress_ratios=0
        // at layers 0-1, then alternates 4/128 from layer 2 on).
        let nL = layers.count
        let traceLayers: Set<Int> = nL <= 12
            ? Set(0..<nL)
            : Set([0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 20, 30, nL - 1])
        for (k, layer) in layers.enumerated() {
            // Pool mode: pread layer K's shard into the rotating
            // slot BEFORE the block's forward references its
            // Tensors. (No-op in `.mmap` / `.preload` strategies.)
            loader?.ensureLayer(k)
            var cmdL = Device.shared.queue.makeCommandBuffer()!
            x = layer(x, startPos: startPos, inputIds: flatIds, in: &cmdL)
            cmdL.commit(); cmdL.waitUntilCompleted()
            loader?.releaseLayer(k)
            MemoryLogger.snapshot(
                "forward:layer-\(String(format: "%02d", k))", force: true)
            if traceLayers.contains(k) {
                traceTensorStats("layer-\(String(format: "%02d", k))", x)
            }
        }

        // 4. Head. `ParallelHead.callAsFunction` commits `cmdH`
        // internally (after rsqrt + mixes) and the remaining work
        // (collapse + norm + slice + lm_head) on a fresh cmd2 that
        // it also commits before returning. Caller must NOT commit
        // cmdH again — double-commit traps inside Metal.
        let cmdH = Device.shared.queue.makeCommandBuffer()!
        let logits = head(x, hcFn: hcHeadFn, hcScale: hcHeadScale, hcBase: hcHeadBase,
                          norm: norm, in: cmdH)
        MemoryLogger.snapshot("forward:complete", force: true)
        traceTensorStats("logits", logits)
        return logits
    }

    /// Drop all runtime KV cache buffers across every layer (main blocks
    /// and MTP blocks), including each compressor's rolling state and any
    /// indexer kvCache. ARC frees the underlying `MTLBuffer`s and the
    /// pages return to the system. The next `forward` re-allocates lazily.
    ///
    /// `MLA.releaseCache` already releases the attention-side compressor's
    /// state, and `Indexer.releaseCache` releases the indexer-owned
    /// compressor — together they cover every Compressor instance in the
    /// model.
    ///
    /// Intended for use between unrelated prompts, when pausing a session,
    /// or under memory pressure. Cheap to call (O(numLayers) host-side, no
    /// GPU work). Must be called only between forward passes — not
    /// thread-safe.
    public func releaseCache() {
        for block in layers {
            block.attn.releaseCache()
            block.attn.indexer?.releaseCache()
        }
        for m in mtp {
            m.block.attn.releaseCache()
            m.block.attn.indexer?.releaseCache()
        }
    }
}
