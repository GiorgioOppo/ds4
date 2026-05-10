import Foundation
import Metal

/// KV compressor: gated softmax-pooling over `compressRatio` consecutive
/// tokens. Mirrors the `Compressor` module in
/// `Reference/inference/model.py` lines 279–377.
///
/// **Implemented**: prefill path (`startPos == 0`), with and without overlap.
/// **Not implemented yet**: decode path (single-token incremental). Decode
/// requires keeping the `kv_state` / `score_state` buffers populated across
/// commits and emitting a compressed token only every `ratio` steps; that
/// state-machine is its own subproject and is left as a `fatalError` here.
///
/// The Compressor's KV cache is owned by the parent (MLA / Indexer); the
/// caller assigns `self.kvCache` and `self.rope` before invoking forward.
public final class Compressor {
    public let dim: Int
    public let headDim: Int
    public let ropeHeadDim: Int
    public let nopeHeadDim: Int
    public let compressRatio: Int
    public let overlap: Bool
    public let rotate: Bool
    public let normEps: Float

    public let ape: Tensor                  // [ratio, coff*head_dim] f32
    public let wkv: Linear
    public let wgate: Linear
    public let norm: RMSNorm

    public var kvState: Tensor              // [maxBatch, coff*ratio, coff*head_dim] f32
    public var scoreState: Tensor           // [maxBatch, coff*ratio, coff*head_dim] f32

    /// Assigned by the parent module (MLA / Indexer) — the slice of the
    /// attention KV cache where compressed tokens are written.
    public var kvCache: Tensor?
    public var rope: RoPE?

    private let pBroadcastAdd: MTLComputePipelineState
    private let pWeightedSum: MTLComputePipelineState

    public init(config: ModelConfig, compressRatio: Int, headDim: Int, rotate: Bool,
                ape: Tensor, wkv: Linear, wgate: Linear, norm: RMSNorm,
                kvState: Tensor, scoreState: Tensor) {
        self.dim = config.dim
        self.headDim = headDim
        self.ropeHeadDim = config.ropeHeadDim
        self.nopeHeadDim = headDim - config.ropeHeadDim
        self.compressRatio = compressRatio
        self.overlap = compressRatio == 4
        self.rotate = rotate
        self.normEps = config.normEps
        self.ape = ape
        self.wkv = wkv
        self.wgate = wgate
        self.norm = norm
        self.kvState = kvState
        self.scoreState = scoreState
        self.pBroadcastAdd = Device.shared.makePipeline("broadcast_add_4d_2d_f32")
        self.pWeightedSum = Device.shared.makePipeline("weighted_sum_axis2_f32")
    }

    /// Prefill path. Returns the compressed KV `[B, S/ratio, head_dim]` after
    /// norm + RoPE on the rope tail and (optionally) Hadamard + FP4 quant or
    /// FP8 act_quant on the nope dims.
    ///
    /// Caller is responsible for writing the result into the attention KV
    /// cache after this returns; the reference Python does that, but pulling
    /// it inside the Compressor would require a blit copy that's clearer at
    /// the call site.
    public func callAsFunction(_ x: Tensor, startPos: Int, in cmd: MTLCommandBuffer) -> Tensor? {
        precondition(x.dtype == .f32 && x.shape.count == 3,
                     "Compressor expects f32 [B, S, dim]")
        guard startPos == 0 else {
            fatalError("Compressor decode path not implemented yet — see file header")
        }
        let B = x.shape[0]
        let S = x.shape[1]
        let ratio = compressRatio
        let coff = overlap ? 2 : 1
        let coffHeadDim = coff * headDim

        // The simplest path: assume seqlen is a multiple of `ratio`. The
        // reference handles a non-multiple by stashing the tail into
        // kv_state for the next decode step; for the prefill-only port we
        // require the caller to pad / chunk so that S % ratio == 0.
        precondition(S % ratio == 0, "Compressor prefill requires S divisible by ratio")
        let numBlocks = S / ratio

        // 1. Linear projections.
        let kvFlat = wkv(x.reshape([B * S, dim]), in: cmd)             // [B*S, coffHeadDim]
        let scoreFlat = wgate(x.reshape([B * S, dim]), in: cmd)        // [B*S, coffHeadDim]
        let kv = kvFlat.reshape([B, numBlocks, ratio, coffHeadDim])
        let score = scoreFlat.reshape([B, numBlocks, ratio, coffHeadDim])

        // 2. score += ape (broadcast over [B, numBlocks])
        broadcastAdd(target: score, weight: ape,
                     B: B, NS: numBlocks, R: ratio, C: coffHeadDim, in: cmd)

        // 3. Optional overlap_transform; otherwise the tensors are already
        //    [B, NS, ratio, head_dim].
        let kvWide: Tensor
        let scoreWide: Tensor
        let axisR: Int
        if overlap {
            // Pad value for score is -inf so masked positions vanish in softmax.
            kvWide = OverlapTransform.apply(kv, padValue: 0, in: cmd)
            scoreWide = OverlapTransform.apply(score, padValue: -.infinity, in: cmd)
            axisR = 2 * ratio
        } else {
            kvWide = kv
            scoreWide = score
            axisR = ratio
        }

        // 4. Softmax along axis=2 (the ratio axis).
        SoftmaxAxis.apply(scoreWide, axis: 2, in: cmd)

        // 5. y[B, NS, head_dim] = Σ_r kv[B, NS, r, head_dim] * score[B, NS, r, head_dim].
        let pooled = Tensor.empty(shape: [B, numBlocks, headDim], dtype: .f32)
        weightedSumAxis2(kv: kvWide, score: scoreWide, out: pooled,
                          B: B, NS: numBlocks, R: axisR, C: headDim, in: cmd)

        // 6. Norm.
        let normed = norm(pooled, in: cmd)

        // 7. RoPE on the rope tail. Reshape to [B*NS, 1, head_dim] so the
        //    existing RoPE kernel (which expects [tokens, heads, head_dim])
        //    treats each compressed token as one head.
        guard let rope = rope else {
            fatalError("Compressor.rope must be set before forward")
        }
        let normedAsTokens = normed.reshape([B * numBlocks, 1, headDim])
        rope.apply(normedAsTokens, startPos: 0, inverse: false, in: cmd)

        // 8. Quantization step. We currently route to act_quant in inplace
        //    mode (round-trip) so the math matches QAT noise but the tensor
        //    stays f32 for downstream consumers.
        if rotate {
            Hadamard.apply(normedAsTokens, in: cmd)
            let aq = ActQuant(format: .fp4)
            _ = aq.quant(normedAsTokens.reshape([B * numBlocks, headDim]),
                         inplace: true, in: cmd)
        } else {
            // Non-overlap path quantises only the non-rope dims. With the
            // current f32-only scaffold we keep both halves in f32; this
            // matches the reference except the QAT noise is not applied.
            // TODO: split the buffer and call ActQuant.fp8 on the leading
            //       (head_dim - rope_head_dim) columns when we add a slicing
            //       primitive.
        }

        return normedAsTokens.reshape([B, numBlocks, headDim])
    }

    private func broadcastAdd(target y: Tensor, weight w: Tensor,
                              B: Int, NS: Int, R: Int, C: Int,
                              in cmd: MTLCommandBuffer) {
        let enc = cmd.makeComputeCommandEncoder()!
        enc.setComputePipelineState(pBroadcastAdd)
        enc.setBuffer(y.buffer, offset: y.offset, index: 0)
        enc.setBuffer(w.buffer, offset: w.offset, index: 1)
        var dims = SIMD4<UInt32>(UInt32(B), UInt32(NS), UInt32(R), UInt32(C))
        enc.setBytes(&dims, length: MemoryLayout.size(ofValue: dims), index: 2)
        enc.dispatchThreads(MTLSize(width: C, height: R, depth: B * NS),
                            threadsPerThreadgroup: MTLSize(width: 16, height: 4, depth: 4))
        enc.endEncoding()
    }

    private func weightedSumAxis2(kv: Tensor, score: Tensor, out: Tensor,
                                   B: Int, NS: Int, R: Int, C: Int,
                                   in cmd: MTLCommandBuffer) {
        let enc = cmd.makeComputeCommandEncoder()!
        enc.setComputePipelineState(pWeightedSum)
        enc.setBuffer(kv.buffer, offset: kv.offset, index: 0)
        enc.setBuffer(score.buffer, offset: score.offset, index: 1)
        enc.setBuffer(out.buffer, offset: 0, index: 2)
        var dims = SIMD4<UInt32>(UInt32(B), UInt32(NS), UInt32(R), UInt32(C))
        enc.setBytes(&dims, length: MemoryLayout.size(ofValue: dims), index: 3)
        enc.dispatchThreads(MTLSize(width: C, height: NS, depth: B),
                            threadsPerThreadgroup: MTLSize(width: 16, height: 8, depth: 1))
        enc.endEncoding()
    }

    // MARK: - Pure-Swift reference (no overlap, no rotate, prefill only)

    /// Mirrors the prefill non-overlap non-rotate path of Compressor.forward
    /// for testing. Skips quantization (which is a round-trip QAT noise that
    /// would only contribute precision noise to the test).
    public static func referenceCPU(
        x: [Float],
        wkv: [Float], wgate: [Float],            // [coffHeadDim, dim] each
        ape: [Float],                             // [ratio, coffHeadDim]
        normWeight: [Float],                      // [headDim]
        normEps: Float,
        ropeFreqs: [Float],                       // [maxSeqLen, ropeHeadDim/2, 2]
        B: Int, S: Int, dim: Int, headDim: Int, ropeHeadDim: Int,
        ratio: Int, overlap: Bool
    ) -> [Float] {
        precondition(!overlap, "CPU reference for overlap=true not implemented")
        precondition(S % ratio == 0)
        let coff = 1
        let coffHeadDim = coff * headDim
        let numBlocks = S / ratio

        // 1. Linear: kv[b, s, c] = Σ_d x[b, s, d] * wkv[c, d]
        var kv = [Float](repeating: 0, count: B * S * coffHeadDim)
        var score = [Float](repeating: 0, count: B * S * coffHeadDim)
        for b in 0..<B {
            for s in 0..<S {
                for c in 0..<coffHeadDim {
                    var av: Float = 0
                    var sv: Float = 0
                    for d in 0..<dim {
                        av += x[(b * S + s) * dim + d] * wkv[c * dim + d]
                        sv += x[(b * S + s) * dim + d] * wgate[c * dim + d]
                    }
                    kv[(b * S + s) * coffHeadDim + c] = av
                    score[(b * S + s) * coffHeadDim + c] = sv
                }
            }
        }

        // 2. Reshape to [B, NS, R, C] and add ape.
        // 3. Softmax along R; weighted sum.
        var pooled = [Float](repeating: 0, count: B * numBlocks * headDim)
        for b in 0..<B {
            for ns in 0..<numBlocks {
                // Build score block + softmax.
                var sm = [Float](repeating: 0, count: ratio * coffHeadDim)
                for r in 0..<ratio {
                    for c in 0..<coffHeadDim {
                        let s = ns * ratio + r
                        sm[r * coffHeadDim + c] = score[(b * S + s) * coffHeadDim + c]
                                                  + ape[r * coffHeadDim + c]
                    }
                }
                // softmax along axis=0 (the ratio dim) — for each c independently.
                for c in 0..<coffHeadDim {
                    var m = -Float.infinity
                    for r in 0..<ratio { m = max(m, sm[r * coffHeadDim + c]) }
                    var sumExp: Float = 0
                    for r in 0..<ratio {
                        let e = exp(sm[r * coffHeadDim + c] - m)
                        sm[r * coffHeadDim + c] = e
                        sumExp += e
                    }
                    for r in 0..<ratio { sm[r * coffHeadDim + c] /= sumExp }
                }
                // Weighted sum over r.
                for c in 0..<headDim {
                    var acc: Float = 0
                    for r in 0..<ratio {
                        let s = ns * ratio + r
                        acc += kv[(b * S + s) * coffHeadDim + c] * sm[r * coffHeadDim + c]
                    }
                    pooled[(b * numBlocks + ns) * headDim + c] = acc
                }
            }
        }

        // 4. RMSNorm with `normWeight`, eps.
        for b in 0..<B {
            for ns in 0..<numBlocks {
                let off = (b * numBlocks + ns) * headDim
                var sq: Float = 0
                for d in 0..<headDim { let v = pooled[off + d]; sq += v * v }
                let r = 1.0 / (sq / Float(headDim) + normEps).squareRoot()
                for d in 0..<headDim { pooled[off + d] = pooled[off + d] * r * normWeight[d] }
            }
        }

        // 5. RoPE on the trailing `ropeHeadDim` of each compressed token.
        let halfRD = ropeHeadDim / 2
        for b in 0..<B {
            for ns in 0..<numBlocks {
                let baseOut = (b * numBlocks + ns) * headDim + (headDim - ropeHeadDim)
                for i in 0..<halfRD {
                    let c = ropeFreqs[2 * (ns * halfRD + i) + 0]
                    let s = ropeFreqs[2 * (ns * halfRD + i) + 1]
                    let a = pooled[baseOut + 2 * i]
                    let bv = pooled[baseOut + 2 * i + 1]
                    pooled[baseOut + 2 * i]     = a * c - bv * s
                    pooled[baseOut + 2 * i + 1] = a * s + bv * c
                }
            }
        }

        return pooled
    }
}
