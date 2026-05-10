import Foundation
import Metal

/// KV compressor: gated softmax-pooling over `compressRatio` consecutive
/// tokens. Mirrors the `Compressor` module in
/// `Reference/inference/model.py` lines 279–377.
///
/// Both prefill (`startPos == 0`, multi-token) and decode (`startPos > 0`,
/// single-token incremental) paths are implemented, with and without
/// overlap. The decode path maintains `kvState`/`scoreState` across calls
/// and only emits a compressed token every `compressRatio` steps.
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
    private let pOverlapConcat: MTLComputePipelineState
    private let pStateShift: MTLComputePipelineState

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
        self.pOverlapConcat = Device.shared.makePipeline("compressor_overlap_concat_f32")
        self.pStateShift = Device.shared.makePipeline("compressor_state_shift_copy_f32")
    }

    /// Forward pass. Returns the compressed KV when one is emitted, else nil.
    /// Prefill (startPos == 0) emits `[B, S/ratio, head_dim]`; decode emits
    /// `[B, 1, head_dim]` only when `(start_pos + 1) % ratio == 0`.
    public func callAsFunction(_ x: Tensor, startPos: Int, in cmd: MTLCommandBuffer) -> Tensor? {
        precondition(x.dtype == .f32 && x.shape.count == 3,
                     "Compressor expects f32 [B, S, dim]")
        if startPos == 0 {
            return forwardPrefill(x, in: cmd)
        }
        return forwardDecode(x, startPos: startPos, in: cmd)
    }

    // MARK: - Prefill

    private func forwardPrefill(_ x: Tensor, in cmd: MTLCommandBuffer) -> Tensor? {
        let B = x.shape[0]
        let S = x.shape[1]
        let ratio = compressRatio
        let coff = overlap ? 2 : 1
        let coffHeadDim = coff * headDim

        precondition(S % ratio == 0, "Compressor prefill requires S divisible by ratio")
        let numBlocks = S / ratio
        if numBlocks == 0 { return nil }

        let kvFlat = wkv(x.reshape([B * S, dim]), in: cmd)
        let scoreFlat = wgate(x.reshape([B * S, dim]), in: cmd)
        let kv = kvFlat.reshape([B, numBlocks, ratio, coffHeadDim])
        let score = scoreFlat.reshape([B, numBlocks, ratio, coffHeadDim])

        broadcastAdd(target: score, weight: ape,
                     B: B, NS: numBlocks, R: ratio, C: coffHeadDim, in: cmd)

        let kvWide: Tensor
        let scoreWide: Tensor
        let axisR: Int
        if overlap {
            kvWide = OverlapTransform.apply(kv, padValue: 0, in: cmd)
            scoreWide = OverlapTransform.apply(score, padValue: -.infinity, in: cmd)
            axisR = 2 * ratio
        } else {
            kvWide = kv
            scoreWide = score
            axisR = ratio
        }

        SoftmaxAxis.apply(scoreWide, axis: 2, in: cmd)

        let pooled = Tensor.empty(shape: [B, numBlocks, headDim], dtype: .f32)
        weightedSumAxis2(kv: kvWide, score: scoreWide, out: pooled,
                          B: B, NS: numBlocks, R: axisR, C: headDim, in: cmd)

        return postProcess(pooled.reshape([B * numBlocks, 1, headDim]),
                           tokens: B * numBlocks, in: cmd)
            .reshape([B, numBlocks, headDim])
    }

    // MARK: - Decode (one token at a time, model.py:343-377)

    private func forwardDecode(_ x: Tensor, startPos: Int,
                               in cmd: MTLCommandBuffer) -> Tensor? {
        let B = x.shape[0]
        precondition(x.shape[1] == 1, "Compressor decode expects seqlen == 1")
        let ratio = compressRatio
        let coff = overlap ? 2 : 1
        let coffHeadDim = coff * headDim
        let shouldEmit = (startPos + 1) % ratio == 0

        // 1. Linear projections — one row per batch.
        let kvRow = wkv(x.reshape([B, dim]), in: cmd)        // [B, coffHeadDim]
        let scoreRow = wgate(x.reshape([B, dim]), in: cmd)   // [B, coffHeadDim]

        // score += ape[startPos % ratio] (broadcast across B)
        addApeRow(scoreRow, apeRow: startPos % ratio,
                  B: B, C: coffHeadDim, in: cmd)

        // 2. Write into kvState / scoreState at the right slot.
        let stateRow = overlap ? (ratio + startPos % ratio) : (startPos % ratio)
        blitRowToState(src: kvRow, state: kvState, B: B,
                       stateRows: kvState.shape[1], rowIndex: stateRow,
                       rowBytes: coffHeadDim * MemoryLayout<Float>.size, in: cmd)
        blitRowToState(src: scoreRow, state: scoreState, B: B,
                       stateRows: scoreState.shape[1], rowIndex: stateRow,
                       rowBytes: coffHeadDim * MemoryLayout<Float>.size, in: cmd)

        if !shouldEmit {
            return nil
        }

        // 3. Build the [B, axisR, headDim] kv/score tensors to pool from.
        let axisR: Int
        let pooledKV: Tensor
        let pooledScore: Tensor

        if overlap {
            // axis = 2*ratio rows, headDim cols per row, gathered with the
            // first/second-half slice trick (model.py:350-351).
            axisR = 2 * ratio
            pooledKV = overlapConcat(kvState, B: B, R: ratio, D: headDim, in: cmd)
            pooledScore = overlapConcat(scoreState, B: B, R: ratio, D: headDim, in: cmd)
        } else {
            axisR = ratio
            pooledKV = kvState                  // [B, ratio, headDim]
            pooledScore = scoreState
        }

        // 4. Softmax along the ratio axis (which is axis=1 of the [B, R, D]
        //    tensors), then weighted sum.
        SoftmaxAxis.apply(pooledScore, axis: 1, in: cmd)

        // weighted_sum_axis2 expects [B, NS, R, C] → [B, NS, C]. With NS=1
        // we reshape and dispatch.
        let kvAsBlocks = pooledKV.reshape([B, 1, axisR, headDim])
        let scoreAsBlocks = pooledScore.reshape([B, 1, axisR, headDim])
        let emitted = Tensor.empty(shape: [B, 1, headDim], dtype: .f32)
        weightedSumAxis2(kv: kvAsBlocks, score: scoreAsBlocks, out: emitted,
                          B: B, NS: 1, R: axisR, C: headDim, in: cmd)

        // 5. Overlap path: shift state[:, R:] → state[:, :R].
        if overlap {
            stateShiftDown(state: kvState, B: B, R: ratio, twoD: coffHeadDim, in: cmd)
            stateShiftDown(state: scoreState, B: B, R: ratio, twoD: coffHeadDim, in: cmd)
        }

        // 6. Shared post-emit: norm + RoPE + (Hadamard+FP4 | FP8 noise stub).
        // RoPE freqs index for decode is (startPos + 1 - ratio).
        let ropePos = startPos + 1 - ratio
        return postProcess(emitted.reshape([B, 1, headDim]),
                           tokens: B, in: cmd, ropeStartPos: ropePos)
    }

    /// Norm + RoPE + (rotate? Hadamard + FP4 quant : FP8 quant). Shared
    /// between prefill and decode. Returns the same tensor reshape as
    /// `[tokens, 1, headDim]` after the in-place ops.
    private func postProcess(_ pooled: Tensor, tokens: Int,
                              in cmd: MTLCommandBuffer,
                              ropeStartPos: Int = 0) -> Tensor {
        guard let rope = rope else { fatalError("Compressor.rope must be set") }

        // pooled comes in as [tokens, 1, headDim]; norm wants 2D-ish input.
        let normed = norm(pooled.reshape([tokens, headDim]), in: cmd)
            .reshape([tokens, 1, headDim])
        rope.apply(normed, startPos: ropeStartPos, inverse: false, in: cmd)

        if rotate {
            Hadamard.apply(normed, in: cmd)
            _ = ActQuant(format: .fp4).quant(normed.reshape([tokens, headDim]),
                                              inplace: true, in: cmd)
        }
        // (Non-rotate FP8 act_quant on nope dims requires a slicing primitive
        // and is handled in Tier 3 — skipping here is structurally correct
        // but skips QAT noise.)
        return normed
    }

    // MARK: - Small helpers wrapping kernels / blits

    private func addApeRow(_ scoreRow: Tensor, apeRow: Int,
                           B: Int, C: Int, in cmd: MTLCommandBuffer) {
        // scoreRow: [B, C]. ape: [ratio, C]. Add ape[apeRow] to every row
        // of scoreRow. Reuse broadcast_add_4d_2d by viewing scoreRow as
        // [B, 1, 1, C] and a one-row slice of ape as [1, C].
        let enc = cmd.makeComputeCommandEncoder()!
        enc.setComputePipelineState(pBroadcastAdd)
        enc.setBuffer(scoreRow.buffer, offset: scoreRow.offset, index: 0)
        let apeOff = ape.offset + apeRow * C * MemoryLayout<Float>.size
        enc.setBuffer(ape.buffer, offset: apeOff, index: 1)
        var dims = SIMD4<UInt32>(UInt32(B), 1, 1, UInt32(C))
        enc.setBytes(&dims, length: MemoryLayout.size(ofValue: dims), index: 2)
        enc.dispatchThreads(MTLSize(width: C, height: 1, depth: B),
                            threadsPerThreadgroup: MTLSize(width: 64, height: 1, depth: 1))
        enc.endEncoding()
    }

    private func blitRowToState(src: Tensor, state: Tensor, B: Int,
                                 stateRows: Int, rowIndex: Int,
                                 rowBytes: Int, in cmd: MTLCommandBuffer) {
        let blit = cmd.makeBlitCommandEncoder()!
        for b in 0..<B {
            let srcOff = src.offset + b * rowBytes
            let dstOff = state.offset + (b * stateRows + rowIndex) * rowBytes
            blit.copy(from: src.buffer, sourceOffset: srcOff,
                      to: state.buffer, destinationOffset: dstOff,
                      size: rowBytes)
        }
        blit.endEncoding()
    }

    private func overlapConcat(_ state: Tensor, B: Int, R: Int, D: Int,
                                in cmd: MTLCommandBuffer) -> Tensor {
        let out = Tensor.empty(shape: [B, 2 * R, D], dtype: .f32)
        let enc = cmd.makeComputeCommandEncoder()!
        enc.setComputePipelineState(pOverlapConcat)
        enc.setBuffer(state.buffer, offset: state.offset, index: 0)
        enc.setBuffer(out.buffer, offset: 0, index: 1)
        var dims = SIMD3<UInt32>(UInt32(B), UInt32(R), UInt32(D))
        enc.setBytes(&dims, length: MemoryLayout.size(ofValue: dims), index: 2)
        enc.dispatchThreads(MTLSize(width: D, height: 2 * R, depth: B),
                            threadsPerThreadgroup: MTLSize(width: 16, height: 4, depth: 4))
        enc.endEncoding()
        return out
    }

    private func stateShiftDown(state: Tensor, B: Int, R: Int, twoD: Int,
                                 in cmd: MTLCommandBuffer) {
        // state[:, :R] = state[:, R:]. Two-step: copy the high half to a
        // temp, then blit-copy the temp back to the low half.
        let tmp = Tensor.empty(shape: [B, R, twoD], dtype: .f32)
        let enc = cmd.makeComputeCommandEncoder()!
        enc.setComputePipelineState(pStateShift)
        enc.setBuffer(state.buffer, offset: state.offset, index: 0)
        enc.setBuffer(tmp.buffer, offset: 0, index: 1)
        var dims = SIMD3<UInt32>(UInt32(B), UInt32(R), UInt32(twoD))
        enc.setBytes(&dims, length: MemoryLayout.size(ofValue: dims), index: 2)
        enc.dispatchThreads(MTLSize(width: twoD, height: R, depth: B),
                            threadsPerThreadgroup: MTLSize(width: 16, height: 4, depth: 4))
        enc.endEncoding()
        // Now copy tmp back into state[:, :R, :]
        let blit = cmd.makeBlitCommandEncoder()!
        let bytesPerRow = twoD * MemoryLayout<Float>.size
        let stateRows = state.shape[1]      // == 2*R
        for b in 0..<B {
            for r in 0..<R {
                blit.copy(from: tmp.buffer, sourceOffset: (b * R + r) * bytesPerRow,
                          to: state.buffer,
                          destinationOffset: state.offset + (b * stateRows + r) * bytesPerRow,
                          size: bytesPerRow)
            }
        }
        blit.endEncoding()
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
