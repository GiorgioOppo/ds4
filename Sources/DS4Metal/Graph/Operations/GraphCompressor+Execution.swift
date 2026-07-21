import Foundation
import Metal

extension GraphContext {
    /// ratio-4 state shift (prev<-cur). kernel_dsv4_ratio4_shift_f32.
    func ratio4ShiftEnc(stateKv: GPUTensor, stateScore: GPUTensor, width: Int) throws {
        var w = UInt32(width)
        let pso = try rt.pipeline("kernel_dsv4_ratio4_shift_f32")
        let e = encoder
        e.setComputePipelineState(pso)
        e.setBytes(&w, length: 4, index: 0)
        e.setBuffer(stateKv.buffer, offset: stateKv.byteOffset, index: 1)
        e.setBuffer(stateScore.buffer, offset: stateScore.byteOffset, index: 2)
        e.dispatchThreadgroups(MTLSize(width: (4 * width + 255) / 256, height: 1, depth: 1),
                               threadsPerThreadgroup: MTLSize(width: 256, height: 1, depth: 1))
    }

    /// In-place E4M3FN fp8 quantization of one row (non-rope part). kernel_dsv4_fp8_kv_quantize_f32.
    func fp8QuantizeRowEnc(_ row: GPUTensor, headDim: Int, nRot: Int) throws {
        if nRot == headDim { return }
        let args = MetalRuntime.fp8KVQuantizeArgs(headDim: headDim, nTok: 1, nRot: nRot)
        let pso = try rt.pipeline("kernel_dsv4_fp8_kv_quantize_f32")
        let e = encoder
        e.setComputePipelineState(pso)
        args.withUnsafeBytes { e.setBytes($0.baseAddress!, length: args.count, index: 0) }
        e.setBuffer(row.buffer, offset: row.byteOffset, index: 1)
        e.setBuffer(row.buffer, offset: row.byteOffset, index: 2)
        e.setThreadgroupMemoryLength(64 * 4, index: 0)
        e.dispatchThreadgroups(MTLSize(width: 1, height: 1, depth: 1),
                               threadsPerThreadgroup: MTLSize(width: 64, height: 1, depth: 1))
    }

    /// How an emitted compressed row is finalized (after pool+norm+rope): the
    /// ATTENTION compressor fp8-rounds the non-rope part; the INDEXER compressor
    /// applies the 128-wide Hadamard + FP4 activation simulation (C:
    /// dsv4_indexer_qat_row_inplace).
    enum CompressorFinalize { case fp8, indexerQat }

    /// Run the compressor for one token on a compressed layer. `attnNorm` is the
    /// rms-normed pre-attention input (s.cur). Updates comp.state + on emit appends one
    /// row to comp.cache and bumps comp.count. Returns n_comp visible to THIS token's
    /// attention (= count after a possible emit).
    func runCompressor(attnNorm: GPUTensor, w: LayerWeights, comp: CompressorState,
                       rope: RopeParams, pos: Int, rmsEps: Float, nRot: Int) throws -> Int {
        guard let compKv = w.compKv, let compGate = w.compGate, let ape = w.compApe, let normW = w.compNorm else {
            return comp.count
        }
        return try runCompressor(x: attnNorm, kv: compKv, gate: compGate, ape: ape, normW: normW,
                                 comp: comp, rope: rope, pos: pos, rmsEps: rmsEps, nRot: nRot,
                                 finalize: .fp8, q8Weights: w.compQ8)
    }

    /// Generalized compressor step (shared by the attention and INDEXER
    /// compressors — same recurrence in the C `compressor_decode_one`, different
    /// weights/width/finalize).
    func runCompressor(x: GPUTensor, kv: GPUTensor, gate: GPUTensor, ape: GPUTensor, normW: GPUTensor,
                       comp: CompressorState, rope: RopeParams, pos: Int, rmsEps: Float, nRot: Int,
                       finalize: CompressorFinalize, q8Weights: Bool = false) throws -> Int {
        let h = comp.headDim, ratio = comp.ratio, width = comp.width
        // 1) project x -> kv_cur / sc_cur. Both F16 matrices have the same
        // shape and activation: the paired path shares activation reads and a
        // dispatch while retaining the two original dot-product orders.
        if q8Weights && GraphContext.fusedCompressorProj {
            try matmulQ8_0Pair(weightA: kv, weightB: gate, x: x,
                               outA: comp.kvCur, outB: comp.scCur,
                               inDim: x.count, outDim: width)
        } else if q8Weights {
            try matmulQ8_0(weight: kv, x: x, out: comp.kvCur, inDim: x.count, outDim: width)
            try matmulQ8_0(weight: gate, x: x, out: comp.scCur, inDim: x.count, outDim: width)
        } else if GraphContext.fusedCompressorProj {
            try matmulF16Pair(weightA: kv, weightB: gate, x: x,
                              outA: comp.kvCur, outB: comp.scCur,
                              inDim: x.count, outDim: width)
        } else {
            try matmulF16(weight: kv, x: x, out: comp.kvCur, inDim: x.count, outDim: width)
            try matmulF16(weight: gate, x: x, out: comp.scCur, inDim: x.count, outDim: width)
        }
        return try runCompressorTail(kvCur: comp.kvCur, scCur: comp.scCur, ape: ape, normW: normW,
                                     comp: comp, rope: rope, pos: pos, rmsEps: rmsEps,
                                     nRot: nRot, finalize: finalize)
    }

    /// The recurrent half of the compressor step — state store (+APE) and the
    /// possible comp-row emit — with the kv/score projections provided by the
    /// caller. The per-token path passes comp.kvCur/scCur (computed just
    /// above); the batched prefill precomputes the projections for the WHOLE
    /// run as two GEMMs and passes each token's row view. Dispatches and order
    /// are identical to the historical body.
    func runCompressorTail(kvCur: GPUTensor, scCur: GPUTensor, ape: GPUTensor, normW: GPUTensor,
                           comp: CompressorState, rope: RopeParams, pos: Int, rmsEps: Float,
                           nRot: Int, finalize: CompressorFinalize) throws -> Int {
        let h = comp.headDim, ratio = comp.ratio, width = comp.width
        // 2) store into recurrent state (+ APE, F16 -> ape_type 1).
        try compressorStoreOneEnc(kvCur: kvCur, scCur: scCur, ape: ape, apeType: 1,
                                  stateKv: comp.stateKv, stateScore: comp.stateScore, width: width, ratio: ratio, pos: pos)
        let emit = ((pos + 1) % ratio) == 0
        if !emit { return comp.count }
        // 3) emit: pool -> rmsNorm -> rope(comp_pos) -> finalize -> write cache[count].
        try compressorPoolEnc(comp, out: comp.rowScratch)
        try rmsNorm(comp.rowScratch, weight: normW, out: comp.rowScratch, rows: 1, n: h, eps: rmsEps)
        let compPos = pos + 1 - ratio
        try ropeTail(x: comp.rowScratch, nTok: 1, nHead: 1, headDim: h, nRot: nRot, nCtxOrig: rope.nCtxOrig,
                     freqBase: rope.freqBase, freqScale: rope.freqScale, extFactor: rope.extFactor,
                     attnFactor: rope.attnFactor, betaFast: rope.betaFast, betaSlow: rope.betaSlow, pos0: compPos, posStep: 1)
        switch finalize {
        case .fp8:        try fp8QuantizeRowEnc(comp.rowScratch, headDim: h, nRot: nRot)
        case .indexerQat: try indexerHadamardFp4Enc(comp.rowScratch, rows: 1, rowStrideBytes: h * 4)
        }
        // copy rowScratch -> cache[count]
        let dstRow = comp.cache.rowView(row: comp.count, cols: h)
        try gatherRowsEnc(src: comp.rowScratch, srcByteOffset: comp.rowScratch.byteOffset, srcRowStride: h,
                          dst: dstRow, dstByteOffset: dstRow.byteOffset, rows: 1, cols: h)
        if ratio == 4 { try ratio4ShiftEnc(stateKv: comp.stateKv, stateScore: comp.stateScore, width: width) }
        comp.count += 1
        return comp.count
    }
}
