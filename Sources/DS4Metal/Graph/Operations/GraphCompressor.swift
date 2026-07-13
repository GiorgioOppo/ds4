import Foundation
import Metal

// NSA attention-compressor integration (decode). Per compressed layer (ratio!=0),
// every token: project attn_norm -> kv_cur/sc_cur (F16 matvec), store into the
// recurrent state (+ APE); every `ratio` tokens emit one pooled+normed+roped+fp8
// compressed KV row into comp_cache. Attention then runs over raw SWA rows + ALL
// emitted compressed rows. Faithful to compressor_decode_one (ds4.c:8478) +
// ds4_gpu_compressor_update_tensor (ds4_metal.m:14689). All sub-ops were validated
// vs CPU (MetalCompressorTests); this wires them into the decode command buffer.

/// Persistent per-layer compressor state (lives across tokens for the whole
/// generation). Allocated only for compressed layers (ratio 4 or 128).
public final class CompressorState {
    public let ratio: Int
    public let headDim: Int          // 512 (attention compressor)
    public let width: Int            // coff*headDim (coff = ratio==4 ? 2 : 1)
    public let maxComp: Int          // capacity of comp_cache (rows)
    public let stateKv: GPUTensor    // [coff*ratio x width] f32
    public let stateScore: GPUTensor // [coff*ratio x width] f32, init -1e30
    public let cache: GPUTensor      // [maxComp x headDim] f32 (the emitted rows)
    public let kvCur: GPUTensor      // [width] projection scratch
    public let scCur: GPUTensor      // [width] projection scratch
    public let rowScratch: GPUTensor // [headDim] emitted-row scratch
    public let packedKv: GPUTensor   // [8 x headDim] ratio-4 pool gather scratch
    public let packedScore: GPUTensor
    // Unfused-pool scratch (decode emits pool ONE row): the window transposed to
    // [headDim x poolRows] plus the per-dim softmax — the same graph sequence
    // ds4_metal.m keeps for n_comp == 1 (the fused kernel reduces in a different
    // order; see compressorPoolEnc).
    let poolKvT: GPUTensor           // transposed kv, then the kv*softmax product
    let poolScoreT: GPUTensor        // transposed score
    let poolSoftmax: GPUTensor       // per-dim softmax of poolScoreT
    var poolRows: Int { ratio == 4 ? 8 : ratio }
    public var count: Int = 0        // n_comp emitted so far

    public init(_ rt: MetalRuntime, ratio: Int, headDim: Int, maxComp: Int) throws {
        self.ratio = ratio; self.headDim = headDim
        let coff = ratio == 4 ? 2 : 1
        self.width = coff * headDim
        self.maxComp = maxComp
        let rows = coff * ratio
        stateKv = try .zeros(rt, floatCount: rows * width)
        stateScore = try .floats(rt, [Float](repeating: -1e30, count: rows * width))
        let poolN = (ratio == 4 ? 8 : ratio) * headDim
        poolKvT = try .zeros(rt, floatCount: poolN)
        poolScoreT = try .zeros(rt, floatCount: poolN)
        poolSoftmax = try .zeros(rt, floatCount: poolN)
        // Sized to the full context (maxComp = maxKeys/ratio) but allocated
        // zero-fill-on-demand: only the rows actually emitted ([0..count], the only
        // region attention/indexer ever read) cost physical RAM. So a 1M-context
        // model no longer commits ~13 GB of compressor caches up front. reset()
        // doesn't touch cache (rows are overwritten as re-emitted), so skipping the
        // zero is safe — unwritten rows are never read.
        cache = try .lazyZeros(rt, floatCount: maxComp * headDim)
        kvCur = try .zeros(rt, floatCount: width)
        scCur = try .zeros(rt, floatCount: width)
        rowScratch = try .zeros(rt, floatCount: headDim)
        packedKv = try .zeros(rt, floatCount: 8 * headDim)
        packedScore = try .zeros(rt, floatCount: 8 * headDim)
    }

    /// Reset for a fresh sequence (pos 0): score=-1e30, count=0.
    public func reset(_ rt: MetalRuntime) throws {
        let coff = ratio == 4 ? 2 : 1
        let rows = coff * ratio
        try stateScore.fill(rt, value: -1e30, floatCount: rows * width)
        try stateKv.fill(rt, value: 0, floatCount: rows * width)
        count = 0
    }
}
