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
    /// Logical session ceiling for emitted rows. This is deliberately separate
    /// from `cacheCapacity`: a 1M-token session must not expose a multi-GB
    /// `MTLBuffer` to Metal while only a handful of rows are live.
    public let maxComp: Int
    public let stateKv: GPUTensor    // [coff*ratio x width] f32
    public let stateScore: GPUTensor // [coff*ratio x width] f32, init -1e30
    public private(set) var cache: GPUTensor // [cacheCapacity x headDim] f32
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

    /// Currently allocated rows in the contiguous Metal cache.
    public var cacheCapacity: Int { cache.count / headDim }

    public init(_ rt: MetalRuntime, ratio: Int, headDim: Int, maxComp: Int,
                initialComp: Int? = nil) throws {
        self.ratio = ratio; self.headDim = headDim
        let coff = ratio == 4 ? 2 : 1
        self.width = coff * headDim
        self.maxComp = max(1, maxComp)
        let rows = coff * ratio
        stateKv = try .zeros(rt, floatCount: rows * width)
        stateScore = try .floats(rt, [Float](repeating: -1e30, count: rows * width))
        let poolN = (ratio == 4 ? 8 : ratio) * headDim
        poolKvT = try .zeros(rt, floatCount: poolN)
        poolScoreT = try .zeros(rt, floatCount: poolN)
        poolSoftmax = try .zeros(rt, floatCount: poolN)
        // Metal accounts residency/mapping against the RESOURCE LENGTH, not only
        // the pages touched through `contents()`. A full-capacity lazy buffer still
        // made a 1M-token decoder expose ~14 GB of resources on every attention
        // command and slowed an empty-context decode 7x. Start from a short,
        // contiguous buffer and grow geometrically with the live high-water mark.
        let requestedInitial = initialComp ?? self.maxComp
        let allocatedRows = min(self.maxComp, max(1, requestedInitial))
        cache = try .lazyZeros(rt, floatCount: allocatedRows * headDim)
        kvCur = try .zeros(rt, floatCount: width)
        scCur = try .zeros(rt, floatCount: width)
        rowScratch = try .zeros(rt, floatCount: headDim)
        packedKv = try .zeros(rt, floatCount: 8 * headDim)
        packedScore = try .zeros(rt, floatCount: 8 * headDim)
    }

    /// Pure geometric policy, exposed internally for CPU-only boundary tests.
    static func grownCacheCapacity(current: Int, required: Int,
                                   maximum: Int) -> Int {
        let cap = max(1, maximum)
        let target = min(max(1, required), cap)
        let have = min(max(1, current), cap)
        guard target > have else { return have }
        let doubled = have > cap / 2 ? cap : have * 2
        return min(cap, max(target, doubled))
    }

    /// Grow the physical Metal resource while preserving only emitted rows.
    /// The caller must have joined every command buffer that can reference the
    /// old cache before entering this method.
    @discardableResult
    func ensureCacheCapacity(_ rt: MetalRuntime, requiredRows: Int) throws -> Bool {
        precondition(requiredRows >= 0 && requiredRows <= maxComp,
                     "compressor cache: \(requiredRows) rows outside 0...\(maxComp)")
        guard requiredRows > cacheCapacity else { return false }
        let next = Self.grownCacheCapacity(current: cacheCapacity,
                                           required: requiredRows,
                                           maximum: maxComp)
        let replacement = try GPUTensor.lazyZeros(rt, floatCount: next * headDim)
        let liveFloats = count * headDim
        precondition(liveFloats <= cache.count,
                     "compressor cache: live rows exceed physical capacity")
        if liveFloats > 0 {
            memcpy(replacement.buffer.contents().advanced(by: replacement.byteOffset),
                   cache.buffer.contents().advanced(by: cache.byteOffset),
                   liveFloats * MemoryLayout<Float>.stride)
        }
        cache = replacement
        return true
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
