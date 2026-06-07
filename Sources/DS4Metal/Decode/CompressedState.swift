import Foundation
import Metal

// Stage 2 of the compressed-attention port (see docs/COMPRESSED-ATTENTION-PORT.md).
//
// Per-layer, persistent-across-tokens state for KV compression + the sparse
// indexer. On compressed layers (ratio != 0) the raw KV cache only keeps the most
// recent `nSWA` tokens; older tokens are pooled into `attnCompCache` rows (one row
// emitted every `ratio` tokens) via the recurrent `state*` buffers. ratio==4 layers
// additionally keep a parallel indexer lane (`index*`) used to pick the visible
// top-512 compressed rows.
//
// Mirrors the C engine's per-layer graph state: g->layer_attn_state_kv/score[il],
// g->layer_attn_comp_cache[il], g->layer_index_state_kv/score[il],
// g->layer_index_comp_cache[il], g->layer_n_comp[il], g->layer_n_index_comp[il].
//
// NOT YET BUILT/VERIFIED: this environment has no Swift/Metal/full-model, so none
// of this has been compiled or run. Build + verify on macOS against the C golden
// vectors (tests/test-vectors in antirez/ds4).

/// Persistent compression state for one compressed layer. `nil` for ratio==0.
public final class CompressedLayerState {
    public let ratio: Int
    public let coff: Int
    public let headDim: Int           // attention compressor head dim (512)
    public let compWidth: Int         // coff * headDim
    public let stateRows: Int         // coff * ratio
    public let compCap: Int           // max emitted rows (ctxSize/4 + 2)

    // Attention compressor recurrent state + emitted compressed-KV cache.
    public let stateKv: GPUTensor     // [stateRows * compWidth] F32
    public let stateScore: GPUTensor  // [stateRows * compWidth] F32
    public let attnCompCache: GPUTensor   // [compCap * headDim] F32 (the visible compressed rows)
    public var nComp: Int = 0         // emitted attention compressed rows so far

    // Per-token projection scratch (compWidth wide; index lane reuses via max width).
    public let compKvCur: GPUTensor   // [maxWidth] F32
    public let compScCur: GPUTensor   // [maxWidth] F32

    // Indexer lane (ratio == 4 only); nil otherwise.
    public let isIndexed: Bool
    public let indexWidth: Int        // coff * 128
    public let indexHeadDim: Int      // 128
    public let indexStateKv: GPUTensor?
    public let indexStateScore: GPUTensor?
    public let indexCompCache: GPUTensor?   // [compCap * 128] F32 (QAT'd indexer rows)
    public var nIndexComp: Int = 0

    // Indexer query/score scratch (ratio==4).
    public let indexerQ: GPUTensor?       // [nIndexerHead * 128] F32
    public let indexerWeights: GPUTensor? // [nIndexerHead] F32
    public let indexerScores: GPUTensor?  // [compCap] F32 (one score per indexer row)
    public let compSelected: GPUTensor?   // [nIndexerTopK] Int32 (selected row indices)

    public init(_ rt: MetalRuntime, ratio: Int, ctxSize: Int) throws {
        precondition(ratio != 0)
        self.ratio = ratio
        self.coff = DSV4Shape.coff(ratio: ratio)
        self.headDim = DSV4Shape.nHeadDim
        self.compWidth = DSV4Shape.attnCompWidth(ratio: ratio)
        self.stateRows = coff * ratio
        self.compCap = DSV4Shape.compCap(ctxSize: ctxSize)
        self.isIndexed = ratio == 4
        self.indexHeadDim = DSV4Shape.nIndexerHeadDim
        self.indexWidth = DSV4Shape.indexCompWidth(ratio: ratio)

        stateKv = try .zeros(rt, floatCount: stateRows * compWidth)
        stateScore = try .zeros(rt, floatCount: stateRows * compWidth)
        attnCompCache = try .zeros(rt, floatCount: compCap * headDim)
        let maxWidth = max(compWidth, indexWidth)
        compKvCur = try .zeros(rt, floatCount: maxWidth)
        compScCur = try .zeros(rt, floatCount: maxWidth)

        if isIndexed {
            indexStateKv = try .zeros(rt, floatCount: stateRows * indexWidth)
            indexStateScore = try .zeros(rt, floatCount: stateRows * indexWidth)
            indexCompCache = try .zeros(rt, floatCount: compCap * indexHeadDim)
            indexerQ = try .zeros(rt, floatCount: DSV4Shape.indexerQDim)
            indexerWeights = try .zeros(rt, floatCount: DSV4Shape.nIndexerHead)
            indexerScores = try .zeros(rt, floatCount: compCap)
            compSelected = try .zerosBytes(rt, byteLength: DSV4Shape.nIndexerTopK * 4)
        } else {
            indexStateKv = nil; indexStateScore = nil; indexCompCache = nil
            indexerQ = nil; indexerWeights = nil; indexerScores = nil; compSelected = nil
        }
    }

    /// Build one state per layer (nil on ratio==0 layers 0,1).
    public static func perLayer(_ rt: MetalRuntime, nLayers: Int, ctxSize: Int) throws -> [CompressedLayerState?] {
        try (0..<nLayers).map { il in
            let ratio = DSV4Shape.compressRatio(layer: il)
            return ratio == 0 ? nil : try CompressedLayerState(rt, ratio: ratio, ctxSize: ctxSize)
        }
    }
}
