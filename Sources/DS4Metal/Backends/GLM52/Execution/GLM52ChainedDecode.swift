import DS4Core
import Foundation
import Metal

// Chained decode — the GLM GraphContext analog. The per-layer functions in
// GLM52DecodeGraph read activations back to the host between stages; here
// the hidden state LIVES on the GPU across the whole layer and the whole
// token, every activation buffer comes from a persistent scratch reused
// across layers and tokens, and the host is tapped only where a decision is
// genuinely CPU-side: the top-k readback on scoring full-indexer layers and
// the 24 KiB ffnIn readback that feeds each sparse layer's F32 router. A
// dense layer is ONE command buffer end to end; a sparse fill-range layer
// is two. Numerically identical to the per-stage path by construction: the
// same kernels encode in the same order on the same buffers.

/// Persistent activation buffers for the chained decode, sized once from the
/// geometry. One instance per engine; never read on the host except through
/// the explicit taps.
public final class GLM52DecodeScratch {
    let hidden: MTLBuffer
    let normed: MTLBuffer
    let attnOut: MTLBuffer
    let ffnIn: MTLBuffer
    let ffnOut: MTLBuffer
    let contribution: MTLBuffer
    let qRank: MTLBuffer
    let qRankNorm: MTLBuffer
    let kvRaw: MTLBuffer
    let cacheReady: MTLBuffer
    let query: MTLBuffer
    let qLow: MTLBuffer
    let attnLora: MTLBuffer
    let heads: MTLBuffer
    let indexerRaw: MTLBuffer
    let indexerQuery: MTLBuffer
    let indexerWeights: MTLBuffer
    let scores: MTLBuffer
    let mid: MTLBuffer
    let geometry: GLM52DecodeGeometry

    public init(runtime: MetalRuntime,
                geometry: GLM52DecodeGeometry,
                scoreCapacity: Int) throws {
        self.geometry = geometry
        let layer = geometry.layer
        func make(_ floats: Int) throws -> MTLBuffer {
            guard let buffer = runtime.device.makeBuffer(
                length: max(floats, 1) * MemoryLayout<Float>.stride,
                options: .storageModeShared) else {
                throw MetalError.bufferAlloc
            }
            return buffer
        }
        hidden = try make(layer.embeddingWidth)
        normed = try make(layer.embeddingWidth)
        attnOut = try make(layer.embeddingWidth)
        ffnIn = try make(layer.embeddingWidth)
        ffnOut = try make(layer.embeddingWidth)
        contribution = try make(layer.embeddingWidth)
        qRank = try make(geometry.qLoraRank)
        qRankNorm = try make(geometry.qLoraRank)
        kvRaw = try make(layer.kvRawWidth)
        cacheReady = try make(layer.kvRawWidth)
        query = try make(geometry.queryWidth)
        qLow = try make(layer.headCount * layer.kvLoraRank)
        attnLora = try make(layer.headCount * layer.kvLoraRank)
        heads = try make(layer.headCount * layer.valueDimension)
        indexerRaw = try make(geometry.indexerHeadDimension)
        indexerQuery = try make(geometry.indexerQueryWidth)
        indexerWeights = try make(geometry.indexerHeadCount)
        scores = try make(scoreCapacity)
        mid = try make(max(layer.denseHiddenWidth, layer.expertHiddenWidth))
    }

    /// Load the embedded token into the resident hidden state.
    func loadHidden(_ values: [Float]) {
        precondition(values.count * MemoryLayout<Float>.stride
                         <= hidden.length)
        values.withUnsafeBytes {
            hidden.contents().copyMemory(
                from: $0.baseAddress!, byteCount: $0.count)
        }
    }

    func readHidden(count: Int) -> [Float] {
        let pointer = hidden.contents().bindMemory(
            to: Float.self, capacity: count)
        return Array(UnsafeBufferPointer(start: pointer, count: count))
    }
}

extension MetalRuntime {
    func glm52EncodeMatvecF32(into commandBuffer: MTLCommandBuffer,
                              rows: MTLBuffer, input: MTLBuffer,
                              output: MTLBuffer, rowCount: Int,
                              inputWidth: Int) throws {
        let width = 64
        try glm52GraphEncode(
            into: commandBuffer, pipelineName: "kernel_glm52_matvec_f32",
            arguments: [UInt32(rowCount), UInt32(inputWidth), 0, 0],
            buffers: [rows, input, output],
            threadgroups: MTLSize(width: (rowCount + width - 1) / width,
                                  height: 1, depth: 1),
            threadsPerThreadgroup: MTLSize(width: width, height: 1, depth: 1))
    }

    /// Validation wrapper for the F32 matvec kernel.
    public func glm52MatvecF32(rows: [Float], input: [Float],
                               rowCount: Int) throws -> [Float] {
        guard rowCount > 0, !input.isEmpty,
              rows.count == rowCount * input.count else {
            throw MetalError.unsupported(
                "GLM 5.2 F32 matvec expects [\(rowCount)]x\(input.count) rows")
        }
        let rowsBuffer = try glm52GraphBuffer(rows)
        let inputBuffer = try glm52GraphBuffer(input)
        let output = try glm52GraphOutputBuffer(floats: rowCount)
        guard let commandBuffer = queue.makeCommandBuffer() else {
            throw MetalError.bufferAlloc
        }
        try glm52EncodeMatvecF32(into: commandBuffer, rows: rowsBuffer,
                                 input: inputBuffer, output: output,
                                 rowCount: rowCount,
                                 inputWidth: input.count)
        try glm52GraphCommit(commandBuffer)
        return glm52GraphReadback(output, count: rowCount)
    }

    /// One full decode layer with the hidden state resident in
    /// `scratch.hidden` (in/out). Encodes attention, residual, FFN norm and
    /// — on dense layers — the whole FFN into as few command buffers as the
    /// host taps allow. Returns the routing (sparse) and the selection.
    public func glm52ChainedDecodeLayer(
        weights: GLM52ResidentDecodeWeights,
        ffn: GLM52ResidentFFN,
        caches: GLM52ResidentDecodeCaches,
        scratch: GLM52DecodeScratch,
        reusedSelection: [UInt32]?,
        position: Int,
        activeExperts: Int? = nil) throws
        -> (routing: GLM52RouterOutput?, selection: [UInt32]) {
        let g = weights.geometry
        let layer = g.layer
        let headsWidth = layer.headCount * layer.valueDimension
        let visible = position + 1
        guard position == caches.rows, visible <= caches.capacity,
              visible <= scratch.scores.length / MemoryLayout<Float>.stride
        else {
            throw MetalError.unsupported(
                "GLM 5.2 chained decode position \(position) does not match "
                + "\(caches.rows) live rows / capacity \(caches.capacity)")
        }
        if weights.indexer != nil {
            guard reusedSelection == nil, caches.indexerKeys != nil else {
                throw MetalError.unsupported(
                    "full-indexer chained layer needs indexer caches and no "
                    + "reused selection")
            }
        } else {
            guard reusedSelection != nil, caches.indexerKeys == nil else {
                throw MetalError.unsupported(
                    "IndexShare chained layer requires the preceding "
                    + "full-indexer selection and owns no indexer keys")
            }
        }

        guard var commandBuffer = queue.makeCommandBuffer() else {
            throw MetalError.bufferAlloc
        }

        // Shared trunk (all on scratch buffers, hidden untouched so far).
        try glm52EncodeRMSNorm(into: commandBuffer, input: scratch.hidden,
                               weight: weights.attnNorm,
                               output: scratch.normed,
                               width: layer.embeddingWidth)
        try glm52EncodeMatvecQ8(into: commandBuffer, input: scratch.normed,
                                weights: weights.qA, output: scratch.qRank,
                                rowCount: g.qLoraRank,
                                inputWidth: layer.embeddingWidth,
                                weightType: weights.types.qA)
        try glm52EncodeMatvecQ8(into: commandBuffer, input: scratch.normed,
                                weights: weights.kvA, output: scratch.kvRaw,
                                rowCount: layer.kvRawWidth,
                                inputWidth: layer.embeddingWidth,
                                weightType: weights.types.kvA)
        try glm52EncodeRMSNorm(into: commandBuffer, input: scratch.qRank,
                               weight: weights.qANorm,
                               output: scratch.qRankNorm,
                               width: g.qLoraRank)
        try glm52GraphEncode(
            into: commandBuffer,
            pipelineName: "kernel_glm52_kv_lora_norm_cache_ready_f32",
            arguments: [1, 0, 0, 0],
            buffers: [scratch.kvRaw, weights.kvANorm, scratch.cacheReady],
            threadgroups: MTLSize(width: 1, height: 1, depth: 1),
            threadsPerThreadgroup: MTLSize(width: 128, height: 1, depth: 1),
            threadgroupMemoryLength: 128 * MemoryLayout<Float>.stride)
        try glm52GraphEncode(
            into: commandBuffer,
            pipelineName: "kernel_glm52_store_compact_row_f16",
            arguments: [UInt32(position), 1, UInt32(caches.capacity), 0],
            buffers: [scratch.cacheReady, caches.compact],
            threadgroups: MTLSize(width: 1, height: 1, depth: 1),
            threadsPerThreadgroup: MTLSize(width: 64, height: 1, depth: 1))
        try glm52EncodeMatvecQ8(into: commandBuffer,
                                input: scratch.qRankNorm,
                                weights: weights.qB, output: scratch.query,
                                rowCount: g.queryWidth,
                                inputWidth: g.qLoraRank,
                                weightType: weights.types.qB)
        try glm52EncodeRope(into: commandBuffer,
                            pipelineName: "kernel_glm52_rope_tail_f32",
                            values: scratch.query,
                            headCount: layer.headCount,
                            headDimension: g.qkDimension,
                            rotationDimension: layer.ropeDimension,
                            position: position)

        var selection: [UInt32]
        if let indexer = weights.indexer, let keyCache = caches.indexerKeys {
            try glm52EncodeMatvecQ8(into: commandBuffer,
                                    input: scratch.hidden,
                                    weights: indexer.key,
                                    output: scratch.indexerRaw,
                                    rowCount: g.indexerHeadDimension,
                                    inputWidth: layer.embeddingWidth,
                                    weightType: weights.types.indexerKey)
            try glm52GraphEncode(
                into: commandBuffer,
                pipelineName: "kernel_glm52_store_indexer_k_f16",
                arguments: [UInt32(position), 1, UInt32(caches.capacity), 0],
                buffers: [scratch.indexerRaw, indexer.keyNorm,
                          indexer.keyNormBias, keyCache],
                threadgroups: MTLSize(width: 1, height: 1, depth: 1),
                threadsPerThreadgroup: MTLSize(width: 32, height: 1, depth: 1),
                threadgroupMemoryLength: 32 * MemoryLayout<Float>.stride)
            if visible <= g.indexerTopK {
                selection = (0..<visible).map(UInt32.init)
            } else {
                // Score on GPU (indexer proj included), then the single
                // host tap of the branch: the top-k over the score row.
                try glm52EncodeMatvecQ8(
                    into: commandBuffer,
                    input: scratch.qRankNorm,
                    weights: indexer.queryB,
                    output: scratch.indexerQuery,
                    rowCount: g.indexerQueryWidth,
                    inputWidth: g.qLoraRank,
                    weightType: weights.types.indexerQueryB)
                try glm52EncodeRope(
                    into: commandBuffer,
                    pipelineName: "kernel_glm52_rope_prefix_f32",
                    values: scratch.indexerQuery,
                    headCount: g.indexerHeadCount,
                    headDimension: g.indexerHeadDimension,
                    rotationDimension: g.indexerRotationDimension,
                    position: position)
                let projBuffer = try glm52GraphBuffer(indexer.proj)
                try glm52EncodeMatvecF32(into: commandBuffer,
                                         rows: projBuffer,
                                         input: scratch.hidden,
                                         output: scratch.indexerWeights,
                                         rowCount: g.indexerHeadCount,
                                         inputWidth: layer.embeddingWidth)
                try glm52GraphEncode(
                    into: commandBuffer,
                    pipelineName: "kernel_glm52_indexer_scores_f16",
                    arguments: [UInt32(visible), 1, UInt32(position),
                                g.indexerScale.bitPattern],
                    buffers: [scratch.indexerQuery, scratch.indexerWeights,
                              keyCache, scratch.scores],
                    threadgroups: MTLSize(width: visible, height: 1,
                                          depth: 1),
                    threadsPerThreadgroup: MTLSize(width: 32, height: 4,
                                                   depth: 1),
                    threadgroupMemoryLength:
                        (128 + 4) * MemoryLayout<Float>.stride)
                try glm52GraphCommit(commandBuffer)
                let hostScores = glm52GraphReadback(scratch.scores,
                                                    count: visible)
                selection = try glm52IndexerTopK(
                    scores: hostScores, rowCount: visible, tokenCount: 1,
                    topK: g.indexerTopK)
                guard let fresh = queue.makeCommandBuffer() else {
                    throw MetalError.bufferAlloc
                }
                commandBuffer = fresh
            }
        } else {
            selection = reusedSelection ?? []
        }
        guard !selection.isEmpty, selection.count <= visible,
              Set(selection).count == selection.count,
              selection.allSatisfy({ Int($0) < visible }) else {
            throw MetalError.unsupported(
                "GLM 5.2 chained decode selection must be unique rows "
                + "inside 0..<\(visible)")
        }
        guard let selectionBuffer = device.makeBuffer(
            bytes: selection,
            length: selection.count * MemoryLayout<UInt32>.stride,
            options: .storageModeShared) else {
            throw MetalError.bufferAlloc
        }

        // Attention tail, residual and FFN norm — hidden updated in place.
        let attentionGeometry = GLM52AttentionGeometry.v5_2
        try glm52GraphEncode(
            into: commandBuffer,
            pipelineName: "kernel_glm52_qk_lowrank_q8_0",
            arguments: [0, 0, 0, 0],
            buffers: [scratch.query, weights.keyB, scratch.qLow],
            threadgroups: MTLSize(width: layer.kvLoraRank / 4,
                                  height: layer.headCount, depth: 1),
            threadsPerThreadgroup: MTLSize(width: 32, height: 4, depth: 1))
        try glm52GraphEncode(
            into: commandBuffer,
            pipelineName: "kernel_glm52_attention_indexed_f16",
            arguments: [UInt32(visible), UInt32(selection.count),
                        attentionGeometry.scale.bitPattern, 1],
            buffers: [scratch.qLow, scratch.query, caches.compact,
                      selectionBuffer, scratch.attnLora],
            threadgroups: MTLSize(width: layer.headCount, height: 1,
                                  depth: 1),
            threadsPerThreadgroup: MTLSize(width: 32, height: 4, depth: 1),
            threadgroupMemoryLength:
                (selection.count + 5) * MemoryLayout<Float>.stride)
        try glm52GraphEncode(
            into: commandBuffer,
            pipelineName: "kernel_glm52_value_project_q8_0",
            arguments: [0, 0, 0, 0],
            buffers: [scratch.attnLora, weights.valueB, scratch.heads],
            threadgroups: MTLSize(width: layer.valueDimension / 4,
                                  height: layer.headCount, depth: 1),
            threadsPerThreadgroup: MTLSize(width: 32, height: 4, depth: 1))
        try glm52EncodeMatvecQ8(into: commandBuffer, input: scratch.heads,
                                weights: weights.attnOutput,
                                output: scratch.attnOut,
                                rowCount: layer.embeddingWidth,
                                inputWidth: headsWidth,
                                weightType: weights.types.attnOutput)
        try glm52EncodeAdd(into: commandBuffer, a: scratch.hidden,
                           b: scratch.attnOut, output: scratch.hidden,
                           count: layer.embeddingWidth)
        try glm52EncodeRMSNorm(into: commandBuffer, input: scratch.hidden,
                               weight: ffn.ffnNorm, output: scratch.ffnIn,
                               width: layer.embeddingWidth)

        var routing: GLM52RouterOutput?
        switch ffn.kind {
        case .dense(let gate, let up, let down):
            // No host tap: the dense FFN rides the same command buffer.
            try glm52EncodePairSwiGLU(
                into: commandBuffer, input: scratch.ffnIn, gate: gate,
                up: up, mid: scratch.mid,
                hiddenWidth: layer.denseHiddenWidth,
                inputWidth: layer.embeddingWidth, routeWeight: 1)
            try glm52EncodeMatvecQ8(into: commandBuffer, input: scratch.mid,
                                    weights: down, output: scratch.ffnOut,
                                    rowCount: layer.embeddingWidth,
                                    inputWidth: layer.denseHiddenWidth)
            try glm52EncodeAdd(into: commandBuffer, a: scratch.hidden,
                               b: scratch.ffnOut, output: scratch.hidden,
                               count: layer.embeddingWidth)
            try glm52GraphCommit(commandBuffer)

        case .sparse(let routerRows, let routerBias, let sharedGate,
                     let sharedUp, let sharedDown, let expertProvider):
            try glm52GraphCommit(commandBuffer)
            // The one genuine host tap of a sparse layer: the F32 router.
            let ffnInHost = glm52GraphReadback(
                scratch.ffnIn, count: layer.embeddingWidth)
            let logits = try GLM52FFNCPUReference.matvec(
                rows: routerRows, input: ffnInHost,
                rowCount: GLM52RouterReference.expertCount)
            let routed = try glm52Route(logits: logits, bias: routerBias)
            routing = routed

            guard let ffnBuffer = queue.makeCommandBuffer() else {
                throw MetalError.bufferAlloc
            }
            let hidden = layer.expertHiddenWidth
            try glm52EncodePairSwiGLU(
                into: ffnBuffer, input: scratch.ffnIn, gate: sharedGate,
                up: sharedUp, mid: scratch.mid, hiddenWidth: hidden,
                inputWidth: layer.embeddingWidth, routeWeight: 1,
                weightType: ffn.sharedWeightTypes.gateUp)
            try glm52EncodeMatvecQ8(into: ffnBuffer, input: scratch.mid,
                                    weights: sharedDown,
                                    output: scratch.contribution,
                                    rowCount: layer.embeddingWidth,
                                    inputWidth: hidden,
                                    weightType: ffn.sharedWeightTypes.down)
            try glm52EncodeAdd(into: ffnBuffer, a: scratch.hidden,
                               b: scratch.contribution,
                               output: scratch.hidden,
                               count: layer.embeddingWidth)
            let usedExperts = min(routed.selected.count,
                                  max(1, activeExperts
                                          ?? routed.selected.count))
            if let stage = ffn.stagedSelection {
                // Zero-copy path: the WHOLE selection lands in one reusable
                // staging buffer with ONE concurrent read burst, and each
                // record is bound by offset — no per-expert allocations or
                // copies. Byte-identical to the provider path by
                // construction (same packed record layout).
                let ids = (0..<usedExperts).map {
                    UInt32(bitPattern: routed.selected[$0])
                }
                let staged = try stage(ids)
                for rank in 0..<usedExperts {
                    let base = staged.recordOffsets[rank]
                    try glm52EncodePairSwiGLU(
                        into: ffnBuffer, input: scratch.ffnIn,
                        gate: staged.buffer, up: staged.buffer,
                        mid: scratch.mid, hiddenWidth: hidden,
                        inputWidth: layer.embeddingWidth,
                        routeWeight: routed.weights[rank],
                        weightType: staged.gateUpType,
                        gateOffset: base,
                        upOffset: base + staged.gateBytes)
                    try glm52EncodeMatvecQ8(
                        into: ffnBuffer, input: scratch.mid,
                        weights: staged.buffer,
                        output: scratch.contribution,
                        rowCount: layer.embeddingWidth,
                        inputWidth: hidden,
                        weightType: staged.downType,
                        weightsOffset: base + staged.gateBytes
                            + staged.upBytes)
                    try glm52EncodeAdd(into: ffnBuffer, a: scratch.hidden,
                                       b: scratch.contribution,
                                       output: scratch.hidden,
                                       count: layer.embeddingWidth)
                }
            } else {
                for rank in 0..<usedExperts {
                    let record = try expertProvider(
                        UInt32(bitPattern: routed.selected[rank]))
                    let gate = try glm52GraphBuffer(record.gate)
                    let up = try glm52GraphBuffer(record.up)
                    let down = try glm52GraphBuffer(record.down)
                    try glm52EncodePairSwiGLU(
                        into: ffnBuffer, input: scratch.ffnIn, gate: gate,
                        up: up, mid: scratch.mid, hiddenWidth: hidden,
                        inputWidth: layer.embeddingWidth,
                        routeWeight: routed.weights[rank],
                        weightType: record.gateUpType)
                    try glm52EncodeMatvecQ8(into: ffnBuffer,
                                            input: scratch.mid,
                                            weights: down,
                                            output: scratch.contribution,
                                            rowCount: layer.embeddingWidth,
                                            inputWidth: hidden,
                                            weightType: record.downType)
                    try glm52EncodeAdd(into: ffnBuffer, a: scratch.hidden,
                                       b: scratch.contribution,
                                       output: scratch.hidden,
                                       count: layer.embeddingWidth)
                }
            }
            try glm52GraphCommit(ffnBuffer)
        }

        caches.appendedRow()
        return (routing, selection)
    }
}
