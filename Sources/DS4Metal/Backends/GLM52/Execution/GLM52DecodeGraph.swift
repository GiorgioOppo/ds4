import DS4Core
import Foundation
import Metal

// Resident decode graph — the persistent form of glm52DecodeAttention. The
// per-dispatch executor re-uploads every weight array on every call; here the
// quantized weights are uploaded ONCE into MTLBuffers, the compact and
// indexer caches live on the GPU and are appended in place, and the whole
// attention step is encoded on chained buffers: in the fill-range path the
// entire step is ONE command buffer; the top-k path splits only around the
// score readback that feeds the host-orchestrated multi-block top-k. The
// CPU keeps exactly what the per-dispatch executor kept host-side minus the
// norms (now kernel_glm52_rms_norm_f32): the residual adds, the router and
// the 32-row F32 indexer-proj matvec. Correctness anchor: parity with
// glm52DecodeAttention, which is itself judged by GLM52DecodeCPUReference.

/// One layer's decode weights resident on the GPU. Upload happens once at
/// construction; the buffers are shared-storage so fixtures stay comparable.
public final class GLM52ResidentDecodeWeights {
    public let geometry: GLM52DecodeGeometry
    let attnNorm: MTLBuffer
    let qA: MTLBuffer
    let qANorm: MTLBuffer
    let qB: MTLBuffer
    let kvA: MTLBuffer
    let kvANorm: MTLBuffer
    let keyB: MTLBuffer
    let valueB: MTLBuffer
    let attnOutput: MTLBuffer

    struct ResidentIndexer {
        let key: MTLBuffer
        let keyNorm: MTLBuffer
        let keyNormBias: MTLBuffer
        let queryB: MTLBuffer
        /// The 32-row F32 matvec stays on CPU beside the router.
        let proj: [Float]
    }
    let indexer: ResidentIndexer?
    public var isFullIndexer: Bool { indexer != nil }

    public init(runtime: MetalRuntime,
                geometry: GLM52DecodeGeometry,
                attention: GLM52QuantizedDecodeAttention,
                indexer quantizedIndexer: GLM52QuantizedDecodeIndexer?) throws {
        try runtime.glm52ValidateDecodeWeights(
            geometry: geometry, attention: attention,
            indexer: quantizedIndexer)
        self.geometry = geometry
        attnNorm = try runtime.glm52GraphBuffer(attention.attnNorm)
        qA = try runtime.glm52GraphBuffer(attention.qA)
        qANorm = try runtime.glm52GraphBuffer(attention.qANorm)
        qB = try runtime.glm52GraphBuffer(attention.qB)
        kvA = try runtime.glm52GraphBuffer(attention.kvA)
        kvANorm = try runtime.glm52GraphBuffer(attention.kvANorm)
        keyB = try runtime.glm52GraphBuffer(attention.keyB)
        valueB = try runtime.glm52GraphBuffer(attention.valueB)
        attnOutput = try runtime.glm52GraphBuffer(attention.attnOutput)
        if let quantizedIndexer {
            indexer = ResidentIndexer(
                key: try runtime.glm52GraphBuffer(quantizedIndexer.key),
                keyNorm: try runtime.glm52GraphBuffer(quantizedIndexer.keyNorm),
                keyNormBias: try runtime.glm52GraphBuffer(
                    quantizedIndexer.keyNormBias),
                queryB: try runtime.glm52GraphBuffer(quantizedIndexer.queryB),
                proj: quantizedIndexer.proj)
        } else {
            indexer = nil
        }
    }
}

/// One layer's decode caches resident on the GPU: interleaved compact rows
/// (`[capacity][576]` F16) and — on full-indexer layers — the indexer key
/// plane (`[capacity][128]` F16), appended in place by the graph.
public final class GLM52ResidentDecodeCaches {
    public let capacity: Int
    public private(set) var rows: Int = 0
    let compact: MTLBuffer
    let indexerKeys: MTLBuffer?
    private let geometry: GLM52DecodeGeometry

    public init(runtime: MetalRuntime,
                geometry: GLM52DecodeGeometry,
                capacity: Int,
                fullIndexer: Bool) throws {
        guard capacity > 0, capacity <= Int(UInt32.max) else {
            throw MetalError.unsupported(
                "GLM 5.2 resident cache capacity \(capacity) is invalid")
        }
        self.capacity = capacity
        self.geometry = geometry
        let rowWidth = geometry.layer.kvLoraRank + geometry.layer.ropeDimension
        guard let compactBuffer = runtime.device.makeBuffer(
            length: capacity * rowWidth * MemoryLayout<UInt16>.stride,
            options: .storageModeShared) else {
            throw MetalError.bufferAlloc
        }
        compact = compactBuffer
        if fullIndexer {
            guard let keyBuffer = runtime.device.makeBuffer(
                length: capacity * geometry.indexerHeadDimension
                    * MemoryLayout<UInt16>.stride,
                options: .storageModeShared) else {
                throw MetalError.bufferAlloc
            }
            indexerKeys = keyBuffer
        } else {
            indexerKeys = nil
        }
    }

    func appendedRow() { rows += 1 }

    /// The live compact rows as F16 bits — for tests and checkpoints.
    public func compactSnapshot() -> [UInt16] {
        let count = rows * (geometry.layer.kvLoraRank
                                + geometry.layer.ropeDimension)
        let pointer = compact.contents().bindMemory(
            to: UInt16.self, capacity: count)
        return Array(UnsafeBufferPointer(start: pointer, count: count))
    }

    /// The live indexer key rows as F16 bits (empty on IndexShare layers).
    public func indexerKeySnapshot() -> [UInt16] {
        guard let indexerKeys else { return [] }
        let count = rows * geometry.indexerHeadDimension
        let pointer = indexerKeys.contents().bindMemory(
            to: UInt16.self, capacity: count)
        return Array(UnsafeBufferPointer(start: pointer, count: count))
    }
}

extension MetalRuntime {
    // MARK: - Buffer helpers

    func glm52GraphBuffer(_ values: [Float]) throws -> MTLBuffer {
        guard let buffer = device.makeBuffer(
            bytes: values,
            length: values.count * MemoryLayout<Float>.stride,
            options: .storageModeShared) else {
            throw MetalError.bufferAlloc
        }
        return buffer
    }

    func glm52GraphBuffer(_ bytes: [UInt8]) throws -> MTLBuffer {
        guard let buffer = device.makeBuffer(
            bytes: bytes, length: bytes.count,
            options: .storageModeShared) else {
            throw MetalError.bufferAlloc
        }
        return buffer
    }

    private func glm52GraphOutputBuffer(floats count: Int) throws -> MTLBuffer {
        guard let buffer = device.makeBuffer(
            length: count * MemoryLayout<Float>.stride,
            options: .storageModeShared) else {
            throw MetalError.bufferAlloc
        }
        return buffer
    }

    private func glm52GraphReadback(_ buffer: MTLBuffer,
                                    count: Int) -> [Float] {
        let pointer = buffer.contents().bindMemory(
            to: Float.self, capacity: count)
        return Array(UnsafeBufferPointer(start: pointer, count: count))
    }

    /// Encode one kernel dispatch into an open command buffer. Sequential
    /// encoders in one command buffer execute in order with automatic hazard
    /// tracking on the shared buffers.
    private func glm52GraphEncode(
        into commandBuffer: MTLCommandBuffer,
        pipelineName: String,
        arguments: [UInt32],
        buffers: [MTLBuffer],
        threadgroups: MTLSize,
        threadsPerThreadgroup: MTLSize,
        threadgroupMemoryLength: Int? = nil) throws {
        let pipeline = try pipeline(pipelineName)
        guard let encoder = commandBuffer.makeComputeCommandEncoder() else {
            throw MetalError.bufferAlloc
        }
        encoder.setComputePipelineState(pipeline)
        arguments.withUnsafeBytes {
            encoder.setBytes($0.baseAddress!, length: $0.count, index: 0)
        }
        for (index, buffer) in buffers.enumerated() {
            encoder.setBuffer(buffer, offset: 0, index: index + 1)
        }
        if let length = threadgroupMemoryLength {
            encoder.setThreadgroupMemoryLength(length, index: 0)
        }
        encoder.dispatchThreadgroups(
            threadgroups, threadsPerThreadgroup: threadsPerThreadgroup)
        encoder.endEncoding()
    }

    private func glm52GraphCommit(_ commandBuffer: MTLCommandBuffer) throws {
        commandBuffer.commit()
        commandBuffer.waitUntilCompleted()
        if let error = commandBuffer.error { throw error }
    }

    // MARK: - Encoded stages

    private func glm52EncodeRMSNorm(into commandBuffer: MTLCommandBuffer,
                                    input: MTLBuffer, weight: MTLBuffer,
                                    output: MTLBuffer, width: Int,
                                    epsilon: Float = 1e-5) throws {
        try glm52GraphEncode(
            into: commandBuffer, pipelineName: "kernel_glm52_rms_norm_f32",
            arguments: [UInt32(width), epsilon.bitPattern, 0, 0],
            buffers: [input, weight, output],
            threadgroups: MTLSize(width: 1, height: 1, depth: 1),
            threadsPerThreadgroup: MTLSize(width: 256, height: 1, depth: 1),
            threadgroupMemoryLength: 256 * MemoryLayout<Float>.stride)
    }

    private func glm52EncodeMatvecQ8(into commandBuffer: MTLCommandBuffer,
                                     input: MTLBuffer, weights: MTLBuffer,
                                     output: MTLBuffer, rowCount: Int,
                                     inputWidth: Int) throws {
        let width = 256
        try glm52GraphEncode(
            into: commandBuffer, pipelineName: "kernel_glm52_moe_down",
            arguments: [GLM52TensorSchema.q8_0, UInt32(rowCount),
                        UInt32(inputWidth), Float(1).bitPattern],
            buffers: [input, weights, output],
            threadgroups: MTLSize(width: (rowCount + width - 1) / width,
                                  height: 1, depth: 1),
            threadsPerThreadgroup: MTLSize(width: width, height: 1, depth: 1))
    }

    private func glm52EncodeRope(into commandBuffer: MTLCommandBuffer,
                                 pipelineName: String, values: MTLBuffer,
                                 headCount: Int, headDimension: Int,
                                 rotationDimension: Int,
                                 position: Int) throws {
        let pairs = headCount * (rotationDimension / 2)
        let width = 256
        try glm52GraphEncode(
            into: commandBuffer, pipelineName: pipelineName,
            arguments: [UInt32(headCount), UInt32(headDimension),
                        UInt32(rotationDimension), UInt32(position)],
            buffers: [values],
            threadgroups: MTLSize(width: (pairs + width - 1) / width,
                                  height: 1, depth: 1),
            threadsPerThreadgroup: MTLSize(width: width, height: 1, depth: 1))
    }

    // MARK: - Resident decode attention

    /// One decode attention step on resident weights and caches. Encoded on
    /// chained GPU buffers: the fill-range path is a single command buffer;
    /// the top-k path splits around the score readback feeding the
    /// host-orchestrated multi-block top-k. Appends this token's cache rows
    /// in place (before selection and attention, upstream's order). On error
    /// the caches are unspecified.
    public func glm52ResidentDecodeAttention(
        weights: GLM52ResidentDecodeWeights,
        caches: GLM52ResidentDecodeCaches,
        input: [Float],
        reusedSelection: [UInt32]?,
        position: Int) throws -> (output: [Float], selection: [UInt32]) {
        let g = weights.geometry
        let layer = g.layer
        let headsWidth = layer.headCount * layer.valueDimension
        let visible = position + 1

        guard input.count == layer.embeddingWidth else {
            throw MetalError.unsupported(
                "GLM 5.2 resident decode input must be "
                + "\(layer.embeddingWidth) wide")
        }
        guard position == caches.rows, visible <= caches.capacity else {
            throw MetalError.unsupported(
                "GLM 5.2 resident decode position \(position) does not "
                + "match \(caches.rows) live rows / capacity "
                + "\(caches.capacity)")
        }
        if weights.indexer != nil {
            guard reusedSelection == nil, caches.indexerKeys != nil else {
                throw MetalError.unsupported(
                    "full-indexer resident layer needs indexer caches and "
                    + "no reused selection")
            }
        } else {
            guard reusedSelection != nil, caches.indexerKeys == nil else {
                throw MetalError.unsupported(
                    "IndexShare resident layer requires the preceding "
                    + "full-indexer selection and owns no indexer keys")
            }
        }

        // Activation buffers for this step.
        let x = try glm52GraphBuffer(input)
        let normed = try glm52GraphOutputBuffer(floats: layer.embeddingWidth)
        let qRank = try glm52GraphOutputBuffer(floats: g.qLoraRank)
        let qRankNorm = try glm52GraphOutputBuffer(floats: g.qLoraRank)
        let kvRaw = try glm52GraphOutputBuffer(floats: layer.kvRawWidth)
        let cacheReady = try glm52GraphOutputBuffer(floats: layer.kvRawWidth)
        let query = try glm52GraphOutputBuffer(floats: g.queryWidth)
        let qLow = try glm52GraphOutputBuffer(
            floats: layer.headCount * layer.kvLoraRank)
        let attnLora = try glm52GraphOutputBuffer(
            floats: layer.headCount * layer.kvLoraRank)
        let heads = try glm52GraphOutputBuffer(floats: headsWidth)
        let output = try glm52GraphOutputBuffer(floats: layer.embeddingWidth)

        guard let stepBuffer = queue.makeCommandBuffer() else {
            throw MetalError.bufferAlloc
        }

        // Shared trunk: norms, LoRA projections, cache stores, query RoPE.
        try glm52EncodeRMSNorm(into: stepBuffer, input: x,
                               weight: weights.attnNorm, output: normed,
                               width: layer.embeddingWidth)
        try glm52EncodeMatvecQ8(into: stepBuffer, input: normed,
                                weights: weights.qA, output: qRank,
                                rowCount: g.qLoraRank,
                                inputWidth: layer.embeddingWidth)
        try glm52EncodeMatvecQ8(into: stepBuffer, input: normed,
                                weights: weights.kvA, output: kvRaw,
                                rowCount: layer.kvRawWidth,
                                inputWidth: layer.embeddingWidth)
        try glm52EncodeRMSNorm(into: stepBuffer, input: qRank,
                               weight: weights.qANorm, output: qRankNorm,
                               width: g.qLoraRank)
        try glm52GraphEncode(
            into: stepBuffer,
            pipelineName: "kernel_glm52_kv_lora_norm_cache_ready_f32",
            arguments: [1, 0, 0, 0],
            buffers: [kvRaw, weights.kvANorm, cacheReady],
            threadgroups: MTLSize(width: 1, height: 1, depth: 1),
            threadsPerThreadgroup: MTLSize(width: 128, height: 1, depth: 1),
            threadgroupMemoryLength: 128 * MemoryLayout<Float>.stride)
        try glm52GraphEncode(
            into: stepBuffer,
            pipelineName: "kernel_glm52_store_compact_row_f16",
            arguments: [UInt32(position), 1, UInt32(caches.capacity), 0],
            buffers: [cacheReady, caches.compact],
            threadgroups: MTLSize(width: 1, height: 1, depth: 1),
            threadsPerThreadgroup: MTLSize(width: 64, height: 1, depth: 1))
        try glm52EncodeMatvecQ8(into: stepBuffer, input: qRankNorm,
                                weights: weights.qB, output: query,
                                rowCount: g.queryWidth,
                                inputWidth: g.qLoraRank)
        try glm52EncodeRope(into: stepBuffer,
                            pipelineName: "kernel_glm52_rope_tail_f32",
                            values: query, headCount: layer.headCount,
                            headDimension: g.qkDimension,
                            rotationDimension: layer.ropeDimension,
                            position: position)

        // Indexer store, then the selection: encoded score path only when
        // the visible range exceeds top-k.
        var selection: [UInt32]
        var scores: MTLBuffer?
        if let indexer = weights.indexer, let keyCache = caches.indexerKeys {
            let idxRaw = try glm52GraphOutputBuffer(
                floats: g.indexerHeadDimension)
            try glm52EncodeMatvecQ8(into: stepBuffer, input: x,
                                    weights: indexer.key, output: idxRaw,
                                    rowCount: g.indexerHeadDimension,
                                    inputWidth: layer.embeddingWidth)
            try glm52GraphEncode(
                into: stepBuffer,
                pipelineName: "kernel_glm52_store_indexer_k_f16",
                arguments: [UInt32(position), 1, UInt32(caches.capacity), 0],
                buffers: [idxRaw, indexer.keyNorm, indexer.keyNormBias,
                          keyCache],
                threadgroups: MTLSize(width: 1, height: 1, depth: 1),
                threadsPerThreadgroup: MTLSize(width: 32, height: 1, depth: 1),
                threadgroupMemoryLength: 32 * MemoryLayout<Float>.stride)
            if visible <= g.indexerTopK {
                selection = (0..<visible).map(UInt32.init)
            } else {
                let indexerQuery = try glm52GraphOutputBuffer(
                    floats: g.indexerQueryWidth)
                try glm52EncodeMatvecQ8(into: stepBuffer, input: qRankNorm,
                                        weights: indexer.queryB,
                                        output: indexerQuery,
                                        rowCount: g.indexerQueryWidth,
                                        inputWidth: g.qLoraRank)
                try glm52EncodeRope(
                    into: stepBuffer,
                    pipelineName: "kernel_glm52_rope_prefix_f32",
                    values: indexerQuery, headCount: g.indexerHeadCount,
                    headDimension: g.indexerHeadDimension,
                    rotationDimension: g.indexerRotationDimension,
                    position: position)
                var headWeights = [Float](
                    repeating: 0, count: g.indexerHeadCount)
                for head in 0..<g.indexerHeadCount {
                    var dot: Float = 0
                    let base = head * layer.embeddingWidth
                    for i in 0..<layer.embeddingWidth {
                        dot += indexer.proj[base + i] * input[i]
                    }
                    headWeights[head] = dot
                }
                let scoreBuffer = try glm52GraphOutputBuffer(floats: visible)
                try glm52GraphEncode(
                    into: stepBuffer,
                    pipelineName: "kernel_glm52_indexer_scores_f16",
                    arguments: [UInt32(visible), 1, UInt32(position),
                                g.indexerScale.bitPattern],
                    buffers: [indexerQuery,
                              try glm52GraphBuffer(headWeights),
                              keyCache, scoreBuffer],
                    threadgroups: MTLSize(width: visible, height: 1, depth: 1),
                    threadsPerThreadgroup: MTLSize(width: 32, height: 4,
                                                   depth: 1),
                    threadgroupMemoryLength:
                        (128 + 4) * MemoryLayout<Float>.stride)
                scores = scoreBuffer
                selection = []
            }
        } else {
            // Validated above: IndexShare layers carry the reused selection.
            selection = reusedSelection ?? []
        }

        // The top-k path must materialize the scores before selecting; the
        // fill-range path continues in the same command buffer.
        if let scores {
            try glm52GraphCommit(stepBuffer)
            let hostScores = glm52GraphReadback(scores, count: visible)
            selection = try glm52IndexerTopK(
                scores: hostScores, rowCount: visible, tokenCount: 1,
                topK: g.indexerTopK)
        }
        guard !selection.isEmpty,
              selection.count <= visible,
              Set(selection).count == selection.count,
              selection.allSatisfy({ Int($0) < visible }) else {
            throw MetalError.unsupported(
                "GLM 5.2 resident decode selection must be unique rows "
                + "inside 0..<\(visible)")
        }
        guard let selectionBuffer = device.makeBuffer(
            bytes: selection,
            length: selection.count * MemoryLayout<UInt32>.stride,
            options: .storageModeShared) else {
            throw MetalError.bufferAlloc
        }

        let attentionBuffer: MTLCommandBuffer
        if scores != nil {
            guard let fresh = queue.makeCommandBuffer() else {
                throw MetalError.bufferAlloc
            }
            attentionBuffer = fresh
        } else {
            attentionBuffer = stepBuffer
        }

        // Attention tail: absorb, indexed softmax with per-row tail
        // rotation over the resident cache, value projection, output.
        let attentionGeometry = GLM52AttentionGeometry.v5_2
        try glm52GraphEncode(
            into: attentionBuffer,
            pipelineName: "kernel_glm52_qk_lowrank_q8_0",
            arguments: [0, 0, 0, 0],
            buffers: [query, weights.keyB, qLow],
            threadgroups: MTLSize(width: layer.kvLoraRank / 4,
                                  height: layer.headCount, depth: 1),
            threadsPerThreadgroup: MTLSize(width: 32, height: 4, depth: 1))
        try glm52GraphEncode(
            into: attentionBuffer,
            pipelineName: "kernel_glm52_attention_indexed_f16",
            arguments: [UInt32(visible), UInt32(selection.count),
                        attentionGeometry.scale.bitPattern, 1],
            buffers: [qLow, query, caches.compact, selectionBuffer, attnLora],
            threadgroups: MTLSize(width: layer.headCount, height: 1, depth: 1),
            threadsPerThreadgroup: MTLSize(width: 32, height: 4, depth: 1),
            threadgroupMemoryLength:
                (selection.count + 5) * MemoryLayout<Float>.stride)
        try glm52GraphEncode(
            into: attentionBuffer,
            pipelineName: "kernel_glm52_value_project_q8_0",
            arguments: [0, 0, 0, 0],
            buffers: [attnLora, weights.valueB, heads],
            threadgroups: MTLSize(width: layer.valueDimension / 4,
                                  height: layer.headCount, depth: 1),
            threadsPerThreadgroup: MTLSize(width: 32, height: 4, depth: 1))
        try glm52EncodeMatvecQ8(into: attentionBuffer, input: heads,
                                weights: weights.attnOutput, output: output,
                                rowCount: layer.embeddingWidth,
                                inputWidth: headsWidth)
        try glm52GraphCommit(attentionBuffer)

        caches.appendedRow()
        return (glm52GraphReadback(output, count: layer.embeddingWidth),
                selection)
    }

    /// One full decode layer on resident attention state: attention
    /// residual, then the shared residual FFN stage (host glue — FFN
    /// residency arrives with the expert-streaming integration).
    public func glm52ResidentDecodeLayer(
        weights: GLM52ResidentDecodeWeights,
        caches: GLM52ResidentDecodeCaches,
        input: [Float],
        reusedSelection: [UInt32]?,
        ffnNorm: [Float],
        ffn: GLM52QuantizedLayerFFN,
        position: Int) throws
        -> (output: [Float], routing: GLM52RouterOutput?,
            selection: [UInt32]) {
        let attn = try glm52ResidentDecodeAttention(
            weights: weights, caches: caches, input: input,
            reusedSelection: reusedSelection, position: position)
        let afterAttn = (0..<input.count).map { input[$0] + attn.output[$0] }
        let ffnResult = try glm52LayerFFNStage(
            geometry: weights.geometry.layer, afterAttention: afterAttn,
            ffnNorm: ffnNorm, ffn: ffn)
        return (ffnResult.output, ffnResult.routing, attn.selection)
    }

    /// Validation wrapper for the generic-width RMSNorm kernel.
    public func glm52RMSNorm(values: [Float], weight: [Float],
                             epsilon: Float = 1e-5) throws -> [Float] {
        guard !values.isEmpty, values.count == weight.count,
              values.count <= Int(UInt32.max), epsilon > 0 else {
            throw MetalError.unsupported(
                "GLM 5.2 RMSNorm expects matching non-empty values/weight")
        }
        let input = try glm52GraphBuffer(values)
        let weights = try glm52GraphBuffer(weight)
        let output = try glm52GraphOutputBuffer(floats: values.count)
        guard let commandBuffer = queue.makeCommandBuffer() else {
            throw MetalError.bufferAlloc
        }
        try glm52EncodeRMSNorm(into: commandBuffer, input: input,
                               weight: weights, output: output,
                               width: values.count, epsilon: epsilon)
        try glm52GraphCommit(commandBuffer)
        return glm52GraphReadback(output, count: values.count)
    }
}
