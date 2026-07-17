import DS4Core
import Foundation
import Metal

// Validation wrappers for the compact DSA attention core kernels. Each stage
// dispatches ONE kernel and reads the result back; nothing here is a decode
// loop. The staged split (absorb → indexed softmax accumulate → value project)
// mirrors GLM52AttentionCPUReference.absorbed, whose output is the oracle the
// chained dispatch is compared against in tests.

private struct GLM52CompactAttentionPlan {
    static let geometry = GLM52AttentionGeometry.v5_2
    /// Architecture cap: the indexer never selects more than top-2048 rows,
    /// and the kernel keeps the score vector in threadgroup memory.
    static let maxSelectedRows = 2_048

    let rowCount: Int
    let selection: [UInt32]

    var geometry: GLM52AttentionGeometry { Self.geometry }

    init(query: [Float],
         cacheBits: [UInt16],
         selection: [UInt32]) throws {
        let g = Self.geometry
        guard query.count == g.headCount * g.qkDimension else {
            throw MetalError.unsupported(
                "GLM 5.2 attention query must be [\(g.headCount)][\(g.qkDimension)]")
        }
        guard !cacheBits.isEmpty,
              cacheBits.count.isMultiple(of: g.cacheRowWidth) else {
            throw MetalError.unsupported(
                "GLM 5.2 attention cache must contain \(g.cacheRowWidth)-wide F16 rows")
        }
        let rowCount = cacheBits.count / g.cacheRowWidth
        guard rowCount <= Int(UInt32.max) else {
            throw MetalError.unsupported("GLM 5.2 attention cache row count overflows")
        }
        guard !selection.isEmpty, selection.count <= Self.maxSelectedRows else {
            throw MetalError.unsupported(
                "GLM 5.2 attention selection must hold 1...\(Self.maxSelectedRows) rows")
        }
        var seen = Set<UInt32>()
        seen.reserveCapacity(selection.count)
        for row in selection {
            guard Int(row) < rowCount else {
                throw MetalError.unsupported(
                    "GLM 5.2 attention selected row \(row) is outside 0..<\(rowCount)")
            }
            guard seen.insert(row).inserted else {
                throw MetalError.unsupported(
                    "GLM 5.2 attention selection repeats row \(row)")
            }
        }
        self.rowCount = rowCount
        self.selection = selection
    }

    var arguments: [UInt32] {
        [UInt32(rowCount), UInt32(selection.count),
         Self.geometry.scale.bitPattern, 0]
    }
}

extension MetalRuntime {
    private func glm52AttentionBuffer(_ values: [Float]) throws -> MTLBuffer {
        guard let buffer = device.makeBuffer(
            bytes: values,
            length: values.count * MemoryLayout<Float>.stride,
            options: .storageModeShared) else {
            throw MetalError.bufferAlloc
        }
        return buffer
    }

    private func glm52AttentionDispatch(
        pipelineName: String,
        arguments: [UInt32],
        buffers: [MTLBuffer],
        threadgroups: MTLSize,
        threadsPerThreadgroup: MTLSize,
        threadgroupMemoryLength: Int? = nil
    ) throws {
        let pipeline = try pipeline(pipelineName)
        guard pipeline.threadExecutionWidth == 32,
              pipeline.maxTotalThreadsPerThreadgroup >= 128 else {
            throw MetalError.unsupported(
                "GLM 5.2 attention kernels require four 32-lane SIMD groups")
        }
        guard let commandBuffer = queue.makeCommandBuffer(),
              let encoder = commandBuffer.makeComputeCommandEncoder() else {
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
        encoder.dispatchThreadgroups(threadgroups,
                                     threadsPerThreadgroup: threadsPerThreadgroup)
        encoder.endEncoding()
        commandBuffer.commit()
        commandBuffer.waitUntilCompleted()
        if let error = commandBuffer.error { throw error }
    }

    private func glm52AttentionReadback(_ buffer: MTLBuffer,
                                        count: Int) -> [Float] {
        let pointer = buffer.contents().bindMemory(to: Float.self, capacity: count)
        return Array(UnsafeBufferPointer(start: pointer, count: count))
    }

    /// Stage 1 — absorb the per-head nope query into attn_k_b:
    /// returns `q_low` as `[64][512]`.
    public func glm52QKLowRank(query: [Float], keyB: [Float]) throws -> [Float] {
        let g = GLM52AttentionGeometry.v5_2
        guard query.count == g.headCount * g.qkDimension,
              keyB.count == g.headCount * g.kvLoraRank * g.nopeDimension else {
            throw MetalError.unsupported(
                "GLM 5.2 qk_lowrank expects v5_2 query/key_b geometry")
        }
        let outputCount = g.headCount * g.kvLoraRank
        let queryBuffer = try glm52AttentionBuffer(query)
        let keyBBuffer = try glm52AttentionBuffer(keyB)
        guard let outputBuffer = device.makeBuffer(
            length: outputCount * MemoryLayout<Float>.stride,
            options: .storageModeShared) else {
            throw MetalError.bufferAlloc
        }
        try glm52AttentionDispatch(
            pipelineName: "kernel_glm52_qk_lowrank_f32",
            arguments: [0, 0, 0, 0],
            buffers: [queryBuffer, keyBBuffer, outputBuffer],
            threadgroups: MTLSize(width: g.kvLoraRank / 4,
                                  height: g.headCount, depth: 1),
            threadsPerThreadgroup: MTLSize(width: 32, height: 4, depth: 1))
        return glm52AttentionReadback(outputBuffer, count: outputCount)
    }

    /// Stage 2 — softmax attention over the selected compact-cache rows,
    /// accumulated in the KV-LoRA domain: returns `attn_lora` as `[64][512]`,
    /// already divided by the clamped softmax denominator.
    public func glm52AttentionIndexed(qLow: [Float],
                                      query: [Float],
                                      cacheBits: [UInt16],
                                      selection: [UInt32]) throws -> [Float] {
        let g = GLM52AttentionGeometry.v5_2
        guard qLow.count == g.headCount * g.kvLoraRank else {
            throw MetalError.unsupported(
                "GLM 5.2 attention q_low must be [\(g.headCount)][\(g.kvLoraRank)]")
        }
        let plan = try GLM52CompactAttentionPlan(
            query: query, cacheBits: cacheBits, selection: selection)
        let outputCount = g.headCount * g.kvLoraRank
        let qLowBuffer = try glm52AttentionBuffer(qLow)
        let queryBuffer = try glm52AttentionBuffer(query)
        guard let cacheBuffer = device.makeBuffer(
                  bytes: cacheBits,
                  length: cacheBits.count * MemoryLayout<UInt16>.stride,
                  options: .storageModeShared),
              let selectionBuffer = device.makeBuffer(
                  bytes: plan.selection,
                  length: plan.selection.count * MemoryLayout<UInt32>.stride,
                  options: .storageModeShared),
              let outputBuffer = device.makeBuffer(
                  length: outputCount * MemoryLayout<Float>.stride,
                  options: .storageModeShared) else {
            throw MetalError.bufferAlloc
        }
        try glm52AttentionDispatch(
            pipelineName: "kernel_glm52_attention_indexed_f16",
            arguments: plan.arguments,
            buffers: [qLowBuffer, queryBuffer, cacheBuffer,
                      selectionBuffer, outputBuffer],
            threadgroups: MTLSize(width: g.headCount, height: 1, depth: 1),
            threadsPerThreadgroup: MTLSize(width: 32, height: 4, depth: 1),
            threadgroupMemoryLength:
                (plan.selection.count + 5) * MemoryLayout<Float>.stride)
        return glm52AttentionReadback(outputBuffer, count: outputCount)
    }

    /// Stage 3 — project the accumulated KV-LoRA rows through attn_v_b:
    /// returns per-head attention output as `[64][256]`.
    public func glm52ValueProject(attnLora: [Float],
                                  valueB: [Float]) throws -> [Float] {
        let g = GLM52AttentionGeometry.v5_2
        guard attnLora.count == g.headCount * g.kvLoraRank,
              valueB.count == g.headCount * g.valueDimension * g.kvLoraRank else {
            throw MetalError.unsupported(
                "GLM 5.2 value projection expects v5_2 attn_lora/value_b geometry")
        }
        let outputCount = g.headCount * g.valueDimension
        let loraBuffer = try glm52AttentionBuffer(attnLora)
        let valueBBuffer = try glm52AttentionBuffer(valueB)
        guard let outputBuffer = device.makeBuffer(
            length: outputCount * MemoryLayout<Float>.stride,
            options: .storageModeShared) else {
            throw MetalError.bufferAlloc
        }
        try glm52AttentionDispatch(
            pipelineName: "kernel_glm52_value_project_f32",
            arguments: [0, 0, 0, 0],
            buffers: [loraBuffer, valueBBuffer, outputBuffer],
            threadgroups: MTLSize(width: g.valueDimension / 4,
                                  height: g.headCount, depth: 1),
            threadsPerThreadgroup: MTLSize(width: 32, height: 4, depth: 1))
        return glm52AttentionReadback(outputBuffer, count: outputCount)
    }

    /// Chained validation path: the three stages dispatched in sequence.
    /// Numerically comparable to `GLM52AttentionCPUReference` within float
    /// tolerance; not a decode loop and not a performance path.
    public func glm52CompactAttention(query: [Float],
                                      keyB: [Float],
                                      valueB: [Float],
                                      cacheBits: [UInt16],
                                      selection: [UInt32]) throws -> [Float] {
        _ = try GLM52CompactAttentionPlan(
            query: query, cacheBits: cacheBits, selection: selection)
        let qLow = try glm52QKLowRank(query: query, keyB: keyB)
        let attnLora = try glm52AttentionIndexed(
            qLow: qLow, query: query, cacheBits: cacheBits, selection: selection)
        return try glm52ValueProject(attnLora: attnLora, valueB: valueB)
    }

    // MARK: - Q8_0 weight variants

    /// GGUF Q8_0 row bytes for `elements` values: consecutive 34-byte blocks
    /// of 32 (2-byte f16 scale + 32 signed int8).
    static func glm52Q8RowBytes(_ elements: Int) -> Int {
        precondition(elements.isMultiple(of: 32))
        return (elements / 32) * 34
    }

    private func glm52AttentionByteBuffer(_ bytes: [UInt8]) throws -> MTLBuffer {
        guard let buffer = device.makeBuffer(
            bytes: bytes, length: bytes.count, options: .storageModeShared) else {
            throw MetalError.bufferAlloc
        }
        return buffer
    }

    /// Stage 1 on quantized weights — `attn_k_b` exactly as stored in the
    /// GGUF: Q8_0 rows of 192 (204 bytes) indexed `[head][kvLora]`.
    public func glm52QKLowRankQ8(query: [Float], keyBQ8: [UInt8]) throws -> [Float] {
        let g = GLM52AttentionGeometry.v5_2
        let rowBytes = Self.glm52Q8RowBytes(g.nopeDimension)
        guard query.count == g.headCount * g.qkDimension,
              keyBQ8.count == g.headCount * g.kvLoraRank * rowBytes else {
            throw MetalError.unsupported(
                "GLM 5.2 qk_lowrank Q8_0 expects v5_2 query/key_b geometry")
        }
        let outputCount = g.headCount * g.kvLoraRank
        let queryBuffer = try glm52AttentionBuffer(query)
        let keyBBuffer = try glm52AttentionByteBuffer(keyBQ8)
        guard let outputBuffer = device.makeBuffer(
            length: outputCount * MemoryLayout<Float>.stride,
            options: .storageModeShared) else {
            throw MetalError.bufferAlloc
        }
        try glm52AttentionDispatch(
            pipelineName: "kernel_glm52_qk_lowrank_q8_0",
            arguments: [0, 0, 0, 0],
            buffers: [queryBuffer, keyBBuffer, outputBuffer],
            threadgroups: MTLSize(width: g.kvLoraRank / 4,
                                  height: g.headCount, depth: 1),
            threadsPerThreadgroup: MTLSize(width: 32, height: 4, depth: 1))
        return glm52AttentionReadback(outputBuffer, count: outputCount)
    }

    /// Stage 3 on quantized weights — `attn_v_b` exactly as stored in the
    /// GGUF: Q8_0 rows of 512 (544 bytes) indexed `[head][value]`.
    public func glm52ValueProjectQ8(attnLora: [Float],
                                    valueBQ8: [UInt8]) throws -> [Float] {
        let g = GLM52AttentionGeometry.v5_2
        let rowBytes = Self.glm52Q8RowBytes(g.kvLoraRank)
        guard attnLora.count == g.headCount * g.kvLoraRank,
              valueBQ8.count == g.headCount * g.valueDimension * rowBytes else {
            throw MetalError.unsupported(
                "GLM 5.2 value projection Q8_0 expects v5_2 attn_lora/value_b geometry")
        }
        let outputCount = g.headCount * g.valueDimension
        let loraBuffer = try glm52AttentionBuffer(attnLora)
        let valueBBuffer = try glm52AttentionByteBuffer(valueBQ8)
        guard let outputBuffer = device.makeBuffer(
            length: outputCount * MemoryLayout<Float>.stride,
            options: .storageModeShared) else {
            throw MetalError.bufferAlloc
        }
        try glm52AttentionDispatch(
            pipelineName: "kernel_glm52_value_project_q8_0",
            arguments: [0, 0, 0, 0],
            buffers: [loraBuffer, valueBBuffer, outputBuffer],
            threadgroups: MTLSize(width: g.valueDimension / 4,
                                  height: g.headCount, depth: 1),
            threadsPerThreadgroup: MTLSize(width: 32, height: 4, depth: 1))
        return glm52AttentionReadback(outputBuffer, count: outputCount)
    }

    /// Chained validation path on quantized projections: Q8_0 absorb, F16
    /// softmax accumulation, Q8_0 value projection. Comparable to the F32
    /// oracle run on the DEQUANTIZED weights within float tolerance.
    public func glm52CompactAttentionQ8(query: [Float],
                                        keyBQ8: [UInt8],
                                        valueBQ8: [UInt8],
                                        cacheBits: [UInt16],
                                        selection: [UInt32]) throws -> [Float] {
        _ = try GLM52CompactAttentionPlan(
            query: query, cacheBits: cacheBits, selection: selection)
        let qLow = try glm52QKLowRankQ8(query: query, keyBQ8: keyBQ8)
        let attnLora = try glm52AttentionIndexed(
            qLow: qLow, query: query, cacheBits: cacheBits, selection: selection)
        return try glm52ValueProjectQ8(attnLora: attnLora, valueBQ8: valueBQ8)
    }
}
