import Foundation
import Metal
import DS4Core

// Swift wrappers over the Laguna decode kernels in `metal/laguna/` (one file
// per area: laguna_quant/rope/kv/attention, GLM style).
// Validation-grade API in the GLM style: [Float] in, [Float] out, one command
// buffer per call, judged against the CPU oracles in `Reference/`. The future
// resident graph will encode the same pipelines without the per-call sync.

/// F16 ring KV cache held in shared Metal buffers, matching the layout of the
/// `key_cache`/`value_cache` arguments: `capacity` rows of
/// `kvHeadCount * headDim` half floats.
public struct LagunaMetalKVCache {
    public let capacity: Int
    public let rowWidth: Int
    public let keys: MTLBuffer
    public let values: MTLBuffer

    public var rowBytes: Int { rowWidth * MemoryLayout<UInt16>.stride }

    /// F16 bit patterns of one cache buffer, for test assertions.
    public func keyBits() -> [UInt16] {
        let pointer = keys.contents().bindMemory(to: UInt16.self,
                                                 capacity: capacity * rowWidth)
        return Array(UnsafeBufferPointer(start: pointer,
                                         count: capacity * rowWidth))
    }
}

extension MetalRuntime {
    // MARK: Buffers

    public func lagunaKVCache(capacity: Int, rowWidth: Int) throws -> LagunaMetalKVCache {
        guard capacity > 0, rowWidth > 0 else {
            throw MetalError.unsupported("Laguna KV cache needs positive dimensions")
        }
        let bytes = capacity * rowWidth * MemoryLayout<UInt16>.stride
        guard let keys = device.makeBuffer(length: bytes, options: .storageModeShared),
              let values = device.makeBuffer(length: bytes, options: .storageModeShared) else {
            throw MetalError.bufferAlloc
        }
        memset(keys.contents(), 0, bytes)
        memset(values.contents(), 0, bytes)
        return LagunaMetalKVCache(capacity: capacity, rowWidth: rowWidth,
                                  keys: keys, values: values)
    }

    // MARK: Per-head RMSNorm + RoPE

    /// Dispatch `kernel_laguna_qk_head_rms_norm_rope_neox` for one token:
    /// per-head RMS norm of Q and K with their shared per-dimension weights,
    /// then NeoX rotation on the head prefix with the spec's YaRN parameters.
    public func lagunaQKHeadRMSNormRope(
        query: [Float], key: [Float],
        queryWeight: [Float], keyWeight: [Float],
        spec: LagunaAttentionSpec, position: Int
    ) throws -> (query: [Float], key: [Float]) {
        guard query.count == spec.queryWidth,
              key.count == spec.keyValueWidth,
              queryWeight.count == spec.headDim,
              keyWeight.count == spec.headDim,
              spec.rotationDims <= spec.headDim,
              spec.rotationDims % 2 == 0,
              position >= 0 else {
            throw MetalError.unsupported("Laguna norm/rope: inconsistent shapes")
        }

        guard let queryBuffer = device.makeBuffer(
                  bytes: query,
                  length: query.count * MemoryLayout<Float>.stride,
                  options: .storageModeShared
              ),
              let keyBuffer = device.makeBuffer(
                  bytes: key,
                  length: key.count * MemoryLayout<Float>.stride,
                  options: .storageModeShared
              ),
              let queryWeightBuffer = device.makeBuffer(
                  bytes: queryWeight,
                  length: queryWeight.count * MemoryLayout<Float>.stride,
                  options: .storageModeShared
              ),
              let keyWeightBuffer = device.makeBuffer(
                  bytes: keyWeight,
                  length: keyWeight.count * MemoryLayout<Float>.stride,
                  options: .storageModeShared
              ) else {
            throw MetalError.bufferAlloc
        }

        // ds4_metal_args_laguna_norm_rope: six uint32, seven float, one pad.
        let combinedHeads = spec.headCount + spec.kvHeadCount
        var arguments = [UInt32](repeating: 0, count: 14)
        arguments[0] = 1 // n_tokens
        arguments[1] = UInt32(combinedHeads)
        arguments[2] = UInt32(spec.headDim)
        arguments[3] = UInt32(spec.rotationDims)
        arguments[4] = UInt32(position)
        arguments[5] = UInt32(spec.ropeOriginalContext)
        arguments[6] = spec.rmsEpsilon.bitPattern
        arguments[7] = spec.ropeFrequencyBase.bitPattern
        arguments[8] = spec.ropeFrequencyScale.bitPattern
        arguments[9] = spec.extrapolationFactor.bitPattern
        arguments[10] = spec.attentionFactor.bitPattern
        arguments[11] = spec.betaFast.bitPattern
        arguments[12] = spec.betaSlow.bitPattern

        let pipeline = try pipeline("kernel_laguna_qk_head_rms_norm_rope_neox")
        // The in-threadgroup reduction assumes a power-of-two thread count.
        let threadCount = min(128, pipeline.maxTotalThreadsPerThreadgroup)
        guard threadCount >= spec.rotationDims / 2,
              threadCount & (threadCount - 1) == 0 else {
            throw MetalError.unsupported("Laguna norm/rope thread count \(threadCount)")
        }
        guard let commandBuffer = queue.makeCommandBuffer(),
              let encoder = commandBuffer.makeComputeCommandEncoder() else {
            throw MetalError.bufferAlloc
        }
        encoder.setComputePipelineState(pipeline)
        arguments.withUnsafeBytes {
            encoder.setBytes($0.baseAddress!, length: $0.count, index: 0)
        }
        encoder.setBuffer(queryBuffer, offset: 0, index: 1)
        encoder.setBuffer(keyBuffer, offset: 0, index: 2)
        encoder.setBuffer(queryWeightBuffer, offset: 0, index: 3)
        encoder.setBuffer(keyWeightBuffer, offset: 0, index: 4)
        var queryHeads = UInt32(spec.headCount)
        encoder.setBytes(&queryHeads, length: MemoryLayout<UInt32>.stride, index: 5)
        encoder.setThreadgroupMemoryLength(
            threadCount * MemoryLayout<Float>.stride, index: 0
        )
        encoder.dispatchThreadgroups(
            MTLSize(width: combinedHeads, height: 1, depth: 1),
            threadsPerThreadgroup: MTLSize(width: threadCount, height: 1, depth: 1)
        )
        encoder.endEncoding()
        commandBuffer.commit()
        commandBuffer.waitUntilCompleted()
        if let error = commandBuffer.error { throw error }

        let queryPointer = queryBuffer.contents().bindMemory(to: Float.self,
                                                             capacity: query.count)
        let keyPointer = keyBuffer.contents().bindMemory(to: Float.self,
                                                         capacity: key.count)
        return (
            Array(UnsafeBufferPointer(start: queryPointer, count: query.count)),
            Array(UnsafeBufferPointer(start: keyPointer, count: key.count))
        )
    }

    /// Multi-token form used by Laguna prefill. Token rows are tightly packed
    /// and positions advance from `position`, matching the Metal grid's
    /// second dimension.
    public func lagunaQKHeadRMSNormRopeBatch(
        queries: [[Float]], keys: [[Float]],
        queryWeight: [Float], keyWeight: [Float],
        spec: LagunaAttentionSpec, position: Int
    ) throws -> (queries: [[Float]], keys: [[Float]]) {
        let count = queries.count
        guard count > 0, keys.count == count,
              queries.allSatisfy({ $0.count == spec.queryWidth }),
              keys.allSatisfy({ $0.count == spec.keyValueWidth }),
              queryWeight.count == spec.headDim,
              keyWeight.count == spec.headDim,
              spec.rotationDims <= spec.headDim,
              spec.rotationDims % 2 == 0,
              position >= 0 else {
            throw MetalError.unsupported(
                "Laguna batch norm/rope: inconsistent shapes")
        }

        let flatQueries = queries.flatMap { $0 }
        let flatKeys = keys.flatMap { $0 }
        guard let queryBuffer = device.makeBuffer(
                  bytes: flatQueries,
                  length: flatQueries.count * MemoryLayout<Float>.stride,
                  options: .storageModeShared),
              let keyBuffer = device.makeBuffer(
                  bytes: flatKeys,
                  length: flatKeys.count * MemoryLayout<Float>.stride,
                  options: .storageModeShared),
              let queryWeightBuffer = device.makeBuffer(
                  bytes: queryWeight,
                  length: queryWeight.count * MemoryLayout<Float>.stride,
                  options: .storageModeShared),
              let keyWeightBuffer = device.makeBuffer(
                  bytes: keyWeight,
                  length: keyWeight.count * MemoryLayout<Float>.stride,
                  options: .storageModeShared) else {
            throw MetalError.bufferAlloc
        }

        let combinedHeads = spec.headCount + spec.kvHeadCount
        var arguments = [UInt32](repeating: 0, count: 14)
        arguments[0] = UInt32(count)
        arguments[1] = UInt32(combinedHeads)
        arguments[2] = UInt32(spec.headDim)
        arguments[3] = UInt32(spec.rotationDims)
        arguments[4] = UInt32(position)
        arguments[5] = UInt32(spec.ropeOriginalContext)
        arguments[6] = spec.rmsEpsilon.bitPattern
        arguments[7] = spec.ropeFrequencyBase.bitPattern
        arguments[8] = spec.ropeFrequencyScale.bitPattern
        arguments[9] = spec.extrapolationFactor.bitPattern
        arguments[10] = spec.attentionFactor.bitPattern
        arguments[11] = spec.betaFast.bitPattern
        arguments[12] = spec.betaSlow.bitPattern

        let pipeline = try pipeline(
            "kernel_laguna_qk_head_rms_norm_rope_neox")
        let threadCount = min(128, pipeline.maxTotalThreadsPerThreadgroup)
        guard threadCount >= spec.rotationDims / 2,
              threadCount & (threadCount - 1) == 0 else {
            throw MetalError.unsupported(
                "Laguna batch norm/rope thread count \(threadCount)")
        }
        guard let commandBuffer = queue.makeCommandBuffer(),
              let encoder = commandBuffer.makeComputeCommandEncoder() else {
            throw MetalError.bufferAlloc
        }
        encoder.setComputePipelineState(pipeline)
        arguments.withUnsafeBytes {
            encoder.setBytes($0.baseAddress!, length: $0.count, index: 0)
        }
        encoder.setBuffer(queryBuffer, offset: 0, index: 1)
        encoder.setBuffer(keyBuffer, offset: 0, index: 2)
        encoder.setBuffer(queryWeightBuffer, offset: 0, index: 3)
        encoder.setBuffer(keyWeightBuffer, offset: 0, index: 4)
        var queryHeads = UInt32(spec.headCount)
        encoder.setBytes(
            &queryHeads, length: MemoryLayout<UInt32>.stride, index: 5)
        encoder.setThreadgroupMemoryLength(
            threadCount * MemoryLayout<Float>.stride, index: 0)
        encoder.dispatchThreadgroups(
            MTLSize(width: combinedHeads, height: count, depth: 1),
            threadsPerThreadgroup: MTLSize(
                width: threadCount, height: 1, depth: 1))
        encoder.endEncoding()
        commandBuffer.commit()
        commandBuffer.waitUntilCompleted()
        if let error = commandBuffer.error { throw error }

        let queryPointer = queryBuffer.contents().bindMemory(
            to: Float.self, capacity: flatQueries.count)
        let keyPointer = keyBuffer.contents().bindMemory(
            to: Float.self, capacity: flatKeys.count)
        let queryRows = (0..<count).map { row in
            Array(UnsafeBufferPointer(
                start: queryPointer + row * spec.queryWidth,
                count: spec.queryWidth))
        }
        let keyRows = (0..<count).map { row in
            Array(UnsafeBufferPointer(
                start: keyPointer + row * spec.keyValueWidth,
                count: spec.keyValueWidth))
        }
        return (queryRows, keyRows)
    }

    // MARK: KV store

    /// Dispatch `kernel_laguna_store_kv_f16`: write one position's K/V rows
    /// into the F16 ring cache at `position % capacity`.
    public func lagunaStoreKV(
        keyRow: [Float], valueRow: [Float],
        cache: LagunaMetalKVCache, position: Int
    ) throws {
        guard keyRow.count == cache.rowWidth,
              valueRow.count == cache.rowWidth,
              position >= 0 else {
            throw MetalError.unsupported("Laguna KV store: inconsistent shapes")
        }
        guard let keyBuffer = device.makeBuffer(
                  bytes: keyRow,
                  length: keyRow.count * MemoryLayout<Float>.stride,
                  options: .storageModeShared
              ),
              let valueBuffer = device.makeBuffer(
                  bytes: valueRow,
                  length: valueRow.count * MemoryLayout<Float>.stride,
                  options: .storageModeShared
              ) else {
            throw MetalError.bufferAlloc
        }

        // The kernel only needs the row width product; report the whole row
        // as one KV head so wrappers stay shape-agnostic.
        var arguments = [UInt32](repeating: 0, count: 4)
        arguments[0] = UInt32(cache.capacity)
        arguments[1] = UInt32(position % cache.capacity)
        arguments[2] = 1
        arguments[3] = UInt32(cache.rowWidth)

        let pipeline = try pipeline("kernel_laguna_store_kv_f16")
        let threadCount = min(256, pipeline.maxTotalThreadsPerThreadgroup)
        let groups = (cache.rowWidth + threadCount - 1) / threadCount
        guard let commandBuffer = queue.makeCommandBuffer(),
              let encoder = commandBuffer.makeComputeCommandEncoder() else {
            throw MetalError.bufferAlloc
        }
        encoder.setComputePipelineState(pipeline)
        arguments.withUnsafeBytes {
            encoder.setBytes($0.baseAddress!, length: $0.count, index: 0)
        }
        encoder.setBuffer(keyBuffer, offset: 0, index: 1)
        encoder.setBuffer(valueBuffer, offset: 0, index: 2)
        encoder.setBuffer(cache.keys, offset: 0, index: 3)
        encoder.setBuffer(cache.values, offset: 0, index: 4)
        encoder.dispatchThreadgroups(
            MTLSize(width: max(groups, 1), height: 1, depth: 1),
            threadsPerThreadgroup: MTLSize(width: threadCount, height: 1, depth: 1)
        )
        encoder.endEncoding()
        commandBuffer.commit()
        commandBuffer.waitUntilCompleted()
        if let error = commandBuffer.error { throw error }
    }

    // MARK: Gated GQA attention (prefill)

    /// Attend a causal token block in parallel. New K/V rows are first
    /// staged as F16, so an SWA block may cross the ring boundary without
    /// overwriting older rows still needed by early queries; the staged rows
    /// are committed only after attention completes.
    public func lagunaAttentionPrefill(
        queries: [[Float]], gates: [[Float]],
        keyRows: [[Float]], valueRows: [[Float]],
        cache: LagunaMetalKVCache, position: Int,
        spec: LagunaAttentionSpec
    ) throws -> [[Float]] {
        let count = queries.count
        guard count > 0, count <= cache.capacity,
              gates.count == count,
              keyRows.count == count,
              valueRows.count == count,
              queries.allSatisfy({ $0.count == spec.queryWidth }),
              gates.allSatisfy({ $0.count == spec.headCount }),
              keyRows.allSatisfy({ $0.count == spec.keyValueWidth }),
              valueRows.allSatisfy({ $0.count == spec.keyValueWidth }),
              cache.rowWidth == spec.keyValueWidth,
              cache.capacity == spec.cacheCapacity,
              spec.headDim == 128,
              spec.kvHeadCount > 0,
              spec.headCount % spec.kvHeadCount == 0,
              position >= 0 else {
            throw MetalError.unsupported(
                "Laguna prefill attention: inconsistent shapes")
        }

        let flatQueries = queries.flatMap { $0 }
        let flatGates = gates.flatMap { $0 }
        let flatKeys = keyRows.flatMap { $0 }
        let flatValues = valueRows.flatMap { $0 }
        let stagedBytes = flatKeys.count * MemoryLayout<UInt16>.stride
        guard let queryBuffer = device.makeBuffer(
                  bytes: flatQueries,
                  length: flatQueries.count * MemoryLayout<Float>.stride,
                  options: .storageModeShared),
              let gateBuffer = device.makeBuffer(
                  bytes: flatGates,
                  length: flatGates.count * MemoryLayout<Float>.stride,
                  options: .storageModeShared),
              let keyBuffer = device.makeBuffer(
                  bytes: flatKeys,
                  length: flatKeys.count * MemoryLayout<Float>.stride,
                  options: .storageModeShared),
              let valueBuffer = device.makeBuffer(
                  bytes: flatValues,
                  length: flatValues.count * MemoryLayout<Float>.stride,
                  options: .storageModeShared),
              let stagedKey = device.makeBuffer(
                  length: stagedBytes, options: .storageModeShared),
              let stagedValue = device.makeBuffer(
                  length: stagedBytes, options: .storageModeShared),
              let outBuffer = device.makeBuffer(
                  length: flatQueries.count * MemoryLayout<Float>.stride,
                  options: .storageModeShared),
              let commandBuffer = queue.makeCommandBuffer(),
              let encoder = commandBuffer.makeComputeCommandEncoder() else {
            throw MetalError.bufferAlloc
        }

        var arguments = [UInt32](repeating: 0, count: 8)
        arguments[0] = UInt32(count)
        arguments[1] = UInt32(position)
        arguments[2] = UInt32(cache.capacity)
        arguments[3] = UInt32(spec.headCount)
        arguments[4] = UInt32(spec.kvHeadCount)
        arguments[5] = UInt32(spec.headDim)
        arguments[6] = (1 / Float(spec.headDim).squareRoot()).bitPattern
        func setArguments() {
            arguments.withUnsafeBytes {
                encoder.setBytes($0.baseAddress!, length: $0.count, index: 0)
            }
        }

        let stage = try pipeline("kernel_laguna_stage_kv_f16")
        encoder.setComputePipelineState(stage)
        setArguments()
        encoder.setBuffer(keyBuffer, offset: 0, index: 1)
        encoder.setBuffer(valueBuffer, offset: 0, index: 2)
        encoder.setBuffer(stagedKey, offset: 0, index: 3)
        encoder.setBuffer(stagedValue, offset: 0, index: 4)
        let values = count * spec.keyValueWidth
        let stageThreads = min(256, stage.maxTotalThreadsPerThreadgroup)
        encoder.dispatchThreadgroups(
            MTLSize(width: (values + stageThreads - 1) / stageThreads,
                    height: 1, depth: 1),
            threadsPerThreadgroup: MTLSize(
                width: stageThreads, height: 1, depth: 1)
        )

        let headsPerKV = spec.headCount / spec.kvHeadCount
        let attentionName: String
        let headGroups: Int
        switch headsPerKV {
        case 6:
            attentionName = "kernel_laguna_attention_prefill_gqa6_f16"
            headGroups = spec.headCount / 6
        case 3:
            attentionName = "kernel_laguna_attention_prefill_gqa3_f16"
            headGroups = spec.headCount / 3
        default:
            attentionName = "kernel_laguna_attention_prefill_gqa_f16"
            headGroups = spec.headCount
        }
        let attention = try pipeline(attentionName)
        encoder.setComputePipelineState(attention)
        setArguments()
        encoder.setBuffer(queryBuffer, offset: 0, index: 1)
        encoder.setBuffer(gateBuffer, offset: 0, index: 2)
        encoder.setBuffer(cache.keys, offset: 0, index: 3)
        encoder.setBuffer(cache.values, offset: 0, index: 4)
        encoder.setBuffer(stagedKey, offset: 0, index: 5)
        encoder.setBuffer(stagedValue, offset: 0, index: 6)
        encoder.setBuffer(outBuffer, offset: 0, index: 7)
        encoder.dispatchThreadgroups(
            MTLSize(width: headGroups, height: count, depth: 1),
            threadsPerThreadgroup: MTLSize(width: 32, height: 1, depth: 1)
        )

        let commit = try pipeline("kernel_laguna_commit_kv_f16")
        encoder.setComputePipelineState(commit)
        setArguments()
        encoder.setBuffer(stagedKey, offset: 0, index: 1)
        encoder.setBuffer(stagedValue, offset: 0, index: 2)
        encoder.setBuffer(cache.keys, offset: 0, index: 3)
        encoder.setBuffer(cache.values, offset: 0, index: 4)
        let commitThreads = min(256, commit.maxTotalThreadsPerThreadgroup)
        encoder.dispatchThreadgroups(
            MTLSize(width: (values + commitThreads - 1) / commitThreads,
                    height: 1, depth: 1),
            threadsPerThreadgroup: MTLSize(
                width: commitThreads, height: 1, depth: 1)
        )

        encoder.endEncoding()
        commandBuffer.commit()
        commandBuffer.waitUntilCompleted()
        if let error = commandBuffer.error { throw error }

        let pointer = outBuffer.contents().bindMemory(
            to: Float.self, capacity: flatQueries.count)
        return (0..<count).map { row in
            let start = pointer + row * spec.queryWidth
            return Array(UnsafeBufferPointer(
                start: start, count: spec.queryWidth))
        }
    }

    // MARK: Routed MoE (multi-token prefill)

    /// Validation wrapper for applying one quantized routed expert to several
    /// token rows in one tiled dispatch pair.
    public func lagunaMoEPrefillExpert(
        inputs: [[Float]],
        routeWeights: [Float],
        gateRows: [UInt8],
        upRows: [UInt8],
        downRows: [UInt8],
        weightType: UInt32,
        expertWidth: Int
    ) throws -> [[Float]] {
        let count = inputs.count
        let inputWidth = inputs.first?.count ?? 0
        let gateBytes = QuantEncode.rowSize(
            type: weightType, columns: inputWidth) * expertWidth
        let downBytes = QuantEncode.rowSize(
            type: weightType, columns: expertWidth) * inputWidth
        guard count > 0, inputWidth > 0, expertWidth > 0,
              routeWeights.count == count,
              inputs.allSatisfy({ $0.count == inputWidth }),
              gateRows.count == gateBytes,
              upRows.count == gateBytes,
              downRows.count == downBytes else {
            throw MetalError.unsupported(
                "Laguna MoE prefill: inconsistent shapes")
        }
        let flatInputs = inputs.flatMap { $0 }
        let appTokens = (0..<count).map(UInt32.init)
        guard let inputBuffer = device.makeBuffer(
                  bytes: flatInputs,
                  length: flatInputs.count * MemoryLayout<Float>.stride,
                  options: .storageModeShared),
              let tokenBuffer = device.makeBuffer(
                  bytes: appTokens,
                  length: appTokens.count * MemoryLayout<UInt32>.stride,
                  options: .storageModeShared),
              let weightBuffer = device.makeBuffer(
                  bytes: routeWeights,
                  length: routeWeights.count * MemoryLayout<Float>.stride,
                  options: .storageModeShared),
              let gateBuffer = device.makeBuffer(
                  bytes: gateRows, length: gateRows.count,
                  options: .storageModeShared),
              let upBuffer = device.makeBuffer(
                  bytes: upRows, length: upRows.count,
                  options: .storageModeShared),
              let downBuffer = device.makeBuffer(
                  bytes: downRows, length: downRows.count,
                  options: .storageModeShared),
              let mids = device.makeBuffer(
                  length: count * expertWidth
                      * MemoryLayout<Float>.stride,
                  options: .storageModeShared),
              let contributions = device.makeBuffer(
                  length: count * inputWidth
                      * MemoryLayout<Float>.stride,
                  options: .storageModeShared),
              let commandBuffer = queue.makeCommandBuffer(),
              let encoder = commandBuffer.makeComputeCommandEncoder() else {
            throw MetalError.bufferAlloc
        }

        let arguments: [UInt32] = [
            weightType, UInt32(expertWidth), UInt32(inputWidth),
            0, UInt32(count), 0, 0, 0,
        ]
        func setArguments() {
            arguments.withUnsafeBytes {
                encoder.setBytes($0.baseAddress!, length: $0.count, index: 0)
            }
        }
        let rowsPerGroup = 4
        let threads = MTLSize(
            width: 32, height: rowsPerGroup, depth: 1)
        let swiglu = try pipeline(
            "kernel_laguna_moe_prefill_swiglu_sg")
        encoder.setComputePipelineState(swiglu)
        setArguments()
        encoder.setBuffer(tokenBuffer, offset: 0, index: 1)
        encoder.setBuffer(weightBuffer, offset: 0, index: 2)
        encoder.setBuffer(inputBuffer, offset: 0, index: 3)
        encoder.setBuffer(gateBuffer, offset: 0, index: 4)
        encoder.setBuffer(upBuffer, offset: 0, index: 5)
        encoder.setBuffer(mids, offset: 0, index: 6)
        encoder.dispatchThreadgroups(
            MTLSize(width: (expertWidth + rowsPerGroup - 1)
                / rowsPerGroup, height: 1, depth: 1),
            threadsPerThreadgroup: threads)

        let down = try pipeline("kernel_laguna_moe_prefill_down_sg")
        encoder.setComputePipelineState(down)
        setArguments()
        encoder.setBuffer(mids, offset: 0, index: 1)
        encoder.setBuffer(downBuffer, offset: 0, index: 2)
        encoder.setBuffer(contributions, offset: 0, index: 3)
        encoder.dispatchThreadgroups(
            MTLSize(width: (inputWidth + rowsPerGroup - 1)
                / rowsPerGroup, height: 1, depth: 1),
            threadsPerThreadgroup: threads)
        encoder.endEncoding()
        commandBuffer.commit()
        commandBuffer.waitUntilCompleted()
        if let error = commandBuffer.error { throw error }

        let pointer = contributions.contents().bindMemory(
            to: Float.self, capacity: count * inputWidth)
        return (0..<count).map { token in
            Array(UnsafeBufferPointer(
                start: pointer + token * inputWidth,
                count: inputWidth))
        }
    }

    // MARK: Gated GQA attention (decode)

    /// Dispatch `kernel_laguna_attention_decode_gqa_f16` for one query token
    /// at `position` over the ring window. The kernel is specialized for the
    /// production 128-dimension heads.
    public func lagunaAttentionDecode(
        query: [Float], gate: [Float],
        cache: LagunaMetalKVCache, position: Int,
        spec: LagunaAttentionSpec
    ) throws -> [Float] {
        guard spec.headDim == 128 else {
            throw MetalError.unsupported(
                "kernel_laguna_attention_decode_gqa_f16 is specialized for headDim 128"
            )
        }
        guard query.count == spec.queryWidth,
              gate.count == spec.headCount,
              cache.rowWidth == spec.keyValueWidth,
              cache.capacity == spec.cacheCapacity,
              spec.headCount % spec.kvHeadCount == 0,
              position >= 0 else {
            throw MetalError.unsupported("Laguna decode attention: inconsistent shapes")
        }

        let keyCount = min(position + 1, spec.cacheCapacity)
        let keyStart = position + 1 - keyCount
        guard let queryBuffer = device.makeBuffer(
                  bytes: query,
                  length: query.count * MemoryLayout<Float>.stride,
                  options: .storageModeShared
              ),
              let gateBuffer = device.makeBuffer(
                  bytes: gate,
                  length: gate.count * MemoryLayout<Float>.stride,
                  options: .storageModeShared
              ),
              let outBuffer = device.makeBuffer(
                  length: query.count * MemoryLayout<Float>.stride,
                  options: .storageModeShared
              ) else {
            throw MetalError.bufferAlloc
        }

        var arguments = [UInt32](repeating: 0, count: 8)
        arguments[0] = UInt32(spec.headCount)
        arguments[1] = UInt32(spec.kvHeadCount)
        arguments[2] = UInt32(spec.headDim)
        arguments[3] = UInt32(spec.cacheCapacity)
        arguments[4] = UInt32(keyStart)
        arguments[5] = UInt32(keyCount)
        arguments[6] = (1 / Float(spec.headDim).squareRoot()).bitPattern

        let pipeline = try pipeline("kernel_laguna_attention_decode_gqa_f16")
        // Eight SIMD groups of 32 lanes; the split path needs all of them.
        let threadCount = 256
        guard pipeline.maxTotalThreadsPerThreadgroup >= threadCount else {
            throw MetalError.unsupported("Laguna decode attention needs 256 threads")
        }
        // scratch: 8 maxima + 8 sums + 8 * headDim partial values.
        let scratchBytes = (16 + 8 * spec.headDim) * MemoryLayout<Float>.stride
        guard let commandBuffer = queue.makeCommandBuffer(),
              let encoder = commandBuffer.makeComputeCommandEncoder() else {
            throw MetalError.bufferAlloc
        }
        encoder.setComputePipelineState(pipeline)
        arguments.withUnsafeBytes {
            encoder.setBytes($0.baseAddress!, length: $0.count, index: 0)
        }
        encoder.setBuffer(queryBuffer, offset: 0, index: 1)
        encoder.setBuffer(gateBuffer, offset: 0, index: 2)
        encoder.setBuffer(cache.keys, offset: 0, index: 3)
        encoder.setBuffer(cache.values, offset: 0, index: 4)
        encoder.setBuffer(outBuffer, offset: 0, index: 5)
        encoder.setThreadgroupMemoryLength(scratchBytes, index: 0)
        encoder.dispatchThreadgroups(
            MTLSize(width: spec.headCount, height: 1, depth: 1),
            threadsPerThreadgroup: MTLSize(width: threadCount, height: 1, depth: 1)
        )
        encoder.endEncoding()
        commandBuffer.commit()
        commandBuffer.waitUntilCompleted()
        if let error = commandBuffer.error { throw error }

        let outPointer = outBuffer.contents().bindMemory(to: Float.self,
                                                         capacity: query.count)
        return Array(UnsafeBufferPointer(start: outPointer, count: query.count))
    }

    /// Validation wrapper for the production full-attention split-K path.
    /// Three query heads sharing one KV head reuse each K/V row, then the
    /// specialized reducer merges workgroups and applies the learned gate.
    public func lagunaAttentionDecodeGQA3Split(
        query: [Float], gate: [Float],
        cache: LagunaMetalKVCache, position: Int,
        spec: LagunaAttentionSpec,
        maximumWorkgroups: Int = 32
    ) throws -> [Float] {
        let headsPerKV = spec.headCount / max(1, spec.kvHeadCount)
        guard spec.headDim == 128,
              spec.headCount % 3 == 0,
              headsPerKV % 3 == 0,
              query.count == spec.queryWidth,
              gate.count == spec.headCount,
              cache.rowWidth == spec.keyValueWidth,
              cache.capacity == spec.cacheCapacity,
              position >= 0,
              position < spec.cacheCapacity else {
            throw MetalError.unsupported(
                "Laguna split-K decode needs contiguous GQA3 full attention")
        }
        let keyCount = position + 1
        let supported = device.maxThreadsPerThreadgroup.width / 32
        guard maximumWorkgroups >= 32, supported >= 32 else {
            throw MetalError.unsupported(
                "Laguna split-K decode requires 32 workgroups")
        }
        let workgroups = 32
        let rows = spec.headCount
        let partialFloats = rows * spec.headDim * workgroups
            + rows * 2 * workgroups
        guard let queryBuffer = device.makeBuffer(
                  bytes: query, length: query.count * 4,
                  options: .storageModeShared),
              let gateBuffer = device.makeBuffer(
                  bytes: gate, length: gate.count * 4,
                  options: .storageModeShared),
              let tmpBuffer = device.makeBuffer(
                  length: partialFloats * 4,
                  options: .storageModeShared),
              let outBuffer = device.makeBuffer(
                  length: query.count * 4,
                  options: .storageModeShared),
              let commandBuffer = queue.makeCommandBuffer(),
              let encoder = commandBuffer.makeComputeCommandEncoder() else {
            throw MetalError.bufferAlloc
        }

        let split = try pipeline(
            "kernel_laguna_attention_decode_gqa3_split_f16")
        var arguments = [UInt32](repeating: 0, count: 8)
        arguments[0] = UInt32(spec.headCount)
        arguments[1] = UInt32(spec.kvHeadCount)
        arguments[2] = UInt32(spec.headDim)
        arguments[3] = UInt32(keyCount)
        arguments[4] = 1
        arguments[5] = 1
        arguments[6] = UInt32(workgroups)
        arguments[7] = (1 / Float(spec.headDim).squareRoot()).bitPattern
        encoder.setComputePipelineState(split)
        arguments.withUnsafeBytes {
            encoder.setBytes($0.baseAddress!, length: $0.count, index: 0)
        }
        encoder.setBuffer(queryBuffer, offset: 0, index: 1)
        encoder.setBuffer(cache.keys, offset: 0, index: 2)
        encoder.setBuffer(cache.values, offset: 0, index: 3)
        encoder.setBuffer(tmpBuffer, offset: 0, index: 4)
        encoder.setThreadgroupMemoryLength(
            3 * (2 + spec.headDim) * 4, index: 0)
        encoder.dispatchThreadgroups(
            MTLSize(width: spec.headCount / 3, height: 1,
                    depth: workgroups),
            threadsPerThreadgroup: MTLSize(width: 32, height: 1, depth: 1))

        let constants = MTLFunctionConstantValues()
        var dv = Int32(spec.headDim)
        var nwg = Int32(workgroups)
        constants.setConstantValue(&dv, type: .int, index: 500)
        constants.setConstantValue(&nwg, type: .int, index: 501)
        let function = try library.makeFunction(
            name: "kernel_laguna_flash_attn_reduce_gate_f32",
            constantValues: constants)
        let reduce = try device.makeComputePipelineState(function: function)
        encoder.setComputePipelineState(reduce)
        var rowCount = Int32(rows)
        encoder.setBytes(
            &rowCount, length: MemoryLayout<Int32>.stride, index: 0)
        encoder.setBuffer(tmpBuffer, offset: 0, index: 1)
        encoder.setBuffer(outBuffer, offset: 0, index: 2)
        encoder.setBuffer(gateBuffer, offset: 0, index: 3)
        encoder.dispatchThreadgroups(
            MTLSize(width: rows, height: 1, depth: 1),
            threadsPerThreadgroup: MTLSize(
                width: 32 * workgroups, height: 1, depth: 1))
        encoder.endEncoding()
        commandBuffer.commit()
        commandBuffer.waitUntilCompleted()
        if let error = commandBuffer.error { throw error }

        let pointer = outBuffer.contents().bindMemory(
            to: Float.self, capacity: query.count)
        return Array(UnsafeBufferPointer(
            start: pointer, count: query.count))
    }
}
