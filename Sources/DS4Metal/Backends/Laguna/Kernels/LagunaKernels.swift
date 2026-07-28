import Foundation
import Metal
import DS4Core

// Swift wrappers over the Laguna decode kernels in `metal/laguna/laguna.metal`.
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
}
