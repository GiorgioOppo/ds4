import DS4Core
import Foundation
import Metal

/// F16 bit patterns for GLM 5.2's two compact DSA cache planes.
public struct GLM52CompactKVOutput: Sendable, Equatable {
    public let kvLoRABits: [UInt16]
    public let kRoPEBits: [UInt16]

    public init(kvLoRABits: [UInt16], kRoPEBits: [UInt16]) {
        self.kvLoRABits = kvLoRABits
        self.kRoPEBits = kRoPEBits
    }
}

private struct GLM52CompactKVPlan {
    let tokenCount: Int
    let kvCacheCount: Int
    let ropeCacheCount: Int
}

private func glm52CompactKVPlan(
    rows: [Float],
    pos0: Int,
    cacheCapacity: Int,
    initialKVLoRABits: [UInt16]?,
    initialKRoPEBits: [UInt16]?
) throws -> GLM52CompactKVPlan {
    let rawWidth = GLM52CompactKVReference.rawWidth
    guard !rows.isEmpty, rows.count.isMultiple(of: rawWidth) else {
        throw MetalError.unsupported(
            "GLM 5.2 compact KV requires non-empty 576-wide rows"
        )
    }
    let tokenCount = rows.count / rawWidth
    let u32Max = Int(UInt32.max)
    guard cacheCapacity > 0, cacheCapacity <= u32Max,
          pos0 >= 0, pos0 <= cacheCapacity,
          tokenCount <= u32Max,
          tokenCount <= cacheCapacity - pos0 else {
        throw MetalError.unsupported(
            "GLM 5.2 compact KV range pos0=\(pos0), tokens=\(tokenCount), " +
            "capacity=\(cacheCapacity) is invalid"
        )
    }

    let (kvCacheCount, kvOverflow) = cacheCapacity.multipliedReportingOverflow(
        by: GLM52CompactKVReference.kvLoRAWidth
    )
    let (ropeCacheCount, ropeOverflow) = cacheCapacity.multipliedReportingOverflow(
        by: GLM52CompactKVReference.kRoPEWidth
    )
    guard !kvOverflow, !ropeOverflow else {
        throw MetalError.unsupported("GLM 5.2 compact KV cache size overflow")
    }
    if let initialKVLoRABits, initialKVLoRABits.count != kvCacheCount {
        throw MetalError.unsupported(
            "GLM 5.2 KV-LoRA cache count \(initialKVLoRABits.count); " +
            "expected \(kvCacheCount)"
        )
    }
    if let initialKRoPEBits, initialKRoPEBits.count != ropeCacheCount {
        throw MetalError.unsupported(
            "GLM 5.2 K-RoPE cache count \(initialKRoPEBits.count); " +
            "expected \(ropeCacheCount)"
        )
    }
    return GLM52CompactKVPlan(
        tokenCount: tokenCount,
        kvCacheCount: kvCacheCount,
        ropeCacheCount: ropeCacheCount
    )
}

/// Independent scalar oracle for the cache-only half conversion and placement.
///
/// Every input row is cache-ready: elements `0..<512` have already undergone
/// KV-LoRA RMS normalization, while elements `512..<576` are the untouched
/// K-RoPE tail. This primitive deliberately performs no normalization or RoPE.
public enum GLM52CompactKVReference {
    public static let rawWidth = 576
    public static let kvLoRAWidth = 512
    public static let kRoPEWidth = 64

    public static func store(
        rows: [Float],
        pos0: Int,
        cacheCapacity: Int,
        initialKVLoRABits: [UInt16]? = nil,
        initialKRoPEBits: [UInt16]? = nil
    ) throws -> GLM52CompactKVOutput {
        let plan = try glm52CompactKVPlan(
            rows: rows,
            pos0: pos0,
            cacheCapacity: cacheCapacity,
            initialKVLoRABits: initialKVLoRABits,
            initialKRoPEBits: initialKRoPEBits
        )
        var kvCache = initialKVLoRABits
            ?? [UInt16](repeating: 0, count: plan.kvCacheCount)
        var ropeCache = initialKRoPEBits
            ?? [UInt16](repeating: 0, count: plan.ropeCacheCount)

        for token in 0..<plan.tokenCount {
            let source = token * rawWidth
            let position = pos0 + token
            let kvDestination = position * kvLoRAWidth
            let ropeDestination = position * kRoPEWidth
            for column in 0..<kvLoRAWidth {
                kvCache[kvDestination + column] = Half.bits(rows[source + column])
            }
            for column in 0..<kRoPEWidth {
                ropeCache[ropeDestination + column] = Half.bits(
                    rows[source + kvLoRAWidth + column]
                )
            }
        }
        return GLM52CompactKVOutput(
            kvLoRABits: kvCache,
            kRoPEBits: ropeCache
        )
    }
}

extension MetalRuntime {
    /// Dispatch the atomic GLM compact-cache store over synthetic or staged rows.
    /// This is a validation wrapper, not a runnable GLM decoder integration.
    public func glm52StoreCompactKV(
        rows: [Float],
        pos0: Int,
        cacheCapacity: Int,
        initialKVLoRABits: [UInt16]? = nil,
        initialKRoPEBits: [UInt16]? = nil
    ) throws -> GLM52CompactKVOutput {
        let plan = try glm52CompactKVPlan(
            rows: rows,
            pos0: pos0,
            cacheCapacity: cacheCapacity,
            initialKVLoRABits: initialKVLoRABits,
            initialKRoPEBits: initialKRoPEBits
        )
        let kvInitial = initialKVLoRABits
            ?? [UInt16](repeating: 0, count: plan.kvCacheCount)
        let ropeInitial = initialKRoPEBits
            ?? [UInt16](repeating: 0, count: plan.ropeCacheCount)

        guard let rowsBuffer = device.makeBuffer(
                  bytes: rows,
                  length: rows.count * MemoryLayout<Float>.stride,
                  options: .storageModeShared
              ),
              let kvBuffer = device.makeBuffer(
                  bytes: kvInitial,
                  length: kvInitial.count * MemoryLayout<UInt16>.stride,
                  options: .storageModeShared
              ),
              let ropeBuffer = device.makeBuffer(
                  bytes: ropeInitial,
                  length: ropeInitial.count * MemoryLayout<UInt16>.stride,
                  options: .storageModeShared
              ) else {
            throw MetalError.bufferAlloc
        }

        var arguments = [UInt32](repeating: 0, count: 4)
        arguments[0] = UInt32(pos0)
        arguments[1] = UInt32(plan.tokenCount)
        arguments[2] = UInt32(cacheCapacity)

        let pipeline = try pipeline("kernel_glm52_store_compact_kv_f16")
        let threadCount = min(256, pipeline.maxTotalThreadsPerThreadgroup)
        guard threadCount > 0 else {
            throw MetalError.unsupported("GLM 5.2 compact KV has no dispatchable threads")
        }
        guard let commandBuffer = queue.makeCommandBuffer(),
              let encoder = commandBuffer.makeComputeCommandEncoder() else {
            throw MetalError.bufferAlloc
        }
        encoder.setComputePipelineState(pipeline)
        arguments.withUnsafeBytes {
            encoder.setBytes($0.baseAddress!, length: $0.count, index: 0)
        }
        encoder.setBuffer(rowsBuffer, offset: 0, index: 1)
        encoder.setBuffer(kvBuffer, offset: 0, index: 2)
        encoder.setBuffer(ropeBuffer, offset: 0, index: 3)
        encoder.dispatchThreadgroups(
            MTLSize(width: plan.tokenCount, height: 2, depth: 1),
            threadsPerThreadgroup: MTLSize(width: threadCount, height: 1, depth: 1)
        )
        encoder.endEncoding()
        commandBuffer.commit()
        commandBuffer.waitUntilCompleted()
        if let error = commandBuffer.error { throw error }

        let kvPointer = kvBuffer.contents().bindMemory(
            to: UInt16.self,
            capacity: plan.kvCacheCount
        )
        let ropePointer = ropeBuffer.contents().bindMemory(
            to: UInt16.self,
            capacity: plan.ropeCacheCount
        )
        return GLM52CompactKVOutput(
            kvLoRABits: Array(UnsafeBufferPointer(
                start: kvPointer,
                count: plan.kvCacheCount
            )),
            kRoPEBits: Array(UnsafeBufferPointer(
                start: ropePointer,
                count: plan.ropeCacheCount
            ))
        )
    }
}
