import DS4Core
import Foundation
import Metal

private struct GLM52IndexerKeyStorePlan {
    let tokenCount: Int
    let cacheCount: Int
}

private func glm52IndexerKeyStorePlan(
    rawKeys: [Float],
    weight: [Float],
    bias: [Float],
    pos0: Int,
    cacheCapacity: Int,
    initialCacheBits: [UInt16]?
) throws -> GLM52IndexerKeyStorePlan {
    let width = GLM52IndexerKeyStoreReference.headDimension
    guard !rawKeys.isEmpty, rawKeys.count.isMultiple(of: width) else {
        throw MetalError.unsupported(
            "GLM 5.2 indexer keys require non-empty 128-wide rows"
        )
    }
    guard weight.count == width, bias.count == width else {
        throw MetalError.unsupported(
            "GLM 5.2 indexer affine vectors must each contain 128 values"
        )
    }
    let tokenCount = rawKeys.count / width
    let u32Max = Int(UInt32.max)
    guard cacheCapacity > 0, cacheCapacity <= u32Max,
          pos0 >= 0, pos0 <= cacheCapacity,
          tokenCount <= u32Max,
          tokenCount <= cacheCapacity - pos0 else {
        throw MetalError.unsupported(
            "GLM 5.2 indexer cache range pos0=\(pos0), tokens=\(tokenCount), " +
            "capacity=\(cacheCapacity) is invalid"
        )
    }
    let (cacheCount, overflow) = cacheCapacity.multipliedReportingOverflow(by: width)
    guard !overflow else {
        throw MetalError.unsupported("GLM 5.2 indexer cache size overflow")
    }
    if let initialCacheBits, initialCacheBits.count != cacheCount {
        throw MetalError.unsupported(
            "GLM 5.2 indexer cache count \(initialCacheBits.count); " +
            "expected \(cacheCount)"
        )
    }
    return GLM52IndexerKeyStorePlan(
        tokenCount: tokenCount,
        cacheCount: cacheCount
    )
}

/// Scalar oracle for GLM's indexer-key normalization, RoPE and F16 placement.
///
/// Despite the GGUF name `indexer.k_norm`, upstream applies centered
/// LayerNorm (mean and centered variance), followed by learned affine weight
/// and bias. RoPE covers the prefix `0..<64`; columns `64..<128` are not
/// rotated. The fixed short-context parameters here are base 8,000,000,
/// frequency scale 1, attention scale 1 and no YaRN extrapolation.
public enum GLM52IndexerKeyStoreReference {
    public static let headDimension = 128
    public static let rotationDimension = 64
    public static let epsilon: Float = 1.0e-6
    public static let frequencyBase: Float = 8_000_000

    public static func store(
        rawKeys: [Float],
        weight: [Float],
        bias: [Float],
        pos0: Int,
        cacheCapacity: Int,
        initialCacheBits: [UInt16]? = nil
    ) throws -> [UInt16] {
        let plan = try glm52IndexerKeyStorePlan(
            rawKeys: rawKeys,
            weight: weight,
            bias: bias,
            pos0: pos0,
            cacheCapacity: cacheCapacity,
            initialCacheBits: initialCacheBits
        )
        var cache = initialCacheBits
            ?? [UInt16](repeating: 0, count: plan.cacheCount)

        for token in 0..<plan.tokenCount {
            let source = token * headDimension
            var sum: Float = 0
            for column in 0..<headDimension {
                sum += rawKeys[source + column]
            }
            let mean = sum / Float(headDimension)
            var sumSquares: Float = 0
            for column in 0..<headDimension {
                let centered = rawKeys[source + column] - mean
                sumSquares += centered * centered
            }
            let inverseDeviation = 1 / sqrt(
                sumSquares / Float(headDimension) + epsilon
            )

            var normalized = [Float](repeating: 0, count: headDimension)
            for column in 0..<headDimension {
                normalized[column] =
                    (rawKeys[source + column] - mean)
                    * inverseDeviation * weight[column] + bias[column]
            }
            let position = Float(pos0 + token)
            for column in stride(from: 0, to: rotationDimension, by: 2) {
                let theta = position * pow(
                    frequencyBase,
                    -Float(column) / Float(rotationDimension)
                )
                let cosine = cos(theta)
                let sine = sin(theta)
                let x0 = normalized[column]
                let x1 = normalized[column + 1]
                normalized[column] = x0 * cosine - x1 * sine
                normalized[column + 1] = x0 * sine + x1 * cosine
            }

            let destination = (pos0 + token) * headDimension
            for column in 0..<headDimension {
                cache[destination + column] = Half.bits(normalized[column])
            }
        }
        return cache
    }
}

extension MetalRuntime {
    /// Runs upstream-compatible centered indexer normalization, prefix RoPE and
    /// F16 cache placement. This is a validation wrapper, not graph integration.
    public func glm52StoreIndexerKeys(
        rawKeys: [Float],
        weight: [Float],
        bias: [Float],
        pos0: Int,
        cacheCapacity: Int,
        initialCacheBits: [UInt16]? = nil
    ) throws -> [UInt16] {
        let plan = try glm52IndexerKeyStorePlan(
            rawKeys: rawKeys,
            weight: weight,
            bias: bias,
            pos0: pos0,
            cacheCapacity: cacheCapacity,
            initialCacheBits: initialCacheBits
        )
        let initial = initialCacheBits
            ?? [UInt16](repeating: 0, count: plan.cacheCount)
        guard let keysBuffer = device.makeBuffer(
                  bytes: rawKeys,
                  length: rawKeys.count * MemoryLayout<Float>.stride,
                  options: .storageModeShared
              ),
              let weightBuffer = device.makeBuffer(
                  bytes: weight,
                  length: weight.count * MemoryLayout<Float>.stride,
                  options: .storageModeShared
              ),
              let biasBuffer = device.makeBuffer(
                  bytes: bias,
                  length: bias.count * MemoryLayout<Float>.stride,
                  options: .storageModeShared
              ),
              let cacheBuffer = device.makeBuffer(
                  bytes: initial,
                  length: initial.count * MemoryLayout<UInt16>.stride,
                  options: .storageModeShared
              ) else {
            throw MetalError.bufferAlloc
        }

        var arguments = [UInt32](repeating: 0, count: 4)
        arguments[0] = UInt32(pos0)
        arguments[1] = UInt32(plan.tokenCount)
        arguments[2] = UInt32(cacheCapacity)

        let pipeline = try pipeline("kernel_glm52_store_indexer_k_f16")
        guard pipeline.maxTotalThreadsPerThreadgroup >= 32 else {
            throw MetalError.unsupported(
                "GLM 5.2 indexer key store requires 32 threads"
            )
        }
        guard let commandBuffer = queue.makeCommandBuffer(),
              let encoder = commandBuffer.makeComputeCommandEncoder() else {
            throw MetalError.bufferAlloc
        }
        encoder.setComputePipelineState(pipeline)
        arguments.withUnsafeBytes {
            encoder.setBytes($0.baseAddress!, length: $0.count, index: 0)
        }
        encoder.setBuffer(keysBuffer, offset: 0, index: 1)
        encoder.setBuffer(weightBuffer, offset: 0, index: 2)
        encoder.setBuffer(biasBuffer, offset: 0, index: 3)
        encoder.setBuffer(cacheBuffer, offset: 0, index: 4)
        encoder.setThreadgroupMemoryLength(
            32 * MemoryLayout<Float>.stride,
            index: 0
        )
        encoder.dispatchThreadgroups(
            MTLSize(width: plan.tokenCount, height: 1, depth: 1),
            threadsPerThreadgroup: MTLSize(width: 32, height: 1, depth: 1)
        )
        encoder.endEncoding()
        commandBuffer.commit()
        commandBuffer.waitUntilCompleted()
        if let error = commandBuffer.error { throw error }

        let pointer = cacheBuffer.contents().bindMemory(
            to: UInt16.self,
            capacity: plan.cacheCount
        )
        return Array(UnsafeBufferPointer(start: pointer, count: plan.cacheCount))
    }
}
