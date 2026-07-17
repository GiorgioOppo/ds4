import DS4Core
import Foundation
import Metal

private struct GLM52KVLoRANormPlan {
    let tokenCount: Int
    let outputCount: Int
}

private func glm52KVLoRANormPlan(
    rawRows: [Float],
    weight: [Float]
) throws -> GLM52KVLoRANormPlan {
    let rawWidth = GLM52KVLoRANormReference.rawWidth
    guard !rawRows.isEmpty, rawRows.count.isMultiple(of: rawWidth) else {
        throw MetalError.unsupported(
            "GLM 5.2 KV-LoRA normalization requires non-empty 576-wide rows"
        )
    }
    guard weight.count == GLM52KVLoRANormReference.kvLoRAWidth else {
        throw MetalError.unsupported(
            "GLM 5.2 KV-LoRA norm weight count \(weight.count); expected 512"
        )
    }
    let tokenCount = rawRows.count / rawWidth
    guard tokenCount <= Int(UInt32.max) else {
        throw MetalError.unsupported("GLM 5.2 KV-LoRA token count exceeds UInt32")
    }
    return GLM52KVLoRANormPlan(
        tokenCount: tokenCount,
        outputCount: rawRows.count
    )
}

/// Scalar oracle for the compact-attention KV-LoRA normalization boundary.
///
/// Only columns `0..<512` participate in RMSNorm and receive the learned
/// weight. Columns `512..<576` are the raw K-RoPE payload and remain unchanged
/// until the attention cache applies its positional transform.
public enum GLM52KVLoRANormReference {
    public static let rawWidth = 576
    public static let kvLoRAWidth = 512
    public static let kRoPEWidth = 64
    public static let epsilon: Float = 1.0e-5

    public static func normalize(
        rawRows: [Float],
        weight: [Float]
    ) throws -> [Float] {
        let plan = try glm52KVLoRANormPlan(rawRows: rawRows, weight: weight)
        var output = rawRows
        for token in 0..<plan.tokenCount {
            let row = token * rawWidth
            var sumSquares: Float = 0
            for column in 0..<kvLoRAWidth {
                let value = rawRows[row + column]
                sumSquares += value * value
            }
            let inverseRMS = 1 / sqrt(
                sumSquares / Float(kvLoRAWidth) + epsilon
            )
            for column in 0..<kvLoRAWidth {
                output[row + column] = rawRows[row + column] *
                    inverseRMS * weight[column]
            }
        }
        return output
    }
}

extension MetalRuntime {
    /// Runs GLM's 512-wide KV-LoRA RMSNorm while preserving each raw RoPE tail.
    /// This validation wrapper does not make the GLM backend runnable.
    public func glm52NormalizeKVLoRA(
        rawRows: [Float],
        weight: [Float]
    ) throws -> [Float] {
        let plan = try glm52KVLoRANormPlan(rawRows: rawRows, weight: weight)
        guard let inputBuffer = device.makeBuffer(
                  bytes: rawRows,
                  length: rawRows.count * MemoryLayout<Float>.stride,
                  options: .storageModeShared
              ),
              let weightBuffer = device.makeBuffer(
                  bytes: weight,
                  length: weight.count * MemoryLayout<Float>.stride,
                  options: .storageModeShared
              ),
              let outputBuffer = device.makeBuffer(
                  length: plan.outputCount * MemoryLayout<Float>.stride,
                  options: .storageModeShared
              ) else {
            throw MetalError.bufferAlloc
        }

        var arguments = [UInt32](repeating: 0, count: 4)
        arguments[0] = UInt32(plan.tokenCount)

        let pipeline = try pipeline(
            "kernel_glm52_kv_lora_norm_cache_ready_f32"
        )
        guard pipeline.maxTotalThreadsPerThreadgroup >= 128 else {
            throw MetalError.unsupported(
                "GLM 5.2 KV-LoRA RMSNorm requires 128 threads"
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
        encoder.setBuffer(inputBuffer, offset: 0, index: 1)
        encoder.setBuffer(weightBuffer, offset: 0, index: 2)
        encoder.setBuffer(outputBuffer, offset: 0, index: 3)
        encoder.setThreadgroupMemoryLength(
            128 * MemoryLayout<Float>.stride,
            index: 0
        )
        encoder.dispatchThreadgroups(
            MTLSize(width: plan.tokenCount, height: 1, depth: 1),
            threadsPerThreadgroup: MTLSize(width: 128, height: 1, depth: 1)
        )
        encoder.endEncoding()
        commandBuffer.commit()
        commandBuffer.waitUntilCompleted()
        if let error = commandBuffer.error { throw error }

        let pointer = outputBuffer.contents().bindMemory(
            to: Float.self,
            capacity: plan.outputCount
        )
        return Array(UnsafeBufferPointer(start: pointer, count: plan.outputCount))
    }
}
