import Foundation
import Metal

public struct GLM52RouterOutput: Sendable, Equatable {
    public let selected: [Int32]
    public let weights: [Float]
    public let probabilities: [Float]

    public init(selected: [Int32], weights: [Float], probabilities: [Float]) {
        self.selected = selected
        self.weights = weights
        self.probabilities = probabilities
    }
}

/// Scalar oracle for the GLM 5.2 sigmoid router.
///
/// Selection is based on `sigmoid(logit) + bias`; the normalized route weight
/// deliberately uses the unbiased sigmoid value. Exact score ties prefer the
/// lower expert id. Keeping this reference independent of Metal gives later
/// graph/streaming work a stable correctness boundary.
public enum GLM52RouterReference {
    public static let expertCount = 256
    public static let expertsUsed = 8
    public static let expertWeightScale: Float = 2.5

    @inline(__always)
    public static func sigmoid(_ value: Float) -> Float {
        if value >= 0 {
            let e = Foundation.exp(-value)
            return 1 / (1 + e)
        }
        let e = Foundation.exp(value)
        return e / (1 + e)
    }

    public static func route(logits: [Float], bias: [Float]) throws
        -> GLM52RouterOutput {
        guard logits.count == expertCount else {
            throw MetalError.unsupported(
                "GLM 5.2 router logits count \(logits.count); expected \(expertCount)"
            )
        }
        guard bias.count == expertCount else {
            throw MetalError.unsupported(
                "GLM 5.2 router bias count \(bias.count); expected \(expertCount)"
            )
        }

        let probabilities = logits.map(sigmoid)
        let selected = (0..<expertCount).sorted { lhs, rhs in
            let a = probabilities[lhs] + bias[lhs]
            let b = probabilities[rhs] + bias[rhs]
            return a == b ? lhs < rhs : a > b
        }.prefix(expertsUsed).map(Int32.init)

        var sum: Float = 0
        for expert in selected { sum += probabilities[Int(expert)] }
        sum = max(sum, 6.103515625e-5)
        let weights = selected.map {
            probabilities[Int($0)] / sum * expertWeightScale
        }
        return GLM52RouterOutput(
            selected: selected,
            weights: weights,
            probabilities: probabilities
        )
    }
}

extension MetalRuntime {
    /// Execute the architecture-exact GLM 5.2 router for one token.
    public func glm52Route(logits: [Float], bias: [Float]) throws
        -> GLM52RouterOutput {
        guard logits.count == GLM52RouterReference.expertCount else {
            throw MetalError.unsupported(
                "GLM 5.2 router logits count \(logits.count); expected 256"
            )
        }
        guard bias.count == GLM52RouterReference.expertCount else {
            throw MetalError.unsupported(
                "GLM 5.2 router bias count \(bias.count); expected 256"
            )
        }

        var args = [UInt8](repeating: 0, count: 16)
        func writeU32(_ offset: Int, _ value: UInt32) {
            withUnsafeBytes(of: value.littleEndian) { bytes in
                for index in 0..<4 { args[offset + index] = bytes[index] }
            }
        }
        writeU32(0, UInt32(GLM52RouterReference.expertCount))
        writeU32(4, UInt32(GLM52RouterReference.expertsUsed))
        writeU32(8, GLM52RouterReference.expertWeightScale.bitPattern)
        writeU32(12, 0)

        guard let logitsBuffer = device.makeBuffer(
                  bytes: logits,
                  length: logits.count * MemoryLayout<Float>.stride,
                  options: .storageModeShared
              ),
              let biasBuffer = device.makeBuffer(
                  bytes: bias,
                  length: bias.count * MemoryLayout<Float>.stride,
                  options: .storageModeShared
              ),
              let selectedBuffer = device.makeBuffer(
                  length: GLM52RouterReference.expertsUsed * MemoryLayout<Int32>.stride,
                  options: .storageModeShared
              ),
              let weightsBuffer = device.makeBuffer(
                  length: GLM52RouterReference.expertsUsed * MemoryLayout<Float>.stride,
                  options: .storageModeShared
              ),
              let probabilitiesBuffer = device.makeBuffer(
                  length: GLM52RouterReference.expertCount * MemoryLayout<Float>.stride,
                  options: .storageModeShared
              ) else {
            throw MetalError.bufferAlloc
        }

        let pipeline = try pipeline("kernel_glm52_router_select")
        guard pipeline.maxTotalThreadsPerThreadgroup >= 256 else {
            throw MetalError.unsupported(
                "GLM 5.2 router requires 256 threads; GPU supports " +
                "\(pipeline.maxTotalThreadsPerThreadgroup)"
            )
        }
        guard let commandBuffer = queue.makeCommandBuffer(),
              let encoder = commandBuffer.makeComputeCommandEncoder() else {
            throw MetalError.bufferAlloc
        }

        encoder.setComputePipelineState(pipeline)
        args.withUnsafeBytes {
            encoder.setBytes($0.baseAddress!, length: args.count, index: 0)
        }
        encoder.setBuffer(logitsBuffer, offset: 0, index: 1)
        encoder.setBuffer(biasBuffer, offset: 0, index: 2)
        encoder.setBuffer(selectedBuffer, offset: 0, index: 3)
        encoder.setBuffer(weightsBuffer, offset: 0, index: 4)
        encoder.setBuffer(probabilitiesBuffer, offset: 0, index: 5)
        encoder.setThreadgroupMemoryLength(
            256 * MemoryLayout<Float>.stride + 256 * MemoryLayout<Int32>.stride,
            index: 0
        )
        encoder.dispatchThreadgroups(
            MTLSize(width: 1, height: 1, depth: 1),
            threadsPerThreadgroup: MTLSize(width: 256, height: 1, depth: 1)
        )
        encoder.endEncoding()
        commandBuffer.commit()
        commandBuffer.waitUntilCompleted()
        if let error = commandBuffer.error { throw error }

        let selectedPointer = selectedBuffer.contents().bindMemory(
            to: Int32.self,
            capacity: GLM52RouterReference.expertsUsed
        )
        let weightsPointer = weightsBuffer.contents().bindMemory(
            to: Float.self,
            capacity: GLM52RouterReference.expertsUsed
        )
        let probabilitiesPointer = probabilitiesBuffer.contents().bindMemory(
            to: Float.self,
            capacity: GLM52RouterReference.expertCount
        )
        return GLM52RouterOutput(
            selected: Array(UnsafeBufferPointer(
                start: selectedPointer,
                count: GLM52RouterReference.expertsUsed
            )),
            weights: Array(UnsafeBufferPointer(
                start: weightsPointer,
                count: GLM52RouterReference.expertsUsed
            )),
            probabilities: Array(UnsafeBufferPointer(
                start: probabilitiesPointer,
                count: GLM52RouterReference.expertCount
            ))
        )
    }
}
