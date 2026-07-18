import DS4Core
import Foundation
import Metal

/// CPU correctness oracle for GLM's tail RoPE — the faithful port of upstream
/// `rope_tail_ext_inplace` with the GLM shape constants (freq base 8e6,
/// freq_scale 1, ext_factor 0, no YaRN): adjacent pairs, iterative
/// `theta *= freq_base^(-2/n_rot)` exactly like the C loop. `inverse` flips
/// the sine sign (upstream's un-rotate path).
public enum GLM52RopeTailReference {
    public static let frequencyBase: Float = 8_000_000

    public static func rotate(values: [Float],
                              headCount: Int,
                              headDimension: Int,
                              rotationDimension: Int,
                              position: Int,
                              inverse: Bool = false) throws -> [Float] {
        guard headCount > 0, rotationDimension > 0,
              rotationDimension.isMultiple(of: 2),
              rotationDimension <= headDimension,
              values.count == headCount * headDimension,
              position >= 0, position <= Int(UInt32.max) else {
            throw MetalError.unsupported(
                "GLM 5.2 rope tail expects [\(headCount)][\(headDimension)] "
                + "values with an even tail of \(rotationDimension)")
        }
        let nope = headDimension - rotationDimension
        let thetaScale = pow(frequencyBase, -2 / Float(rotationDimension))
        let sinSign: Float = inverse ? -1 : 1
        var output = values
        for head in 0..<headCount {
            let tail = head * headDimension + nope
            var theta = Float(position)
            for i in stride(from: 0, to: rotationDimension, by: 2) {
                let c = cos(theta)
                let s = sinSign * sin(theta)
                let x0 = output[tail + i]
                let x1 = output[tail + i + 1]
                output[tail + i] = x0 * c - x1 * s
                output[tail + i + 1] = x0 * s + x1 * c
                theta *= thetaScale
            }
        }
        return output
    }
}

extension MetalRuntime {
    /// Dispatch the tail RoPE kernel (forward only): one thread per pair,
    /// closed-form theta like the indexer-key store. Validation wrapper —
    /// the persistent graph will rotate resident tensors instead.
    public func glm52RopeTail(values: [Float],
                              headCount: Int,
                              headDimension: Int,
                              rotationDimension: Int,
                              position: Int) throws -> [Float] {
        // Reuse the oracle's validation so both paths reject identically.
        _ = try GLM52RopeTailReference.rotate(
            values: values, headCount: headCount,
            headDimension: headDimension,
            rotationDimension: rotationDimension, position: 0)

        guard let buffer = device.makeBuffer(
            bytes: values, length: values.count * MemoryLayout<Float>.stride,
            options: .storageModeShared) else {
            throw MetalError.bufferAlloc
        }
        var arguments = [UInt32](repeating: 0, count: 4)
        arguments[0] = UInt32(headCount)
        arguments[1] = UInt32(headDimension)
        arguments[2] = UInt32(rotationDimension)
        arguments[3] = UInt32(position)

        let pipeline = try pipeline("kernel_glm52_rope_tail_f32")
        guard let commandBuffer = queue.makeCommandBuffer(),
              let encoder = commandBuffer.makeComputeCommandEncoder() else {
            throw MetalError.bufferAlloc
        }
        encoder.setComputePipelineState(pipeline)
        arguments.withUnsafeBytes {
            encoder.setBytes($0.baseAddress!, length: $0.count, index: 0)
        }
        encoder.setBuffer(buffer, offset: 0, index: 1)
        let pairs = headCount * (rotationDimension / 2)
        let width = min(256, pipeline.maxTotalThreadsPerThreadgroup)
        encoder.dispatchThreadgroups(
            MTLSize(width: (pairs + width - 1) / width, height: 1, depth: 1),
            threadsPerThreadgroup: MTLSize(width: width, height: 1, depth: 1))
        encoder.endEncoding()
        commandBuffer.commit()
        commandBuffer.waitUntilCompleted()
        if let error = commandBuffer.error { throw error }

        let pointer = buffer.contents().bindMemory(
            to: Float.self, capacity: values.count)
        return Array(UnsafeBufferPointer(start: pointer, count: values.count))
    }
}
