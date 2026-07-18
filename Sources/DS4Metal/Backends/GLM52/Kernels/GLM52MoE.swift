import DS4Core
import Foundation
import Metal

// Validation wrappers for the routed expert kernels: each dispatch reads the
// K-quant weight bytes exactly as stored in the GGUF (one thread per output
// row, reference element pairing) and is compared against
// GLM52FFNCPUReference on the dequantized weights. Not a decode loop and not
// a performance path — the tuned per-quant families come later.

/// One routed expert's quantized weights: gate/up share a K-quant type, down
/// may use its own. Byte layouts are the GGUF expert slices the payload
/// reader delivers.
public struct GLM52QuantizedExpert: Sendable {
    public let gateUpType: UInt32
    public let downType: UInt32
    public let gate: [UInt8]
    public let up: [UInt8]
    public let down: [UInt8]

    public init(gateUpType: UInt32, downType: UInt32,
                gate: [UInt8], up: [UInt8], down: [UInt8]) {
        self.gateUpType = gateUpType
        self.downType = downType
        self.gate = gate
        self.up = up
        self.down = down
    }
}

extension MetalRuntime {
    /// Quantized row bytes for `width` elements (multiple of the type's block
    /// size), or nil for a type outside the GLM FFN contract: Q8_0 for
    /// dense/shared/output-head weights, the four K-quants for routed experts.
    static func glm52KQuantRowBytes(type: UInt32, width: Int) -> Int? {
        guard width > 0,
              let info = GGUF.typeInfo(type),
              width.isMultiple(of: Int(info.blockElems)),
              [GLM52TensorSchema.q8_0, GLM52TensorSchema.q2_K,
               GLM52TensorSchema.q4_K, GLM52TensorSchema.q5_K,
               GLM52TensorSchema.q6_K].contains(type)
        else { return nil }
        return (width / Int(info.blockElems)) * Int(info.blockBytes)
    }

    private func glm52MoEDispatch(pipelineName: String,
                                  weightType: UInt32,
                                  rowCount: Int,
                                  inputWidth: Int,
                                  routeWeight: Float,
                                  input: [Float],
                                  weightBuffers: [[UInt8]]) throws -> [Float] {
        guard let rowBytes = Self.glm52KQuantRowBytes(type: weightType,
                                                      width: inputWidth) else {
            throw MetalError.unsupported(
                "GLM 5.2 MoE weight type \(weightType) / width \(inputWidth) "
                + "is outside the routed contract")
        }
        guard rowCount > 0, input.count == inputWidth,
              weightBuffers.allSatisfy({ $0.count == rowCount * rowBytes })
        else {
            throw MetalError.unsupported(
                "GLM 5.2 MoE expects [\(rowCount)]x\(rowBytes)-byte rows and "
                + "a \(inputWidth)-wide input")
        }
        guard let inputBuffer = device.makeBuffer(
            bytes: input, length: input.count * MemoryLayout<Float>.stride,
            options: .storageModeShared) else {
            throw MetalError.bufferAlloc
        }
        var buffers: [MTLBuffer] = [inputBuffer]
        for bytes in weightBuffers {
            guard let buffer = device.makeBuffer(
                bytes: bytes, length: bytes.count,
                options: .storageModeShared) else {
                throw MetalError.bufferAlloc
            }
            buffers.append(buffer)
        }
        guard let outputBuffer = device.makeBuffer(
            length: rowCount * MemoryLayout<Float>.stride,
            options: .storageModeShared) else {
            throw MetalError.bufferAlloc
        }
        buffers.append(outputBuffer)

        var arguments = [UInt32](repeating: 0, count: 4)
        arguments[0] = weightType
        arguments[1] = UInt32(rowCount)
        arguments[2] = UInt32(inputWidth)
        arguments[3] = routeWeight.bitPattern

        let pipeline = try pipeline(pipelineName)
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
        let width = min(256, pipeline.maxTotalThreadsPerThreadgroup)
        encoder.dispatchThreadgroups(
            MTLSize(width: (rowCount + width - 1) / width, height: 1, depth: 1),
            threadsPerThreadgroup: MTLSize(width: width, height: 1, depth: 1))
        encoder.endEncoding()
        commandBuffer.commit()
        commandBuffer.waitUntilCompleted()
        if let error = commandBuffer.error { throw error }

        let pointer = outputBuffer.contents().bindMemory(
            to: Float.self, capacity: rowCount)
        return Array(UnsafeBufferPointer(start: pointer, count: rowCount))
    }

    /// One expert's fused gate/up stage:
    /// `mid[r] = silu(gate_r·x) * (up_r·x) * routeWeight`.
    public func glm52MoEPairSwiGLU(input: [Float],
                                   gateRows: [UInt8],
                                   upRows: [UInt8],
                                   weightType: UInt32,
                                   hiddenWidth: Int,
                                   routeWeight: Float) throws -> [Float] {
        try glm52MoEDispatch(
            pipelineName: "kernel_glm52_moe_pair_swiglu",
            weightType: weightType,
            rowCount: hiddenWidth,
            inputWidth: input.count,
            routeWeight: routeWeight,
            input: input,
            weightBuffers: [gateRows, upRows])
    }

    /// One expert's down projection: `out[r] = down_r · mid`.
    public func glm52MoEDown(mid: [Float],
                             downRows: [UInt8],
                             weightType: UInt32,
                             outputWidth: Int) throws -> [Float] {
        try glm52MoEDispatch(
            pipelineName: "kernel_glm52_moe_down",
            weightType: weightType,
            rowCount: outputWidth,
            inputWidth: mid.count,
            routeWeight: 1,
            input: mid,
            weightBuffers: [downRows])
    }

    /// One dense or shared-expert FFN block on quantized weights (Q8_0 in the
    /// real model): fused pair-SwiGLU (route weight 1) then down. Comparable
    /// to `GLM52FFNCPUReference.ffnBlock` on the dequantized weights.
    public func glm52FFNBlock(input: [Float],
                              gateRows: [UInt8],
                              upRows: [UInt8],
                              downRows: [UInt8],
                              weightType: UInt32,
                              hiddenWidth: Int) throws -> [Float] {
        let mid = try glm52MoEPairSwiGLU(
            input: input, gateRows: gateRows, upRows: upRows,
            weightType: weightType, hiddenWidth: hiddenWidth, routeWeight: 1)
        return try glm52MoEDown(
            mid: mid, downRows: downRows, weightType: weightType,
            outputWidth: input.count)
    }

    /// Output-head logits from an ALREADY-normalized hidden state (the RMSNorm
    /// stays with the CPU oracle in this validation path): one quantized
    /// matvec over the vocabulary rows.
    public func glm52OutputHeadLogits(normalized: [Float],
                                      headRows: [UInt8],
                                      weightType: UInt32,
                                      vocabularySize: Int) throws -> [Float] {
        try glm52MoEDown(
            mid: normalized, downRows: headRows, weightType: weightType,
            outputWidth: vocabularySize)
    }

    /// Chained validation path over the selected experts (router rank order):
    /// per expert, GPU pair-SwiGLU (route weight on the mid, upstream's
    /// association) then GPU down; contributions sum host-side. Comparable to
    /// `GLM52FFNCPUReference.routedFFN` on the dequantized weights.
    public func glm52RoutedFFN(input: [Float],
                               experts: [GLM52QuantizedExpert],
                               weights: [Float],
                               hiddenWidth: Int) throws -> [Float] {
        guard !experts.isEmpty, experts.count == weights.count else {
            throw MetalError.unsupported(
                "GLM 5.2 routed FFN expects one weight per expert")
        }
        var output = [Float](repeating: 0, count: input.count)
        for (rank, expert) in experts.enumerated() {
            let mid = try glm52MoEPairSwiGLU(
                input: input,
                gateRows: expert.gate,
                upRows: expert.up,
                weightType: expert.gateUpType,
                hiddenWidth: hiddenWidth,
                routeWeight: weights[rank])
            let contribution = try glm52MoEDown(
                mid: mid,
                downRows: expert.down,
                weightType: expert.downType,
                outputWidth: input.count)
            for i in 0..<output.count { output[i] += contribution[i] }
        }
        return output
    }
}
