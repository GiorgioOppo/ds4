import DS4Core
import Foundation
import Metal

extension GraphContext {
    func glm53Dispatch(_ name: String, _ arguments: [UInt32], _ buffers: [GPUTensor],
                       groups: MTLSize, threads: Int = 128, shared: Int = 0) throws {
        let p = try rt.pipeline(name)
        guard p.threadExecutionWidth == 32, threads <= p.maxTotalThreadsPerThreadgroup,
              shared <= rt.device.maxThreadgroupMemoryLength else {
            throw MetalError.unsupported("GLM 5.3 kernel dispatch exceeds device limits: \(name)")
        }
        let e = encoder
        e.setComputePipelineState(p)
        arguments.withUnsafeBytes { e.setBytes($0.baseAddress!, length: $0.count, index: 0) }
        for (i, b) in buffers.enumerated() { e.setBuffer(b.buffer, offset: b.byteOffset, index: i + 1) }
        if shared > 0 { e.setThreadgroupMemoryLength((shared + 15) & ~15, index: 0) }
        e.dispatchThreadgroups(groups, threadsPerThreadgroup: .init(width: threads, height: 1, depth: 1))
    }

    func glm53Projection(_ weight: GPUTensor, type: UInt32, input: GPUTensor,
                          output: GPUTensor, columns: Int, width: Int,
                          rows: Int, heads: Int = 1) throws {
        guard let bytes = GGUF.tensorNBytes(type: type, elements: UInt64(columns)),
              weight.byteLength >= Int(bytes) * width * heads,
              input.count >= columns * rows * heads, output.count >= width * rows * heads else {
            throw MetalError.unsupported("Invalid GLM 5.3 projection bounds")
        }
        if heads == 1 {
            if rows > 1 && [1, 8, 12].contains(type) {
                switch type {
                case 1: try encodeMMDenseF16(weight: weight, act: input, actBase: 0, out: output, inDim: columns, outDim: width, nTok: rows)
                case 8: try encodeMMDenseQ8(weight: weight, act: input, actBase: 0, out: output, inDim: columns, outDim: width, nTok: rows)
                default: try encodeMMDenseQ4K(weight: weight, act: input, actBase: 0, out: output, inDim: columns, outDim: width, nTok: rows)
                }
                return
            }
            if type == 30 {
                if rows <= 8 {
                    try glm53Dispatch("kernel_glm53_mul_mv_bf16_f32", [UInt32(columns), UInt32(width), UInt32(rows)],
                        [weight, input, output], groups: .init(width: (width + 3) / 4, height: rows, depth: 1))
                } else {
                    let boundary = width % 64 != 0 || rows % 32 != 0
                    let pipeline = try rt.mulMMPipeline("kernel_glm53_mul_mm_bf16_f32", bcInp: columns % 32 != 0, bcOut: boundary)
                    let args = MetalRuntime.mulMMArgs(inDim: columns, outDim: width, nTok: rows, rowBytes: bytes)
                    let e = encoder
                    e.setComputePipelineState(pipeline)
                    args.withUnsafeBytes { e.setBytes($0.baseAddress!, length: $0.count, index: 0) }
                    for (i, tensor) in [weight, input, output].enumerated() {
                        e.setBuffer(tensor.buffer, offset: tensor.byteOffset, index: i + 1)
                    }
                    e.setThreadgroupMemoryLength(boundary ? 8192 : 6144, index: 0)
                    e.dispatchThreadgroups(.init(width: (rows + 31) / 32, height: (width + 63) / 64, depth: 1),
                                           threadsPerThreadgroup: .init(width: 128, height: 1, depth: 1))
                }
                return
            }
            if rows == 1 && type == 8 {
                try matmulQ8_0(weight: weight, x: input, out: output, inDim: columns, outDim: width); return
            }
            if rows == 1 && type == 12 {
                try matmulQ4_K(weight: weight, x: input, out: output, inDim: columns, outDim: width); return
            }
        }
        try glm53Dispatch("kernel_glm53_grouped_projection",
            [type, UInt32(columns), UInt32(width), UInt32(rows), UInt32(heads), UInt32(bytes)],
            [weight, input, output], groups: .init(width: (width + 3) / 4, height: rows, depth: heads))
    }

    func glm53KDA(q: GPUTensor, k: GPUTensor, v: GPUTensor,
                   decay: GPUTensor, beta: GPUTensor, gate: GPUTensor,
                   qConv: GPUTensor, kConv: GPUTensor, vConv: GPUTensor,
                   aLog: GPUTensor, dtBias: GPUTensor, norm: GPUTensor,
                   convolutionState: GPUTensor, recurrentState: GPUTensor,
                   out: GPUTensor, heads: Int = 64, rows: Int) throws {
        guard rows > 0, heads > 0, heads <= 64,
              [q, k, v, decay, gate, out].allSatisfy({ $0.count >= rows * heads * 128 }),
              beta.count >= rows * heads, convolutionState.count >= 9 * heads * 128,
              recurrentState.count >= heads * 128 * 128,
              [qConv, kConv, vConv].allSatisfy({ $0.count >= heads * 128 * 4 }),
              aLog.count >= heads, dtBias.count >= heads * 128, norm.count >= 128 else {
            throw MetalError.unsupported("GLM 5.3 KDA buffer bounds")
        }
        let args: [UInt32] = [UInt32(heads), UInt32(rows), Float(-5).bitPattern, Float(1e-5).bitPattern]
        if rows == 1 {
            try glm53Dispatch("kernel_glm53_kda_decode", args,
                [q, k, v, decay, beta, gate, qConv, kConv, vConv, aLog, dtBias, norm,
                 convolutionState, recurrentState, out],
                groups: .init(width: 1, height: heads, depth: 1), shared: 656 * 4)
        } else {
            try glm53Dispatch("kernel_glm53_kda_prefill_prepare", args,
                [q, k, v, decay, qConv, kConv, vConv, aLog, dtBias, convolutionState],
                groups: .init(width: heads, height: 1, depth: 1), shared: 264 * 4)
            try glm53Dispatch("kernel_glm53_kda_prefill_recurrence", args,
                [q, k, v, decay, beta, recurrentState, out],
                groups: .init(width: heads, height: 32, depth: 1))
            try glm53Dispatch("kernel_glm53_kda_prefill_output", args, [out, gate, norm],
                groups: .init(width: rows, height: heads, depth: 1), shared: 16)
        }
    }

    func glm53PoolUpdate(rawKeys: GPUTensor, gates: GPUTensor,
                           norm: GPUTensor, bias: GPUTensor, ape: GPUTensor,
                           cache: GPUTensor, tailKeys: GPUTensor, tailGates: GPUTensor,
                           position: Int, rows: Int, capacity: Int) throws {
        guard position >= 0, rows > 0, capacity >= rows, position <= capacity - rows,
              rawKeys.count >= rows * 128, gates.count >= rows * 128,
              norm.count >= 128, bias.count >= 128, ape.byteLength >= 512 * 2,
              cache.byteLength >= ((capacity + 3) / 4) * 128 * 2,
              tailKeys.count >= 512, tailGates.count >= 512 else {
            throw MetalError.unsupported("GLM 5.3 pool update bounds")
        }
        func dispatch(offset: Int, count: Int) throws {
            let keySlice = rawKeys.subview(byteOffset: offset * 128 * 4,
                byteLength: count * 128 * 4, count: count * 128)
            let gateSlice = gates.subview(byteOffset: offset * 128 * 4,
                byteLength: count * 128 * 4, count: count * 128)
            try glm53Dispatch("kernel_glm53_indexer_pool_update",
                [UInt32(count), UInt32(position + offset), UInt32(capacity), 128, 4, 1, Float(1e-6).bitPattern],
                [keySlice, gateSlice, norm, bias, ape, cache, tailKeys, tailGates],
                groups: .init(width: ((position + offset) % 4 + count + 3) / 4, height: 1, depth: 1), shared: 520 * 4)
        }
        // Complete the previous partial pool before a later threadgroup can
        // overwrite its tail. Separate dispatches preserve the data dependency.
        let leading = position % 4
        let first = leading == 0 ? 0 : min(rows, 4 - leading)
        if first > 0 { try dispatch(offset: 0, count: first) }
        if first < rows { try dispatch(offset: first, count: rows - first) }
    }

    func glm53Rows(_ name: String, width: Int, rows: Int, buffers: [GPUTensor], clamp: Float = 10) throws {
        // These kernels use thread_position_in_grid uint2 and therefore the
        // x workgroup spans only columns; y indexes independent rows.
        try glm53Dispatch(name, [UInt32(width), UInt32(rows), clamp.bitPattern], buffers,
            groups: .init(width: (width + 127) / 128, height: rows, depth: 1))
    }
}
