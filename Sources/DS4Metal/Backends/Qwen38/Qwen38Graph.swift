import Foundation
import Metal
import DS4Core

extension Qwen38Model {
    func dispatch(_ graph: GraphContext, _ name: String, _ words: [UInt32], _ buffers: [GPUTensor],
                  grid: (Int, Int, Int), threads: Int = 128, quant: UInt32? = nil) throws {
        let pipeline: MTLComputePipelineState
        if let quant {
            let key = name + ":" + String(quant)
            if let cached = specializedPipelines[key] { pipeline = cached }
            else {
                let constants = MTLFunctionConstantValues()
                var type = quant, tail: UInt32 = 0, addresses = false
                constants.setConstantValue(&type, type: .uint, index: 900)
                constants.setConstantValue(&tail, type: .uint, index: 905)
                constants.setConstantValue(&addresses, type: .bool, index: 906)
                let function = try rt.library.makeFunction(name: name, constantValues: constants)
                pipeline = try rt.device.makeComputePipelineState(function: function)
                specializedPipelines[key] = pipeline
            }
        } else if name == "kernel_qwen4_moe_mid" || name == "kernel_qwen4_moe_down" {
            // Even optional function constants require Metal's specialized
            // function API. The plain makeFunction(name:) returns a template
            // that cannot be used to create a compute pipeline.
            let key = "\(name):\(words[4]):\(words[2])"
            if let cached = specializedPipelines[key] { pipeline = cached }
            else {
                let constants = MTLFunctionConstantValues()
                var type = words[4], sharedType: UInt32 = 0, dimension = words[2], rows: UInt32 = 2
                var addresses = false
                constants.setConstantValue(&type, type: .uint, index: 901)
                constants.setConstantValue(&sharedType, type: .uint, index: 902)
                constants.setConstantValue(&dimension, type: .uint, index: 903)
                constants.setConstantValue(&rows, type: .uint, index: 904)
                constants.setConstantValue(&addresses, type: .bool, index: 906)
                let function = try rt.library.makeFunction(name: name, constantValues: constants)
                pipeline = try rt.device.makeComputePipelineState(function: function)
                specializedPipelines[key] = pipeline
            }
        } else { pipeline = try rt.pipeline(name) }
        guard threads <= pipeline.maxTotalThreadsPerThreadgroup else { throw MetalError.unsupported("Qwen kernel \(name) needs \(threads) threads") }
        let e = graph.encoder
        e.setComputePipelineState(pipeline)
        words.withUnsafeBytes { e.setBytes($0.baseAddress!, length: $0.count, index: 0) }
        for (i, b) in buffers.enumerated() { e.setBuffer(b.buffer, offset: b.byteOffset, index: i + 1) }
        e.dispatchThreadgroups(MTLSize(width: grid.0, height: grid.1, depth: grid.2), threadsPerThreadgroup: MTLSize(width: threads, height: 1, depth: 1))
    }

    func finish(_ graph: GraphContext) throws {
        graph.commit()
        if let error = graph.lastError { throw SwiftModelDecoderError.gpu(error.localizedDescription) }
    }

    func project(_ graph: GraphContext, _ name: String, _ x: GPUTensor, _ out: GPUTensor, _ count: Int) throws {
        let t = tensor(name), w = try weight(name)
        let input = Int(t.dims[0]), rows = Int(t.dims[1]), row = try Qwen38Weights.rowBytes(type: t.type, width: input)
        if count > 8 && t.type == 8 {
            try graph.encodeMMDenseQ8(weight: w, act: x, actBase: 0, out: out, inDim: input, outDim: rows, nTok: count)
        } else if count > 8 && t.type == 1 {
            try graph.encodeMMDenseF16(weight: w, act: x, actBase: 0, out: out, inDim: input, outDim: rows, nTok: count)
        } else if count > 8 && t.type == 0 {
            // Preserve full-F32 router/HC projections in the upstream prefill
            // policy. The generic half-operand GEMM changes routing decisions.
            try dispatch(graph, "kernel_qwen4_dense_mm", [UInt32(count), UInt32(input), UInt32(rows), t.type, UInt32(row), 1, 0, 0], [w, x, out], grid: ((rows + 31) / 32, (count + 31) / 32, 1))
        } else if count == 1 && t.type == 8 {
            try graph.matmulQ8_0(weight: w, x: x, out: out, inDim: input, outDim: rows)
        } else if count == 1 && t.type == 1 {
            try graph.matmulF16(weight: w, x: x, out: out, inDim: input, outDim: rows)
        } else {
            let args: [UInt32] = [UInt32(count), UInt32(input), 1, 0, UInt32(rows), 0, 0, 0, t.type, 0, 0, 0, UInt32(row), 0, 0, 0]
            try dispatch(graph, "kernel_qwen4_multi_gemv", args, [x, w, w, w, w, out, out, out, out], grid: ((rows + 7) / 8, count, 1))
        }
    }

    func mix(_ graph: GraphContext, prefix: String, residual: GPUTensor, count: Int, inject: Bool = true) throws {
        let norm = try weight(prefix + "_norm.weight"), down = prefix + "_down.weight", upName = prefix + "_up.weight"
        let up = try weight(upName), upType = tensor(upName).type
        let injection = inject ? try weight(prefix + "_inject.weight") : norm
        let injectionType = inject ? tensor(prefix + "_inject.weight").type : UInt32(1)
        func suffix(_ t: UInt32) -> String { t == 0 ? "f32" : t == 8 ? "q8" : "f16" }
        try dispatch(graph, "kernel_qwen4_hc_norm_" + suffix(injectionType),
                     [UInt32(count), UInt32(c.embedding), UInt32(c.hcCount), inject ? UInt32(c.hcCount) : 0, c.epsilon.bitPattern, 0, 0, 0],
                     [residual, norm, injection, s("xn"), s("inj")], grid: (c.hcCount * 8, count, 1))
        try project(graph, down, s("xn"), s("lo"), count)
        if count > 8 && upType != 8 {
            let args = [UInt32(count), UInt32(c.embedding), UInt32(c.hcCount), UInt32(c.hcRank)]
            try dispatch(graph, "kernel_qwen4_hc_lo_act", args, [s("lo"), s("loAct")], grid: ((count * c.hcRank + 255) / 256, 1, 1), threads: 256)
            try project(graph, upName, s("loAct"), s("hcUp"), count)
            try dispatch(graph, "kernel_qwen4_hc_mix_rows", args, [s("hcUp"), s("xn"), s("mixed")], grid: ((c.embedding + 255) / 256, count, 1), threads: 256)
        } else {
            try dispatch(graph, "kernel_qwen4_hc_gate_mix_" + suffix(upType),
                         [UInt32(count), UInt32(c.embedding), UInt32(c.hcCount), UInt32(c.hcRank)],
                         [s("xn"), s("lo"), up, s("mixed")], grid: ((c.embedding + 3) / 4, count, 1))
        }
    }

    func ple(_ graph: GraphContext, prefix p: String, count: Int) throws {
        try project(graph, p + "ple_key.weight", s("pleEmbedding"), s("pleKey"), count)
        try project(graph, p + "ple_value.weight", s("pleEmbedding"), s("pleValue"), count)
        try dispatch(graph, "kernel_qwen4_ple_gate", [UInt32(count), UInt32(c.embedding), UInt32(c.hcCount), c.epsilon.bitPattern],
                     [s("R"), s("pleKey"), s("pleValue"), try weight(p + "ple_norm_key.weight"), try weight(p + "ple_norm_query.weight"), try weight(p + "ple_norm_conv.weight"), s("pleGated"), s("pleNormed")], grid: (count, 1, 1), threads: 256)
        let conv = p + "ple_conv1d.weight"
        try dispatch(graph, "kernel_qwen4_ple_conv", [UInt32(count), UInt32(c.hcDimension), 4, 3, tensor(conv).type == 1 ? 1 : 0, .max, .max, 0],
                     [s("R"), s("pleGated"), s("pleNormed"), s("pleHistory"), try weight(conv), s("pleHistory"), s("pleHistory")], grid: ((c.hcDimension + 255) / 256, 1, 1), threads: 256)
    }

    func linear(_ graph: GraphContext, layer: Int, count: Int) throws {
        let p = "blk.\(layer).", state = states[layer]
        for (name, out) in [("attn_qkv.weight", "qkv"), ("attn_gate.weight", "z"), ("ssm_alpha.weight", "alpha"), ("ssm_beta.weight", "beta")] {
            try project(graph, p + name, s("mixed"), s(out), count)
        }
        try dispatch(graph, "kernel_qwen4_conv_stream", [UInt32(count), UInt32(c.linearQKVDimension), 4, 1],
                     [s("qkv"), state["conv"]!, try weight(p + "ssm_conv1d.weight")], grid: ((c.linearQKVDimension + 255) / 256, 1, 1), threads: 256)
        let shape: [UInt32] = [UInt32(count), UInt32(c.linearKHeads), UInt32(c.linearVHeads), UInt32(c.linearDimension)]
        try dispatch(graph, "kernel_qwen4_gdn_prep", shape,
                     [s("qkv"), s("alpha"), s("beta"), try weight(p + "ssm_a"), try weight(p + "ssm_dt.bias")], grid: (c.linearKHeads, count, 1), threads: 32)
        let optimized = c.linearDimension == 128
        try dispatch(graph, optimized ? "kernel_qwen4_gdn_scan_r4" : "kernel_qwen4_gdn_scan", shape + [.max, .max, 0, 0],
                     [s("qkv"), s("alpha"), s("beta"), state["gdn"]!, s("linearOut"), state["gdn"]!, state["gdn"]!],
                     grid: (optimized ? c.linearDimension / 4 : c.linearDimension, c.linearVHeads, 1), threads: 32)
        try dispatch(graph, "kernel_qwen4_gdn_out", [UInt32(count), UInt32(c.linearVHeads), UInt32(c.linearDimension), c.epsilon.bitPattern],
                     [s("linearOut"), s("z"), try weight(p + "ssm_norm.weight")], grid: (c.linearVHeads, count, 1), threads: 32)
        try project(graph, p + "ssm_out.weight", s("linearOut"), s("blk"), count)
    }

    func forward(_ tokens: [Int], cancelled: @Sendable () -> Bool) throws {
        try stage(tokens)
        let T = tokens.count
        let graph = GraphContext(rt)
        try graph.begin()
        // A thrown I/O/pipeline/cancellation error still drains submitted and
        // encoded work before buffers/state can be reset or released.
        defer { graph.commit() }
        for layer in 0..<c.layers {
            if cancelled() { throw SwiftModelDecoderError.cancelled }
            let p = "blk.\(layer)."
            if layer == 1 { try ple(graph, prefix: p, count: T) }
            try mix(graph, prefix: p + "hc_attn", residual: s("R"), count: T)
            if c.isLinear(layer: layer) { try linear(graph, layer: layer, count: T) }
            else { try attention(graph, layer: layer, count: T) }
            try dispatch(graph, "kernel_qwen4_hc_combine", [UInt32(T), UInt32(c.embedding), UInt32(c.hcCount), 0],
                         [s("R"), s("blk"), s("inj")], grid: ((c.embedding + 255) / 256, T, 1), threads: 256)
            try mix(graph, prefix: p + "hc_ffn", residual: s("R"), count: T)
            try moe(graph, prefix: p, count: T, cancelled: cancelled)
        }
        let last = s("R").subview(byteOffset: (T - 1) * c.hcDimension * 4, byteLength: c.hcDimension * 4, count: c.hcDimension)
        try mix(graph, prefix: "output_hc", residual: last, count: 1, inject: false)
        try project(graph, "output.weight", s("mixed"), s("logits"), 1)
        try finish(graph)
        if cancelled() { throw SwiftModelDecoderError.cancelled }
    }
}
