import DS4Core
import Foundation

/// Schema is validated before any GPU allocation; expert matrices remain
/// descriptors and are mapped only one selected expert at a time.
struct GLM53Weights {
    struct Weight {
        let tensor: GGUFModel.Tensor
        var type: UInt32 { tensor.type }
        var columns: Int { Int(tensor.dims[0]) }
        var rows: Int { Int(tensor.dims[1]) }
        func map(_ runtime: MetalRuntime, model: GGUFModel, expert: Int? = nil) throws -> GPUTensor {
            let parts = tensor.dims.count == 3 ? Int(tensor.dims[2]) : 1
            let bytes = Int(tensor.bytes) / (expert == nil ? 1 : parts)
            let offset = Int(tensor.absOffset) + (expert ?? 0) * bytes
            guard bytes > 0, bytes <= runtime.device.maxBufferLength,
                  expert == nil || (expert! >= 0 && expert! < parts) else {
                throw GLM53ConfigurationError.invalidTensor(tensor.name, expected: "bounded mapped tensor")
            }
            return try .mappedNoCopy(runtime, ptr: model.mapBase + offset,
                byteLength: bytes, elementCount: expert == nil ? Int(tensor.elements) : columns * rows)
        }
    }
    let embedding, output, outputNorm: Weight
    let layers: [[String: Weight]]

    init(model: GGUFModel) throws {
        let dense: Set<UInt32> = [0, 1, 8, 12, 30]
        let expert: Set<UInt32> = [8, 10, 11, 12, 13, 14, 16, 30]
        func require(_ name: String, _ dims: [Int], types: Set<UInt32>) throws -> Weight {
            guard let t = model.findTensor(name), t.dims == dims.map(UInt64.init),
                  types.contains(t.type), t.bytes <= UInt64(Int.max),
                  t.absOffset % (t.type == 0 ? 4 : 2) == 0,
                  let expected = GGUF.tensorNBytes(type: t.type, elements: t.elements),
                  expected == t.bytes else {
                throw GLM53ConfigurationError.invalidTensor(name, expected: "dimensions \(dims), GGUF type in \(types.sorted())")
            }
            return Weight(tensor: t)
        }
        embedding = try require("token_embd.weight", [4096, 154880], types: dense)
        output = try require("output.weight", [4096, 154880], types: dense)
        outputNorm = try require("output_norm.weight", [4096], types: [0])
        var all: [[String: Weight]] = []
        for layer in 0..<45 {
            let prefix = "blk.\(layer)."
            var weights: [String: Weight] = [:]
            func add(_ name: String, _ dims: [Int], _ types: Set<UInt32> = dense) throws {
                weights[name] = try require(prefix + name, dims, types: types)
            }
            for part in ["attn", "ffn"] {
                try add(part + "_norm.weight", [4096], [0])
                try add("hc_" + part + "_fn.weight", [16384, 24], [30])
                try add("hc_" + part + "_scale.weight", [3], [0])
                try add("hc_" + part + "_base.weight", [24], [0])
            }
            if GLM53Configuration.isKDALayer(layer) {
                for n in ["q", "k", "v"] {
                    try add("kda_" + n + ".weight", [4096, 8192])
                    try add("kda_" + n + "_conv.weight", [4, 1, 8192], [0])
                }
                for n in ["f", "g"] {
                    try add("kda_" + n + "_a.weight", [4096, 128])
                    try add("kda_" + n + "_b.weight", [128, 8192])
                }
                try add("kda_beta.weight", [4096, 64])
                try add("kda_dt_bias.weight", [8192], [0])
                try add("kda_a_log.weight", [64], [0])
                try add("kda_o_norm.weight", [128], [0])
                try add("kda_output.weight", [8192, 4096])
            } else {
                try add("attn_q_a.weight", [4096, 1536])
                try add("attn_q_a_norm.weight", [1536], [0])
                try add("attn_q_b.weight", [1536, 16384])
                try add("attn_kv_a_mqa.weight", [4096, 512])
                try add("attn_kv_a_norm.weight", [512], [0])
                try add("attn_k_b.weight", [256, 512, 64])
                try add("attn_v_b.weight", [512, 256, 64])
                try add("attn_output.weight", [16384, 4096])
                try add("indexer.attn_q_b.weight", [1536, 4096])
                try add("indexer.attn_k.weight", [4096, 128])
                try add("indexer.k_norm.weight", [128], [0])
                try add("indexer.k_norm.bias", [128], [0])
                try add("indexer.proj.weight", [4096, 32])
                try add("indexer.pool_ape.weight", [128, 4], [30])
                try add("indexer.pool_gate.weight", [4096, 128], [30])
            }
            if layer < 3 {
                for n in ["gate", "up"] { try add("ffn_" + n + ".weight", [4096, 12288]) }
                try add("ffn_down.weight", [12288, 4096])
            } else {
                try add("ffn_gate_inp.weight", [4096, 288], [0])
                try add("exp_probs_b.bias", [288], [0])
                for n in ["gate", "up"] {
                    try add("ffn_" + n + "_exps.weight", [4096, 2048, 288], expert)
                    try add("ffn_" + n + "_shexp.weight", [4096, 2048])
                }
                try add("ffn_down_exps.weight", [2048, 4096, 288], expert)
                try add("ffn_down_shexp.weight", [2048, 4096])
                guard weights["ffn_gate_exps.weight"]!.type == weights["ffn_up_exps.weight"]!.type else {
                    throw GLM53ConfigurationError.invalidTensor(prefix + "ffn_gate/up_exps", expected: "matching quantization")
                }
            }
            all.append(weights)
        }
        layers = all
    }
}
