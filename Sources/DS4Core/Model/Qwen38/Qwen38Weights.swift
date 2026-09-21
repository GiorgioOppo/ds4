import Foundation

/// Validated tensor directory. Expert/PLE tensors remain file ranges, never
/// resident GPU allocations. Validation runs before constructing any GPU state.
public struct Qwen38Weights {
    public let tensors: [String: GGUFModel.Tensor]
    public subscript(_ name: String) -> GGUFModel.Tensor { tensors[name]! }

    public static func rowBytes(type: UInt32, width: Int) throws -> Int {
        let block: Int, bytes: Int
        switch type {
        case 0: block = 1; bytes = 4
        case 1, 30: block = 1; bytes = 2
        case 2: block = 32; bytes = 18
        case 8: block = 32; bytes = 34
        case 10: block = 256; bytes = 84
        case 12: block = 256; bytes = 144
        case 16: block = 256; bytes = 66
        case 39: block = 32; bytes = 17
        default: throw Qwen38Error.invalidModel("Unsupported tensor type \(type)")
        }
        guard width > 0, width % block == 0 else { throw Qwen38Error.invalidModel("Unaligned quantized row width \(width)") }
        let (result, overflow) = (width / block).multipliedReportingOverflow(by: bytes)
        guard !overflow else { throw Qwen38Error.invalidModel("Quantized row byte length overflows") }
        return result
    }

    public init(model: GGUFModel, configuration c: Qwen38Configuration) throws {
        var directory = [String: GGUFModel.Tensor]()
        for t in model.tensors {
            guard directory.updateValue(t, forKey: t.name) == nil else { throw Qwen38Error.invalidModel("Duplicate tensor \(t.name)") }
        }
        let dense: Set<UInt32> = [0, 1, 2, 8, 30], hc: Set<UInt32> = [0, 1, 8]
        let expert: Set<UInt32> = [0, 1, 8, 10, 12, 16, 39]
        func require(_ name: String, _ dims: [Int], _ types: Set<UInt32>) throws -> GGUFModel.Tensor {
            guard let t = directory[name], t.dims == dims.map(UInt64.init), types.contains(t.type) else {
                throw Qwen38Error.invalidModel("Missing or unsupported tensor \(name), expected \(dims)")
            }
            let row = try Self.rowBytes(type: t.type, width: dims[0])
            let bytes = UInt64(row) * dims.dropFirst().reduce(UInt64(1)) { $0 * UInt64($1) }
            guard t.bytes == bytes, t.absOffset <= model.size, bytes <= model.size - t.absOffset else {
                throw Qwen38Error.invalidModel("Invalid byte span for \(name)")
            }
            return t
        }
        let E = c.embedding, HC = c.hcDimension, Q = c.queryDimension
        _ = try require("token_embd.weight", [E, c.vocabulary], dense)
        guard let ng = directory["per_layer_token_embd.weight"], ng.type == 30, ng.dims.count == 2,
              ng.dims[0] == UInt64(c.pleHeadDimension), ng.dims[1] >= c.ngramRows,
              ng.dims[1] <= UInt64(Int32.max), ng.bytes == ng.dims[0] * ng.dims[1] * 2,
              ng.absOffset <= model.size, ng.bytes <= model.size - ng.absOffset else {
            throw Qwen38Error.invalidModel("Original BF16 PLE n-gram table is missing, truncated, or too small")
        }
        _ = try require("output_hc_norm.weight", [HC], [0])
        _ = try require("output_hc_down.weight", [HC, c.hcRank], dense)
        _ = try require("output_hc_up.weight", [c.hcRank, HC], hc)
        _ = try require("output.weight", [E, c.vocabulary], dense)
        for i in 0..<c.layers {
            let p = "blk.\(i)."
            for sub in ["hc_attn", "hc_ffn"] {
                _ = try require(p + sub + "_norm.weight", [HC], [0])
                _ = try require(p + sub + "_down.weight", [HC, c.hcRank], dense)
                _ = try require(p + sub + "_up.weight", [c.hcRank, HC], hc)
                _ = try require(p + sub + "_inject.weight", [HC, c.hcCount], hc)
            }
            if c.isLinear(layer: i) {
                for (name, width) in [("attn_qkv", c.linearQKVDimension), ("attn_gate", c.linearValueDimension), ("ssm_beta", c.linearVHeads), ("ssm_alpha", c.linearVHeads)] {
                    _ = try require(p + name + ".weight", [E, width], dense)
                }
                _ = try require(p + "ssm_conv1d.weight", [4, c.linearQKVDimension], [0])
                _ = try require(p + "ssm_dt.bias", [c.linearVHeads], [0])
                _ = try require(p + "ssm_a", [c.linearVHeads], [0])
                _ = try require(p + "ssm_norm.weight", [c.linearDimension], [0])
                _ = try require(p + "ssm_out.weight", [c.linearValueDimension, E], dense)
            } else {
                for (name, width) in [("attn_q", 2 * Q), ("attn_k", c.kvHeads * c.headDimension), ("attn_v", c.kvHeads * c.headDimension), ("indexer.q_proj", c.indexHeads * c.indexDimension), ("indexer.k_proj", c.indexDimension)] {
                    _ = try require(p + name + ".weight", [E, width], dense)
                }
                _ = try require(p + "attn_output.weight", [Q, E], dense)
                for name in ["attn_q_norm", "attn_k_norm"] { _ = try require(p + name + ".weight", [c.headDimension], [0]) }
                for name in ["indexer.q_norm", "indexer.k_norm"] { _ = try require(p + name + ".weight", [c.indexDimension], [0]) }
            }
            if i == 1 {
                _ = try require(p + "ple_key.weight", [E, HC], dense)
                _ = try require(p + "ple_value.weight", [E, E], dense)
                for name in ["ple_norm_key", "ple_norm_query", "ple_norm_conv"] { _ = try require(p + name + ".weight", [HC], [0]) }
                _ = try require(p + "ple_conv1d.weight", [4, HC], [0, 1])
            }
            _ = try require(p + "ffn_gate_inp.weight", [E, c.experts], [0])
            let gate = try require(p + "ffn_gate_exps.weight", [E, c.expertWidth, c.experts], expert)
            _ = try require(p + "ffn_up_exps.weight", [E, c.expertWidth, c.experts], [gate.type])
            let downType = directory[p + "ffn_down_exps.weight"]?.type ?? UInt32.max
            let downWidth = downType == 10 ? (c.expertWidth + 255) / 256 * 256 : c.expertWidth
            _ = try require(p + "ffn_down_exps.weight", [downWidth, E, c.experts], expert)
            let gateDims = directory[p + "ffn_gate_inp_shexp.weight"]?.dims.count == 1 ? [E] : [E, 1]
            _ = try require(p + "ffn_gate_inp_shexp.weight", gateDims, [0])
            for name in ["ffn_gate_shexp", "ffn_up_shexp"] { _ = try require(p + name + ".weight", [E, c.expertWidth], dense) }
            _ = try require(p + "ffn_down_shexp.weight", [c.expertWidth, E], dense)
        }
        tensors = directory
    }
}
