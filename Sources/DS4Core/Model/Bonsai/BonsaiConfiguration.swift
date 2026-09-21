import Foundation

/// Validates the complete supported Prism checkpoint before any GPU allocation.
/// Shape and Hadamard metadata are part of the format, not optional hints.
public struct BonsaiConfiguration: Sendable {
    public let layerCount = 64
    public let embeddingSize = 5120
    public let vocabularySize = 248320
    public let feedForwardSize = 17408
    public let contextLimit = 262144
    public let headCount = 24, kvHeadCount = 4, headDimension = 256
    public let rotaryDimension = 64
    public let keyStateHeads = 16, valueStateHeads = 48, stateDimension = 128
    public let convolutionWidth = 4, fullAttentionInterval = 4
    public let epsilon: Float = 1e-6, rotaryBase: Float = 10_000_000
    public let groupedValueHeads: Bool
    public let signs: [Int: [Int32]]
    public let weights: [String: Weight]

    public struct Weight: Sendable {
        public let tensor: GGUFModel.Tensor
        public let columns: Int, rows: Int, rowBytes: Int
        public let folded: Bool
    }

    public init(model: GGUFModel) throws {
        func fail(_ detail: String) -> GGUFError { .message("Bonsai 2: \(detail)") }
        let integers: [(String, UInt32)] = [
            ("block_count",64), ("embedding_length",5120), ("feed_forward_length",17408),
            ("attention.head_count",24), ("attention.head_count_kv",4), ("attention.key_length",256),
            ("attention.value_length",256), ("rope.dimension_count",64), ("ssm.conv_kernel",4),
            ("ssm.state_size",128), ("ssm.group_count",16), ("ssm.time_step_rank",48),
            ("ssm.inner_size",6144), ("full_attention_interval",4), ("context_length",262144)
        ]
        for (key, value) in integers where model.u32("qwen35." + key) != value {
            throw fail("unsupported or missing qwen35.\(key)")
        }
        guard model.u32("prism.hadamard.version") == 1, model.u32("prism.hadamard.block_size") == 1024 else {
            throw fail("unsupported Hadamard version/block size")
        }
        for (key, value) in ["general.architecture":"qwen35", "prism.hadamard.transform":"normalized-sylvester-walsh-hadamard",
                             "prism.hadamard.axis":"input-last-dimension", "prism.hadamard.sign_mode":"explicit", "tokenizer.ggml.pre":"qwen35"] {
            guard model.string(key) == value else { throw fail("unsupported or missing \(key)") }
        }
        func int32Array(_ key: String, count: Int) throws -> [Int32] {
            guard let array = model.array(key), array.type == 5, array.len == UInt64(count),
                  let values = model.intArray(key), values.count == count,
                  values.allSatisfy({ Int32(exactly: $0) != nil }) else { throw fail("invalid \(key)") }
            return values.map { Int32($0) }
        }
        guard try int32Array("qwen35.rope.dimension_sections", count: 4) == [11,11,10,0] else { throw fail("unsupported MRoPE sections") }
        let forbidden = Set(["qwen35.attention.recurrent_layers", "qwen35.rope.scaling.type", "qwen35.attention.scale"])
        guard !model.kvs.contains(where: { forbidden.contains($0.key) }),
              (model.u32("qwen35.nextn_predict_layers") ?? 0) == 0,
              model.f32Compat("qwen35.attention.layer_norm_rms_epsilon") == epsilon,
              model.f32Compat("qwen35.rope.freq_base") == rotaryBase,
              let grouped = model.bool("prism.hadamard.gdn_v_grouped") else { throw fail("unsupported normalization, attention or GDN metadata") }
        groupedValueHeads = grouped
        guard let tokens = model.array("tokenizer.ggml.tokens"), tokens.type == 8,
              tokens.len == UInt64(vocabularySize) else { throw fail("vocabulary must contain 248320 strings") }
        let widths = try int32Array("prism.hadamard.sign_widths", count: 3)
        guard widths == [5120,6144,17408] else { throw fail("unsupported Hadamard widths") }
        let values = try int32Array("prism.hadamard.sign_values", count: 28672)
        guard values.allSatisfy({ $0 == 1 || $0 == -1 }) else { throw fail("Hadamard signs must be +1 or -1") }
        var signTable: [Int: [Int32]] = [:], offset = 0
        for width in widths.map(Int.init) { signTable[width] = Array(values[offset..<offset+width]); offset += width }
        signs = signTable
        func strings(_ key: String, count: Int) throws -> [String] {
            guard let array = model.array(key), array.type == 8, array.len == UInt64(count),
                  let bytes = model.stringArrayBytes(key), bytes.count == count else { throw fail("invalid \(key)") }
            return try bytes.map {
                guard !$0.contains(0), $0.count < 128, let string = String(bytes: $0, encoding: .utf8) else { throw fail("invalid folded weight name") }
                return string
            }
        }
        let names = try strings("prism.hadamard.weight_names", count: 401)
        guard Set(names).count == names.count, !names.contains("token_embd.weight"),
              try strings("prism.hadamard.inverse_weight_names", count: 1) == ["token_embd.weight"] else { throw fail("invalid folded/inverse weight coverage") }
        let foldedNames = Set(names + ["token_embd.weight"])
        var table: [String: Weight] = [:], used = Set<String>()
        func bind(_ name: String, _ columns: Int, _ rows: Int = 1, matrix: Bool = false, folded: Bool = false, f32: Bool = false) throws {
            guard let tensor = model.findTensor(name), tensor.dims == (matrix ? [UInt64(columns),UInt64(rows)] : [UInt64(columns)]),
                  foldedNames.contains(name) == folded,
                  folded ? [142,143].contains(tensor.type) : (matrix && !f32 ? [0,1,30].contains(tensor.type) : tensor.type == 0),
                  let rowBytes = BonsaiQuantization.rowBytes(type: tensor.type, columns: columns),
                  tensor.bytes == UInt64(rowBytes) * UInt64(rows), tensor.absOffset <= model.size,
                  tensor.bytes <= model.size - tensor.absOffset,
                  tensor.absOffset % (tensor.type == 0 ? 4 : 2) == 0 else { throw fail("invalid shape, encoding, alignment or folded status: \(name)") }
            table[name] = Weight(tensor: tensor, columns: columns, rows: rows, rowBytes: rowBytes, folded: folded)
            if folded { used.insert(name) }
        }
        try bind("token_embd.weight",5120,248320,matrix:true,folded:true)
        try bind("output.weight",5120,248320,matrix:true,folded:true)
        try bind("output_norm.weight",5120)
        for i in 0..<64 {
            let prefix = "blk.\(i)."
            try bind(prefix+"attn_norm.weight",5120); try bind(prefix+"post_attention_norm.weight",5120)
            try bind(prefix+"ffn_gate.weight",5120,17408,matrix:true,folded:true)
            try bind(prefix+"ffn_up.weight",5120,17408,matrix:true,folded:true)
            try bind(prefix+"ffn_down.weight",17408,5120,matrix:true,folded:true)
            if (i+1)%4 != 0 {
                try bind(prefix+"attn_qkv.weight",5120,10240,matrix:true,folded:true)
                try bind(prefix+"attn_gate.weight",5120,6144,matrix:true,folded:true)
                try bind(prefix+"ssm_out.weight",6144,5120,matrix:true,folded:true)
                for name in ["ssm_alpha.weight","ssm_beta.weight"] { try bind(prefix+name,5120,48,matrix:true) }
                try bind(prefix+"ssm_conv1d.weight",4,10240,matrix:true,f32:true)
                try bind(prefix+"ssm_a",48); try bind(prefix+"ssm_dt.bias",48); try bind(prefix+"ssm_norm.weight",128)
            } else {
                try bind(prefix+"attn_q.weight",5120,12288,matrix:true,folded:true)
                for name in ["attn_k.weight","attn_v.weight"] { try bind(prefix+name,5120,1024,matrix:true,folded:true) }
                try bind(prefix+"attn_output.weight",6144,5120,matrix:true,folded:true)
                try bind(prefix+"attn_q_norm.weight",256); try bind(prefix+"attn_k_norm.weight",256)
            }
        }
        guard used == foldedNames else { throw fail("folded weight list does not match the complete graph") }
        weights = table
    }
}
