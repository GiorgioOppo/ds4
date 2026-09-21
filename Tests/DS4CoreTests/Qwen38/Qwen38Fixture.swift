import Foundation
import DS4Core

/// Reduced upstream qwen4exp geometry with deterministic original-F32 weights.
/// The real vocabulary width exercises the output head, without a model download.
enum Qwen38Fixture {
    static func metadata() -> [(key: String, value: GGUFMetadataValue)] {
        var result: [(key: String, value: GGUFMetadataValue)] = [("general.architecture", .text("qwen4exp"))]
        let values: [(String, Int)] = [
            ("embedding_length", 64), ("block_count", 8), ("context_length", 128),
            ("attention.head_count", 4), ("attention.head_count_kv", 2), ("attention.key_length", 32), ("attention.value_length", 32),
            ("rope.dimension_count", 8), ("expert_count", 32), ("expert_used_count", 10),
            ("expert_feed_forward_length", 32), ("expert_shared_feed_forward_length", 32),
            ("ssm.conv_kernel", 4), ("ssm.state_size", 32), ("ssm.group_count", 2), ("ssm.time_step_rank", 6), ("ssm.inner_size", 192),
            ("full_attention_interval", 4), ("hyper_connection.count", 4), ("hyper_connection.low_rank", 8),
            ("attention.indexer.head_count", 4), ("attention.indexer.key_length", 32), ("attention.indexer.top_k", 8),
            ("ple.ngram_size", 3), ("ple.heads_per_ngram", 8), ("ple.conv_kernel", 4), ("ple.eos_token_id", 248044),
            ("embedding_length_per_layer_input", 4)
        ]
        result += values.map { ("qwen4exp." + $0.0, .uint32(UInt32($0.1))) }
        result += [("qwen4exp.rope.freq_base", .float32(10_000_000)), ("qwen4exp.attention.layer_norm_rms_epsilon", .float32(1e-6))]
        func array(_ name: String, _ values: [UInt64]) {
            result.append(("qwen4exp." + name, .array(elementType: .uint64, elements: values.map(GGUFMetadataValue.uint64))))
        }
        array("attention.compress_ratios", [0, 0, 0, 4, 0, 0, 0, 4])
        array("ple.layers", [1]); array("ple.layer_multipliers", [UInt64.max - 4, 0xdeadbeef12345679, 0xfedcba9876543211])
        array("ple.head_offsets", (0..<16).map { UInt64($0 * 8) }); array("ple.head_vocab_sizes", [UInt64](repeating: 7, count: 16))
        return result
    }

    static func write(path: String, full: Bool = false, override: (String, GGUFMetadataValue)? = nil) throws {
        var metadata = metadata()
        if let override {
            metadata.removeAll { $0.key == override.0 }
            metadata.append((override.0, override.1))
        }
        var writer = try GGUFWriter(metadata: metadata)
        if full {
            func add(_ name: String, _ dims: [Int], type: UInt32 = 0) {
                let count = dims.reduce(1, *), bytes = count * (type == 30 ? 2 : 4)
                let seed = name.utf8.reduce(UInt32(2166136261)) { ($0 ^ UInt32($1)) &* 16777619 }
                writer.add(.init(name: name, dims: dims.map(UInt64.init), type: type, byteCount: bytes) {
                    var data = Data(count: bytes)
                    data.withUnsafeMutableBytes { (raw: UnsafeMutableRawBufferPointer) in
                        var random = seed
                        for i in 0..<count {
                            random = random &* 1664525 &+ 1013904223
                            var value = (Float(Int(random >> 16) - 32768) / 32768) * 0.025
                            if name.contains("norm") { value += 1 }
                            if name.hasSuffix("ssm_a") { value = -1 }
                            if name.hasSuffix("ssm_dt.bias") { value = 0 }
                            if name == "token_embd.weight" { value *= 8 }
                            if type == 30 { raw.storeBytes(of: UInt16(value.bitPattern >> 16), toByteOffset: i * 2, as: UInt16.self) }
                            else { raw.storeBytes(of: value, toByteOffset: i * 4, as: Float.self) }
                        }
                    }
                    return data
                })
            }
            add("token_embd.weight", [64, 248320]); add("per_layer_token_embd.weight", [4, 128], type: 30)
            add("output_hc_norm.weight", [256]); add("output_hc_down.weight", [256, 8]); add("output_hc_up.weight", [8, 256]); add("output.weight", [64, 248320])
            for i in 0..<8 {
                let p = "blk.\(i)."
                for hc in ["hc_attn", "hc_ffn"] {
                    add(p + hc + "_norm.weight", [256]); add(p + hc + "_down.weight", [256, 8]); add(p + hc + "_up.weight", [8, 256]); add(p + hc + "_inject.weight", [256, 4])
                }
                if (i + 1) % 4 != 0 {
                    add(p + "attn_qkv.weight", [64, 320]); add(p + "attn_gate.weight", [64, 192]); add(p + "ssm_conv1d.weight", [4, 320])
                    add(p + "ssm_dt.bias", [6]); add(p + "ssm_a", [6]); add(p + "ssm_beta.weight", [64, 6]); add(p + "ssm_alpha.weight", [64, 6]); add(p + "ssm_norm.weight", [32]); add(p + "ssm_out.weight", [192, 64])
                } else {
                    add(p + "attn_q.weight", [64, 256]); add(p + "attn_k.weight", [64, 64]); add(p + "attn_v.weight", [64, 64]); add(p + "attn_output.weight", [128, 64]); add(p + "attn_q_norm.weight", [32]); add(p + "attn_k_norm.weight", [32])
                    add(p + "indexer.q_proj.weight", [64, 128]); add(p + "indexer.k_proj.weight", [64, 32]); add(p + "indexer.q_norm.weight", [32]); add(p + "indexer.k_norm.weight", [32])
                }
                if i == 1 {
                    add(p + "ple_key.weight", [64, 256]); add(p + "ple_value.weight", [64, 64]); add(p + "ple_norm_key.weight", [256]); add(p + "ple_norm_query.weight", [256]); add(p + "ple_norm_conv.weight", [256]); add(p + "ple_conv1d.weight", [4, 256])
                }
                add(p + "ffn_gate_inp.weight", [64, 32]); add(p + "ffn_gate_exps.weight", [64, 32, 32]); add(p + "ffn_up_exps.weight", [64, 32, 32]); add(p + "ffn_down_exps.weight", [32, 64, 32])
                add(p + "ffn_gate_inp_shexp.weight", [64]); add(p + "ffn_gate_shexp.weight", [64, 32]); add(p + "ffn_up_shexp.weight", [64, 32]); add(p + "ffn_down_shexp.weight", [32, 64])
            }
        }
        try writer.write(to: path)
    }
}
