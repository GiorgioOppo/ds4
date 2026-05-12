import Foundation

/// Translation from HuggingFace tensor-leaf names to the DeepSeek-V4
/// inference-side naming convention. Ports `Reference/inference/convert.py:55-79`
/// plus the identity entries that already match (so passing through
/// a previously-converted shard is a no-op).
public let leafMapping: [String: String] = [
    "embed_tokens": "embed",
    "input_layernorm": "attn_norm",
    "post_attention_layernorm": "ffn_norm",
    "q_proj": "wq",
    "q_a_proj": "wq_a",
    "q_a_layernorm": "q_norm",
    "q_b_proj": "wq_b",
    "kv_a_proj_with_mqa": "wkv_a",
    "kv_a_layernorm": "kv_norm",
    "kv_b_proj": "wkv_b",
    "o_proj": "wo",
    "gate_proj": "w1",
    "down_proj": "w2",
    "up_proj": "w3",
    "lm_head": "head",
    // Identity / passthrough names that already match.
    "embed": "embed",
    "wq_b": "wq_b",
    "wo_a": "wo_a",
    "wo_b": "wo_b",
    "head": "head",
    "attn_sink": "attn_sink",
    "weights_proj": "weights_proj",
]

/// Map a HuggingFace tensor name into the inference-side namespace.
/// Strips the `model.` prefix, rewrites a couple of phrases
/// (`self_attn → attn`, `mlp → ffn`, `weight_scale_inv → scale`,
/// `e_score_correction_bias → bias`), then rewrites the leaf via
/// `leafMapping`.
public func renameKey(_ name: String) -> String {
    var n = name
    if n.hasPrefix("model.") { n = String(n.dropFirst("model.".count)) }
    n = n.replacingOccurrences(of: "self_attn", with: "attn")
    n = n.replacingOccurrences(of: "mlp", with: "ffn")
    n = n.replacingOccurrences(of: "weight_scale_inv", with: "scale")
    n = n.replacingOccurrences(of: "e_score_correction_bias", with: "bias")

    let parts = n.split(separator: ".").map(String.init)
    let leaf: String
    if parts.contains(where: { ["hc", "attn_sink", "tie2eid", "ape"].contains($0) ||
                                $0.hasPrefix("hc_") }) {
        leaf = parts.last ?? n
    } else {
        leaf = parts.count >= 2 ? parts[parts.count - 2] : (parts.last ?? n)
    }
    if let mapped = leafMapping[leaf] {
        n = n.replacingOccurrences(of: leaf, with: mapped)
    }
    return n
}

/// `convert.py:105` — MTP-tied embed/head alias the main ones in the
/// HF checkpoint; skip them at conversion time.
public func shouldSkip(_ name: String) -> Bool {
    if name.hasPrefix("mtp.") {
        if name.contains("emb") { return true }
        if name.hasSuffix("head.weight") { return true }
    }
    return false
}

/// Given a `<base>.weight` tensor name, return the matching
/// `<base>.scale` companion name. Used to pair INT8/INT4/INT2 packed
/// weights with their F16 group-scale tensors.
public func scaleNameFor(_ weightName: String) -> String {
    if weightName.hasSuffix(".weight") {
        return weightName.dropLast(".weight".count) + ".scale"
    }
    return weightName + ".scale"
}
