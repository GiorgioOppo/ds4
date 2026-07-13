import Foundation
import DS4Core

/// Pesi MTP (Multi-Token Prediction) nel GGUF: se presenti, la decodifica
/// speculativa (draft con la testa MTP + verifica batch) è possibile. Le
/// conversioni llama.cpp li chiamano blk.N.nextn.* (eh_proj, embed_tokens,
/// enorm, hnorm, shared_head.*); altri converter usano nomi mtp.*.
func mtpReport(_ model: GGUFModel) -> String {
    let pats = ["nextn", "mtp", "eh_proj"]
    let found = model.tensors.filter { t in pats.contains { t.name.lowercased().contains($0) } }
    guard !found.isEmpty else {
        return "  MTP: nessun peso nel GGUF -> decodifica speculativa NON possibile con questo file"
    }
    var s = "  MTP: \(found.count) tensori presenti -> decodifica speculativa POSSIBILE"
    for t in found.prefix(8) { s += "\n    \(t.name)  \(t.typeName)  \(ByteCountFormatter.string(fromByteCount: Int64(t.bytes), countStyle: .memory))" }
    if found.count > 8 { s += "\n    … e altri \(found.count - 8)" }
    return s
}

/// I knob DS4_* attivi: rende ogni run auto-documentante (i confronti A/B
/// hanno senso solo a knob uguali).
func knobReport() -> String {
    let knobs = ["DS4_EXPERT_CACHE_SLOTS", "DS4_EXPERT_CACHE_UNIFORM", "DS4_EXPERT_PREAD", "DS4_PREAD_SPLIT",
                 "DS4_EXPERT_BUNDLE", "DS4_WILLNEED_EXPERTS", "DS4_PREFETCH", "DS4_PREFETCH_EXPERTS",
                 "DS4_EXPERT_LOOKAHEAD", "DS4_ASYNC_FFN",
                 "DS4_PREFILL_UNION", "DS4_PREFILL_FFN_BATCH", "DS4_PREFILL_ROUTE_BATCH",
                 "DS4_PREFILL_CHUNK", "DS4_PREFILL_MM", "DS4_POOL_INTERLEAVE", "DS4_Q8_NSG", "DS4_MOE_NSG", "DS4_DENSE_Q4_NSG",
                 "DS4_ACTIVE_EXPERTS", "DS4_RAW_RING", "DS4_RESIDENT_DENSE",
                 "DS4_DENSE_STREAM", "DS4_DENSE_AHEAD", "DS4_DENSE_Q4", "DS4_SHARED_Q4",
                 "DS4_QKV_Q4", "DS4_COMP_Q8", "DS4_GPU_INDEXER_TOPK", "DS4_DENSE_Q4_KERNEL", "DS4_FUSED_ROUTER_PROBS", "DS4_FUSED_ROUTER_FINALIZE", "DS4_FUSED_COMP_PROJ",
                 "DS4_MTLIO", "DS4_MTLIO_MIN_GBS", "DS4_MLOCK", "DS4_PROFILE_ROUTE", "DS4_SPEC_K", "DS4_SPEC_DRAFT_EXPERTS",
                 "DS4_DEMO_TEMPERATURE", "DS4_DEMO_TOP_K", "DS4_DEMO_TOP_P", "DS4_DEMO_MIN_P",
                 "DS4_DEMO_REPEAT_PENALTY", "DS4_DEMO_REPEAT_LAST_N"]
    let env = ProcessInfo.processInfo.environment
    return "  knob: " + knobs.map { "\($0)=\(env[$0] ?? "·")" }.joined(separator: "  ")
}

