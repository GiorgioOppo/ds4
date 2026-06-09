import Foundation
import DS4Core
import DS4Metal
///Users/oppog/Downloads/ds4-main/gguf/DeepSeek-V4-Flash-IQ2XXS-w2Q2K-AProjQ8-SExpQ8-OutQ8-chat-v2-imatrix.gguf
// DS4Demo: a tiny CLI that drives the PURE-SWIFT DeepSeek-V4 engine (DS4Core +
// DS4Metal) with no external links — no C engine, no prebuilt static lib. It
// brings up the Metal runtime (compiling the vendored metal/ kernels), runs a
// GPU self-test, and — if a GGUF path is given — streams a few tokens from the
// real model via StreamingDecoder (per-layer load/compute/evict, 16GB-friendly).
//
// Usage:
//   DS4Demo                       # Metal bring-up self-test only
//   DS4Demo <gguf-path> [maxNew]  # + stream <maxNew> tokens (heavy I/O)

func log(_ s: String) { FileHandle.standardError.write((s + "\n").data(using: .utf8)!) }

do {
    let rt = try MetalRuntime()   // kernels embedded in the binary — no folder needed
    log("DS4Demo: Metal runtime up on \(rt.deviceName), \(rt.functionNames.count) kernels compiled")
    let ok = try rt.runTouchSelfTest()
    log("DS4Demo: GPU self-test \(ok ? "PASSED" : "FAILED")")

    let args = CommandLine.arguments
    guard args.count >= 2 else {
        log("DS4Demo: no GGUF path given — bring-up only. Pass a .gguf to stream tokens.")
        exit(ok ? 0 : 1)
    }
    let ggufPath = args[1]
    let maxNew = args.count >= 3 ? (Int(args[2]) ?? 4) : 4

    log("DS4Demo: opening \(ggufPath) …")
    let model = try GGUFModel(path: ggufPath, metalMapping: true, prefetchCPU: false)
    // Quant-format audit (DS4_TYPES_ONLY=1): print the GGUF dtype of the per-layer
    // weights the engine assumes (experts=Q4_K, router=Q8). Mismatch => garbage.
    if ProcessInfo.processInfo.environment["DS4_TYPES_ONLY"] != nil {
        for nm in ["blk.2.ffn_gate_exps.weight", "blk.2.ffn_up_exps.weight", "blk.2.ffn_down_exps.weight",
                   "blk.2.ffn_gate_shexp.weight", "blk.2.ffn_up_shexp.weight", "blk.2.ffn_down_shexp.weight",
                   "blk.2.attn_q_a.weight", "blk.2.attn_q_a_norm.weight", "blk.2.attn_q_b.weight",
                   "blk.2.attn_kv.weight", "blk.2.attn_kv_a_norm.weight", "blk.2.attn_sinks.weight",
                   "blk.2.attn_output_a.weight", "blk.2.attn_output_b.weight", "blk.2.ffn_gate_inp.weight",
                   "blk.2.attn_norm.weight", "blk.2.ffn_norm.weight",
                   "blk.2.hc_attn_fn.weight", "blk.2.hc_attn_scale.weight", "blk.2.hc_attn_base.weight",
                   "blk.2.hc_ffn_fn.weight", "blk.0.attn_q_a.weight",
                   "output.weight", "output_norm.weight", "output_hc_fn.weight", "output_hc_scale.weight",
                   "token_embd.weight"] {
            if let t = model.findTensor(nm) { log("  TYPE \(nm) = \(t.typeName) (code \(t.type))") }
            else { log("  TYPE \(nm) = <missing>") }
        }
        let tok = try Tokenizer(model: model)
        log("  SPECIAL bos=\(tok.bosId) eos=\(tok.eosId) user=\(tok.userId) assistant=\(tok.assistantId) thinkEnd=\(tok.thinkEndId)")
        let ids = tok.encodeChatPrompt(system: nil, prompt: "ciao come stai?", think: .none)
        log("  PROMPT ids = \(ids)")
        for id in ids { log("    \(id) -> '\(String(bytes: tok.tokenText(id), encoding: .utf8) ?? "?")'") }
        exit(0)
    }
    var dims = DSV4Shape.dims
    let mq = GGUFWeights.detectMoEQuant(model)
    dims.gateQuant = mq.gate; dims.upQuant = mq.up; dims.downQuant = mq.down; dims.routerF16 = mq.routerF16
    log("DS4Demo: MoE quant gate=\(mq.gate) up=\(mq.up) down=\(mq.down) routerF16=\(mq.routerF16)")
    // Optional: reduce active experts per token (DS4_ACTIVE_EXPERTS=2..6). Fewer
    // experts = less expert I/O (faster on RAM-starved machines), lower quality.
    if let s = ProcessInfo.processInfo.environment["DS4_ACTIVE_EXPERTS"], let kk = Int(s) {
        dims.activeExperts = max(1, min(kk, dims.k))
        log("DS4Demo: active experts (top-k) = \(dims.activeExperts) of \(dims.k)")
    }
    let rope = RopeParams(nCtxOrig: 4096, freqBase: 10000, freqScale: 1, extFactor: 0,
                          attnFactor: 1, betaFast: 32, betaSlow: 1)
    // Fast path: non-routed weights resident (memoized), only the 6 selected experts
    // gathered per token (~6/256 of expert IO) — the C --ssd-streaming model.
    let dec = try StreamingDecoder.fromGGUFExpertCachedMapped(rt: rt, model: model, dims: dims, rope: rope,
                                                              nLayers: DSV4Shape.nLayer, maxKeys: 4096)
    log("DS4Demo: no-copy mmap non-routed + gather 6 experts/token (C --ssd-streaming model)…")
    let t0 = Date()
    let logits = try dec.forward(token: 0, pos: 0, nKeys: 1)
    let dt = Date().timeIntervalSince(t0)
    let finite = logits.allSatisfy { $0.isFinite }
    var argmax = 0; var best = -Float.greatestFiniteMagnitude
    for (i, v) in logits.enumerated() where v > best { best = v; argmax = i }
    log(String(format: "DS4Demo: 1 forward in %.1fs — logits[%d] finite=%@ argmax=%d (logit %.3f)",
               dt, logits.count, finite ? "YES" : "NO", argmax, best))
    if maxNew > 0 {
        // Real chat generation: tokenize the prompt (3rd arg) with the model's
        // tokenizer + chat template, greedy-decode, detokenize, print the answer.
        let prompt = args.count >= 4 ? args[3] : "ciao come stai? rispondi in 1 parola"
        let tok = try Tokenizer(model: model)
        let ids = tok.encodeChatPrompt(system: nil, prompt: prompt, think: .none).map { Int($0) }
        log("DS4Demo: prompt '\(prompt)' -> \(ids.count) tokens; generating \(maxNew) (greedy, streaming)…")
        let stdout = FileHandle.standardOutput
        // Prefill: process the prompt with per-token progress (each forward is slow).
        var pos = 0
        var last: [Float] = []
        for (i, t) in ids.enumerated() {
            let t0 = Date()
            last = try dec.forward(token: t, pos: pos, nKeys: pos + 1); pos += 1
            log(String(format: "  prefill %d/%d  %.1fs", i + 1, ids.count, Date().timeIntervalSince(t0)))
        }
        // Decode: stream each token's bytes to stdout AS it is produced (like ds4).
        dec.resetProfile()   // profila solo la fase di decode (non il prefill)
        stdout.write(Data("\nRisposta: ".utf8))
        var rng: UInt64 = 1
        var genTokens = 0
        let genStart = Date()
        for _ in 0..<maxNew {
            let next = Sampler.sample(last, temperature: 0, topK: 0, topP: 1, minP: 0, rng: &rng)
            if Int32(next) == tok.eosId { break }
            stdout.write(Data(tok.tokenText(Int32(next))))   // stream immediately (unbuffered)
            let t0 = Date()
            last = try dec.forward(token: next, pos: pos, nKeys: pos + 1); pos += 1
            genTokens += 1
            let dt = Date().timeIntervalSince(t0)
            log(String(format: "  [tok %d  %.1fs  %.2f tok/s]", genTokens, dt, dt > 0 ? 1.0 / dt : 0))
        }
        stdout.write(Data("\n".utf8))
        let total = Date().timeIntervalSince(genStart)
        log(String(format: "DS4Demo: %d tokens in %.1fs (%.2f tok/s)", genTokens, total,
                   total > 0 ? Double(genTokens) / total : 0))
        log("")
        log(dec.profile.report())
    }
    exit(0)
} catch {
    log("DS4Demo error: \(error)")
    exit(1)
}
