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

// ── DS4_DIAG=1: diagnosi delle ottimizzazioni di streaming ──
// Stampa i numeri che decidono la PROSSIMA ottimizzazione: banda grezza del
// disco (il tetto del gather), presenza dei pesi MTP (decodifica speculativa
// possibile?), knob attivi, e — dopo la generazione — concentrazione del
// routing + allocazione slot per layer e il verdetto gather vs SSD.

/// Banda del file modello BYPASSANDO la page cache (F_NOCACHE) — misura il
/// disco vero, ripetibile. Tre scenari con slab da ~2 MB (la taglia di un
/// esperto 2-bit): sequenziale a coda 1, random a coda 1 (il gather senza
/// hint), random parallelo (il gather con madvise/pread paralleli). Il TETTO
/// del disco è il massimo dei tre: sugli NVMe Apple il parallelo random può
/// superare il sequenziale a coda 1 (serve profondità di coda, non località).
func diskBench(path: String) -> (report: String, ceilingGBs: Double) {
    let fd = open(path, O_RDONLY)
    guard fd >= 0 else { return ("  SSD: open fallita (\(String(cString: strerror(errno))))", 0) }
    defer { close(fd) }
    _ = fcntl(fd, F_NOCACHE, 1)
    var st = stat()
    guard fstat(fd, &st) == 0, st.st_size > 0 else { return ("  SSD: fstat fallita", 0) }
    let fileSize = Int(st.st_size)
    let slab = min(2 << 20, fileSize)            // ~2 MB
    let align = 1 << 14                          // offset a pagine da 16 KB
    let nSlabs = 24
    var rng = SystemRandomNumberGenerator()
    func randOff() -> Int {
        (Int.random(in: 0..<max(1, fileSize - slab), using: &rng) / align) * align
    }
    let buf = UnsafeMutableRawPointer.allocate(byteCount: slab, alignment: align)
    defer { buf.deallocate() }
    // 1) sequenziale (fino a 1 GB)
    let seqTotal = min(1 << 30, fileSize)
    var done = 0
    var t0 = Date()
    while done < seqTotal {
        let n = pread(fd, buf, min(slab, seqTotal - done), off_t(done))
        if n <= 0 { break }
        done += n
    }
    let seq = Double(done) / max(1e-9, Date().timeIntervalSince(t0)) / 1e9
    // 2) random, coda 1: uno slab alla volta
    let offs1 = (0..<nSlabs).map { _ in randOff() }
    t0 = Date()
    for off in offs1 { _ = pread(fd, buf, slab, off_t(off)) }
    let qd1 = Double(nSlabs * slab) / max(1e-9, Date().timeIntervalSince(t0)) / 1e9
    // 3) random, in parallelo: tutti gli slab insieme (pread è thread-safe)
    let offs2 = (0..<nSlabs).map { _ in randOff() }
    t0 = Date()
    DispatchQueue.concurrentPerform(iterations: offs2.count) { i in
        let b = UnsafeMutableRawPointer.allocate(byteCount: slab, alignment: align)
        defer { b.deallocate() }
        _ = pread(fd, b, slab, off_t(offs2[i]))
    }
    let qdN = Double(nSlabs * slab) / max(1e-9, Date().timeIntervalSince(t0)) / 1e9
    let ceiling = max(seq, qd1, qdN)
    let report = String(format: """
      SSD (F_NOCACHE, slab %d MB):
        sequenziale coda 1  %6.2f GB/s
        random coda 1       %6.2f GB/s   <- gather senza hint/parallelismo
        random parallelo    %6.2f GB/s   <- gather con madvise/pread paralleli
        TETTO               %6.2f GB/s   <- riferimento per la banda effettiva
    """, slab >> 20, seq, qd1, qdN, ceiling)
    return (report, ceiling)
}

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
    let knobs = ["DS4_EXPERT_CACHE_SLOTS", "DS4_EXPERT_CACHE_UNIFORM", "DS4_EXPERT_PREAD",
                 "DS4_WILLNEED_EXPERTS", "DS4_PREFETCH", "DS4_PREFETCH_EXPERTS",
                 "DS4_PREFILL_UNION", "DS4_Q8_NSG",
                 "DS4_ACTIVE_EXPERTS", "DS4_RAW_RING", "DS4_RESIDENT_DENSE", "DS4_PROFILE_ROUTE"]
    let env = ProcessInfo.processInfo.environment
    return "  knob: " + knobs.map { "\($0)=\(env[$0] ?? "·")" }.joined(separator: "  ")
}

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

    let diag = ProcessInfo.processInfo.environment["DS4_DIAG"] == "1"
    var diskCeilingGBs = 0.0
    if diag {
        log("── Diagnosi (DS4_DIAG=1) ──")
        log(knobReport())
        let bench = diskBench(path: ggufPath)
        diskCeilingGBs = bench.ceilingGBs
        log(bench.report)
    }

    log("DS4Demo: opening \(ggufPath) …")
    let model = try GGUFModel(path: ggufPath, metalMapping: true, prefetchCPU: false)
    if diag { log(mtpReport(model)) }
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
        // Prefill: LAYER-MAJOR — load each layer's weights once and apply to all
        // prompt tokens (amortizes the dominant weight I/O). Returns the last
        // token's logits; KV cache is populated for positions 0..N-1.
        let pf0 = Date()
        var last = try dec.prefill(tokens: ids)
        var pos = ids.count
        log(String(format: "DS4Demo: prefill %d token (layer-major) in %.1fs (%.1fs/token)",
                   ids.count, Date().timeIntervalSince(pf0),
                   Date().timeIntervalSince(pf0) / Double(max(1, ids.count))))
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

        // ── Diagnosi post-run: routing, allocazione slot, verdetto gather ──
        if diag, let usage = dec.usage {
            log("")
            log("── Diagnosi cache esperti ──")
            let envSlots = ProcessInfo.processInfo.environment["DS4_EXPERT_CACHE_SLOTS"].flatMap(Int.init) ?? 0
            if envSlots == 0 {
                log("  slot-cache OFF (imposta DS4_EXPERT_CACHE_SLOTS=8.. per misurare gli hit)")
            }
            let base = max(8, envSlots)
            let alloc = usage.slotAllocation(base: base)
            if alloc == nil {
                log("  allocazione usage-driven: storia insufficiente (~43+ token/layer) -> uniforme \(base); rilancia con più token")
            }
            log("  layer   route  conc(top8)  conc(top16)  slot")
            for il in 0..<DSV4Shape.nLayer {
                let r = usage.routes(layer: il)
                guard r > 0 else { continue }
                log(String(format: "  %5d %7d       %.2f        %.2f  %4d", il, r,
                           usage.concentration(layer: il, n: 8),
                           usage.concentration(layer: il, n: 16),
                           alloc?[il] ?? base))
            }
            // Verdetto: banda effettiva del gather vs tetto sequenziale del disco.
            // Sotto ~60% del tetto il sidecar expert-bundle (slab contigui) può
            // ancora rendere; sopra, il gather è vicino alla fisica del disco e
            // conviene puntare su hit-rate (slot) o decodifica speculativa (MTP).
            if dec.profile.gatherBytes > 0, dec.profile.gatherS > 0 {
                let eff = Double(dec.profile.gatherBytes) / dec.profile.gatherS / 1e9
                if diskCeilingGBs > 0 {
                    let pct = eff / diskCeilingGBs * 100
                    log(String(format: "  gather effettivo %.2f GB/s = %.0f%% del tetto SSD (%.2f GB/s) -> %@",
                               eff, pct, diskCeilingGBs,
                               pct < 60 ? "margine: il sidecar expert-bundle può rendere"
                                        : "vicino alla fisica del disco: puntare su hit-rate/MTP"))
                } else {
                    log(String(format: "  gather effettivo %.2f GB/s (banda SSD non misurata)", eff))
                }
            }
        }
    }
    exit(0)
} catch {
    log("DS4Demo error: \(error)")
    exit(1)
}
