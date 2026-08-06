import Foundation
import DS4Core
import DS4Metal
import Metal
///Users/oppog/Downloads/ds4-main/gguf/DeepSeek-V4-Flash-IQ2XXS-w2Q2K-AProjQ8-SExpQ8-OutQ8-chat-v2-imatrix.gguf
// DS4Demo: a tiny CLI that drives the PURE-SWIFT DeepSeek-V4 engine (DS4Core +
// DS4Metal) with no external links — no C engine, no prebuilt static lib. It
// brings up the Metal runtime (compiling the vendored metal/ kernels), runs a
// GPU self-test, and — if a GGUF path is given — streams a few tokens from the
// real model via StreamingDecoder (per-layer load/compute/evict, 16GB-friendly).
//
// Usage:
//   DS4Demo                                # Metal bring-up self-test only
//   DS4Demo <gguf-path> [maxNew] [prompt]  # + stream <maxNew> tokens (heavy I/O)
//   DS4Demo <shardA.gguf,shardB.gguf> …    # Pro Q4 layer-range split (comma list)
//   DS4Demo requantize <in> <out> RULE …   # offline GGUF→GGUF requant (no GPU)
//   prompt "@/path/file" = usa il CONTENUTO del file come prompt (testi lunghi,
//   benchmark prefill; troncato a DS4_PROMPT_MAX_CHARS, default 12000).

// Offline, GPU-free subcommand: intercept BEFORE the Metal runtime is created so
// the requantizer runs on machines without an Apple GPU.
if CommandLine.arguments.count >= 2, CommandLine.arguments[1] == "requantize" {
    exit(runRequantizeCLI(Array(CommandLine.arguments.dropFirst(2))))
}
if CommandLine.arguments.count >= 2, CommandLine.arguments[1] == "inspect-dspark" {
    exit(runDSparkInspectCLI(Array(CommandLine.arguments.dropFirst(2))))
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
    let abTrace = ABLogitTrace.fromEnvironment()
    if let trace = abTrace {
        log("DS4Demo: A/B logit trace attiva — max \(trace.frameLimit) frame, output \(trace.prefix).{json,f32}")
    }
    let ggufPath = args[1]
    let maxNew = args.count >= 3 ? (Int(args[2]) ?? 4) : 4
    let maxKeys: Int
    if let rawContext = ProcessInfo.processInfo.environment["DS4_DEMO_CONTEXT"] {
        guard let parsed = Int(rawContext), (1_024...1_000_000).contains(parsed) else {
            log("DS4Demo: DS4_DEMO_CONTEXT deve essere un intero tra 1024 e 1000000")
            exit(2)
        }
        maxKeys = parsed
    } else {
        maxKeys = 4_096
    }
    log("DS4Demo: capacita' contesto = \(maxKeys) token (DS4_DEMO_CONTEXT)")
    let syntheticLiveContext: Int?
    if let rawLive = ProcessInfo.processInfo.environment["DS4_DEMO_LIVE_CONTEXT"] {
        guard let parsed = Int(rawLive), parsed >= 8,
              parsed + maxNew + 1 <= maxKeys else {
            log("DS4Demo: DS4_DEMO_LIVE_CONTEXT deve essere >= 8 e lasciare spazio a maxNew+1 entro DS4_DEMO_CONTEXT")
            exit(2)
        }
        syntheticLiveContext = parsed
        log("DS4Demo: target contesto realmente usato = \(parsed) token sintetici")
    } else {
        syntheticLiveContext = nil
    }

    let diag = ProcessInfo.processInfo.environment["DS4_DIAG"] == "1"
    var diskCeilingGBs = 0.0
    if diag {
        log("── Diagnosi (DS4_DIAG=1) ──")
        log(knobReport())
        // Self-test del requantizer Q4_K (DS4_DENSE_Q4): roundtrip sintetico.
        // Un bug di packing farebbe esplodere l'errore; l'atteso è ~1-4%.
        do {
            let n = 256 * 64
            var xs = [Float](repeating: 0, count: n)
            var state: UInt64 = 0x9E3779B97F4A7C15
            for i in 0..<n {   // xorshift riproducibile
                state ^= state << 13; state ^= state >> 7; state ^= state << 17
                xs[i] = Float(Int64(bitPattern: state) % 10007) / 10007.0
            }
            var packed = [UInt8](repeating: 0, count: n / 256 * 144)
            var back = [Float](repeating: 0, count: n)
            xs.withUnsafeBufferPointer { xp in
                packed.withUnsafeMutableBytes { pp in
                    Quantize.quantizeQ4_K(xp.baseAddress!, count: n, into: pp.baseAddress!)
                }
            }
            packed.withUnsafeBytes { pp in
                back.withUnsafeMutableBufferPointer { bp in
                    Quantize.dequantQ4_K(pp.baseAddress!, count: n, into: bp.baseAddress!)
                }
            }
            var se: Float = 0, sx: Float = 0
            for i in 0..<n { let d = xs[i] - back[i]; se += d * d; sx += xs[i] * xs[i] }
            let rel = (se / max(sx, 1e-12)).squareRoot()
            // Dati UNIFORMI in (-1,1): l'errore Q4_K atteso è ~5-6% (su pesi
            // reali, ~gaussiani, è ~2-3%). Un bug di packing darebbe >20%.
            log(String(format: "  Q4_K roundtrip: errore relativo %.2f%% %@", rel * 100,
                       rel < 0.08 ? "(OK — atteso ~5-6% su dati uniformi)" : "(SOSPETTO: probabile bug di packing)"))
        }
        // Costo FISSO di un command buffer (commit + waitUntilCompleted, vuoto):
        // il decode streaming ne usa ~3 per layer × 43 layer ≈ 130+ per token.
        // Se questo round-trip costa millisecondi, il "compute" del profilo è in
        // realtà latenza di sincronizzazione GPU<->CPU e la cura è FONDERE i
        // command buffer, non ottimizzare i kernel.
        do {
            // warm-up (primo cb paga la creazione delle code)
            for _ in 0..<5 { let c = GraphContext(rt); try c.begin(); c.commit() }
            let n = 100
            let t0 = Date()
            for _ in 0..<n { let c = GraphContext(rt); try c.begin(); c.commit() }
            let usPerCB = Date().timeIntervalSince(t0) / Double(n) * 1e6
            let perToken = usPerCB * 130 / 1000
            log(String(format: "  command buffer vuoto: %.0f µs/round-trip  (×130 cb/token ≈ %.0f ms/token di pura sincronizzazione)",
                       usPerCB, perToken))
        }
        let bench = diskBench(path: ggufPath)
        diskCeilingGBs = bench.ceilingGBs
        log(bench.report)
    }

    log("DS4Demo: opening \(ggufPath) …")
    // Multi-shard input: a comma-separated path list opens the DeepSeek V4 Pro Q4
    // layer-range split (…Layers00-30.gguf,…Layers-31-output.gguf) as one logical
    // model via GGUFShardSet + fromGGUFShards. The primary (first) shard supplies
    // the tokenizer/config/global metadata; a single path keeps prior behavior.
    let shardPaths = ggufPath.split(separator: ",").map(String.init)
    let shardSet: GGUFShardSet? = shardPaths.count > 1
        ? try GGUFShardSet(paths: shardPaths, metalMapping: true) : nil
    if let shardSet {
        log("DS4Demo: multi-shard Pro Q4 split — \(shardSet.shards.count) shards, "
            + "\(shardSet.n_tensors) tensors merged")
    }
    let model: GGUFModel
    if let shardSet {
        model = shardSet.primary
    } else {
        model = try GGUFModel(path: ggufPath, metalMapping: true, prefetchCPU: false)
    }
    // DS4Demo cannot import DS4Engine without changing the XcodeGen target graph,
    // so it shares the same canonical DS4Core detector used by the engine factory.
    // Select before constructing DeepSeek tokenizer/dims: Qwen must never fall
    // through into misleading errors about missing deepseek4.* metadata.
    let detectedArchitecture = try ModelArchitectureDetector.detect(in: model)
    do {
        try ModelArchitectureDetector.requireImplemented(detectedArchitecture)
    } catch {
        if detectedArchitecture.family == .glm, GLM52RuntimeGate.enabled {
            // GLM 5.2 greedy demo on the resident engine (validation-grade
            // speed). Prompt from DS4_PROMPT; token budget from DS4_MAX_NEW.
            let prompt = ProcessInfo.processInfo.environment["DS4_PROMPT"]
                ?? "Ciao! Presentati in una frase."
            let maxNew = Int(ProcessInfo.processInfo
                .environment["DS4_MAX_NEW"] ?? "") ?? 128
            log("DS4Demo: GLM 5.2 greedy — prompt: \(prompt)")
            let tokenizer = try GLM52Tokenizer(model: model)
            let runtime = try MetalRuntime()
            // Knob condivisi: DS4_RESIDENT_LAYERS (quanti layer restano
            // residenti; gli altri streammano da SSD), DS4_ACTIVE_EXPERTS
            // (meno esperti = meno I/O, qualità ridotta),
            // DS4_EXPERT_CACHE_SLOTS (slot cache esperti per layer sparse).
            let environment = ProcessInfo.processInfo.environment
            var glmOptions = GLM52ResidentModelOptions()
            glmOptions.cacheCapacity = maxKeys
            glmOptions.residentLayerCount = DS4RuntimeEnvironment.integer(
                .residentLayers,
                backend: .glm52,
                environment: environment)
                ?? GLM52ResidentModelOptions.adaptiveResidentLayerCount()
            glmOptions.activeExperts = DS4RuntimeEnvironment.integer(
                .activeExperts,
                backend: .glm52,
                environment: environment)
            if let slots = DS4RuntimeEnvironment.integer(
                .expertCacheSlots,
                backend: .glm52,
                environment: environment
            ) {
                glmOptions.expertSlotCount = slots
            }
            // DS4_STREAM_SLOTS: slot di staging del layer streamer
            // (default 3 = due fill SSD in volo mentre un layer computa).
            if let slots = DS4RuntimeEnvironment.integer(
                .streamSlots,
                backend: .glm52,
                environment: environment
            ) {
                glmOptions.streamSlotCount = slots
            }
            if let resident = glmOptions.residentLayerCount {
                log("DS4Demo: GLM streaming — \(resident) layer residenti, "
                    + "il resto da SSD per token")
            } else {
                log("DS4Demo: caricamento pesi residenti GLM (decine di GB)…")
            }
            // Prepass bundle: DS4_GLM_BUILD_BUNDLES=1 + DS4_BUNDLE_DIR
            // riimpacchetta i record esperti di ogni layer sparse in file
            // contigui (una lettura per esperto invece di tre; duplica su
            // disco il payload routed). Resumabile: i bundle validi vengono
            // saltati.
            if environment["DS4_GLM_BUILD_BUNDLES"] == "1" {
                // Convenzione sidecar: accanto al GGUF, salvo override.
                let bundleDir = DS4RuntimeEnvironment.value(
                    .bundleDirectory,
                    backend: .glm52,
                    environment: environment)
                    ?? (ggufPath + ".glm-experts")
                let map = try GLM52WeightMap(model: model)
                let reader = try GLM52PayloadReader(path: ggufPath,
                                                    weightMap: map)
                let start = Date()
                // Parziale-friendly: DS4_GLM_BUNDLE_LAYERS è il tetto sui
                // bundle TOTALI (esistenti inclusi — rilanciare non ne
                // aggiunge); senza limite si ferma comunque con grazia
                // quando lo spazio scende sotto la riserva. I layer senza
                // bundle continuano a servire dal GGUF.
                let summary = try GLM52ExpertBundle.buildAvailable(
                    directory: bundleDir, weightMap: map, reader: reader,
                    maxBundles: environment["DS4_GLM_BUNDLE_LAYERS"]
                        .flatMap(Int.init)) { layer, built in
                    log("DS4Demo: bundle blk\(layer) "
                        + (built ? "creato" : "già valido"))
                }
                log(String(
                    format: "DS4Demo: bundle in %.0f s — %d creati, "
                        + "%d già validi, %d mancanti%@ (dir %@)",
                    Date().timeIntervalSince(start), summary.created,
                    summary.alreadyValid, summary.remaining,
                    summary.stoppedBecause.map { " — stop: \($0)" } ?? "",
                    bundleDir))
                exit(0)
            }
            // Prepass sidecar UNIFICATO: DS4_GLM_BUILD_LAYERQ4=1 costruisce
            // per ogni layer sparse UN file con i tensori grossi
            // riquantizzati Q4_K e i record esperti contigui. Migra da solo
            // i file legacy (tensori dal sidecar v1, esperti dal bundle
            // .experts, che viene eliminato a pack validato — il disco
            // resta ~invariato). DS4_GLM_LAYERQ4_LAYERS è il tetto sui
            // sidecar TOTALI; parziale-friendly come sempre.
            if environment["DS4_GLM_BUILD_LAYERQ4"] == "1" {
                let sidecarDir = environment["DS4_GLM_LAYERQ4_DIR"]
                    ?? (ggufPath + ".glm-layers-q4")
                let legacyBundles = DS4RuntimeEnvironment.value(
                    .bundleDirectory,
                    backend: .glm52,
                    environment: environment)
                    ?? (ggufPath + ".glm-experts")
                let map = try GLM52WeightMap(model: model)
                let reader = try GLM52PayloadReader(path: ggufPath,
                                                    weightMap: map)
                let start = Date()
                let summary = try GLM52LayerQuantSidecar.buildAvailable(
                    directory: sidecarDir, weightMap: map, reader: reader,
                    legacyBundleDirectory: legacyBundles,
                    maxSidecars: environment["DS4_GLM_LAYERQ4_LAYERS"]
                        .flatMap(Int.init)) { layer, built in
                    log("DS4Demo: pack blk\(layer) "
                        + (built ? "creato" : "già valido"))
                }
                log(String(
                    format: "DS4Demo: pack unificati in %.0f s — %d creati, "
                        + "%d già validi, %d mancanti%@ (dir %@)",
                    Date().timeIntervalSince(start), summary.created,
                    summary.alreadyValid, summary.remaining,
                    summary.stoppedBecause.map { " — stop: \($0)" } ?? "",
                    sidecarDir))
                exit(0)
            }
            // Auto-tune dei knob di caricamento (DS4_GLM_AUTOTUNE=1):
            // l'analogo pragmatico del record-holder DeepSeek — prova le
            // alternative ESATTE un gradino alla volta ricaricando il
            // motore (~3 s a load) e stampa il campione. Solo knob a
            // logits invariati; l'ambiente resta sulla config vincente.
            if environment["DS4_GLM_AUTOTUNE"] == "1" {
                let tuneTokens = try tokenizer.encodeChatPrompt(
                    prompt: prompt)
                let outcome = try GLM52ResidentModel.autoTune(
                    runtime: runtime, path: ggufPath, prompt: tuneTokens,
                    genTokens: min(16, maxNew)) {
                    log("DS4Demo: autotune " + $0)
                }
                log("DS4Demo: auto-tune GLM completato\n" + outcome.report)
                exit(0)
            }

            let loadStart = Date()
            let glm = try GLM52ResidentModel(
                runtime: runtime, path: ggufPath, options: glmOptions)
            let loadSeconds = Date().timeIntervalSince(loadStart)
            log(String(format: "DS4Demo: GLM caricato in %.1f s (%d layer, %d residenti)",
                       loadSeconds, glm.loadedLayerCount,
                       glmOptions.residentLayerCount ?? glm.loadedLayerCount))

            // Output standard della demo (stesse etichette del ramo
            // DeepSeek): prefill layer-major, "Risposta:", righe per-token
            // "[tok N ...]", totale, riga REGIME con DS4_WARMUP e profilo
            // streaming come rapporto per-fase.
            let tokens = try tokenizer.encodeChatPrompt(prompt: prompt)
            let prefillStart = Date()
            var logits = try glm.prefill(tokens)
            let prefillSeconds = Date().timeIntervalSince(prefillStart)
            log(String(format: "DS4Demo: prefill %d token (layer-major) in %.1fs (%.2f tok/s)",
                       tokens.count, prefillSeconds,
                       Double(tokens.count) / max(prefillSeconds, 0.001)))
            log("")
            log(glm.profileReport(title: "Profilo prefill"))
            log("DS4Demo: streaming prefill — " + glm.streamingReport())
            glm.resetStreamingStats()

            let stdout = FileHandle.standardOutput
            stdout.write(Data("\nRisposta: ".utf8))
            let warmup = environment["DS4_WARMUP"].flatMap(Int.init) ?? 0
            let decodeStart = Date()
            var steadyStart = decodeStart
            var produced = 0
            // DS4_GLM_SPEC_K=N (>=2): decode speculativo prompt-lookup —
            // il draft è il transcript stesso (n-gramma 4→2) e la VERIFICA
            // è un forward layer-major della finestra (pesi letti una volta
            // per l'intera finestra: è l'unico modo di battere il tetto
            // dell'I/O per token). Accettazione greedy; il rifiuto fa
            // rollback delle cache e ricommitta solo il prefisso valido.
            let specK = DS4RuntimeEnvironment.integer(
                .speculativeTokens,
                backend: .glm52,
                environment: environment) ?? 0
            if specK >= 2 {
                log("DS4Demo: decode speculativo prompt-lookup GLM — "
                    + "finestra \(specK), n-gramma 4→2")
                var history = tokens.map { Int($0) }
                var rounds = 0, drafted = 0, accepted = 0, misses = 0
                specLoop: while produced < maxNew {
                    guard let tP = GLM52GreedyDecoding.argmax(logits) else {
                        break
                    }
                    produced += 1
                    if tokenizer.isStopToken(tP, reasoning: .none) { break }
                    stdout.write(Data(tokenizer.tokenText(tP)))
                    history.append(Int(tP))
                    if produced >= maxNew { break }
                    let cands = PromptLookup.draft(history: history,
                                                   count: specK - 1)
                    let t0 = Date()
                    if cands.isEmpty {
                        misses += 1
                        logits = try glm.forwardNext(tP)
                        let dt = Date().timeIntervalSince(t0)
                        log(String(format: "  [tok %d  %.1fs  %.2f tok/s]",
                                   produced, dt, dt > 0 ? 1.0 / dt : 0))
                        continue
                    }
                    let base = glm.position
                    let window: [Int32] = [tP] + cands.map(Int32.init)
                    var logitsAll = try glm.forwardBatch(window)
                    var j = 0
                    while j + 1 < window.count {
                        guard let a = GLM52GreedyDecoding.argmax(
                                  logitsAll[j]),
                              a == window[j + 1] else { break }
                        j += 1
                    }
                    rounds += 1
                    drafted += window.count - 1
                    accepted += j
                    if j + 1 < window.count {
                        try glm.rollback(to: base)
                        logitsAll = try glm.forwardBatch(
                            Array(window.prefix(j + 1)))
                    }
                    if j >= 1 {
                        for candidate in window[1...j] {
                            if tokenizer.isStopToken(candidate,
                                                     reasoning: .none) {
                                break specLoop
                            }
                            stdout.write(Data(
                                tokenizer.tokenText(candidate)))
                            history.append(Int(candidate))
                            produced += 1
                            if produced >= maxNew { break }
                        }
                    }
                    logits = logitsAll[j]
                    let dt = Date().timeIntervalSince(t0)
                    log(String(format: "  [round %d  +%d tok  %.1fs  %.2f tok/s  accept %.2f  miss %d]",
                               rounds, j + 1, dt,
                               dt > 0 ? Double(j + 1) / dt : 0,
                               drafted > 0
                                   ? Double(accepted) / Double(drafted) : 0,
                               misses))
                }
                if rounds > 0 || misses > 0 {
                    log(String(format: "DS4Demo: prompt-lookup — %d round speculativi, %d forward diretti (miss), accettazione draft %.0f%%",
                               rounds, misses,
                               drafted > 0
                                   ? 100 * Double(accepted) / Double(drafted)
                                   : 0))
                }
            } else {
            // Greedy con ARGMAX SUL DEVICE: dal secondo token in poi i
            // logits non tornano sul host (readback 4 byte, stessa regola
            // di pareggio dell'argmax CPU — output identico).
            var current = GLM52GreedyDecoding.argmax(logits)
            while produced < maxNew, let token = current {
                produced += 1
                if tokenizer.isStopToken(token, reasoning: .none) { break }
                stdout.write(Data(tokenizer.tokenText(token)))
                if produced == maxNew { break }
                let t0 = Date()
                current = try glm.forwardNextGreedy(token)
                let dt = Date().timeIntervalSince(t0)
                log(String(format: "  [tok %d  %.1fs  %.2f tok/s]",
                           produced, dt, dt > 0 ? 1.0 / dt : 0))
                if produced == warmup {
                    // I primi token pagano costi una-tantum (arena fredda,
                    // page cache, wiring): il profilo riparte da qui e la
                    // media di regime è riportata a parte, come in DeepSeek.
                    glm.resetStreamingStats()
                    steadyStart = Date()
                }
            }
            }
            stdout.write(Data("\n".utf8))
            let decodeSeconds = Date().timeIntervalSince(decodeStart)
            log(String(format: "DS4Demo: %d tokens in %.1fs (%.2f tok/s)",
                       produced, decodeSeconds,
                       Double(produced) / max(decodeSeconds, 0.001)))
            if warmup > 0, produced > warmup {
                let steadyTokens = produced - warmup
                let steadySeconds = Date().timeIntervalSince(steadyStart)
                log(String(format: "DS4Demo: REGIME (dal token %d): %d token in %.1fs (%.2f tok/s) — profilo sotto = solo regime",
                           warmup + 1, steadyTokens, steadySeconds,
                           Double(steadyTokens) / max(steadySeconds, 0.001)))
            }
            log("")
            log(glm.profileReport(title: "Profilo decode"))
            log("DS4Demo: streaming decode — " + glm.streamingReport())
            log(String(format: "DS4Demo: totale %.1fs (load %.1f + prefill %.1f + decode %.1f)",
                       Date().timeIntervalSince(loadStart), loadSeconds,
                       prefillSeconds, decodeSeconds))
            // Usage imatrix persistita tra i run (come la .usage.json di
            // DeepSeek): <gguf>.glm-usage.json, DS4_USAGE_FILE=off per
            // disattivare.
            glm.saveUsageProfile()
            exit(0)
        }
        if detectedArchitecture.family == .laguna, LagunaRuntimeGate.enabled {
            // Laguna S 2.1 demo on the first-cut resident engine (bring-up
            // grade: full residency, token-by-token prefill). Mirrors the
            // reference CLI: default Poolside system prompt, the family
            // sampling defaults (temp 0.7, top-k 20, top-p 0.95, min-p 0.05
            // — DS4_DEMO_TEMPERATURE=0 for greedy parity runs), thinking on
            // by default (DS4_NOTHINK=1 for direct replies, with the
            // think-mode stop policy of `ds4_token_is_stop_for_think_mode`).
            // Prompt from DS4_PROMPT; token budget from DS4_MAX_NEW;
            // DS4_LAGUNA_LAYERS truncates the stack for bring-up runs.
            let environment = ProcessInfo.processInfo.environment
            var prompt = environment["DS4_PROMPT"]
                ?? "Ciao! Presentati in una frase."
            if let file = environment["DS4_PROMPT_FILE"], !file.isEmpty {
                let path = (file as NSString).expandingTildeInPath
                guard var text = try? String(
                    contentsOfFile: path, encoding: .utf8
                ) else {
                    log("DS4Demo: impossibile leggere il file prompt '\(path)'")
                    exit(2)
                }
                let cap = environment["DS4_PROMPT_MAX_CHARS"]
                    .flatMap(Int.init) ?? 12_000
                if text.count > cap {
                    text = String(text.prefix(cap))
                    log("DS4Demo: prompt Laguna troncato a \(cap) caratteri")
                }
                prompt = text
            }
            let maxNew = Int(environment["DS4_MAX_NEW"] ?? "") ?? 128
            let reasoning: ThinkMode =
                environment["DS4_NOTHINK"] == "1" ? .none : .high
            let defaults = LagunaConversationProtocol.SamplingDefaults.self
            let temperature = environment["DS4_DEMO_TEMPERATURE"].flatMap(Float.init)
                ?? defaults.temperature
            let topK = environment["DS4_DEMO_TOP_K"].flatMap(Int.init)
                ?? defaults.topK
            let topP = environment["DS4_DEMO_TOP_P"].flatMap(Float.init)
                ?? defaults.topP
            let minP = environment["DS4_DEMO_MIN_P"].flatMap(Float.init)
                ?? defaults.minP
            var rng = UInt64(environment["DS4_DEMO_SEED"].flatMap(Int.init) ?? 1)
            let samplingLabel = temperature <= 0
                ? "greedy"
                : String(format: "temp %.2f, top-k %d, top-p %.2f, min-p %.2f",
                         temperature, topK, topP, minP)
            log("DS4Demo: Laguna S 2.1 (\(samplingLabel)"
                + (reasoning.enabled ? ", think" : ", nothink")
                + ") — prompt: \(prompt)")
            let tokenizer = try LagunaTokenizer(model: model)
            // Quant-format audit (DS4_TYPES_ONLY=1), il gemello del blocco
            // DeepSeek: stampa i dtype che il motore assume (ricetta Q8_0 sul
            // segnale, esperti instradati Q2_K/Q3_K/Q4_K per layer nel file
            // misto), gli id speciali e i token del prompt. Mismatch => garbage.
            // Nessun Metal: utile anche su macchine senza RAM per il modello.
            if environment["DS4_TYPES_ONLY"] != nil {
                func typeLine(_ nm: String) -> String {
                    if let t = model.findTensor(nm) {
                        return "  TYPE \(nm) = \(t.typeName) (code \(t.type))"
                    }
                    return "  TYPE \(nm) = <missing>"
                }
                for nm in ["token_embd.weight", "output_norm.weight",
                           "output.weight", "blk.0.attn_q.weight",
                           "blk.0.attn_gate.weight", "blk.0.ffn_gate.weight",
                           "blk.0.ffn_down.weight",
                           "blk.1.ffn_gate_inp.weight",
                           "blk.1.exp_probs_b.bias",
                           "blk.1.ffn_gate_shexp.weight"] {
                    log(typeLine(nm))
                }
                var layer = 1
                while model.findTensor("blk.\(layer).attn_norm.weight") != nil {
                    let types = ["ffn_gate_exps", "ffn_up_exps", "ffn_down_exps"]
                        .map {
                            model.findTensor("blk.\(layer).\($0).weight")?
                                .typeName ?? "<missing>"
                        }
                    log("  ROUTED blk.\(String(format: "%2d", layer)) "
                        + "gate/up/down = " + types.joined(separator: "/"))
                    layer += 1
                }
                log("  SPECIAL bos=\(tokenizer.special.beginOfSequence)"
                    + " eos=\(tokenizer.special.endOfSequence)"
                    + " eot=\(tokenizer.special.endOfTurn)"
                    + " assistant=\(tokenizer.special.assistant)"
                    + " think=\(tokenizer.special.thinkOpen)"
                    + "/\(tokenizer.special.thinkClose)"
                    + " tool=\(tokenizer.special.toolCallOpen)"
                    + "/\(tokenizer.special.toolCallClose)")
                let ids = try tokenizer.encodeChatPrompt(
                    system: environment["DS4_SYSTEM"]
                        ?? LagunaConversationProtocol.defaultSystemPrompt,
                    prompt: prompt,
                    reasoning: reasoning
                )
                log("  PROMPT ids (\(ids.count)) = \(ids)")
                for id in ids {
                    let text = String(bytes: tokenizer.tokenText(id),
                                      encoding: .utf8) ?? "?"
                    log("    \(id) -> '\(text)'")
                }
                exit(0)
            }
            let runtime = try MetalRuntime()
            var options = LagunaResidentModelOptions()
            options.cacheCapacity = maxKeys
            options.initialFullCacheCapacity =
                DS4RuntimeEnvironment.integer(
                    .kvInitial,
                    backend: .laguna,
                    environment: environment)
            options.layerCount = environment["DS4_LAGUNA_LAYERS"].flatMap(Int.init)
            // Streaming SSD sperimentale degli esperti instradati
            // (divergenza dichiarata dal C, che per Laguna impone la
            // residenza): segnale Q8_0 residente, ~1.6 GB di letture per
            // token. Rende eseguibile il file da 45 GiB su macchine da
            // 32 GB, a pochi tok/s.
            options.expertStreaming = DS4RuntimeEnvironment.flag(
                .ssdStream,
                backend: .laguna,
                default: false,
                environment: environment)
            // Cache LRU degli slab esperti streamati (slot da un esperto,
            // stile ExpertSlotCache DeepSeek): gli hit non pagano né SSD né
            // copia. DS4_EXPERT_CACHE_MB regola il budget; default
            // 2048 quando lo streaming è attivo: sul target M1 Pro 16 GB
            // è il miglior compromesso end-to-end per prefill e decode.
            // Budget più larghi riducono l'I/O ma introducono pressione
            // memoria; 0 disattiva la cache.
            options.expertCacheMB =
                DS4RuntimeEnvironment.integer(
                    .expertCacheMB,
                    backend: .laguna,
                    environment: environment)
                ?? (options.expertStreaming ? 2_048 : 0)
            options.expertCacheSlots = DS4RuntimeEnvironment.integer(
                .expertCacheSlots,
                backend: .laguna,
                environment: environment)
            options.activeExperts = DS4RuntimeEnvironment.integer(
                .activeExperts,
                backend: .laguna,
                environment: environment)
            options.residentExpertLayers = DS4RuntimeEnvironment.integer(
                .residentLayers,
                backend: .laguna,
                environment: environment)
            options.prefillChunk = DS4RuntimeEnvironment.integer(
                .prefillChunk,
                backend: .laguna,
                environment: environment)
            options.expertPread = DS4RuntimeEnvironment.flag(
                .expertPread,
                backend: .laguna,
                default: true,
                environment: environment)
            options.willNeedExperts = DS4RuntimeEnvironment.flag(
                .willNeedExperts,
                backend: .laguna,
                default: true,
                environment: environment)
            options.preadSplit = DS4RuntimeEnvironment.integer(
                .preadSplit,
                backend: .laguna,
                environment: environment) ?? 1
            options.metalIO = DS4RuntimeEnvironment.flag(
                .metalIO,
                backend: .laguna,
                default: false,
                environment: environment)
            options.lockResident = DS4RuntimeEnvironment.flag(
                .mlock,
                backend: .laguna,
                default: false,
                environment: environment)
            options.simdgroupsPerThreadgroup =
                DS4RuntimeEnvironment.integer(
                    .simdgroups,
                    backend: .laguna,
                    environment: environment)
            options.longAttentionIndex = DS4RuntimeEnvironment.flag(
                .indexedAttention,
                backend: .laguna,
                default: true,
                environment: environment)
            options.longAttentionBlockSize =
                DS4RuntimeEnvironment.integer(
                    .longAttentionBlock,
                    backend: .laguna,
                    environment: environment)
            options.longAttentionTopBlocks =
                DS4RuntimeEnvironment.integer(
                    .longAttentionTopBlocks,
                    backend: .laguna,
                    environment: environment)
            options.longAttentionRecentTokens =
                DS4RuntimeEnvironment.integer(
                    .longAttentionRecent,
                    backend: .laguna,
                    environment: environment)
            options.longAttentionThreshold =
                DS4RuntimeEnvironment.integer(
                    .longAttentionThreshold,
                    backend: .laguna,
                    environment: environment)
            let loadStart = Date()
            let laguna = try LagunaResidentModel(
                runtime: runtime, path: ggufPath, options: options
            )
            let loadSeconds = Date().timeIntervalSince(loadStart)
            let streamingNote = laguna.isExpertStreaming
                ? ", esperti in streaming SSD"
                    + (laguna.expertCacheSlots > 0
                           ? " · cache \(laguna.expertCacheSlots) slot"
                               + (laguna.isMultiQuantExpertCacheEnabled
                                    ? " mixed \(laguna.expertCacheCompatibleSlotRange.lowerBound)"
                                        + "...\(laguna.expertCacheCompatibleSlotRange.upperBound)/classe"
                                    : "")
                               + " (\(laguna.expertCacheAllocatedBytes >> 20) MiB)"
                           : "")
                    + (laguna.isExpertPreadEnabled
                           ? " · pread×\(laguna.expertPreadSplit)"
                           : " · mmap")
                    + (laguna.isMetalIOEnabled ? " · MetalIO" : "")
                : ""
            let effectiveChunk = laguna.effectivePrefillChunkSize
            log(String(
                format: "DS4Demo: Laguna caricato in %.1fs (%d layer · top-%d"
                    + " · chunk %d%@%@ · KV %d MiB · decode %@%@"
                    + " · weights %@ · mmap %@)",
                loadSeconds, laguna.loadedLayerCount,
                laguna.activeExpertCount, laguna.prefillChunkSize,
                effectiveChunk == laguna.prefillChunkSize
                    ? "" : "→\(effectiveChunk)",
                streamingNote,
                laguna.allocatedKVCacheBytes >> 20,
                laguna.isChainedDecodeEnabled ? "chained" : "sync",
                laguna.isDecodeSplitKEnabled ? "+splitK" : "",
                laguna.usesPrivateResidentWeights ? "private" : "shared",
                laguna.discardsUploadedFilePages ? "drop" : "keep"))
            if laguna.residentExpertLayerCount > 0 {
                log("DS4Demo: Laguna routed layer residenti = "
                    + "\(laguna.residentExpertLayerCount)")
            }
            if laguna.isLongAttentionIndexEnabled {
                let indexed = laguna.longAttentionConfiguration
                log("DS4Demo: Laguna attention lunga indicizzata"
                    + " — blocco \(indexed.blockSize)"
                    + " · top-\(indexed.topBlocks) blocchi"
                    + " · recente \(indexed.recentTokens)"
                    + " · soglia \(indexed.threshold)")
            }
            log("DS4Demo: Laguna shared expert/I/O overlap = "
                + (laguna.isSharedExpertIOOverlapEnabled ? "on" : "off"))
            // The reference CLI passes the Poolside default system prompt
            // into `ds4_encode_chat_prompt` when none is given.
            var tokens = try tokenizer.encodeChatPrompt(
                system: environment["DS4_SYSTEM"]
                    ?? LagunaConversationProtocol.defaultSystemPrompt,
                prompt: prompt,
                reasoning: reasoning
            )
            if let target = syntheticLiveContext {
                guard target >= tokens.count,
                      target + maxNew + 1 <= maxKeys else {
                    log("DS4Demo: il contesto sintetico Laguna deve essere "
                        + "almeno \(tokens.count) token e lasciare spazio a "
                        + "DS4_MAX_NEW+1 entro DS4_DEMO_CONTEXT")
                    exit(2)
                }
                // Keep the complete, valid chat frame and insert ordinary
                // prompt pieces immediately before Assistant/Think. This is
                // a performance frontier, not a semantic quality benchmark.
                let suffixCount = 2
                let suffix = Array(tokens.suffix(suffixCount))
                var synthetic = Array(tokens.dropLast(suffixCount))
                let encodedFiller = tokenizer.tokenize(prompt + " ")
                let filler = encodedFiller.isEmpty
                    ? [tokenizer.special.endOfSequence] : encodedFiller
                let contentEnd = target - suffixCount
                var fillerIndex = 0
                while synthetic.count < contentEnd {
                    synthetic.append(filler[fillerIndex % filler.count])
                    fillerIndex += 1
                }
                synthetic.append(contentsOf: suffix)
                tokens = synthetic
                log("DS4Demo: contesto sintetico Laguna esatto: "
                    + "\(tokens.count) token")
            }
            let prefillStart = Date()
            var logits = try laguna.prefill(tokens)
            let prefillSeconds = Date().timeIntervalSince(prefillStart)
            log(String(format: "DS4Demo: prefill %d token in %.1fs",
                       tokens.count, prefillSeconds))
            // Ripartizione per-fase del prompt (come la demo GLM): senza
            // questa riga la composizione del prefill non è mai visibile.
            log(laguna.profileReport(title: "Profilo prefill"))
            // Il profilo per-fase riparte qui: il report a fine decode
            // descrive solo il regime di generazione, come nella demo GLM.
            laguna.resetProfile()
            // Come il percorso GLM: DS4_WARMUP=N esclude i primi N token dal
            // regime, così il tok/s a regime non paga caches fredde e ramp-up.
            let warmup = environment["DS4_WARMUP"].flatMap(Int.init) ?? 0
            var generated: [Int32] = []
            let decodeStart = Date()
            var steadyStart = decodeStart
            for _ in 0..<maxNew {
                let next = Int32(Sampler.sample(
                    logits, temperature: temperature, topK: topK,
                    topP: topP, minP: minP, rng: &rng
                ))
                if tokenizer.isStopToken(next, reasoning: reasoning) { break }
                generated.append(next)
                let piece = tokenizer.tokenText(next)
                FileHandle.standardOutput.write(Data(piece))
                logits = try laguna.forwardNext(next)
                if generated.count == warmup {
                    steadyStart = Date()
                    laguna.resetProfile()
                }
            }
            FileHandle.standardOutput.write(Data("\n".utf8))
            let decodeSeconds = Date().timeIntervalSince(decodeStart)
            if !generated.isEmpty {
                log(String(format: "DS4Demo: %d token in %.1fs (%.2f tok/s)",
                           generated.count, decodeSeconds,
                           Double(generated.count) / max(decodeSeconds, 0.001)))
            }
            if warmup > 0, generated.count > warmup {
                let steadyTokens = generated.count - warmup
                let steadySeconds = Date().timeIntervalSince(steadyStart)
                log(String(format: "DS4Demo: REGIME (dal token %d): %d token in %.1fs (%.2f tok/s)",
                           warmup + 1, steadyTokens, steadySeconds,
                           Double(steadyTokens) / max(steadySeconds, 0.001)))
            }
            log("")
            log(laguna.profileReport(title: "Profilo decode"))
            log(String(format: "DS4Demo: totale %.1fs (load %.1f + prefill %.1f + decode %.1f)",
                       Date().timeIntervalSince(loadStart), loadSeconds,
                       prefillSeconds, decodeSeconds))
            exit(0)
        }
        if detectedArchitecture.family == .qwen
            || detectedArchitecture.family == .glm
            || detectedArchitecture.family == .laguna {
            log("DS4Demo: backend \(detectedArchitecture.id.rawValue) non ancora implementato")
        } else {
            log("DS4Demo: \(error)")
        }
        exit(2)
    }
    guard detectedArchitecture.id == .deepSeekV4 else {
        log("DS4Demo: architettura GGUF non supportata: \(detectedArchitecture.id.rawValue)")
        exit(2)
    }
    log("DS4Demo: architecture=\(detectedArchitecture.id.rawValue) model=\(model.string("general.name") ?? "GGUF")")
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
    let config = try ModelConfig(model: model)
    let geometry = DSV4RuntimeGeometry(configuration: config)
    var dims = geometry.dims
    let mq = GGUFWeights.detectMoEQuant(model)
    dims.gateQuant = mq.gate; dims.upQuant = mq.up; dims.downQuant = mq.down; dims.routerF16 = mq.routerF16
    log("DS4Demo: profilo=\(config.shape.name) layer=\(geometry.nLayers) esperti=\(dims.nExperts)")
    log("DS4Demo: MoE quant gate=\(mq.gate) up=\(mq.up) down=\(mq.down) routerF16=\(mq.routerF16)")
    // DS4_DSPARK_GGUF opens and strictly validates the multi-stage DSpark
    // support model. `=1` locates the checkpoint-matched canonical accessory
    // next to the main GGUF; an explicit path also supports custom conversions.
    // A valid support path is attached to the decoder below: target-hidden
    // capture + the resident stage-0 main projection then run on every target
    // forward. Transformer draft stages remain behind the next runtime gate.
    var dsparkRuntimePath: String?
    if let dsparkEnv = ProcessInfo.processInfo.environment["DS4_DSPARK_GGUF"],
       !dsparkEnv.isEmpty {
        let supportPath = dsparkEnv == "1"
            ? DSparkSupportModel.locate(near: ggufPath)
            : dsparkEnv
        if let path = supportPath, FileManager.default.fileExists(atPath: path) {
            do {
                let support = try DSparkSupportModel(
                    path: path,
                    targetDims: dims,
                    targetLayerCount: geometry.nLayers
                )
                log(support.report(mainModelPath: ggufPath))
                if !support.isRunnable(withMainModelPath: ggufPath) {
                    DSparkSupportModel.log("supporto rifiutato: correggi gli errori prima di abilitare il forward")
                } else {
                    dsparkRuntimePath = path
                }
            } catch {
                DSparkSupportModel.log("apertura supporto FALLITA (\(path)): \(error)")
            }
        } else {
            DSparkSupportModel.log(
                "supporto compatibile non trovato (DS4_DSPARK_GGUF=\(dsparkEnv)); "
                    + "scarica l'accessorio DSpark dalla GUI o passa il percorso esplicito"
            )
        }
    }
    // DS4_MTP_GGUF (Fase M1, docs/SELF-SPECULATIVE.md § Fase M): apre il
    // sidecar MTP e stampa inventario + validazione dell'interfaccia draft.
    // Solo diagnostica — nessun effetto sul decode. `=1` cerca *MTP*.gguf
    // accanto al modello; altrimenti è il percorso esplicito del sidecar.
    if let mtpEnv = ProcessInfo.processInfo.environment["DS4_MTP_GGUF"], !mtpEnv.isEmpty {
        let mtpPath = (mtpEnv == "1") ? MTPSidecar.locate(near: ggufPath) : mtpEnv
        if let p = mtpPath, FileManager.default.fileExists(atPath: p) {
            do {
                let side = try MTPSidecar(path: p)
                log(side.report(vocab: Int(dims.vocab), nEmbd: Int(dims.nEmbd)))
            } catch {
                MTPSidecar.log("apertura sidecar FALLITA (\(p)): \(error)")
            }
        } else {
            MTPSidecar.log("sidecar non trovato (DS4_MTP_GGUF=\(mtpEnv)) — scaricalo dal catalogo " +
                "(id 'mtp': DeepSeek-V4-Flash-MTP-Q4K-Q8_0-F32.gguf) o passa il percorso esplicito")
        }
    }
    // Optional: reduce active experts per token (DS4_ACTIVE_EXPERTS=2..6). Fewer
    // experts = less expert I/O (faster on RAM-starved machines), lower quality.
    if let s = ProcessInfo.processInfo.environment["DS4_ACTIVE_EXPERTS"], let kk = Int(s) {
        dims.activeExperts = max(1, min(kk, dims.k))
        log("DS4Demo: active experts (top-k) = \(dims.activeExperts) of \(dims.k)")
    }
    let rope = geometry.ropeParams(layer: 0)
    // Fast path: non-routed weights resident (memoized), only the 6 selected experts
    // gathered per token (~6/256 of expert IO) — the C --ssd-streaming model.
    let dec: StreamingDecoder
    if let shardSet {
        // Pro Q4 split: resident per-layer load routed to the owning shard.
        dec = try StreamingDecoder.fromGGUFShards(rt: rt, shards: shardSet, dims: dims, rope: rope,
                                                  nLayers: geometry.nLayers, maxKeys: maxKeys,
                                                  geometry: geometry)
        log("DS4Demo: multi-shard resident decoder (Pro Q4 split, per-layer shard routing)…")
    } else {
        dec = try StreamingDecoder.fromGGUFExpertCachedMapped(rt: rt, model: model, dims: dims, rope: rope,
                                                              nLayers: geometry.nLayers, maxKeys: maxKeys,
                                                              geometry: geometry)
        log("DS4Demo: no-copy mmap non-routed + gather 6 experts/token (C --ssd-streaming model)…")
    }
    if let dsparkRuntimePath {
        do {
            try dec.enableDSparkStage0(supportPath: dsparkRuntimePath,
                                       mainModelPath: ggufPath)
            let runtime = dec.dsparkStage0Runtime
            let residentMiB = (runtime?.residentWeightBytes ?? 0) >> 20
            let mappedGiB = Double(runtime?.mappedTransformerBytes ?? 0)
                / Double(1 << 30)
            let kvMiB = (runtime?.privateKVBytes ?? 0) >> 20
            DSparkStage0Runtime.log(String(format:
                "capture/stage0 + 3 transformer legati: %.2f GiB mmap, %d MiB residenti, KV privato %d MiB",
                mappedGiB, residentMiB, kvMiB))
        } catch {
            DSparkStage0Runtime.log("stage 0 non attivato: \(error)")
        }
    }
    // Persistenza della usage imatrix TRA i run della demo (l'app la persiste
    // per agente). Senza, i pool della slot-cache nascono al primo token con
    // storia vuota: l'allocazione usage-driven resterebbe un'anteprima nella
    // tabella diagnostica senza mai applicarsi. Col file, dal secondo run i
    // pool partono GIÀ caldi (pre-warm sugli esperti storicamente hot) e con
    // la ridistribuzione attiva. Default: <gguf>.usage.json; DS4_USAGE_FILE
    // per cambiare percorso, DS4_USAGE_FILE=off per disattivare.
    let usagePath = ProcessInfo.processInfo.environment["DS4_USAGE_FILE"] ?? (ggufPath + ".usage.json")
    if usagePath != "off", let data = FileManager.default.contents(atPath: usagePath) {
        dec.usage?.load(data)
        log("DS4Demo: usage imatrix caricata (\(dec.usage?.totalRoutes ?? 0) route) da \(usagePath)")
    }
    let t0 = Date()
    let logits = try dec.forward(token: 0, pos: 0, nKeys: 1)
    let dt = Date().timeIntervalSince(t0)
    let finite = logits.allSatisfy { $0.isFinite }
    var argmax = 0; var best = -Float.greatestFiniteMagnitude
    for (i, v) in logits.enumerated() where v > best { best = v; argmax = i }
    log(String(format: "DS4Demo: 1 forward in %.1fs — logits[%d] finite=%@ argmax=%d (logit %.3f)",
               dt, logits.count, finite ? "YES" : "NO", argmax, best))
    if let position = dec.dsparkStage0Position,
       let hidden = dec.dsparkStage0Hidden() {
        let rms = sqrt(hidden.reduce(0) { $0 + $1 * $1 } / Float(max(1, hidden.count)))
        DSparkStage0Runtime.log(String(
            format: "stage 0 pronto a pos=%d — hidden=%d RMS=%.5f",
            position, hidden.count, rms))
    }
    if maxNew > 0 {
        // Real chat generation: tokenize the prompt (3rd arg) with the model's
        // tokenizer + chat template, greedy-decode, detokenize, print the answer.
        var prompt = args.count >= 4 ? args[3] : "ciao come stai? rispondi in 1 parola"
        // "@/percorso/file": il prompt e' il CONTENUTO del file — niente quoting
        // shell per i testi lunghi (benchmark del prefill). Troncato a
        // DS4_PROMPT_MAX_CHARS (default 12000 ≈ 3k token) per stare nel KV
        // configurato della demo insieme ai token generati.
        // DS4_PROMPT_FILE: come "@path" ma via env — immune allo splitting
        // della shell sugli argomenti (visto in campo: il prompt arrivava
        // spezzato alla prima parola nonostante le virgolette).
        if let pf = ProcessInfo.processInfo.environment["DS4_PROMPT_FILE"], !pf.isEmpty {
            prompt = "@" + pf
        }
        if prompt.hasPrefix("@") {
            let path = (String(prompt.dropFirst()) as NSString).expandingTildeInPath
            guard var text = try? String(contentsOfFile: path, encoding: .utf8) else {
                log("DS4Demo: impossibile leggere il file prompt '\(path)'")
                exit(2)
            }
            let cap = ProcessInfo.processInfo.environment["DS4_PROMPT_MAX_CHARS"].flatMap(Int.init) ?? 12_000
            if text.count > cap {
                text = String(text.prefix(cap))
                log("DS4Demo: prompt dal file \(path): troncato a \(cap) caratteri (DS4_PROMPT_MAX_CHARS per cambiare)")
            } else {
                log("DS4Demo: prompt dal file \(path): \(text.count) caratteri")
            }
            prompt = text
        }
        let tok = try Tokenizer(model: model)
        // DS4_RAW_PROMPT=1: equivalente di `ds4 --raw-prompt` — testo puro,
        // niente template chat e niente BOS (ds4_tokenize_text non lo mette).
        // Serve agli A/B cross-engine: stessa sequenza di token nei due motori.
        let rawPrompt = ProcessInfo.processInfo.environment["DS4_RAW_PROMPT"] == "1"
        var ids = rawPrompt
            ? tok.tokenize(prompt).map { Int($0) }
            : tok.encodeChatPrompt(system: nil, prompt: prompt, think: .none).map { Int($0) }
        if rawPrompt { log("DS4Demo: prompt RAW (senza template): \(ids.count) token") }
        if let target = syntheticLiveContext {
            // Exact live-context benchmark: preserve a valid chat frame while
            // tiling only ordinary user-text tokens. This avoids depending on
            // bytes/token of an external corpus and makes A/B frontiers exact.
            let suffix = [Int(tok.assistantId), Int(tok.thinkEndId)]
            var synthetic = [Int(tok.bosId), Int(tok.userId)]
            let encodedFiller = tok.tokenize(prompt + " ").map { Int($0) }
            let filler = encodedFiller.isEmpty ? [Int(tok.eosId)] : encodedFiller
            let contentEnd = target - suffix.count
            var i = 0
            while synthetic.count < contentEnd {
                synthetic.append(filler[i % filler.count])
                i += 1
            }
            if synthetic.count > contentEnd { synthetic.removeLast(synthetic.count - contentEnd) }
            synthetic.append(contentsOf: suffix)
            ids = synthetic
            log("DS4Demo: contesto sintetico esatto: \(ids.count) token (frame BOS/User/Assistant preservato)")
        }
        // DS4_DUMP_TOKENS=/path: scrive la sequenza token nel formato di
        // `ds4 --dump-tokens` ("[id, id, ...]") per il diff cross-engine.
        if let dumpPath = ProcessInfo.processInfo.environment["DS4_DUMP_TOKENS"], !dumpPath.isEmpty {
            let line = "[" + ids.map(String.init).joined(separator: ", ") + "]\n"
            try? line.write(toFile: dumpPath, atomically: true, encoding: .utf8)
            log("DS4Demo: dump token → \(dumpPath) (\(ids.count) token)")
        }
        // Sampling defaults remain greedy for benchmark/backward compatibility,
        // but the 2-bit model can fall into a coherent yet unrelated greedy
        // continuation after one quantization flip. Demo-specific env knobs let
        // users exercise the same focused top-k path as the GUI.
        let sampleTemperature = ProcessInfo.processInfo.environment["DS4_DEMO_TEMPERATURE"].flatMap(Float.init) ?? 0
        let sampleTopK = ProcessInfo.processInfo.environment["DS4_DEMO_TOP_K"].flatMap(Int.init) ?? 0
        let sampleTopP = ProcessInfo.processInfo.environment["DS4_DEMO_TOP_P"].flatMap(Float.init) ?? 1
        let sampleMinP = ProcessInfo.processInfo.environment["DS4_DEMO_MIN_P"].flatMap(Float.init) ?? 0
        let sampleRepeat = ProcessInfo.processInfo.environment["DS4_DEMO_REPEAT_PENALTY"].flatMap(Float.init) ?? 1
        let sampleRepeatN = max(0, ProcessInfo.processInfo.environment["DS4_DEMO_REPEAT_LAST_N"].flatMap(Int.init) ?? 64)
        let shown = prompt.count > 120 ? String(prompt.prefix(120)) + "…" : prompt
        let samplingLabel = sampleTemperature <= 0
            ? "greedy"
            : String(format: "temp %.2f, top-k %d, top-p %.2f, min-p %.2f, repeat %.2f",
                     sampleTemperature, sampleTopK, sampleTopP, sampleMinP, sampleRepeat)
        log("DS4Demo: prompt '\(shown)' (\(prompt.count) car.) -> \(ids.count) tokens; generating \(maxNew) (\(samplingLabel), streaming)…")
        if ids.count + maxNew + 1 > maxKeys {
            log("DS4Demo: ERRORE il prompt (\(ids.count) token) + \(maxNew) generati supera il KV della demo (maxKeys \(maxKeys)) — abbassa DS4_PROMPT_MAX_CHARS o aumenta DS4_DEMO_CONTEXT")
            exit(2)
        }
        let stdout = FileHandle.standardOutput
        // Prefill: LAYER-MAJOR — load each layer's weights once and apply to all
        // prompt tokens (amortizes the dominant weight I/O). Returns the last
        // token's logits; KV cache is populated for positions 0..N-1.
        dec.resetProfile()   // il profilo qui sotto misura SOLO il prefill
        let pf0 = Date()
        var last = try dec.prefill(tokens: ids)
        var pos = ids.count
        let pfS = Date().timeIntervalSince(pf0)
        log(String(format: "DS4Demo: prefill %d token (layer-major) in %.1fs (%.2f tok/s)",
                   ids.count, pfS, pfS > 0 ? Double(ids.count) / pfS : 0))
        abTrace?.capture(phase: "prefill", step: 0, inputToken: ids.last, logits: last)
        // Per-phase prefill breakdown (route/attn vs gather IO vs experts): the
        // phases are timed by the same counters as the decode profile, reset at
        // the prefill boundary above. gather IO is the EXPOSED (non-overlapped)
        // wait — the pipelined group I/O that ran under the GPU doesn't show.
        if diag {
            log(dec.profile.report(title: "Profilo prefill"))
            log("")
        }
        // Decode: stream each token's bytes to stdout AS it is produced (like ds4).
        dec.resetProfile()   // profila solo la fase di decode (non il prefill)
        // Warm-up escluso dal profilo: i primi token pagano costi una-tantum
        // (wiring dei densi residenti, riassestamento della memoria, cache
        // fredde) che su run corti dominano la media e falsano la diagnosi.
        // Il profilo riparte dal token `warmup`+1; il tempo di regime è
        // riportato a parte. DS4_WARMUP per cambiarlo (0 = come prima).
        let warmup = ProcessInfo.processInfo.environment["DS4_WARMUP"].flatMap(Int.init)
            ?? (diag ? min(4, max(0, maxNew - 1)) : 0)
        stdout.write(Data("\nRisposta: ".utf8))
        var rng: UInt64 = 1
        var recent = ids
        var genTokens = 0
        var generatedTraceTokens: [Int] = []
        generatedTraceTokens.reserveCapacity(maxNew)
        let genStart = Date()
        var steadyStart = genStart
        // DS4_SPEC_K=N (≥2): decode SELF-SPECULATIVE greedy (docs/SELF-SPECULATIVE.md).
        // Round: snapshot → draft di N-1 candidati con DS4_SPEC_DRAFT_EXPERTS
        // (default 2) esperti attivi → rollback → verifica batch full-config con
        // logit per posizione → accettazione greedy del prefisso + bonus token.
        // Parità bit-per-bit col decode normale per costruzione (stessi argmax
        // full-config): DS4_SPEC_K=1 o assente = percorso storico.
        let requestedSpecK = ProcessInfo.processInfo.environment["DS4_SPEC_K"].flatMap(Int.init) ?? 0
        let specK = (sampleTemperature <= 0 && sampleRepeat <= 1) ? requestedSpecK : 0
        let dsparkSpec = dec.dsparkStage0Runtime != nil
            && sampleTemperature <= 0 && sampleRepeat <= 1
            && requestedSpecK < 2 && abTrace == nil
        if requestedSpecK >= 2 && specK == 0 {
            log("DS4Demo: self-speculative disattivata — richiede sampling greedy senza repetition penalty")
        }
        // DS4_SPEC_DRAFT=ngram (prompt-lookup, docs/SELF-SPECULATIVE.md § Fase N):
        // il draft non è un modello ma il transcript stesso — l'occorrenza più
        // recente del suffisso corrente propone la sua continuazione. Zero
        // forward di draft: quando il lookup non trova nulla il round degrada a
        // un forward normale, quindi si paga la verifica SOLO dove c'è davvero
        // da copiare (codice, output di tool, testo ripetitivo). Stessa
        // verifica batch full-config e stessa accettazione greedy del loop a
        // esperti ridotti: parità bit-per-bit col decode normale per costruzione.
        let specDraft = ProcessInfo.processInfo.environment["DS4_SPEC_DRAFT"] ?? "experts"
        if dsparkSpec {
            var rounds = 0, proposed = 0, accepted = 0, misses = 0
            log("DS4Demo: decode DSpark — transformer support + Markov/confidence + verifica target")
            dsparkLoop: while genTokens < maxNew, pos < maxKeys {
                let remaining = min(maxNew - genTokens, maxKeys - pos)
                var committed: [Int] = []
                let roundStart = Date()
                do {
                    if let current = recent.last,
                       let proposal = try dec.dsparkPropose(
                            currentToken: current, position: pos - 1),
                       !proposal.tokens.isEmpty {
                        var candidates = Array(proposal.tokens.prefix(remaining))
                        if let eosIndex = candidates.firstIndex(
                            where: { Int32($0) == tok.eosId }) {
                            candidates = Array(candidates.prefix(upTo: eosIndex))
                        }
                        proposed += candidates.count
                        if !candidates.isEmpty {
                            let result = try dec.dsparkVerifyAndCommit(
                                proposal: candidates,
                                currentLogits: last, startPos: pos)
                            committed = result.acceptedTokens
                            if !committed.isEmpty { last = result.nextLogits }
                        }
                    }
                } catch {
                    dec.disableDSpark()
                    log("DS4Demo: DSpark disattivato dopo errore runtime: \(error)")
                }

                if committed.isEmpty {
                    misses += 1
                    let next = Sampler.sample(
                        last, temperature: 0, topK: 0,
                        topP: 1, minP: 0, rng: &rng)
                    if Int32(next) == tok.eosId { break dsparkLoop }
                    stdout.write(Data(tok.tokenText(Int32(next))))
                    generatedTraceTokens.append(next)
                    recent.append(next)
                    last = try dec.forward(
                        token: next, pos: pos, nKeys: pos + 1)
                    pos += 1
                    genTokens += 1
                    committed = [next]
                } else {
                    rounds += 1
                    accepted += committed.count
                    for next in committed {
                        stdout.write(Data(tok.tokenText(Int32(next))))
                        generatedTraceTokens.append(next)
                        recent.append(next)
                        pos += 1
                        genTokens += 1
                        if genTokens == warmup {
                            dec.resetProfile()
                            steadyStart = Date()
                        }
                    }
                }
                if genTokens == warmup {
                    dec.resetProfile()
                    steadyStart = Date()
                }
                let elapsed = Date().timeIntervalSince(roundStart)
                log(String(format:
                    "  [DSpark +%d tok  %.1fs  %.2f tok/s  accept %.0f%%]",
                    committed.count, elapsed,
                    elapsed > 0 ? Double(committed.count) / elapsed : 0,
                    proposed > 0 ? 100 * Double(accepted) / Double(proposed) : 0))
            }
            log(String(format:
                "DS4Demo: DSpark — %d round verificati, %d fallback, %d/%d draft accettati (%.0f%%)",
                rounds, misses, accepted, proposed,
                proposed > 0 ? 100 * Double(accepted) / Double(proposed) : 0))
        } else if specK >= 2, specDraft == "ngram" {
            var history = ids
            var rounds = 0, drafted = 0, acceptedCands = 0, misses = 0
            log("DS4Demo: decode speculativo prompt-lookup — finestra \(specK), n-gramma 4→2")
            ngramLoop: while genTokens < maxNew {
                // Il prossimo token VERO esce sempre da logit full-config.
                let tP = Sampler.sample(last, temperature: 0, topK: 0, topP: 1, minP: 0, rng: &rng)
                if Int32(tP) == tok.eosId { break }
                stdout.write(Data(tok.tokenText(Int32(tP))))
                generatedTraceTokens.append(tP)
                history.append(tP)
                genTokens += 1
                if genTokens == warmup { dec.resetProfile(); steadyStart = Date() }
                if genTokens >= maxNew { break }
                // 1) draft dal transcript. Nessun match = nessuna speculazione:
                //    forward normale, niente snapshot/verifica da pagare.
                let cands = PromptLookup.draft(history: history, count: specK - 1)
                if cands.isEmpty {
                    misses += 1
                    last = try dec.forward(token: tP, pos: pos, nKeys: pos + 1)
                    pos += 1
                    continue
                }
                let t0 = Date()
                // 2) snapshot (serve solo per l'eventuale rollback: il draft
                //    n-gram NON tocca lo stato del modello) + verifica batch
                //    full-config con logit per posizione.
                let snap = dec.specSnapshot(nKeys: pos)
                var window = [tP]; window.append(contentsOf: cands)
                let logitsAll = try dec.specVerifyStep(tokens: window, startPos: pos)
                // 3) accettazione greedy: window[i+1] deve essere l'argmax
                //    full-config di logitsAll[i].
                var j = 0
                while j + 1 < window.count {
                    let a = Sampler.sample(logitsAll[j], temperature: 0, topK: 0, topP: 1, minP: 0, rng: &rng)
                    if a == window[j + 1] { j += 1 } else { break }
                }
                rounds += 1; drafted += window.count - 1; acceptedCands += j
                // 4) Il verify batch usa kernel di compressione numericamente
                //    diversi dal decode ordinario: rollback e replay one-token
                //    di OGNI prefisso accettato, anche quando l'intera finestra
                //    passa, per conservare l'identità greedy.
                dec.specRestore(snap)
                let replayLogits = try dec.specReplay(
                    window.prefix(j + 1), startPos: pos)
                // 5) emetti gli accettati; l'ultimo logit del replay dà il tP
                //    del round successivo (bonus token).
                if j >= 1 {
                    for c in window[1...j] {
                        if Int32(c) == tok.eosId { break ngramLoop }
                        stdout.write(Data(tok.tokenText(Int32(c))))
                        generatedTraceTokens.append(c)
                        history.append(c)
                        genTokens += 1
                        if genTokens == warmup { dec.resetProfile(); steadyStart = Date() }
                        if genTokens >= maxNew { break }
                    }
                }
                pos += j + 1
                last = replayLogits
                let dt = Date().timeIntervalSince(t0)
                log(String(format: "  [round %d  +%d tok  %.1fs  %.2f tok/s  accept %.2f  miss %d]",
                           rounds, j + 1, dt, dt > 0 ? Double(j + 1) / dt : 0,
                           drafted > 0 ? Double(acceptedCands) / Double(drafted) : 0, misses))
                if genTokens >= maxNew { break }
            }
            if rounds > 0 || misses > 0 {
                log(String(format: "DS4Demo: prompt-lookup — %d round speculativi, %d forward diretti (miss), accettazione draft %.0f%%",
                           rounds, misses,
                           drafted > 0 ? 100 * Double(acceptedCands) / Double(drafted) : 0))
            }
        } else if specK >= 2 {
            let fullExperts = dec.activeExpertsNow
            let draftExperts = max(1, min(fullExperts - 1,
                ProcessInfo.processInfo.environment["DS4_SPEC_DRAFT_EXPERTS"].flatMap(Int.init) ?? 2))
            var rounds = 0, drafted = 0, acceptedCands = 0
            log("DS4Demo: decode speculativo — finestra \(specK), draft a \(draftExperts) esperti (full \(fullExperts))")
            specLoop: while genTokens < maxNew {
                // Il prossimo token VERO esce sempre da logit full-config
                // (prefill al primo giro, verifica ai successivi).
                let tP = Sampler.sample(last, temperature: 0, topK: 0, topP: 1, minP: 0, rng: &rng)
                if Int32(tP) == tok.eosId { break }
                stdout.write(Data(tok.tokenText(Int32(tP))))
                generatedTraceTokens.append(tP)
                genTokens += 1
                if genTokens == warmup { dec.resetProfile(); steadyStart = Date() }
                if genTokens >= maxNew { break }
                let t0 = Date()
                // 1) snapshot dello stato ricorrente + DRAFT economico:
                //    catena tP → c1 → … (la finestra sono gli INPUT alle
                //    posizioni pos..pos+|w|-1).
                let snap = dec.specSnapshot(nKeys: pos)
                dec.setActiveExperts(draftExperts)
                var window = [tP]
                do {
                    var dLast = try dec.forward(token: tP, pos: pos, nKeys: pos + 1)
                    while window.count < specK {
                        let c = Sampler.sample(dLast, temperature: 0, topK: 0, topP: 1, minP: 0, rng: &rng)
                        if Int32(c) == tok.eosId { break }   // la verifica decide comunque
                        window.append(c)
                        guard window.count < specK else { break }
                        dLast = try dec.forward(token: c, pos: pos + window.count - 1,
                                                nKeys: pos + window.count)
                    }
                }
                dec.setActiveExperts(fullExperts)
                // 2) rollback + VERIFICA batch full-config (riscrive KV e
                //    stato per pos..pos+|w|-1 con i valori veri).
                dec.specRestore(snap)
                let logitsAll = try dec.specVerifyStep(tokens: window, startPos: pos)
                // 3) accettazione greedy: window[i+1] deve essere l'argmax
                //    full-config di logitsAll[i].
                var j = 0
                while j + 1 < window.count {
                    let a = Sampler.sample(logitsAll[j], temperature: 0, topK: 0, topP: 1, minP: 0, rng: &rng)
                    if a == window[j + 1] { j += 1 } else { break }
                }
                rounds += 1; drafted += window.count - 1; acceptedCands += j
                // 4) Il verify batch aggiorna il compressore con kernel diversi
                //    dal decode one-token. Ripristina sempre il frontier e
                //    rigioca il prefisso accettato sul percorso ordinario:
                //    accettazioni complete e parziali restano greedy-identiche.
                dec.specRestore(snap)
                let replayLogits = try dec.specReplay(
                    window.prefix(j + 1), startPos: pos)
                // 5) emetti i candidati accettati; l'ultimo logit verificato
                //    fornisce il tP del round successivo (bonus token).
                if j >= 1 {
                    for c in window[1...j] {
                        if Int32(c) == tok.eosId { break specLoop }
                        stdout.write(Data(tok.tokenText(Int32(c))))
                        generatedTraceTokens.append(c)
                        genTokens += 1
                        if genTokens == warmup { dec.resetProfile(); steadyStart = Date() }
                        if genTokens >= maxNew { break }
                    }
                }
                pos += j + 1
                last = replayLogits
                let dt = Date().timeIntervalSince(t0)
                log(String(format: "  [round %d  +%d tok  %.1fs  %.2f tok/s  accept %.2f]",
                           rounds, j + 1, dt, dt > 0 ? Double(j + 1) / dt : 0,
                           drafted > 0 ? Double(acceptedCands) / Double(drafted) : 0))
                if genTokens >= maxNew { break }
            }
            if rounds > 0 {
                log(String(format: "DS4Demo: speculativo — %d round, %.2f token/round, accettazione draft %.0f%%",
                           rounds, Double(genTokens) / Double(rounds),
                           drafted > 0 ? 100 * Double(acceptedCands) / Double(drafted) : 0))
            }
        } else {
        for _ in 0..<maxNew {
            let recentLo = max(0, recent.count - sampleRepeatN)
            let next = Sampler.sample(last, temperature: sampleTemperature, topK: sampleTopK,
                                      topP: sampleTopP, minP: sampleMinP,
                                      repetitionPenalty: sampleRepeat,
                                      recent: recent[recentLo...], rng: &rng)
            if Int32(next) == tok.eosId { break }
            stdout.write(Data(tok.tokenText(Int32(next))))   // stream immediately (unbuffered)
            generatedTraceTokens.append(next)
            recent.append(next)
            let t0 = Date()
            last = try dec.forward(token: next, pos: pos, nKeys: pos + 1); pos += 1
            genTokens += 1
            let dt = Date().timeIntervalSince(t0)
            log(String(format: "  [tok %d  %.1fs  %.2f tok/s]", genTokens, dt, dt > 0 ? 1.0 / dt : 0))
            abTrace?.capture(phase: "decode", step: genTokens,
                             inputToken: next, logits: last)
            if genTokens == warmup {
                dec.resetProfile()
                steadyStart = Date()
            }
        }
        }
        stdout.write(Data("\n".utf8))
        let total = Date().timeIntervalSince(genStart)
        log(String(format: "DS4Demo: %d tokens in %.1fs (%.2f tok/s)", genTokens, total,
                   total > 0 ? Double(genTokens) / total : 0))
        if warmup > 0 && genTokens > warmup {
            let steadyTokens = genTokens - warmup
            let steadyS = Date().timeIntervalSince(steadyStart)
            log(String(format: "DS4Demo: REGIME (dal token %d): %d token in %.1fs (%.2f tok/s) — profilo sotto = solo regime",
                       warmup + 1, steadyTokens, steadyS,
                       steadyS > 0 ? Double(steadyTokens) / steadyS : 0))
        }
        log("")
        log(dec.profile.report())
        if let trace = abTrace {
            try trace.finish(generatedTokens: generatedTraceTokens)
            log("DS4Demo: A/B logit trace scritta: \(trace.prefix).json + .f32")
        }

        // ── Diagnosi post-run: routing, allocazione slot, verdetto gather ──
        if diag, let usage = dec.usage {
            log("")
            log("── Diagnosi cache esperti ──")
            let liveCache = dec.slotCache
            if liveCache == nil {
                log("  slot-cache OFF (imposta DS4_EXPERT_CACHE_SLOTS=8.. per misurare gli hit)")
            } else if let liveCache {
                // These are the pools that actually served this run. Do not
                // recompute usage.slotAllocation here: usage changed during
                // generation, while the cache froze its plan at first use.
                let slots = liveCache.allocatedSlotsByLayer
                let bytes = liveCache.allocatedBytesByLayer
                let totalBytes = bytes.values.reduce(0, +)
                log(String(format: "  pool effettivi: %d layer, %.2f GiB residenti (piano congelato alla creazione)",
                           slots.count, Double(totalBytes) / 1_073_741_824))
            }
            let actualSlots = liveCache?.allocatedSlotsByLayer ?? [:]
            let actualBytes = liveCache?.allocatedBytesByLayer ?? [:]
            log("  layer   route  conc(top8)  conc(top16)  slot(pool)  pool MiB")
            for il in 0..<geometry.nLayers {
                let r = usage.routes(layer: il)
                guard r > 0 else { continue }
                if let slots = actualSlots[il] {
                    log(String(format: "  %5d %7d       %.2f        %.2f       %4d  %8.1f", il, r,
                               usage.concentration(layer: il, n: 8),
                               usage.concentration(layer: il, n: 16), slots,
                               Double(actualBytes[il] ?? 0) / 1_048_576))
                } else {
                    log(String(format: "  %5d %7d       %.2f        %.2f          -         -", il, r,
                               usage.concentration(layer: il, n: 8),
                               usage.concentration(layer: il, n: 16)))
                }
            }
            // Verdetto: banda effettiva del gather vs tetto sequenziale del disco.
            // Sotto ~60% del tetto conviene aumentare la profondità delle pread;
            // sopra, il gather è vicino alla fisica del disco e conviene puntare
            // su hit-rate (slot) o decodifica speculativa (MTP).
            if dec.profile.gatherBytes > 0, dec.profile.gatherS > 0 {
                let eff = Double(dec.profile.gatherBytes) / dec.profile.gatherS / 1e9
                if diskCeilingGBs > 0 {
                    let pct = eff / diskCeilingGBs * 100
                    log(String(format: "  gather effettivo %.2f GB/s = %.0f%% del tetto SSD (%.2f GB/s) -> %@",
                               eff, pct, diskCeilingGBs,
                               pct < 60 ? "margine: prova DS4_PREAD_SPLIT=3"
                                        : "vicino alla fisica del disco: puntare su hit-rate/MTP"))
                } else {
                    log(String(format: "  gather effettivo %.2f GB/s (banda SSD non misurata)", eff))
                }
            }
        }
    }
    // Salva la usage imatrix per il prossimo run (vedi load sopra).
    if usagePath != "off", let data = dec.usage?.serialize() {
        try? data.write(to: URL(fileURLWithPath: usagePath))
        log("DS4Demo: usage imatrix salvata (\(dec.usage?.totalRoutes ?? 0) route) in \(usagePath)")
    }
    exit(0)
} catch {
    log("DS4Demo error: \(error)")
    exit(1)
}
