import SwiftUI
import AppKit
import UniformTypeIdentifiers
import DS4Engine
import DS4Core

extension ChatStore {
    /// Misura DS4_PREFILL_UNION (64/192/256), DS4_PREFILL_CHUNK (512/1024) e
    /// DS4_PREFILL_ROUTE_BATCH (16/32/64/128) sul
    /// modello caricato con il benchmark sintetico del motore (prefill 512 o
    /// 1024 token + 8 di decode), poi APPLICA e persiste la combinazione col
    /// prefill più veloce. ~5 run, alcuni minuti; la chat resta inutilizzabile
    /// nel frattempo (il motore è un actor seriale). I knob che richiedono un
    /// reload (DS4_PREFILL_MM, FFN_BATCH) non sono coperti.
    /// `quick`: SOLO l'unione esperti su un prefill da 128 token (~2-3 min) —
    /// il divario fra le unioni emerge già a decine di token, mentre il chunk
    /// richiede prefill > 512 token per esistere e resta nella modalità
    /// completa (che usa 512 + 1024 token, ~10 min).
    func runSettingsBenchmark(quick: Bool = false) {
        guard let service else { benchStatus = "Carica prima il modello."; return }
        guard phase == .ready else { benchStatus = "Attendi che il modello sia pronto."; return }
        guard !benchRunning else { return }
        benchRunning = true
        benchResults = ""
        benchStatus = quick ? "Benchmark rapido in corso… (~3 min, non usare la chat)"
                            : "Benchmark in corso… (~15 min, non usare la chat)"
        Task {
            // Nested funcs non ereditano l'isolamento MainActor in Swift 6:
            // annotazione esplicita (chiamata sincrona dal Task, che eredita
            // il MainActor del contesto).
            @MainActor func log(_ s: String) {
                benchResults += s + "\n"
                FileHandle.standardError.write(Data(("DS4 bench: " + s + "\n").utf8))
            }
            do {
                // 1) unione esperti, a chunk fisso 512. Il rapido (256 token)
                //    confronta SOLO 192 e 256: a scala corta il rumore può
                //    premiare 64, che sui prefill reali è il valore
                //    catastrofico (misurato sul campo: ~1.7 GB/token di
                //    riletture contro ~0.6 con 192) — il completo lo include
                //    solo perché a 512 token la misura è affidabile.
                let unionCtx = quick ? 256 : 512
                var bestUnion = prefillUnion
                var bestTps = 0.0
                _ = setenv("DS4_PREFILL_CHUNK", "512", 1)
                // No-op se già caldo: la prima misura union non deve pagare
                // la partenza fredda (pool esperti) e vincere/perdere per quello.
                await service.warmup()
                for union in (quick ? [192, 256] : [64, 192, 256]) {
                    _ = setenv("DS4_PREFILL_UNION", String(union), 1)
                    benchStatus = "Benchmark union=\(union)…"
                    let p = try await service.benchmark(contextTokens: unionCtx, genTokens: quick ? 4 : 8)
                    log(String(format: "union=%d (%d token): prefill %.2f tok/s, decode %.2f tok/s",
                               union, unionCtx, p.prefillTps, p.genTps))
                    if p.prefillTps > bestTps { bestTps = p.prefillTps; bestUnion = union }
                }
                _ = setenv("DS4_PREFILL_UNION", String(bestUnion), 1)
                // 2) chunk (solo modalità completa: sotto i 512 token un secondo
                //    chunk non esiste — con l'unione vincente, 1024 token).
                var bestChunk = prefillChunk
                if !quick {
                    var bestChunkTps = 0.0
                    for chunk in [512, 1024] {
                        _ = setenv("DS4_PREFILL_CHUNK", String(chunk), 1)
                        benchStatus = "Benchmark chunk=\(chunk)…"
                        let p = try await service.benchmark(contextTokens: 1024, genTokens: 8)
                        log(String(format: "union=%d chunk=%d (1024 token): prefill %.2f tok/s, decode %.2f tok/s",
                                   bestUnion, chunk, p.prefillTps, p.genTps))
                        if p.prefillTps > bestChunkTps { bestChunkTps = p.prefillTps; bestChunk = chunk }
                    }
                }
                // 3) Route batch (solo completo). Ora è letto a ogni layer di
                // prefill, quindi non richiede più un costoso reload del modello.
                // Conserviamo il valore precedente salvo un vantaggio >2% per
                // evitare di persistere rumore termico/SSD.
                var bestRouteBatch = prefillRouteBatch
                if !quick {
                    _ = setenv("DS4_PREFILL_CHUNK", "512", 1)
                    var routeScores: [Int: Double] = [:]
                    for routeBatch in [16, 32, 64, 128] {
                        _ = setenv("DS4_PREFILL_ROUTE_BATCH", String(routeBatch), 1)
                        benchStatus = "Benchmark route batch=\(routeBatch)…"
                        let p = try await service.benchmark(contextTokens: 512, genTokens: 4)
                        log(String(format: "union=%d routeBatch=%d (512 token): prefill %.2f tok/s",
                                   bestUnion, routeBatch, p.prefillTps))
                        routeScores[routeBatch] = p.prefillTps
                    }
                    if let winner = routeScores.max(by: { $0.value < $1.value }),
                       let current = routeScores[prefillRouteBatch],
                       winner.value > current * 1.02 {
                        bestRouteBatch = winner.key
                    }
                }
                // Applica e persisti i vincitori (i didSet rifanno i setenv).
                prefillUnion = bestUnion
                prefillChunk = bestChunk
                prefillRouteBatch = bestRouteBatch
                log("MIGLIORI: union=\(bestUnion) chunk=\(bestChunk) routeBatch=\(bestRouteBatch) — applicati e salvati.")
                benchStatus = "Fatto: union=\(bestUnion), chunk=\(bestChunk), route batch=\(bestRouteBatch)."
            } catch {
                // Ripristina i valori persistiti dopo un errore/annullamento.
                _ = setenv("DS4_PREFILL_UNION", String(prefillUnion), 1)
                _ = setenv("DS4_PREFILL_CHUNK", String(prefillChunk), 1)
                _ = setenv("DS4_PREFILL_ROUTE_BATCH", String(prefillRouteBatch), 1)
                benchStatus = "Benchmark fallito: \(error)"
            }
            benchRunning = false
        }
    }

    // MARK: Auto-tune per-macchina (M1/M2/…/base/Pro/Max: RAM e SSD diversi)

    /// Nome del chip (es. "Apple M1 Pro") per il referto dell'auto-tune.
    nonisolated static func chipName() -> String {
        var size = 0
        sysctlbyname("machdep.cpu.brand_string", nil, &size, nil, 0)
        guard size > 0 else { return "Apple Silicon" }
        var buf = [CChar](repeating: 0, count: size)
        sysctlbyname("machdep.cpu.brand_string", &buf, &size, nil, 0)
        return String(cString: buf).trimmingCharacters(in: .whitespacesAndNewlines)
    }

    /// Trova i migliori knob di CARICAMENTO per QUESTA macchina — slot della
    /// cache esperti, dense-ahead, async FFN, expert look-ahead — con una
    /// coordinate descent: un reload del modello per candidato (quei knob sono
    /// letti alla creazione del decoder), misurato con un decode breve.
    ///
    /// - Candidati RAM-gated (una base M1 da 16 GB non prova mai 32 slot; una
    ///   Max da 96 GB sì), in ordine crescente di RAM: al primo COLLASSO da
    ///   pressione di memoria i candidati più grossi si saltano.
    /// - Metrica: genTpsP99 (velocità di regime), ma un candidato conta solo
    ///   se STABILE — media/p99 ≥ 0.72. Il collasso da swap ha la firma
    ///   opposta (p99 alto raggiunto presto, media che crolla token dopo
    ///   token): è esattamente ciò che distingue "più slot = più hit" da
    ///   "più slot = spirale di swap" (misurato: 24 slot su 16 GB → 1.55 GB/s).
    /// - Un candidato vince solo con un margine >2% (anti-rumore): a parità
    ///   resta il valore persistito.
    /// I vincitori sono applicati e PERSISTITI (didSet → UserDefaults+setenv);
    /// il referto resta in `benchResults` e nel log motore.
    func runAutoTune() {
        guard service != nil else { benchStatus = "Carica prima il modello."; return }
        guard phase == .ready else { benchStatus = "Attendi che il modello sia pronto."; return }
        guard !benchRunning else { return }
        benchRunning = true
        benchResults = ""
        benchStatus = "Auto-tune in corso… (~15-25 min, non usare la chat)"
        Task {
            @MainActor func log(_ s: String) {
                benchResults += s + "\n"
                FileHandle.standardError.write(Data(("DS4 autotune: " + s + "\n").utf8))
            }
            let ramGB = Int(ProcessInfo.processInfo.physicalMemory >> 30)
            let chip = Self.chipName()
            log("Auto-tune su \(chip), \(ramGB) GB RAM — un reload del modello per configurazione.")

            /// Reload col set di knob corrente e misura: prefill corto (il
            /// prefill ha il suo benchmark) + 28 token di decode.
            @MainActor func measure(_ label: String) async throws -> (score: Double, note: String) {
                benchStatus = "Auto-tune: \(label) — teardown…"
                service = nil                      // libera la RAM wired PRIMA del nuovo load
                // Il teardown di ~GB di buffer wired/mlock non è istantaneo:
                // caricare subito sopra fa girare i primi token del benchmark
                // in piena pressione di memoria (falso collasso, misurato).
                try await Task.sleep(nanoseconds: 4_000_000_000)
                benchStatus = "Auto-tune: \(label) — reload…"
                load()
                while phase == .loading { try await Task.sleep(nanoseconds: 500_000_000) }
                guard phase == .ready, let svc = service else {
                    throw NSError(domain: "DS4AutoTune", code: 1,
                                  userInfo: [NSLocalizedDescriptionKey: "reload fallito (\(label))"])
                }
                try await Task.sleep(nanoseconds: 2_000_000_000)   // assestamento memoria
                // Warmup NON misurato (pool esperti, ring denso, transitorio
                // post-load) — l'equivalente del "REGIME dal token 5" della demo.
                benchStatus = "Auto-tune: \(label) — warmup…"
                _ = try await svc.benchmark(contextTokens: 16, genTokens: 4)
                benchStatus = "Auto-tune: \(label) — misura…"
                let p = try await svc.benchmark(contextTokens: 96, genTokens: 28)
                // Stabilità dal PROFILO temporale, non da media/p99: la spirale
                // di swap ha la CODA più lenta della TESTA (degrado progressivo,
                // misurato con 24 slot su 16 GB); una partenza fredda ha la
                // firma opposta e non deve squalificare la configurazione.
                func median(_ a: [Double]) -> Double {
                    guard !a.isEmpty else { return 0 }
                    return a.sorted()[a.count / 2]
                }
                let speeds = p.genSpeeds.count > 12 ? Array(p.genSpeeds.dropFirst(4)) : p.genSpeeds
                let head = median(Array(speeds.prefix(speeds.count / 2)))
                let tail = median(Array(speeds.suffix(speeds.count / 2)))
                let ratio = head > 0 ? tail / head : 1
                let stable = ratio >= 0.75
                let score = stable ? p.genTpsP99 : 0
                let note = String(format: "%@: %.2f tok/s regime (media %.2f, coda/testa %.0f%%)%@",
                                  label, p.genTpsP99, p.genTps, ratio * 100,
                                  stable ? "" : "  ← COLLASSO PROGRESSIVO (pressione memoria)")
                log(note)
                return (score, note)
            }

            // Snapshot per il ripristino su errore: un reload fallito a metà
            // sweep non deve lasciare persistito il candidato perdente.
            let snapSlots = expertCacheSlots, snapAhead = denseAhead
            let snapAsync = asyncFFNEnabled, snapLook = expertLookahead
            let snapNSG = q8NSG, snapMoeNSG = moeNSG
            do {
                let baseLabel = "baseline slot=\(expertCacheSlots) ahead=\(denseAhead) " +
                                "async=\(asyncFFNEnabled ? 1 : 0) look=\(expertLookahead) " +
                                "q8nsg=\(q8NSG) moensg=\(moeNSG)"
                var best = try await measure(baseLabel).score
                if best == 0 {
                    // Un solo retry: il primo giro paga il transitorio peggiore
                    // (teardown del motore della chat + primo reload).
                    log("baseline instabile — riprovo una volta dopo l'assestamento…")
                    try await Task.sleep(nanoseconds: 8_000_000_000)
                    best = try await measure(baseLabel + " (retry)").score
                }
                guard best > 0 else {
                    throw NSError(domain: "DS4AutoTune", code: 2, userInfo: [
                        NSLocalizedDescriptionKey: "la baseline è instabile anche al retry: " +
                        "liberare RAM (chiudere le altre app) e riprovare"])
                }

                // 1) Slot cache esperti — il knob più importante e più rischioso.
                // 24 anche sui 16 GB: misurato in demo (2026-07-08, con QKV_Q4)
                // 73% hit e 3.33 tok/s senza collasso — la guardia sulla
                // stabilità qui sotto protegge comunque le macchine dove non regge.
                let slotCands: [Int] = ramGB >= 96 ? [16, 24, 32, 48]
                                     : ramGB >= 48 ? [16, 24, 32]
                                     : ramGB >= 24 ? [12, 16, 24]
                                     : [20, 22, 24]
                for cand in slotCands where cand != expertCacheSlots {
                    let saved = expertCacheSlots
                    expertCacheSlots = cand
                    let r = try await measure("slot=\(cand)")
                    if r.score > best * 1.02 { best = r.score }
                    else {
                        expertCacheSlots = saved
                        if r.score == 0 && cand > saved { break }   // collasso: niente candidati più grossi
                    }
                }

                // 2) Dense-ahead (profondità dello staging ring: 1-3 slot extra).
                for cand in [1, 2, 3] where cand != denseAhead {
                    let saved = denseAhead
                    denseAhead = cand
                    let r = try await measure("ahead=\(cand)")
                    if r.score > best * 1.02 { best = r.score } else { denseAhead = saved }
                }

                // 3) FFN asincrona (misurata +10% su M1 Pro, ma va verificata per macchina).
                do {
                    let saved = asyncFFNEnabled
                    asyncFFNEnabled = !saved
                    let r = try await measure("asyncFFN=\(asyncFFNEnabled ? 1 : 0)")
                    if r.score > best * 1.02 { best = r.score } else { asyncFFNEnabled = saved }
                }

                // 4) Expert look-ahead speculativo (0 = solo layer hash, esatto).
                for cand in [4, 8] where cand != expertLookahead {
                    let saved = expertLookahead
                    expertLookahead = cand
                    let r = try await measure("look=\(cand)")
                    if r.score > best * 1.02 { best = r.score } else { expertLookahead = saved }
                }

                // 5) DS4_Q8_NSG: partizione K dei matvec Q8 — l'unico knob
                //    davvero legato ai CORE GPU (occupancy). Numerica identica
                //    per costruzione; il motore lo rilegge a ogni load.
                for cand in [2, 8] where cand != q8NSG {
                    let saved = q8NSG
                    q8NSG = cand
                    let r = try await measure("q8nsg=\(cand)")
                    if r.score > best * 1.02 { best = r.score } else { q8NSG = saved }
                }

                // 6) DS4_MOE_NSG: occupancy dei kernel MoE id (FFN routed +
                //    matvec densi Q4) — partizione per righe, numerica
                //    identica; come q8NSG l'ottimo dipende dai core GPU.
                for cand in [2, 8] where cand != moeNSG {
                    let saved = moeNSG
                    moeNSG = cand
                    let r = try await measure("moensg=\(cand)")
                    if r.score > best * 1.02 { best = r.score } else { moeNSG = saved }
                }

                // Reload finale con i vincitori (le proprietà sono già persistite).
                _ = try await measure("finale slot=\(expertCacheSlots) ahead=\(denseAhead) " +
                                      "async=\(asyncFFNEnabled ? 1 : 0) look=\(expertLookahead) " +
                                      "q8nsg=\(q8NSG) moensg=\(moeNSG)")
                let summary = "slot=\(expertCacheSlots) ahead=\(denseAhead) " +
                              "async=\(asyncFFNEnabled ? 1 : 0) look=\(expertLookahead) " +
                              "q8nsg=\(q8NSG) moensg=\(moeNSG)"
                log("MIGLIORI per \(chip)/\(ramGB)GB: \(summary) — applicati e salvati.")
                UserDefaults.standard.set("\(summary) @ \(Date().formatted())",
                                          forKey: "DS4AutoTune-\(chip)-\(ramGB)")
                benchStatus = "Auto-tune completato: \(summary)"
            } catch {
                expertCacheSlots = snapSlots; denseAhead = snapAhead
                asyncFFNEnabled = snapAsync; expertLookahead = snapLook
                q8NSG = snapNSG; moeNSG = snapMoeNSG
                benchStatus = "Auto-tune fallito: \(error.localizedDescription)"
                log("INTERROTTO: \(error.localizedDescription) — knob ripristinati ai valori di partenza; " +
                    "se il modello è scarico, ricaricalo dalle Settings.")
                if phase != .loading && phase != .ready { load() }
            }
            benchRunning = false
        }
    }

    /// Verifica/crea l'expert-bundle ORA, senza aspettare un load del modello.
    /// Gira in background (una build da ~72 GB dura minuti); l'esito compare
    /// accanto al bottone e i dettagli nel Log motore ("DS4 expbundle:").
    func buildExpertBundleNow() {
        guard !modelPath.isEmpty else { bundleBuildStatus = "Nessun modello selezionato."; return }
        guard phase != .loading else { bundleBuildStatus = "Attendi la fine del load in corso."; return }
        bundleBuildStatus = "Verifica/creazione in corso… (dettagli nel Log motore)"
        let path = modelPath
        Task.detached(priority: .userInitiated) {
            // Su thread GCD, non sul pool cooperativo: la build del bundle fa
            // fan-out con concurrentPerform (copyExpert) — vedi ChatStore.load.
            let outcome = await withCheckedContinuation { (cont: CheckedContinuation<String, Never>) in
                DispatchQueue.global(qos: .userInitiated).async {
                    cont.resume(returning: ExpertBundleTool.ensure(modelPath: path))
                }
            }
            await MainActor.run { self.bundleBuildStatus = outcome }
        }
    }
}
