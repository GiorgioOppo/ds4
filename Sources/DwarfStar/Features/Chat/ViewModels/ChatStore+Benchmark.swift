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
        if glmService != nil {
            runGLMBenchmark(quick: quick)
            return
        }
        guard let service else { benchStatus = "Carica prima il modello."; return }
        guard phase == .ready else { benchStatus = "Attendi che il modello sia pronto."; return }
        guard !isGenerating else { benchStatus = "Ferma la generazione prima del benchmark."; return }
        guard !benchRunning else { return }
        let activityGate = EngineActivityGate.shared
        guard let lease = activityGate.acquire(.benchmark) else {
            let owner = activityGate.activeOwner?.displayName ?? "un'altra operazione"
            benchStatus = "Motore occupato da \(owner)."
            return
        }
        benchRunning = true
        benchSucceeded = nil
        benchResults = ""
        benchStatus = quick ? "Benchmark rapido in corso… (~3 min, non usare la chat)"
                            : "Benchmark in corso… (~15 min, non usare la chat)"
        benchTask = Task {
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
                benchSucceeded = true
            } catch is CancellationError {
                _ = setenv("DS4_PREFILL_UNION", String(prefillUnion), 1)
                _ = setenv("DS4_PREFILL_CHUNK", String(prefillChunk), 1)
                _ = setenv("DS4_PREFILL_ROUTE_BATCH", String(prefillRouteBatch), 1)
                benchStatus = "Benchmark annullato."
                benchSucceeded = false
            } catch {
                // Ripristina i valori persistiti dopo un errore/annullamento.
                _ = setenv("DS4_PREFILL_UNION", String(prefillUnion), 1)
                _ = setenv("DS4_PREFILL_CHUNK", String(prefillChunk), 1)
                _ = setenv("DS4_PREFILL_ROUTE_BATCH", String(prefillRouteBatch), 1)
                benchStatus = "Benchmark fallito: \(error)"
                benchSucceeded = false
            }
            benchRunning = false
            benchTask = nil
            activityGate.release(lease)
        }
    }

    /// Benchmark di MISURA per GLM 5.2 (il tuning DeepSeek dei knob di
    /// prefill non si applica): prefill sintetico + decode greedy, con il
    /// profilo streaming per fase nel referto. Rapido: 64+8 token (~2-3
    /// min); completo: 192+16 (~6-10 min alla velocità attuale).
    private func runGLMBenchmark(quick: Bool) {
        guard let glm = glmService else { return }
        guard phase == .ready else {
            benchStatus = "Attendi che il modello sia pronto."
            return
        }
        guard !isGenerating else {
            benchStatus = "Ferma la generazione prima del benchmark."
            return
        }
        guard !benchRunning else { return }
        let activityGate = EngineActivityGate.shared
        guard let lease = activityGate.acquire(.benchmark) else {
            let owner = activityGate.activeOwner?.displayName
                ?? "un'altra operazione"
            benchStatus = "Motore occupato da \(owner)."
            return
        }
        benchRunning = true
        benchSucceeded = nil
        benchResults = ""
        let contextTokens = quick ? 64 : 192
        let genTokens = quick ? 8 : 16
        benchStatus = quick
            ? "Benchmark GLM rapido in corso… (~2-3 min, non usare la chat)"
            : "Benchmark GLM in corso… (~6-10 min, non usare la chat)"
        benchTask = Task {
            do {
                let numbers = try await glm.benchmark(
                    contextTokens: contextTokens, genTokens: genTokens)
                let summary = String(
                    format: "prefill %d token: %.2f tok/s · decode %d "
                        + "token: %.2f tok/s",
                    contextTokens, numbers.prefillTps,
                    genTokens, numbers.genTps)
                benchResults = summary + "\n" + numbers.report
                DS4Log.info("bench", "GLM " + summary)
                benchStatus = "Fatto: " + summary
                benchSucceeded = true
            } catch is CancellationError {
                benchStatus = "Benchmark annullato."
                benchSucceeded = false
            } catch {
                benchStatus = "Benchmark fallito: \(error)"
                benchSucceeded = false
            }
            benchRunning = false
            benchTask = nil
            activityGate.release(lease)
        }
    }

    // MARK: Machine auto-tune support

    /// Nome del chip (es. "Apple M1 Pro") per il referto dell'auto-tune.
    nonisolated static func chipName() -> String {
        var size = 0
        sysctlbyname("machdep.cpu.brand_string", nil, &size, nil, 0)
        guard size > 0 else { return "Apple Silicon" }
        var buf = [CChar](repeating: 0, count: size)
        sysctlbyname("machdep.cpu.brand_string", &buf, &size, nil, 0)
        let bytes = buf.prefix { $0 != 0 }.map { UInt8(bitPattern: $0) }
        return String(decoding: bytes, as: UTF8.self)
            .trimmingCharacters(in: .whitespacesAndNewlines)
    }

    /// Verifica/crea l'expert-bundle ORA, senza aspettare un load del modello.
    /// Gira in background (una build da ~72 GB dura minuti); l'esito compare
    /// accanto al bottone e i dettagli nel Log motore ("DS4 expbundle:").
    func buildExpertBundleNow() {
        guard !modelPath.isEmpty else { bundleBuildStatus = "Nessun modello selezionato."; return }
        guard phase != .loading else { bundleBuildStatus = "Attendi la fine del load in corso."; return }
        guard !isGenerating, generation == nil, engineSetupTask == nil else {
            bundleBuildStatus = "Ferma la generazione e attendi il warmup prima di creare il bundle."
            return
        }
        guard !bundleBuildRunning else {
            bundleBuildStatus = "Creazione expert bundle già in corso."
            return
        }
        let activityGate = EngineActivityGate.shared
        guard let lease = activityGate.acquire(.expertBundleBuild) else {
            let owner = activityGate.activeOwner?.displayName ?? "un'altra operazione"
            bundleBuildStatus = "Motore occupato da \(owner). Arrestalo prima di creare il bundle."
            return
        }
        bundleBuildRunning = true
        bundleBuildStatus = "Verifica/creazione in corso… (dettagli nel Log motore)"
        let path = modelPath
        let serviceBarrier = service
        let task = Task(priority: .userInitiated) { [weak self] in
            // A completed chat may still be streaming a large DiskKV snapshot.
            // Join it before giving the same SSD to a tens-of-gigabytes bundle
            // build; the activity lease prevents a new generation meanwhile.
            await serviceBarrier?.quiesceForTeardown()
            // Su thread GCD, non sul pool cooperativo: la build del bundle fa
            // fan-out con concurrentPerform (copyExpert) — vedi ChatStore.load.
            let outcome = await withCheckedContinuation { (cont: CheckedContinuation<String, Never>) in
                DispatchQueue.global(qos: .userInitiated).async {
                    cont.resume(returning: ExpertBundleTool.ensure(modelPath: path))
                }
            }
            guard let self else {
                EngineActivityGate.shared.release(lease)
                return
            }
            self.bundleBuildStatus = outcome
            self.bundleBuildRunning = false
            self.bundleBuildTask = nil
            EngineActivityGate.shared.release(lease)
        }
        bundleBuildTask = task
    }
}
