import SwiftUI
import AppKit
import UniformTypeIdentifiers
import DS4Engine
import DS4Core

extension ChatStore {
    func applyFastDemoDefaults(persistExplicitly: Bool = false) {
        expertCacheSlots = 22      // +3,8% regime vs 20, senza collasso su 16 GB
        multiQuantCacheEnabled = true // +28,9% decode, logits esatti, stesso budget RAM
        expertCacheUniform = false // usage-driven; l'auto-tune confronta entrambe
        rawRingEnabled = true       // necessario su questa macchina: KV RAM costante
        willNeedEnabled = true
        expertPreadEnabled = true
        denseStreamEnabled = true
        mlockEnabled = true
        denseQ4Enabled = true
        qkvQ4Enabled = true        // +10% misurato (2.78 → 3.06 tok/s), output coerente
        sharedQ4Enabled = false    // +7% decode a contesto corto ma AFFOSSA il prefill
                                   // (bisect 2026-07-22: experts 368 ms/token) — per
                                   // carichi misti chat il netto è negativo
        prefillUnion = 192         // 256+PREFILL_MM misurato PEGGIORE (GEMM diluito)
        prefillChunk = 2048        // leve 1-8: 4096 rende di più in demo nuda, ma con
                                   // lo stack decode i transienti stringono i 16 GB
        prefillRouteBatch = 128    // misurato 2026-07-22: 21.4 → 24.2 t/s su 2.7k reali
        metalIOEnabled = true      // fallback automatico a pread sotto 4.0 GB/s sul M1 Pro
        expertLookahead = 12       // A/B 2026-07-26: +9.1% decode a contesto lungo (nasconde I/O esperti sotto il compute), neutro a corto
        denseAhead = 2             // staging un layer avanti: +1,5% misurato
        asyncFFNEnabled = true     // pipeline FFN asincrona: +10% misurato, parita' certificata
        q8NSG = 4
        moeNSG = 4
        preadSplit = 3
        denseQ4NSG = 4
        if persistExplicitly {
            let d = UserDefaults.standard
            d.set(22, forKey: "DS4ExpertCacheSlots")
            d.set(true, forKey: "DS4MultiQuantCache")
            d.set(false, forKey: "DS4ExpertCacheUniform")
            d.set(true, forKey: "DS4RawRing")
            d.set(true, forKey: "DS4WillNeed")
            d.set(true, forKey: "DS4ExpertPread")
            d.set(true, forKey: "DS4DenseStream")
            d.set(true, forKey: "DS4MLock")
            d.set(true, forKey: "DS4DenseQ4")
            d.set(true, forKey: "DS4QkvQ4")
            d.set(false, forKey: "DS4SharedQ4")
            d.set(192, forKey: "DS4PrefillUnion")
            d.set(2048, forKey: "DS4PrefillChunk")
            d.set(128, forKey: "DS4PrefillRouteBatch")
            d.set(true, forKey: "DS4MetalIO")
            d.set(12, forKey: "DS4ExpertLookahead")
            d.set(2, forKey: "DS4DenseAhead")
            d.set(true, forKey: "DS4AsyncFFN")
            d.set(4, forKey: "DS4Q8NSG")
            d.set(4, forKey: "DS4MoeNSG")
            d.set(3, forKey: "DS4PreadSplit")
            d.set(4, forKey: "DS4DenseQ4NSG")
            _ = setenv("DS4_RAW_RING", "1", 1)
            _ = setenv("DS4_MULTI_QUANT_CACHE", "1", 1)
            _ = setenv("DS4_EXPERT_CACHE_UNIFORM", "0", 1)
            _ = setenv("DS4_WILLNEED_EXPERTS", "1", 1)
            _ = setenv("DS4_EXPERT_PREAD", "1", 1)
            _ = setenv("DS4_DENSE_STREAM", "1", 1)
            _ = setenv("DS4_MLOCK", "1", 1)
            _ = setenv("DS4_DENSE_Q4", "1", 1)
            _ = setenv("DS4_QKV_Q4", "1", 1)
            _ = setenv("DS4_SHARED_Q4", "0", 1)
            _ = setenv("DS4_PREFILL_UNION", "192", 1)
            _ = setenv("DS4_PREFILL_CHUNK", "2048", 1)
            _ = setenv("DS4_PREFILL_ROUTE_BATCH", "128", 1)
            _ = setenv("DS4_MTLIO", "1", 1)
            _ = setenv("DS4_MTLIO_MIN_GBS", "4.0", 1)
            _ = setenv("DS4_POOL_INTERLEAVE", "1", 1)
            _ = setenv(DS4RuntimeKnob.prefillMoEBatch.rawValue, "1", 1)
            _ = setenv("DS4_GPU_INDEXER_TOPK", "1", 1)
            _ = setenv("DS4_DENSE_Q4_KERNEL", "1", 1)
            _ = setenv("DS4_FUSED_ROUTER_PROBS", "1", 1)
            _ = setenv("DS4_FUSED_ROUTER_FINALIZE", "1", 1)
            // Store fp8 del ring + probabilita'/finalize del router batchati
            // per l'intero run invece che per token (bit-identico).
            _ = setenv("DS4_PREFILL_MICRO_BATCH", "1", 1)
            _ = setenv("DS4_FUSED_COMP_PROJ", "1", 1)
            _ = setenv("DS4_EXPERT_LOOKAHEAD", "12", 1)
            _ = setenv("DS4_DENSE_AHEAD", "2", 1)
            _ = setenv("DS4_ASYNC_FFN", "1", 1)
            _ = setenv("DS4_Q8_NSG", "4", 1)
            _ = setenv("DS4_MOE_NSG", "4", 1)
            _ = setenv("DS4_PREAD_SPLIT", "3", 1)
            _ = setenv("DS4_DENSE_Q4_NSG", "4", 1)
        }
    }

    /// Preset veloce GLM misurato (M1 Pro 16 GB, 2026-07-19): MetalIO OFF
    /// (+18% decode — accodava lavoro GPU davanti ai commit sincroni),
    /// 6 esperti attivi (−25% I/O esperti, lieve trade-off qualità), resto
    /// ai default adattivi del motore (residenza al floor su RAM stretta,
    /// fusione commit e fill parallelo del prefill sono già default).
    /// Complessivo misurato: 6.0 → 4.2 s/token di decode.
    func applyGLMFastDefaults() {
        glmMetalIOEnabled = false
        glmActiveExperts = 6
        glmResidentLayers = 0
        glmExpertArena = 0
        glmStreamSlots = 0
        glmSpeculativeExperts = false
        glmUseQ4Sidecar = true
        glmFuseEnabled = true
        glmMoEBatchEnabled = true
        glmGpuRouterEnabled = true
        glmMlockEnabled = true
        glmReadSplit = 0
        glmNSG = 0
    }

    /// Snapshot hit/miss dell'arena esperti GLM nella riga di tuning.
    func refreshGLMArenaCounters() {
        guard let glmService else { glmArenaCounters = nil; return }
        Task { [weak self] in
            let counters = await glmService.expertArenaCounters()
            await MainActor.run { self?.glmArenaCounters = counters }
        }
    }

    static var residentDenseAuto: Bool { MemoryInfo.physicalBytes >= 24 * 1_073_741_824 }

    static var diskKVDirectory: URL {
        let base = FileManager.default.urls(for: .applicationSupportDirectory, in: .userDomainMask)[0]
        return base.appendingPathComponent("DwarfStar/kv-cache", isDirectory: true)
    }

    /// Application Support/DwarfStar/q4-cache: i .q4dense del requant Q4
    /// (~1.4 GB per modello) — scrivibile anche in sandbox.
    static var q4CacheDirectory: URL {
        let base = FileManager.default.urls(for: .applicationSupportDirectory, in: .userDomainMask)[0]
        return base.appendingPathComponent("DwarfStar/q4-cache", isDirectory: true)
    }

    /// Dove l'app costruisce l'expert-bundle quando non può scrivere accanto al
    /// GGUF (sandbox). ATTENZIONE: il sidecar duplica la regione esperti (~72 GB
    /// sul Flash 2-bit) — la GUI lo dice esplicitamente nel toggle.
    static var bundleDirectory: URL {
        let base = FileManager.default.urls(for: .applicationSupportDirectory, in: .userDomainMask)[0]
        return base.appendingPathComponent("DwarfStar/expert-bundle", isDirectory: true)
    }

    /// Sidecar GLM (layer Q4 unificati + bundle legacy da migrare): l'engine
    /// li cerca in DS4_GLM_LAYERQ4_DIR / DS4_BUNDLE_DIR con default
    /// ACCANTO al GGUF. STESSA politica del bundle DeepSeek: in lettura si
    /// riusa il sibling quando esiste GIÀ (es. costruito dalla demo CLI — e
    /// un sidecar parziale avviato lì non va frammentato su due cartelle);
    /// altrimenti l'app POSSIEDE i suoi artefatti sotto Application Support
    /// (sopravvivono allo spostamento del modello, la cartella dell'utente
    /// non viene mai toccata — anche se scrivibile). La demo senza env
    /// continua a costruire accanto al modello, come la demo DeepSeek.
    /// Deterministico a ogni chiamata, come le directory DeepSeek in init.
    nonisolated static func prepareGLMSidecarEnvironment(modelPath: String) {
        guard !modelPath.isEmpty else { return }
        _ = setenv("DS4_GLM_LAYERQ4_DIR",
                   glmResolvedLayerQ4Directory(modelPath: modelPath), 1)
        _ = setenv("DS4_BUNDLE_DIR",
                   glmResolvedLegacyBundleDirectory(modelPath: modelPath), 1)
    }

    /// Directory EFFETTIVA del sidecar Q4 per questo modello (la stessa che
    /// prepareGLMSidecarEnvironment esporta): sibling se contiene già un
    /// pack (<gguf>.q4dense, o i nomi storici .glmsidecar) o file per-layer
    /// legacy (.layerq4), altrimenti la casa gestita in Application Support.
    nonisolated static func glmResolvedLayerQ4Directory(modelPath: String)
        -> String {
        let sibling = modelPath + ".glm-layers-q4"
        let contents = (try? FileManager.default
            .contentsOfDirectory(atPath: sibling)) ?? []
        if contents.contains(where: {
            $0.hasSuffix(".layerq4") || $0.hasSuffix(".glmsidecar")
                || $0.hasSuffix(".q4dense") || $0.hasSuffix(".expbundle")
        }) {
            return sibling
        }
        return glmSidecarDirectory(modelPath: modelPath)
            .appendingPathComponent("layers-q4", isDirectory: true).path
    }

    /// Come sopra per il bundle esperti legacy (.glm-experts, in migrazione
    /// verso il sidecar unificato).
    nonisolated static func glmResolvedLegacyBundleDirectory(
        modelPath: String) -> String {
        let sibling = modelPath + ".glm-experts"
        if FileManager.default.fileExists(atPath: sibling) { return sibling }
        return glmSidecarDirectory(modelPath: modelPath)
            .appendingPathComponent("experts", isDirectory: true).path
    }

    /// Application Support/DwarfStar/glm-sidecar/<file>-<size>: la casa dei
    /// sidecar GLM costruiti dall'app (politica DeepSeek: l'app possiede i
    /// suoi artefatti qui e non tocca la cartella del modello; il sibling
    /// vale solo se già popolato, es. dalla demo CLI).
    nonisolated static func glmSidecarDirectory(modelPath: String) -> URL {
        let base = FileManager.default.urls(for: .applicationSupportDirectory, in: .userDomainMask)[0]
        let file = (modelPath as NSString).lastPathComponent
        let attributes = try? FileManager.default.attributesOfItem(atPath: modelPath)
        let size = (attributes?[.size] as? UInt64) ?? 0
        return base.appendingPathComponent("DwarfStar/glm-sidecar/\(file)-\(size)",
                                           isDirectory: true)
    }
}
