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
        sharedQ4Enabled = true     // +7% misurato (3.13 → 3.36 tok/s), output coerente
        prefillUnion = 256         // +19% di prefill misurato a 3.5k token (8.63 tok/s)
        prefillChunk = 512         // stabile su prompt brevi/lunghi entro il budget RAM
        prefillRouteBatch = 32     // baseline stabile; il benchmark completo prova 16/32/64/128
        expertBundleEnabled = true
        metalIOEnabled = true      // fallback automatico a pread sotto 4.0 GB/s sul M1 Pro
        expertLookahead = 0        // speculativo misurato neutro; i layer hash restano sempre attivi
        denseAhead = 2             // staging un layer avanti: +1,5% misurato
        asyncFFNEnabled = true     // pipeline FFN asincrona: +10% misurato, parita' certificata
        q8NSG = 4
        moeNSG = 4
        preadSplit = 4
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
            d.set(true, forKey: "DS4SharedQ4")
            d.set(256, forKey: "DS4PrefillUnion")
            d.set(512, forKey: "DS4PrefillChunk")
            d.set(32, forKey: "DS4PrefillRouteBatch")
            d.set(true, forKey: "DS4ExpertBundle")
            d.set(true, forKey: "DS4MetalIO")
            d.set(0, forKey: "DS4ExpertLookahead")
            d.set(2, forKey: "DS4DenseAhead")
            d.set(true, forKey: "DS4AsyncFFN")
            d.set(4, forKey: "DS4Q8NSG")
            d.set(4, forKey: "DS4MoeNSG")
            d.set(4, forKey: "DS4PreadSplit")
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
            _ = setenv("DS4_SHARED_Q4", "1", 1)
            _ = setenv("DS4_PREFILL_UNION", "256", 1)
            _ = setenv("DS4_PREFILL_CHUNK", "512", 1)
            _ = setenv("DS4_PREFILL_ROUTE_BATCH", "32", 1)
            _ = setenv("DS4_EXPERT_BUNDLE", "1", 1)
            _ = setenv("DS4_MTLIO", "1", 1)
            _ = setenv("DS4_MTLIO_MIN_GBS", "4.0", 1)
            _ = setenv("DS4_POOL_INTERLEAVE", "1", 1)
            _ = setenv("DS4_PREFILL_FFN_BATCH", "1", 1)
            _ = setenv("DS4_GPU_INDEXER_TOPK", "1", 1)
            _ = setenv("DS4_DENSE_Q4_KERNEL", "1", 1)
            _ = setenv("DS4_FUSED_ROUTER_PROBS", "1", 1)
            _ = setenv("DS4_FUSED_ROUTER_FINALIZE", "1", 1)
            _ = setenv("DS4_FUSED_COMP_PROJ", "1", 1)
            _ = setenv("DS4_EXPERT_LOOKAHEAD", "0", 1)
            _ = setenv("DS4_DENSE_AHEAD", "2", 1)
            _ = setenv("DS4_ASYNC_FFN", "1", 1)
            _ = setenv("DS4_Q8_NSG", "4", 1)
            _ = setenv("DS4_MOE_NSG", "4", 1)
            _ = setenv("DS4_PREAD_SPLIT", "4", 1)
            _ = setenv("DS4_DENSE_Q4_NSG", "4", 1)
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
}
