import Foundation
import DS4Core
import DS4Metal

extension InferenceService {
// MARK: - Expert usage ("usage imatrix") persistence + tuning info

    /// Per-(model, agent) usage-profile file. Nonisolated so the actor's init
    /// can resolve it before isolation is established.
    nonisolated static func usageURL(modelName: String, agentId: String) -> URL {
        let dir = FileManager.default.urls(for: .applicationSupportDirectory, in: .userDomainMask)[0]
            .appendingPathComponent("DwarfStar", isDirectory: true)
        try? FileManager.default.createDirectory(at: dir, withIntermediateDirectories: true)
        return dir.appendingPathComponent("expert-usage-\(modelName)-\(agentId).json")
    }

    /// Usage profile for `agentId`, SEEDED when absent: a brand-new agent gets
    /// the RICHEST profile saved for the same model (largest file) instead of
    /// an empty one — the routing is mostly model-driven, so the slot-cache
    /// warms from the first token; the agent's own history then takes over
    /// (its file is written at every generation as before). Without this, a
    /// fresh agent paid dozens of cold-cache turns that the demo (one big
    /// warm usage file) never saw.
    public nonisolated static func usageDataSeeded(modelName: String, agentId: String) -> Data? {
        if let own = try? Data(contentsOf: usageURL(modelName: modelName, agentId: agentId)) {
            return own
        }
        let dir = FileManager.default.urls(for: .applicationSupportDirectory, in: .userDomainMask)[0]
            .appendingPathComponent("DwarfStar", isDirectory: true)
        let prefix = "expert-usage-\(modelName)-"
        let candidates = ((try? FileManager.default.contentsOfDirectory(
            at: dir, includingPropertiesForKeys: [.fileSizeKey])) ?? [])
            .filter { $0.lastPathComponent.hasPrefix(prefix) }
        let best = candidates.max {
            ((try? $0.resourceValues(forKeys: [.fileSizeKey]).fileSize) ?? 0)
                < ((try? $1.resourceValues(forKeys: [.fileSizeKey]).fileSize) ?? 0)
        }
        return best.flatMap { try? Data(contentsOf: $0) }
    }

    /// Persist the routing-frequency stats (called automatically after each
    /// generation; cheap — at most nLayer×nExpert small ints).
    public func saveExpertUsage() {
        guard let data = decoder.usage?.serialize() else { return }
        try? data.write(to: Self.usageURL(modelName: modelName, agentId: agentId))
    }

    public func resetExpertUsage() {
        decoder.usage?.reset()
        try? FileManager.default.removeItem(at: Self.usageURL(modelName: modelName, agentId: agentId))
    }

    public struct TuningInfo: Sendable {
        public let agentId: String          // whose usage profile is active
        public let cacheSlots: Int          // 0 = cache off
        public let cacheHits: Int
        public let cacheMisses: Int
        public let totalRoutes: Int
        /// Per-layer summary: "L7 · top8 = 43% · expert 17,4,99,…".
        public let layerSummaries: [String]
    }

    public func tuningInfo() -> TuningInfo {
        let usage = decoder.usage
        var summaries: [String] = []
        if let usage, usage.totalRoutes > 0 {
            for il in 0..<runtimeGeometry.nLayers {
                let conc = usage.concentration(layer: il, n: 8)
                let top = usage.top(layer: il, n: 8).map(String.init).joined(separator: ",")
                guard conc > 0 else { continue }
                summaries.append(String(format: "L%-3d top8 = %2.0f%%  ·  expert %@", il, conc * 100, top))
            }
        }
        return TuningInfo(agentId: agentId,
                          cacheSlots: decoder.slotCache?.slotsPerLayer ?? 0,
                          cacheHits: decoder.slotCache?.hits ?? 0,
                          cacheMisses: decoder.slotCache?.misses ?? 0,
                          totalRoutes: usage?.totalRoutes ?? 0,
                          layerSummaries: summaries)
    }

    /// Physical expert-cache budget contributed by one configured base slot,
    /// derived from the already-open GGUF and its resolved runtime geometry.
    /// The GUI auto-tuner uses it before tearing down the current service to
    /// reject RAM-unsafe slot growth without guessing from the model filename.
    public func expertCacheBudgetBytesPerBaseSlot() -> UInt64 {
        UInt64(max(0, StreamingDecoder.expertCacheBudgetBytesPerBaseSlot(
            model: model,
            dims: dims,
            nLayers: runtimeGeometry.nLayers
        )))
    }

    /// Exact MTLBuffer bytes added by one increment of dense read-ahead for the
    /// currently loaded model and quantization configuration.
    public func denseStagingBytesPerAheadSlot() -> UInt64 {
        UInt64(max(0, decoder.denseStagingBytesPerAheadSlot))
    }

    /// Whether DS4_PREAD_SPLIT reaches at least one routed-layer fill in the
    /// current load (including GGUF fallback layers beside a valid bundle).
    public func expertPreadSplitIsEffective() -> Bool {
        decoder.usesDirectExpertPread
    }

    public func modelInfo() -> ModelInfo {
        let routedBits = dims.gateQuant == .q4_K ? 4 : 2
        return ModelInfo(name: modelName, layers: runtimeGeometry.nLayers, nEmbd: dims.nEmbd,
                         nVocab: dims.vocab, contextSize: contextSize,
                         routedQuantBits: routedBits,
                         kvCacheBytes: estimatedKVCacheBytes(),
                         architecture: backendDescriptor.architecture,
                         displayName: backendDescriptor.displayName,
                         quantizationSummary: "routed \(routedBits)-bit",
                         capabilities: backendDescriptor.capabilities)
    }

    /// Worst-case (full-context) KV-cache RAM, matching what the decoder actually
    /// allocates — so the figure reflects the raw-KV ring. Without the ring the raw
    /// cache is nLayer × ctx × headDim × F32 (dominant); with it the raw cache is a
    /// constant nSWA-row window and only the NSA-compressed rows (ctx/ratio) + the
    /// indexer scale with the context. A static ctx×headDim formula would keep
    /// reporting the huge number even with the ring on.
    func estimatedKVCacheBytes() -> UInt64 {
        let ringOn = getenv("DS4_RAW_RING").map { String(cString: $0) == "1" } ?? false
        let headDim = UInt64(dims.headDim)
        let ctx = UInt64(contextSize)
        let rawRows = ringOn ? UInt64(min(dims.nSWA, contextSize)) : ctx
        var bytes: UInt64 = 0
        for il in 0..<runtimeGeometry.nLayers {
            bytes += rawRows * headDim * 4                                  // raw cache (every layer)
            let ratio = runtimeGeometry.compressRatio(layer: il)
            guard ratio > 0 else { continue }
            bytes += (ctx / UInt64(ratio)) * headDim * 4                   // NSA compressor cache
            if ratio == 4 {
                bytes += (ctx / 4) * UInt64(dims.nIndexerHeadDim) * 4       // indexer compressor (ratio-4 only)
            }
        }
        return bytes
    }

    /// The raw Jinja chat template embedded in the GGUF (for inspection), if any.
    public func chatTemplate() -> String? { model.string("tokenizer.chat_template") }

    /// Tokens currently committed to the KV (used to warn before the context fills).
    public func committedTokens() -> Int { committedIds.count }
}
