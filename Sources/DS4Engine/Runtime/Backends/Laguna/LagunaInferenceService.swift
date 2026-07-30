import DS4Core
import DS4Metal
import Foundation

/// Chat inference over the Laguna S 2.1 resident engine — the deliberate
/// separate-backend counterpart of the DeepSeek `InferenceService` and the
/// GLM `GLM52InferenceService`: no code or state is shared with the other
/// hot loops.
/// @unchecked Sendable: the stored references are immutable (`tokenizer`,
/// `engine`) and the USAGE contract serializes mutation — one generation at
/// a time, enforced by the `LagunaChatService` actor (its producer tasks run
/// strictly one per send, and ChatStore additionally gates on
/// `isGenerating`).
public final class LagunaInferenceService: @unchecked Sendable {
    public let tokenizer: LagunaTokenizer
    public let engine: LagunaResidentModel

    /// Backend-specific GUI preset measured on the M1 Pro with roughly
    /// 10 GiB available before model load. These names deliberately use the
    /// `DS4_LAGUNA_*` aliases: ChatStore also publishes DeepSeek-oriented
    /// common knobs (for example chunk 2048, pread split 4, MetalIO/mlock on),
    /// and backend-specific values must win when Laguna is selected.
    ///
    /// Every entry is installed with overwrite=0, so an explicit launch
    /// environment remains authoritative.
    public static let guiEnvironmentDefaults: [String: String] = [
        "DS4_LAGUNA_SSD_STREAM": "1",
        "DS4_LAGUNA_EXPERT_CACHE_MB": "2048",
        "DS4_LAGUNA_ACTIVE_EXPERTS": "6",
        "DS4_LAGUNA_RESIDENT_LAYERS": "0",
        "DS4_LAGUNA_KV_INITIAL": "512",
        "DS4_LAGUNA_PREFILL_CHUNK": "256",
        "DS4_LAGUNA_PREFILL_BATCH": "1",
        "DS4_LAGUNA_PREFILL_DENSE_MM": "1",
        "DS4_LAGUNA_PREFILL_MOE_BATCH": "0",
        "DS4_LAGUNA_DECODE_CHAINED": "0",
        "DS4_LAGUNA_DECODE_SPLIT_K": "0",
        "DS4_LAGUNA_DECODE_SPLIT_K_MIN": "384",
        "DS4_LAGUNA_SHARED_EXPERT_OVERLAP": "0",
        "DS4_LAGUNA_INDEXED_ATTN": "1",
        "DS4_LAGUNA_INDEXED_ATTN_BLOCK": "16",
        "DS4_LAGUNA_INDEXED_ATTN_TOP_BLOCKS": "32",
        "DS4_LAGUNA_INDEXED_ATTN_RECENT": "512",
        "DS4_LAGUNA_INDEXED_ATTN_THRESHOLD": "4096",
        "DS4_LAGUNA_DISCARD_UPLOAD_PAGES": "0",
        "DS4_LAGUNA_RESIDENT_PRIVATE": "0",
        "DS4_LAGUNA_EXPERT_CACHE_PARTITIONED": "0",
        "DS4_LAGUNA_EXPERT_PREAD": "1",
        "DS4_LAGUNA_WILLNEED_EXPERTS": "1",
        "DS4_LAGUNA_PREAD_SPLIT": "1",
        "DS4_LAGUNA_MTLIO": "0",
        "DS4_LAGUNA_MLOCK": "0",
        "DS4_LAGUNA_NSG": "4",
    ]

    public static func installGUIEnvironmentDefaults() {
        for (name, value) in guiEnvironmentDefaults {
            _ = setenv(name, value, 0)
        }
    }

    public init(modelPath: String,
                options: LagunaResidentModelOptions) throws {
        if let issue = ModelFileDiagnostics.openabilityIssue(path: modelPath) {
            throw GGUFError.cannotOpen(issue)
        }
        let model = try GGUFModel(path: modelPath, metalMapping: false,
                                  prefetchCPU: false)
        tokenizer = try LagunaTokenizer(model: model)
        let runtime = try MetalRuntime()
        engine = try LagunaResidentModel(runtime: runtime, path: modelPath,
                                         options: options)
    }

    /// Default resolver used by the GUI and standalone service callers. The
    /// app installs `guiEnvironmentDefaults` before entering here; without
    /// that preset, streaming still turns on automatically when the GGUF plus
    /// ~3 GiB does not fit physical RAM (the 45 GiB mixed recipe on 16/32 GB
    /// machines). Environment knobs stay authoritative:
    /// DS4_LAGUNA_SSD_STREAM forces the mode, and
    /// DS4_LAGUNA_EXPERT_CACHE_MB sizes the LRU slot cache (default 2048
    /// while streaming, selected by end-to-end prefill/decode timing on the
    /// 16 GB target; larger budgets remain available for explicit A/B runs).
    public static func defaultOptions(modelPath: String, contextSize: Int)
        -> LagunaResidentModelOptions {
        let environment = ProcessInfo.processInfo.environment
        var options = LagunaResidentModelOptions()
        options.cacheCapacity = max(256, contextSize)
        options.initialFullCacheCapacity =
            DS4RuntimeEnvironment.integer(
                "DS4_KV_INITIAL",
                overrides: ["DS4_LAGUNA_KV_INITIAL"],
                environment: environment)
        let fileBytes = (try? FileManager.default
            .attributesOfItem(atPath: modelPath)[.size] as? UInt64) ?? 0
        let headroom: UInt64 = 3 << 30
        if let forced = environment["DS4_LAGUNA_SSD_STREAM"] {
            options.expertStreaming = forced == "1"
        } else {
            options.expertStreaming = fileBytes + headroom
                > ProcessInfo.processInfo.physicalMemory
        }
        options.expertCacheMB =
            DS4RuntimeEnvironment.integer(
                "DS4_EXPERT_CACHE_MB",
                overrides: ["DS4_LAGUNA_EXPERT_CACHE_MB"],
                environment: environment)
            ?? (options.expertStreaming ? 2_048 : 0)
        options.expertCacheSlots = DS4RuntimeEnvironment.integer(
            "DS4_EXPERT_CACHE_SLOTS",
            overrides: ["DS4_LAGUNA_EXPERT_CACHE_SLOTS"],
            environment: environment)
        options.activeExperts = DS4RuntimeEnvironment.integer(
            "DS4_ACTIVE_EXPERTS",
            overrides: ["DS4_LAGUNA_ACTIVE_EXPERTS"],
            environment: environment)
        options.residentExpertLayers = DS4RuntimeEnvironment.integer(
            "DS4_RESIDENT_LAYERS",
            overrides: ["DS4_LAGUNA_RESIDENT_LAYERS"],
            environment: environment)
        options.prefillChunk = DS4RuntimeEnvironment.integer(
            "DS4_PREFILL_CHUNK",
            overrides: ["DS4_LAGUNA_PREFILL_CHUNK"],
            environment: environment)
        options.expertPread = DS4RuntimeEnvironment.flag(
            "DS4_EXPERT_PREAD",
            overrides: ["DS4_LAGUNA_EXPERT_PREAD"],
            default: true,
            environment: environment)
        options.willNeedExperts = DS4RuntimeEnvironment.flag(
            "DS4_WILLNEED_EXPERTS",
            overrides: ["DS4_LAGUNA_WILLNEED_EXPERTS"],
            default: true,
            environment: environment)
        options.preadSplit = DS4RuntimeEnvironment.integer(
            "DS4_PREAD_SPLIT",
            overrides: ["DS4_LAGUNA_PREAD_SPLIT"],
            environment: environment) ?? 1
        options.metalIO = DS4RuntimeEnvironment.flag(
            "DS4_MTLIO",
            overrides: ["DS4_LAGUNA_MTLIO"],
            default: false,
            environment: environment)
        options.lockResident = DS4RuntimeEnvironment.flag(
            "DS4_MLOCK",
            overrides: ["DS4_LAGUNA_MLOCK"],
            default: false,
            environment: environment)
        options.simdgroupsPerThreadgroup = DS4RuntimeEnvironment.integer(
            "DS4_NSG",
            overrides: ["DS4_LAGUNA_NSG"],
            environment: environment)
        options.longAttentionIndex = DS4RuntimeEnvironment.flag(
            "DS4_INDEXED_ATTN",
            overrides: ["DS4_LAGUNA_INDEXED_ATTN"],
            default: true,
            environment: environment)
        options.longAttentionBlockSize = DS4RuntimeEnvironment.integer(
            "DS4_LONG_ATTN_BLOCK",
            overrides: ["DS4_LAGUNA_INDEXED_ATTN_BLOCK"],
            environment: environment)
        options.longAttentionTopBlocks = DS4RuntimeEnvironment.integer(
            "DS4_LONG_ATTN_TOP_BLOCKS",
            overrides: ["DS4_LAGUNA_INDEXED_ATTN_TOP_BLOCKS"],
            environment: environment)
        options.longAttentionRecentTokens = DS4RuntimeEnvironment.integer(
            "DS4_LONG_ATTN_RECENT",
            overrides: ["DS4_LAGUNA_INDEXED_ATTN_RECENT"],
            environment: environment)
        options.longAttentionThreshold = DS4RuntimeEnvironment.integer(
            "DS4_LONG_ATTN_THRESHOLD",
            overrides: ["DS4_LAGUNA_INDEXED_ATTN_THRESHOLD"],
            environment: environment)
        return options
    }
}
