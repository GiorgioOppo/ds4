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

    /// GUI preset measured on the M1 Pro with roughly 10 GiB available before
    /// model load. Shared concepts use only their canonical `DS4_*` names;
    /// genuinely Laguna-only controls retain the backend prefix.
    ///
    /// Callers normally keep `overwrite` false so an explicit launch
    /// environment remains authoritative. The GUI passes true after model
    /// inspection because its process-global common knobs may still contain
    /// the preset of the previously loaded backend.
    public static let guiEnvironmentDefaults: [String: String] = [
        DS4RuntimeKnob.ssdStream.rawValue: "1",
        DS4RuntimeKnob.expertCacheMB.rawValue: "3072",
        DS4RuntimeKnob.multiQuantExpertCache.rawValue: "1",
        DS4RuntimeKnob.activeExperts.rawValue: "10",
        DS4RuntimeKnob.residentLayers.rawValue: "0",
        DS4RuntimeKnob.kvInitial.rawValue: "512",
        DS4RuntimeKnob.prefillChunk.rawValue: "256",
        DS4RuntimeKnob.prefillBatch.rawValue: "1",
        DS4RuntimeKnob.prefillDenseMM.rawValue: "1",
        DS4RuntimeKnob.prefillAttentionMultiKey.rawValue: "1",
        DS4RuntimeKnob.prefillLastRow.rawValue: "1",
        DS4RuntimeKnob.prefillMoEBatch.rawValue: "0",
        DS4RuntimeKnob.decodeChained.rawValue: "0",
        DS4RuntimeKnob.decodeSplitK.rawValue: "0",
        DS4RuntimeKnob.decodeSplitKMinimum.rawValue: "384",
        DS4RuntimeKnob.decodeSWAGQA3.rawValue: "0",
        DS4RuntimeKnob.decodeFusedRopeKV.rawValue: "0",
        DS4RuntimeKnob.ropeSIMD.rawValue: "0",
        DS4RuntimeKnob.sharedExpertOverlap.rawValue: "0",
        DS4RuntimeKnob.indexedAttention.rawValue: "1",
        DS4RuntimeKnob.longAttentionBlock.rawValue: "16",
        DS4RuntimeKnob.longAttentionTopBlocks.rawValue: "32",
        DS4RuntimeKnob.longAttentionRecent.rawValue: "512",
        DS4RuntimeKnob.longAttentionThreshold.rawValue: "4096",
        DS4RuntimeKnob.discardUploadPages.rawValue: "0",
        DS4RuntimeKnob.residentPrivate.rawValue: "0",
        DS4RuntimeKnob.partitionExpertCache.rawValue: "0",
        DS4RuntimeKnob.expertPread.rawValue: "1",
        DS4RuntimeKnob.willNeedExperts.rawValue: "1",
        DS4RuntimeKnob.preadSplit.rawValue: "1",
        DS4RuntimeKnob.metalIO.rawValue: "0",
        DS4RuntimeKnob.mlock.rawValue: "0",
        DS4RuntimeKnob.simdgroups.rawValue: "4",
    ]

    public static func installGUIEnvironmentDefaults(
        overwrite: Bool = false
    ) {
        for (name, value) in guiEnvironmentDefaults {
            _ = setenv(name, value, overwrite ? 1 : 0)
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
    /// DS4_SSD_STREAM forces the mode, and DS4_EXPERT_CACHE_MB sizes the
    /// LRU slot cache (the engine fallback remains 2048 MiB; the measured GUI
    /// profile installs 3072 MiB on the target with about 10 GiB free).
    public static func defaultOptions(modelPath: String, contextSize: Int)
        -> LagunaResidentModelOptions {
        let environment = ProcessInfo.processInfo.environment
        var options = LagunaResidentModelOptions()
        options.cacheCapacity = max(256, contextSize)
        options.initialFullCacheCapacity =
            DS4RuntimeEnvironment.integer(
                .kvInitial,
                backend: .laguna,
                environment: environment)
        let fileBytes = (try? FileManager.default
            .attributesOfItem(atPath: modelPath)[.size] as? UInt64) ?? 0
        let headroom: UInt64 = 3 << 30
        if let forced = DS4RuntimeEnvironment.value(
            .ssdStream,
            backend: .laguna,
            environment: environment
        ) {
            options.expertStreaming = forced == "1"
        } else {
            options.expertStreaming = fileBytes + headroom
                > ProcessInfo.processInfo.physicalMemory
        }
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
        options.simdgroupsPerThreadgroup = DS4RuntimeEnvironment.integer(
            .simdgroups,
            backend: .laguna,
            environment: environment)
        options.longAttentionIndex = DS4RuntimeEnvironment.flag(
            .indexedAttention,
            backend: .laguna,
            default: true,
            environment: environment)
        options.longAttentionBlockSize = DS4RuntimeEnvironment.integer(
            .longAttentionBlock,
            backend: .laguna,
            environment: environment)
        options.longAttentionTopBlocks = DS4RuntimeEnvironment.integer(
            .longAttentionTopBlocks,
            backend: .laguna,
            environment: environment)
        options.longAttentionRecentTokens = DS4RuntimeEnvironment.integer(
            .longAttentionRecent,
            backend: .laguna,
            environment: environment)
        options.longAttentionThreshold = DS4RuntimeEnvironment.integer(
            .longAttentionThreshold,
            backend: .laguna,
            environment: environment)
        return options
    }
}
