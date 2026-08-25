/// Backend identifiers used only to isolate deprecated aliases. New backends
/// should consume canonical `DS4_*` names and normally need no alias entries.
public enum DS4RuntimeBackend: String, Sendable {
    case deepSeekV4
    case glm52
    case laguna
    case kimiK3
}

public enum DS4RuntimeKnobArea: String, Sendable {
    case experts
    case prefill
    case memoryIO
    case attention
    case decode
}

public enum DS4RuntimeKnobValueKind: String, Sendable {
    case boolean
    case integer
    case path
}

public struct DS4RuntimeKnobDefinition: Sendable {
    public let name: String
    public let area: DS4RuntimeKnobArea
    public let valueKind: DS4RuntimeKnobValueKind
}

/// Canonical environment schema shared by every inference backend.
///
/// A knob belongs here when its operational meaning can be stated without
/// naming a model. Architecture-only controls remain backend-prefixed. This
/// keeps future backends from creating parallel spellings such as
/// `DS4_FUTURE_MTLIO` for an already-defined concept.
public enum DS4RuntimeKnob: String, CaseIterable, Sendable {
    case activeExperts = "DS4_ACTIVE_EXPERTS"
    case expertCacheSlots = "DS4_EXPERT_CACHE_SLOTS"
    case expertCacheMB = "DS4_EXPERT_CACHE_MB"
    case multiQuantExpertCache = "DS4_MULTI_QUANT_CACHE"
    case expertAsyncSplit = "DS4_EXPERT_ASYNC_SPLIT"
    case residentLayers = "DS4_RESIDENT_LAYERS"
    case streamSlots = "DS4_STREAM_SLOTS"
    case ssdStream = "DS4_SSD_STREAM"

    case prefillBatch = "DS4_PREFILL_BATCH"
    case prefillAttentionMultiKey = "DS4_PREFILL_ATTN_MULTIKEY"
    case prefillDenseMM = "DS4_PREFILL_DENSE_MM"
    case prefillLastRow = "DS4_PREFILL_LAST_ROW"
    case prefillMoEBatch = "DS4_PREFILL_MOE_BATCH"
    case prefillRouteBatch = "DS4_PREFILL_ROUTE_BATCH"
    case prefillChunk = "DS4_PREFILL_CHUNK"
    case prefillUnion = "DS4_PREFILL_UNION"

    case expertPread = "DS4_EXPERT_PREAD"
    case preadSplit = "DS4_PREAD_SPLIT"
    case willNeedExperts = "DS4_WILLNEED_EXPERTS"
    case metalIO = "DS4_MTLIO"
    case mlock = "DS4_MLOCK"
    case bundleDirectory = "DS4_BUNDLE_DIR"
    case usageFile = "DS4_USAGE_FILE"
    case simdgroups = "DS4_NSG"

    case kvInitial = "DS4_KV_INITIAL"
    case indexedAttention = "DS4_INDEXED_ATTN"
    case indexerTopKFast = "DS4_INDEXER_TOPK_FAST"
    case longAttentionBlock = "DS4_LONG_ATTN_BLOCK"
    case longAttentionTopBlocks = "DS4_LONG_ATTN_TOP_BLOCKS"
    case longAttentionRecent = "DS4_LONG_ATTN_RECENT"
    case longAttentionThreshold = "DS4_LONG_ATTN_THRESHOLD"
    case rawRing = "DS4_RAW_RING"
    case ropeSIMD = "DS4_ROPE_SIMD"

    case sharedExpertOverlap = "DS4_SHARED_EXPERT_OVERLAP"
    case speculativeTokens = "DS4_SPEC_K"
    case decodeChained = "DS4_DECODE_CHAINED"
    case decodeFusedRopeKV = "DS4_DECODE_FUSED_ROPE_KV"
    case decodeSWAGQA3 = "DS4_DECODE_SWA_GQA3"
    case decodeSplitK = "DS4_DECODE_SPLIT_K"
    case decodeSplitKMinimum = "DS4_DECODE_SPLIT_K_MIN"
    case discardUploadPages = "DS4_DISCARD_UPLOAD_PAGES"
    case residentPrivate = "DS4_RESIDENT_PRIVATE"
    case partitionExpertCache = "DS4_EXPERT_CACHE_PARTITIONED"

    public var definition: DS4RuntimeKnobDefinition {
        let area: DS4RuntimeKnobArea
        switch self {
        case .activeExperts, .expertCacheSlots, .expertCacheMB,
             .multiQuantExpertCache, .expertAsyncSplit,
             .sharedExpertOverlap, .partitionExpertCache:
            area = .experts
        case .prefillBatch, .prefillAttentionMultiKey, .prefillDenseMM,
             .prefillLastRow, .prefillMoEBatch, .prefillRouteBatch,
             .prefillChunk, .prefillUnion:
            area = .prefill
        case .residentLayers, .streamSlots, .ssdStream, .expertPread,
             .preadSplit, .willNeedExperts, .metalIO, .mlock,
             .bundleDirectory, .usageFile, .simdgroups,
             .discardUploadPages, .residentPrivate:
            area = .memoryIO
        case .kvInitial, .indexedAttention, .indexerTopKFast,
             .longAttentionBlock,
             .longAttentionTopBlocks, .longAttentionRecent,
             .longAttentionThreshold, .rawRing, .ropeSIMD:
            area = .attention
        case .speculativeTokens, .decodeChained, .decodeFusedRopeKV,
             .decodeSWAGQA3, .decodeSplitK, .decodeSplitKMinimum:
            area = .decode
        }

        let valueKind: DS4RuntimeKnobValueKind
        switch self {
        case .bundleDirectory, .usageFile:
            valueKind = .path
        case .activeExperts, .expertCacheSlots, .expertCacheMB,
             .residentLayers, .streamSlots, .prefillRouteBatch,
             .prefillChunk, .prefillUnion, .preadSplit, .simdgroups,
             .kvInitial, .longAttentionBlock, .longAttentionTopBlocks,
             .longAttentionRecent, .longAttentionThreshold,
             .speculativeTokens, .decodeSplitKMinimum:
            valueKind = .integer
        default:
            valueKind = .boolean
        }
        return DS4RuntimeKnobDefinition(
            name: rawValue, area: area, valueKind: valueKind)
    }

    /// Backend-prefixed spellings accepted for old scripts and preferences.
    /// Canonical values always win when both are present.
    public func deprecatedAliases(
        for backend: DS4RuntimeBackend
    ) -> [String] {
        switch (backend, self) {
        case (.deepSeekV4, .prefillBatch):
            return ["DS4_PREFILL_BATCH_ATTN"]
        case (.deepSeekV4, .prefillMoEBatch):
            return ["DS4_PREFILL_FFN_BATCH"]

        case (.glm52, .activeExperts):
            return ["DS4_GLM_ACTIVE_EXPERTS"]
        case (.glm52, .expertCacheSlots):
            return ["DS4_GLM_EXPERT_SLOTS"]
        case (.glm52, .residentLayers):
            return ["DS4_GLM_RESIDENT_LAYERS"]
        case (.glm52, .streamSlots):
            return ["DS4_GLM_STREAM_SLOTS"]
        case (.glm52, .prefillBatch):
            return ["DS4_GLM_PREFILL_BATCH"]
        case (.glm52, .prefillMoEBatch):
            return ["DS4_GLM_PREFILL_MOE"]
        case (.glm52, .prefillRouteBatch):
            return ["DS4_GLM_PREFILL_ROUTE_BATCH"]
        case (.glm52, .preadSplit):
            return ["DS4_GLM_READ_SPLIT"]
        case (.glm52, .metalIO):
            return ["DS4_GLM_MTLIO"]
        case (.glm52, .mlock):
            return ["DS4_GLM_MLOCK"]
        case (.glm52, .bundleDirectory):
            return ["DS4_GLM_BUNDLE_DIR"]
        case (.glm52, .usageFile):
            return ["DS4_GLM_USAGE_FILE"]
        case (.glm52, .speculativeTokens):
            return ["DS4_GLM_SPEC_K"]
        case (.glm52, .simdgroups):
            return ["DS4_GLM_NSG"]

        case (.laguna, .activeExperts):
            return ["DS4_LAGUNA_ACTIVE_EXPERTS"]
        case (.laguna, .expertCacheSlots):
            return ["DS4_LAGUNA_EXPERT_CACHE_SLOTS"]
        case (.laguna, .expertCacheMB):
            return ["DS4_LAGUNA_EXPERT_CACHE_MB"]
        case (.laguna, .residentLayers):
            return ["DS4_LAGUNA_RESIDENT_LAYERS"]
        case (.laguna, .ssdStream):
            return ["DS4_LAGUNA_SSD_STREAM"]
        case (.laguna, .prefillBatch):
            return ["DS4_LAGUNA_PREFILL_BATCH"]
        case (.laguna, .prefillDenseMM):
            return ["DS4_LAGUNA_PREFILL_DENSE_MM"]
        case (.laguna, .prefillMoEBatch):
            return ["DS4_LAGUNA_PREFILL_MOE_BATCH"]
        case (.laguna, .prefillChunk):
            return ["DS4_LAGUNA_PREFILL_CHUNK"]
        case (.laguna, .expertPread):
            return ["DS4_LAGUNA_EXPERT_PREAD"]
        case (.laguna, .preadSplit):
            return ["DS4_LAGUNA_PREAD_SPLIT"]
        case (.laguna, .willNeedExperts):
            return ["DS4_LAGUNA_WILLNEED_EXPERTS"]
        case (.laguna, .metalIO):
            return ["DS4_LAGUNA_MTLIO"]
        case (.laguna, .mlock):
            return ["DS4_LAGUNA_MLOCK"]
        case (.laguna, .simdgroups):
            return ["DS4_LAGUNA_NSG"]
        case (.laguna, .kvInitial):
            return ["DS4_LAGUNA_KV_INITIAL"]
        case (.laguna, .indexedAttention):
            return ["DS4_LAGUNA_INDEXED_ATTN"]
        case (.laguna, .longAttentionBlock):
            return ["DS4_LAGUNA_INDEXED_ATTN_BLOCK"]
        case (.laguna, .longAttentionTopBlocks):
            return ["DS4_LAGUNA_INDEXED_ATTN_TOP_BLOCKS"]
        case (.laguna, .longAttentionRecent):
            return ["DS4_LAGUNA_INDEXED_ATTN_RECENT"]
        case (.laguna, .longAttentionThreshold):
            return ["DS4_LAGUNA_INDEXED_ATTN_THRESHOLD"]
        case (.laguna, .sharedExpertOverlap):
            return ["DS4_LAGUNA_SHARED_EXPERT_OVERLAP"]
        case (.laguna, .decodeChained):
            return ["DS4_LAGUNA_DECODE_CHAINED"]
        case (.laguna, .decodeSplitK):
            return ["DS4_LAGUNA_DECODE_SPLIT_K"]
        case (.laguna, .decodeSplitKMinimum):
            return ["DS4_LAGUNA_DECODE_SPLIT_K_MIN"]
        case (.laguna, .discardUploadPages):
            return ["DS4_LAGUNA_DISCARD_UPLOAD_PAGES"]
        case (.laguna, .residentPrivate):
            return ["DS4_LAGUNA_RESIDENT_PRIVATE"]
        case (.laguna, .partitionExpertCache):
            return ["DS4_LAGUNA_EXPERT_CACHE_PARTITIONED"]
        default:
            return []
        }
    }
}
