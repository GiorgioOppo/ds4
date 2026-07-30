import Foundation
import Metal
import DS4Core
#if canImport(Darwin)
import Darwin
#endif

// First-cut Laguna S 2.1 resident engine, modeled on GLM52ResidentModel but
// deliberately simpler: Laguna requires full residency upstream, so every
// validated tensor is uploaded once into shared MTLBuffers and there is no
// SSD streaming, sidecar or expert cache. The per-token graph mirrors
// `laguna_graph_forward_token` in the reference `laguna-s2.1` branch,
// dispatching the shared GLM primitives where upstream shares them
// (rms_norm/matvec/add/argmax-free readback, `kernel_glm52_router_select`
// with top-N active experts, the K-quant MoE matvecs) plus the Laguna kernels
// for per-head norm/RoPE, the F16 ring KV store and gated GQA attention.
//
// Scope limits of this cut, all enforced at load with distinct errors:
// - Q8_0-signal layout only (official Q4_K routed recipe and the mixed
//   RoutedQ2_K/Last27Q3_K file, whose K-quant matvecs are wired); the
//   legacy F16/Q4_K recipe is refused until its Q6_K/F16 paths are wired;
// - multi-token prompts prefill LAYER-MAJOR in chunks (two syncs per layer
//   for the whole chunk, one union gather per layer over the slot cache);
//   the sequential decode path remains the fallback and the numerics are
//   identical by construction (same kernels, same causal and rank order);
// - decode keeps two command-buffer syncs per MoE layer (the router
//   selection is read back on the host to pick expert slabs, like the GLM
//   chained decode).
//
// Optional divergence from upstream: `options.expertStreaming` keeps only
// the Q8_0 signal path resident (~5 GiB) and reads the routed expert
// slabs of the selected top-N experts per token, after the host router
// readback, with concurrent F_NOCACHE preads (mmap memcpy walk as
// fallback if the second descriptor cannot open). `options.expertCacheMB`
// adds an LRU slot cache over those slabs — hits bind the cached buffers
// with zero I/O and zero copy, exploiting the skew of the router. Upstream
// REFUSES streaming for Laguna and mandates residency; this exists so
// 32 GB machines can run the 45 GiB file at all (~1.6 GB of reads per
// token uncached — expect low single-digit tok/s). Numerics are
// unchanged: same kernels, same slabs, different storage.

public struct LagunaResidentModelOptions: Sendable {
    /// Number of leading transformer blocks to run (nil = all 48). Front
    /// truncation exists for bring-up tests against partial uploads; the
    /// output head then reads whatever hidden state the last kept layer
    /// produced.
    public var layerCount: Int?
    /// KV positions per full-attention layer (sliding-window layers stay at
    /// the 512-row ring regardless).
    public var cacheCapacity: Int = 4_096
    /// Initially allocated rows on the 12 full-attention layers. The cache
    /// grows geometrically up to `cacheCapacity`; nil uses 512. This keeps a
    /// large configured chat context from consuming GiBs before it is used.
    public var initialFullCacheCapacity: Int?
    /// Experimental SSD streaming of the routed experts (see the divergence
    /// note in the file header). Off by default: the resident path is the
    /// one whose parity against the C engine is being certified.
    public var expertStreaming = false
    /// RAM budget (MiB) for the LRU slot cache of streamed expert slabs
    /// (DeepSeek ExpertSlotCache pattern): hits bind the cached buffers with
    /// zero I/O and zero copy, misses pread into the reassigned slot. 0 keeps
    /// the plain per-token staging triples. Ignored when experts are resident.
    public var expertCacheMB = 0
    /// Explicit number of global streamed-expert cache slots. nil derives the
    /// count from `expertCacheMB`; 0 disables the persistent cache. Unlike the
    /// per-layer DeepSeek pool, Laguna shares these slots across all layers.
    public var expertCacheSlots: Int?
    /// Routed experts actually executed per token. nil preserves the upstream
    /// top-10. Smaller values run top-N and renormalize its route weights.
    public var activeExperts: Int?
    /// Number of leading routed-expert layers kept resident even when expert
    /// streaming is enabled. nil/0 streams every routed layer.
    public var residentExpertLayers: Int?
    /// Maximum token width of the layer-major prefill sweep. nil keeps the
    /// measured Laguna default (256); the cache may reduce it further.
    public var prefillChunk: Int?
    /// Use direct `pread` through an F_NOCACHE descriptor for streamed expert
    /// misses. false falls back to byte-identical mmap copies.
    public var expertPread = true
    /// Issue WILLNEED hints for selected expert ranges before mmap copies.
    /// It is intentionally ignored by the direct-pread path.
    public var willNeedExperts = true
    /// Concurrent ranges per expert slab for the direct-pread path.
    public var preadSplit = 1
    /// Apple Metal fast resource loading for streamed expert misses. It is
    /// opt-in; any initialization or read failure falls back to pread/mmap.
    public var metalIO = false
    /// Best-effort pinning of the output head and streamed-expert buffers.
    public var lockResident = false
    /// Cooperative SIMD groups per threadgroup for Laguna/GLM matvec kernels.
    public var simdgroupsPerThreadgroup: Int?
    /// Sparse long-context attention on the 12 global layers. A 16:1
    /// centroid index scores old blocks on GPU; attention still consumes the
    /// original F16 K/V rows of the selected blocks plus a recent dense tail.
    public var longAttentionIndex = true
    public var longAttentionBlockSize: Int?
    public var longAttentionTopBlocks: Int?
    public var longAttentionRecentTokens: Int?
    public var longAttentionThreshold: Int?

    public init() {}
}

public enum LagunaResidentModelError: Error, Sendable, CustomStringConvertible {
    case unsupportedRecipe(String)
    case bufferAllocation(String)
    case invalidToken(Int32)
    case contextFull(capacity: Int)
    case expertRead(String)
    case kvCheckpoint(String)

    public var description: String {
        switch self {
        case .unsupportedRecipe(let reason):
            return "unsupported Laguna recipe for the resident engine: \(reason)"
        case .bufferAllocation(let what):
            return "Laguna resident engine could not allocate \(what)"
        case .invalidToken(let token):
            return "Laguna token id \(token) is outside the vocabulary"
        case .contextFull(let capacity):
            return "Laguna context is full (\(capacity) positions)"
        case .expertRead(let what):
            return "Laguna expert streaming read failed: \(what)"
        case .kvCheckpoint(let what):
            return "Laguna disk-KV checkpoint failed: \(what)"
        }
    }
}

public final class LagunaResidentModel {
    static func resolvedActiveExperts(_ requested: Int?) -> Int {
        max(1, min(requested ?? LagunaRouterReference.expertsUsed,
                   LagunaRouterReference.expertsUsed))
    }

    static func resolvedPrefillChunk(_ requested: Int?) -> Int {
        max(1, min(requested ?? 256, 1_024))
    }

    static func resolvedPreadSplit(_ requested: Int) -> Int {
        max(1, min(requested, 8))
    }

    static func resolvedSimdgroups(_ requested: Int?) -> Int {
        max(1, min(requested ?? 4, 8))
    }

    static func resolvedLongAttentionBlockSize(_ requested: Int?) -> Int {
        let value = requested ?? 16
        return [4, 8, 16, 32, 64].min {
            abs($0 - value) < abs($1 - value)
        } ?? 16
    }

    static func resolvedLongAttentionTopBlocks(_ requested: Int?) -> Int {
        max(1, min(requested ?? 32, 128))
    }

    static func resolvedLongAttentionRecentTokens(_ requested: Int?) -> Int {
        max(128, min(requested ?? 512, 4_096))
    }

    private static func makeMetalIO(
        runtime: MetalRuntime, path: String
    ) -> (queue: MTLIOCommandQueue, handle: MTLIOFileHandle)? {
        do {
            let descriptor = MTLIOCommandQueueDescriptor()
            descriptor.type = .concurrent
            descriptor.priority = .high
            descriptor.maxCommandBufferCount = 2
            descriptor.maxCommandsInFlight = 32
            let queue = try runtime.device.makeIOCommandQueue(
                descriptor: descriptor)
            let handle = try runtime.device.makeIOFileHandle(
                url: URL(fileURLWithPath: path))
            guard let scratch = runtime.device.makeBuffer(
                length: 4_096, options: .storageModeShared
            ) else { return nil }
            let warmup = queue.makeCommandBuffer()
            warmup.load(scratch, offset: 0, size: 4_096,
                        sourceHandle: handle, sourceHandleOffset: 0)
            warmup.commit()
            warmup.waitUntilCompleted()
            guard warmup.status == .complete else { return nil }
            return (queue, handle)
        } catch {
            return nil
        }
    }

    private static func makeLongAttentionIndex(
        runtime: MetalRuntime,
        capacity: Int,
        rowWidth: Int,
        blockSize: Int
    ) throws -> LongAttentionIndex {
        let blocks = max(1, (capacity + blockSize - 1) / blockSize)
        let bytes = blocks * rowWidth * MemoryLayout<UInt16>.stride
        guard let keys = runtime.device.makeBuffer(
            length: bytes, options: .storageModeShared
        ) else {
            throw LagunaResidentModelError.bufferAllocation(
                "long-attention compressed index")
        }
        memset(keys.contents(), 0, bytes)
        return LongAttentionIndex(
            blockSize: blockSize, blockCapacity: blocks,
            rowWidth: rowWidth, keys: keys)
    }

    public let configuration: LagunaConfiguration
    public private(set) var position = 0
    public var loadedLayerCount: Int { layers.count }
    public var contextCapacity: Int { cacheCapacity }
    public var allocatedKVCacheBytes: Int {
        layers.reduce(0) {
            $0 + $1.cache.keys.length + $1.cache.values.length
                + ($1.longAttentionIndex?.keys.length ?? 0)
        }
    }
    public var isExpertStreaming: Bool {
        !expertStaging.isEmpty || !expertSlots.isEmpty
    }
    public var expertCacheSlots: Int { expertSlots.count }
    public var isExpertCacheLayerPartitioned: Bool {
        !expertLayerSlotRanges.isEmpty
    }
    public var activeExpertCount: Int { activeExperts }
    public var prefillChunkSize: Int { prefillChunk }
    public var expertPreadSplit: Int { preadSplit }
    public var isExpertPreadEnabled: Bool { streamFD >= 0 }
    public var isMetalIOEnabled: Bool { metalIO != nil }
    public var isDecodeSplitKEnabled: Bool { Self.decodeSplitK }
    public var isChainedDecodeEnabled: Bool { Self.chainedDecode }
    public var isSharedExpertIOOverlapEnabled: Bool {
        Self.sharedExpertIOOverlap
    }
    public var isLongAttentionIndexEnabled: Bool {
        longAttentionIndexEnabled
    }
    public var longAttentionConfiguration:
        (blockSize: Int, topBlocks: Int, recentTokens: Int, threshold: Int) {
        (longAttentionBlockSize, longAttentionTopBlocks,
         longAttentionRecentTokens, longAttentionThreshold)
    }
    public var discardsUploadedFilePages: Bool {
        Self.discardUploadedFilePages
    }
    public var usesPrivateResidentWeights: Bool {
        Self.privateResidentWeights
    }
    public var residentExpertLayerCount: Int {
        layers.reduce(into: 0) { count, layer in
            if case .moe(_, _, _, .resident, _, _, _, _, _) = layer.ffn {
                count += 1
            }
        }
    }

    /// Per-phase profile in the shared DeepSeek/GLM `DecodeProfile` format.
    /// route/attn covers the attention batch plus the router (its host
    /// readback shows up as the wall−gpu remainder); experts covers the
    /// routed+shared batch; layer (alt) is the fused dense layer 0; gather
    /// IO counts the per-token slab reads of expert streaming.
    public private(set) var profile = DecodeProfile()
    private var gpuAttnS = 0.0
    private var gpuExpertsS = 0.0
    private var gpuDenseS = 0.0
    private var gpuHeadS = 0.0
    private var indexedAttentionDispatches = 0

    private let runtime: MetalRuntime
    private let model: GGUFModel
    private let shape: LagunaShape
    private let cacheCapacity: Int
    private let initialFullCacheCapacity: Int
    private let embeddingDescriptor: LagunaWeightDescriptor
    private let activeExperts: Int
    private let residentExpertLayers: Int
    private let prefillChunk: Int
    private let preadSplit: Int
    private let willNeedExperts: Bool
    private let lockResident: Bool
    private let simdgroupsPerThreadgroup: Int
    private let longAttentionIndexEnabled: Bool
    private let longAttentionBlockSize: Int
    private let longAttentionTopBlocks: Int
    private let longAttentionRecentTokens: Int
    private let longAttentionThreshold: Int

    private static let q8Type: UInt32 = 8
    /// Default-on batched QK/RoPE + causal attention. Keep a runtime escape
    /// hatch for real-GGUF logits audits and performance A/B runs.
    private static let batchedPrefillAttention =
        DS4RuntimeEnvironment.flag(
            .prefillBatch,
            backend: .laguna,
            default: true)
    /// Experimental expert-major MoE prefill. The real-model A/B currently
    /// shows a regression, so it remains opt-in for further kernel tuning.
    private static let batchedPrefillMoE =
        DS4RuntimeEnvironment.flag(
            .prefillMoEBatch,
            backend: .laguna,
            default: false)
    /// Dense multi-token GEMMs for the Laguna signal path. This is the same
    /// portable Q8_0 prefill family used by DeepSeek: mathematically
    /// equivalent, but not bit-identical because the MMA kernel stages
    /// activations as F16. Real-model A/B: 414 tokens fell from 73.8 s to
    /// 32.6 s on M1 Pro; the switch remains as an exact-path escape hatch.
    private static let batchedPrefillDenseMM =
        DS4RuntimeEnvironment.flag(
            .prefillDenseMM,
            backend: .laguna,
            default: true)
    private static let prefillDenseMMMinimumTokens = 8
    /// Once a resident tensor has been copied out of the GGUF mmap, its clean
    /// source pages are dead weight. Drop only the page-aligned interior so
    /// neighbouring descriptors that share a boundary page stay untouched.
    /// This avoids retaining a second file-backed copy of the ~5 GiB signal
    /// path after model load. Exact bytes in the Metal buffers are unchanged.
    private static let discardUploadedFilePages =
        DS4RuntimeEnvironment.flag(
            .discardUploadPages,
            backend: .laguna,
            default: false)
    /// GPU-only storage for immutable resident weights. Upload uses one
    /// reusable shared staging buffer and preserves every tensor byte.
    private static let privateResidentWeights =
        DS4RuntimeEnvironment.flag(
            .residentPrivate,
            backend: .laguna,
            default: false)
    private static let partitionExpertCache =
        DS4RuntimeEnvironment.flag(
            .partitionExpertCache,
            backend: .laguna,
            default: false)
    /// Long-context full-attention decode. The grouped kernel evaluates three
    /// query heads sharing one KV head, then a specialized reducer applies the
    /// learned gate. Sliding-window layers keep the ring-aware legacy kernel.
    private static let decodeSplitK =
        DS4RuntimeEnvironment.flag(
            .decodeSplitK,
            backend: .laguna,
            default: false)
    private static let decodeSplitKMinimumKeys = max(
        257,
        DS4RuntimeEnvironment.integer(
            .decodeSplitKMinimum,
            backend: .laguna) ?? 384)
    private static let decodeSplitKMaximumWorkgroups = 32
    /// Queue the expert tail of layer N without a CPU wait, then append the
    /// attention/router trunk of N+1 on the same in-order Metal queue. Waiting
    /// for the latter completes both, halving decode wait round-trips while
    /// preserving every kernel and dependency.
    private static let chainedDecode =
        DS4RuntimeEnvironment.flag(
            .decodeChained,
            backend: .laguna,
            default: false)
    /// Start the resident shared expert before the first routed-expert SSD
    /// wait. Its result is kept in `hiddenNext`; routed experts continue to
    /// accumulate in `ffnOut`, and the final two adds preserve the upstream
    /// `(after_attn + routed) + shared` association exactly.
    private static let sharedExpertIOOverlap =
        DS4RuntimeEnvironment.flag(
            .sharedExpertOverlap,
            backend: .laguna,
            default: false)

    /// Routed-expert types with a wired matvec path (the K-quant dot helpers
    /// shared with GLM): Q2_K, Q3_K and Q4_K — the published official and
    /// mixed recipes. The tensor schema already guarantees the three routed
    /// projections of one layer share this type on the Q8_0-signal layout.
    private static let routedTypes: Set<UInt32> = [10, 11, 12]

    private enum RoutedWeights {
        case resident(gate: MTLBuffer, up: MTLBuffer, down: MTLBuffer)
        /// Absolute file offsets of the three routed expert tables; the
        /// slabs of the selected experts are copied per token from the mmap
        /// into `expertStaging` before the expert batch is encoded.
        case streamed(gate: Int, up: Int, down: Int)
    }

    private enum FFN {
        case dense(gate: MTLBuffer, up: MTLBuffer, down: MTLBuffer)
        case moe(routedType: UInt32,
                 routerRows: MTLBuffer, routerBias: MTLBuffer,
                 routed: RoutedWeights,
                 gateUpExpertBytes: Int, downExpertBytes: Int,
                 sharedGate: MTLBuffer, sharedUp: MTLBuffer,
                 sharedDown: MTLBuffer)
    }

    private struct LongAttentionIndex {
        let blockSize: Int
        let blockCapacity: Int
        let rowWidth: Int
        let keys: MTLBuffer
    }

    private struct Layer {
        let index: Int
        var spec: LagunaAttentionSpec
        let attnNorm: MTLBuffer
        let query: MTLBuffer
        let key: MTLBuffer
        let value: MTLBuffer
        let gate: MTLBuffer
        let queryNorm: MTLBuffer
        let keyNorm: MTLBuffer
        let attnOutput: MTLBuffer
        let ffnNorm: MTLBuffer
        let ffn: FFN
        var cache: LagunaMetalKVCache
        var longAttentionIndex: LongAttentionIndex?
    }

    private var layers: [Layer] = []
    private let outputNorm: MTLBuffer
    private let outputHead: MTLBuffer
    /// One (gate, up, down) staging triple per active-expert rank, reused
    /// across layers and tokens; empty when experts are resident. Sized for
    /// the largest expert slab in the file, so mixed Q2_K/Q3_K files share
    /// the same triples. All active ranks stay alive until the expert command
    /// batch completes, hence one triple per rank rather than one in total.
    private var expertStaging: [(gate: MTLBuffer, up: MTLBuffer,
                                 down: MTLBuffer)] = []
    /// Second descriptor on the GGUF with F_NOCACHE set (owned and closed by
    /// `GGUFModel`): the per-token slab preads go straight from disk to the
    /// staging buffers without churning the page cache — the mmap memcpy walk
    /// faulted one ~32 KB page run at a time at queue depth 1 AND evicted the
    /// resident signal path on tight-RAM machines. -1 = fall back to the mmap
    /// walk.
    private var streamFD: Int32 = -1
    /// Optional direct SSD → MTLBuffer path. It is disabled permanently on
    /// the first failed command and the same fill retries through pread.
    private var metalIO: (queue: MTLIOCommandQueue,
                          handle: MTLIOFileHandle)?

    private struct ExpertRead {
        let buffer: MTLBuffer
        let bufferOffset: Int
        let fileOffset: Int
        let bytes: Int

        var destination: UnsafeMutableRawPointer {
            buffer.contents() + bufferOffset
        }
    }

    /// LRU slot cache of streamed expert slabs, keyed by (layer, expert).
    /// Slots are sized for the largest expert in the file so mixed
    /// Q2_K/Q3_K recipes share the pool; a hit binds the slot's buffers
    /// directly into the expert batch (no I/O, no copy), a miss reassigns
    /// the least-recently-used slot not touched by the current selection.
    /// Reuse is safe because the expert batch is waited on before the next
    /// layer's gather can evict anything. Empty = cache disabled
    /// (`expertStaging` triples carry the token instead).
    private struct ExpertSlot {
        let gate: MTLBuffer
        let up: MTLBuffer
        let down: MTLBuffer
        var key = -1        // layer << 16 | expert, -1 = never filled
        var lastUse = 0
    }
    private var expertSlots: [ExpertSlot] = []
    private var expertSlotIndex: [Int: Int] = [:]
    /// When the budget can provide at least `activeExperts` slots to every
    /// streamed layer, keep disjoint per-layer ranges. A small global LRU
    /// otherwise thrashes deterministically: layer 47 evicts layer 1 before
    /// the next token revisits it.
    private var expertLayerSlotRanges: [Int: Range<Int>] = [:]
    private var expertClock = 0

    /// Shared event pacing the overlapped expert batch: a signaler thread
    /// sets `base+rank+1` as soon as that rank's slabs are in RAM (in rank
    /// order), the GPU waits on the same value right before the rank's
    /// encoder. nil (device without shared events) keeps the synchronous
    /// gather-then-encode path.
    private var expertEvent: MTLSharedEvent?
    private var expertEventBase: UInt64 = 0

    private final class CompletionStamp: @unchecked Sendable {
        private let lock = NSLock()
        private var value: Date?

        func mark() {
            lock.lock()
            value = Date()
            lock.unlock()
        }

        func read() -> Date? {
            lock.lock()
            defer { lock.unlock() }
            return value
        }
    }

    private struct PendingExpertCommands {
        let buffer: MTLCommandBuffer
        let wallStart: Date
        let completed: CompletionStamp
    }
    /// At most one expert tail can be in flight. The following layer's trunk
    /// waits before it reads its router result or reassigns any expert slot.
    private var pendingExpertCommands: PendingExpertCommands?

    /// Lazily-grown scratch and metadata for the opt-in expert-major prefill.
    private final class PrefillMoEScratch {
        private var buffers: [String: MTLBuffer] = [:]

        func buffer(
            device: MTLDevice,
            name: String,
            minimumBytes: Int
        ) throws -> MTLBuffer {
            if let buffer = buffers[name],
               buffer.length >= minimumBytes {
                return buffer
            }
            var grownBytes = 256
            while grownBytes < minimumBytes { grownBytes <<= 1 }
            guard let buffer = device.makeBuffer(
                length: grownBytes, options: .storageModeShared
            ) else {
                throw LagunaResidentModelError.bufferAllocation(
                    "prefill MoE \(name)")
            }
            buffers[name] = buffer
            return buffer
        }
    }
    private let prefillMoEScratch = PrefillMoEScratch()

    // Persistent activation scratch (F32 unless noted).
    private var hidden: MTLBuffer
    private var hiddenNext: MTLBuffer
    private let normed: MTLBuffer
    private let queryRows: MTLBuffer
    private let keyRows: MTLBuffer
    private let valueRows: MTLBuffer
    private let gateRows: MTLBuffer
    private let heads: MTLBuffer
    private let attnOut: MTLBuffer
    private let afterAttn: MTLBuffer
    private let ffnNormed: MTLBuffer
    private let ffnMid: MTLBuffer
    private let ffnOut: MTLBuffer
    private let routerLogits: MTLBuffer
    private let routerSelected: MTLBuffer   // Int32 × expertsUsed
    private let routerWeights: MTLBuffer    // Float × expertsUsed
    private let routerProbs: MTLBuffer      // Float × expertCount
    private let logits: MTLBuffer
    /// Generic FlashAttention partial layout for the 48 global query heads,
    /// sized for the maximum 32-way split and reused by every layer/token.
    private let splitAttentionTmp: MTLBuffer
    /// Shared across global layers: only one layer is encoded at a time.
    /// Scores grow lazily with the full KV; selected ids have fixed top-K.
    private var longAttentionScores: MTLBuffer?
    private var longAttentionSelected: MTLBuffer?
    private var longAttentionScratchBlocks = 0
    /// Largest activation-plane set allocated so far. Prefill calls reuse it
    /// until a wider chunk is needed instead of allocating roughly twenty
    /// shared Metal buffers for every prompt.
    private var cachedPrefillPlanes: PrefillPlanes?

    public init(runtime: MetalRuntime, path: String,
                options: LagunaResidentModelOptions = .init()) throws {
        let model = try GGUFModel(path: path, metalMapping: false,
                                  prefetchCPU: false)
        let map = try LagunaWeightMap(model: model)
        guard map.quantizationLayout == .signalQ8 else {
            throw LagunaResidentModelError.unsupportedRecipe(
                "the first cut runs the official Q8_0 signal-path file; the "
                + "legacy F16/Q4_K recipe needs its matvec kernels wired first"
            )
        }

        self.runtime = runtime
        self.model = model
        self.configuration = map.configuration
        self.shape = map.configuration.shape
        self.cacheCapacity = max(1, options.cacheCapacity)
        self.initialFullCacheCapacity = max(
            1, min(options.initialFullCacheCapacity ?? 512,
                   self.cacheCapacity))
        self.embeddingDescriptor = try map.global(.tokenEmbedding)
        self.activeExperts = Self.resolvedActiveExperts(
            options.activeExperts)
        self.residentExpertLayers = max(
            0,
            min(options.residentExpertLayers ?? 0,
                Int(shape.nLayer - shape.nLeadingDense)))
        self.prefillChunk = Self.resolvedPrefillChunk(
            options.prefillChunk)
        self.preadSplit = Self.resolvedPreadSplit(options.preadSplit)
        self.willNeedExperts = options.willNeedExperts
        self.lockResident = options.lockResident
        self.simdgroupsPerThreadgroup = Self.resolvedSimdgroups(
            options.simdgroupsPerThreadgroup)
        self.longAttentionIndexEnabled = options.longAttentionIndex
        self.longAttentionBlockSize =
            Self.resolvedLongAttentionBlockSize(
                options.longAttentionBlockSize)
        self.longAttentionTopBlocks =
            Self.resolvedLongAttentionTopBlocks(
                options.longAttentionTopBlocks)
        self.longAttentionRecentTokens =
            Self.resolvedLongAttentionRecentTokens(
                options.longAttentionRecentTokens)
        self.longAttentionThreshold = max(
            self.longAttentionRecentTokens,
            options.longAttentionThreshold ?? 4_096)

        var uploadStaging: MTLBuffer?
        func upload(_ descriptor: LagunaWeightDescriptor) throws -> MTLBuffer {
            let byteCount = Int(descriptor.bytes)
            let buffer: MTLBuffer
            if Self.privateResidentWeights {
                if uploadStaging == nil
                    || uploadStaging!.length < byteCount {
                    guard let grown = runtime.device.makeBuffer(
                        length: byteCount, options: .storageModeShared
                    ) else {
                        throw LagunaResidentModelError.bufferAllocation(
                            descriptor.name + " staging")
                    }
                    uploadStaging = grown
                }
                guard let staging = uploadStaging,
                      let target = runtime.device.makeBuffer(
                        length: byteCount, options: .storageModePrivate),
                      let commandBuffer = runtime.queue.makeCommandBuffer(),
                      let blit = commandBuffer.makeBlitCommandEncoder() else {
                    throw LagunaResidentModelError.bufferAllocation(
                        descriptor.name)
                }
                memcpy(
                    staging.contents(),
                    model.mapBase + Int(descriptor.absOffset),
                    byteCount)
                blit.copy(
                    from: staging, sourceOffset: 0,
                    to: target, destinationOffset: 0,
                    size: byteCount)
                blit.endEncoding()
                commandBuffer.commit()
                commandBuffer.waitUntilCompleted()
                if let error = commandBuffer.error { throw error }
                buffer = target
            } else {
                guard let shared = runtime.device.makeBuffer(
                    bytes: model.mapBase + Int(descriptor.absOffset),
                    length: byteCount,
                    options: .storageModeShared
                ) else {
                    throw LagunaResidentModelError.bufferAllocation(
                        descriptor.name)
                }
                buffer = shared
            }
            if Self.discardUploadedFilePages {
                let page = Int(getpagesize())
                let start = Int(descriptor.absOffset)
                let end = start + Int(descriptor.bytes)
                let alignedStart = ((start + page - 1) / page) * page
                let alignedEnd = (end / page) * page
                if alignedEnd > alignedStart {
                    _ = posix_madvise(
                        UnsafeMutableRawPointer(
                            mutating: model.mapBase + alignedStart),
                        alignedEnd - alignedStart,
                        POSIX_MADV_DONTNEED)
                }
            }
            return buffer
        }
        func scratch(_ count: Int, _ what: String) throws -> MTLBuffer {
            guard let buffer = runtime.device.makeBuffer(
                length: count * MemoryLayout<Float>.stride,
                options: .storageModeShared
            ) else {
                throw LagunaResidentModelError.bufferAllocation(what)
            }
            memset(buffer.contents(), 0, count * MemoryLayout<Float>.stride)
            return buffer
        }

        self.outputNorm = try upload(map.global(.outputNorm))
        self.outputHead = try upload(map.global(.output))

        let embd = Int(shape.nEmbd)
        let maxQueryWidth = Int(shape.nHead) * Int(shape.nHeadDim)
        self.hidden = try scratch(embd, "hidden state")
        self.hiddenNext = try scratch(embd, "next hidden state")
        self.normed = try scratch(embd, "attention norm")
        self.queryRows = try scratch(maxQueryWidth, "query projection")
        self.keyRows = try scratch(Int(shape.keyValueProjectionWidth), "key projection")
        self.valueRows = try scratch(Int(shape.keyValueProjectionWidth), "value projection")
        self.gateRows = try scratch(Int(shape.nHead), "gate projection")
        self.heads = try scratch(maxQueryWidth, "attention heads")
        self.attnOut = try scratch(embd, "attention output")
        self.afterAttn = try scratch(embd, "post-attention hidden")
        self.ffnNormed = try scratch(embd, "FFN norm")
        self.ffnMid = try scratch(Int(shape.nFFDense), "FFN intermediate")
        self.ffnOut = try scratch(embd, "FFN output")
        self.routerLogits = try scratch(LagunaRouterReference.expertCount, "router logits")
        self.routerSelected = try scratch(activeExperts, "router selection")
        self.routerWeights = try scratch(activeExperts, "router weights")
        self.routerProbs = try scratch(LagunaRouterReference.expertCount, "router probabilities")
        self.logits = try scratch(Int(shape.nVocab), "logits")
        let splitRows = Int(shape.nHeadFull)
        let splitWorkgroups = Self.decodeSplitKMaximumWorkgroups
        let splitValues = splitRows * Int(shape.nHeadDim) * splitWorkgroups
        let splitStats = splitRows * 2 * splitWorkgroups
        self.splitAttentionTmp = try scratch(
            splitValues + splitStats, "split-K attention partials")

        let totalLayers = Int(shape.nLayer)
        let keptLayers = min(max(options.layerCount ?? totalLayers, 1), totalLayers)
        var layers: [Layer] = []
        layers.reserveCapacity(keptLayers)
        var maxGateUpExpertBytes = 0
        var maxDownExpertBytes = 0
        for index in 0..<keptLayers {
            let spec = LagunaAttentionSpec.spec(
                forLayer: index, shape: shape,
                contextSize: cacheCapacity
            )
            let cacheRows = shape.isSlidingWindowLayer(index)
                ? min(Int(shape.nSWA), cacheCapacity)
                : initialFullCacheCapacity
            let boundedSpec = LagunaAttentionSpec(
                headCount: spec.headCount, kvHeadCount: spec.kvHeadCount,
                headDim: spec.headDim, rotationDims: spec.rotationDims,
                cacheCapacity: cacheRows,
                ropeFrequencyBase: spec.ropeFrequencyBase,
                ropeFrequencyScale: spec.ropeFrequencyScale,
                extrapolationFactor: spec.extrapolationFactor,
                attentionFactor: spec.attentionFactor,
                betaFast: spec.betaFast, betaSlow: spec.betaSlow,
                ropeOriginalContext: spec.ropeOriginalContext,
                rmsEpsilon: spec.rmsEpsilon
            )

            let ffn: FFN
            if index < Int(shape.nLeadingDense) {
                ffn = .dense(
                    gate: try upload(map.layer(index, .denseGate)),
                    up: try upload(map.layer(index, .denseUp)),
                    down: try upload(map.layer(index, .denseDown))
                )
            } else {
                let routed = try map.routedExperts(layer: index)
                guard Self.routedTypes.contains(routed.gate.type),
                      routed.up.type == routed.gate.type,
                      routed.down.type == routed.gate.type else {
                    throw LagunaResidentModelError.unsupportedRecipe(
                        "layer \(index) routed experts are "
                        + "\(GGUF.typeName(routed.gate.type))/"
                        + "\(GGUF.typeName(routed.down.type)); this engine "
                        + "runs coherent Q2_K/Q3_K/Q4_K routed experts"
                    )
                }
                let gateUpExpertBytes = Int(routed.gate.bytes)
                    / Int(shape.nExpert)
                let downExpertBytes = Int(routed.down.bytes)
                    / Int(shape.nExpert)
                let routedWeights: RoutedWeights
                let routedLayer = index - Int(shape.nLeadingDense)
                if options.expertStreaming
                    && routedLayer >= residentExpertLayers {
                    routedWeights = .streamed(
                        gate: Int(routed.gate.absOffset),
                        up: Int(routed.up.absOffset),
                        down: Int(routed.down.absOffset)
                    )
                    maxGateUpExpertBytes = max(maxGateUpExpertBytes,
                                               gateUpExpertBytes)
                    maxDownExpertBytes = max(maxDownExpertBytes,
                                             downExpertBytes)
                } else {
                    routedWeights = .resident(
                        gate: try upload(routed.gate),
                        up: try upload(routed.up),
                        down: try upload(routed.down)
                    )
                }
                ffn = .moe(
                    routedType: routed.gate.type,
                    routerRows: try upload(map.layer(index, .router)),
                    routerBias: try upload(map.layer(index, .routerBias)),
                    routed: routedWeights,
                    gateUpExpertBytes: gateUpExpertBytes,
                    downExpertBytes: downExpertBytes,
                    sharedGate: try upload(map.layer(index, .sharedGate)),
                    sharedUp: try upload(map.layer(index, .sharedUp)),
                    sharedDown: try upload(map.layer(index, .sharedDown))
                )
            }

            let longIndex: LongAttentionIndex?
            if longAttentionIndexEnabled
                && !shape.isSlidingWindowLayer(index) {
                longIndex = try Self.makeLongAttentionIndex(
                    runtime: runtime, capacity: cacheRows,
                    rowWidth: Int(shape.keyValueProjectionWidth),
                    blockSize: longAttentionBlockSize)
            } else {
                longIndex = nil
            }
            layers.append(Layer(
                index: index,
                spec: boundedSpec,
                attnNorm: try upload(map.layer(index, .attentionNorm)),
                query: try upload(map.layer(index, .attentionQuery)),
                key: try upload(map.layer(index, .attentionKey)),
                value: try upload(map.layer(index, .attentionValue)),
                gate: try upload(map.layer(index, .attentionGate)),
                queryNorm: try upload(map.layer(index, .attentionQueryNorm)),
                keyNorm: try upload(map.layer(index, .attentionKeyNorm)),
                attnOutput: try upload(map.layer(index, .attentionOutput)),
                ffnNorm: try upload(map.layer(index, .feedForwardNorm)),
                ffn: ffn,
                cache: try runtime.lagunaKVCache(
                    capacity: cacheRows,
                    rowWidth: Int(shape.keyValueProjectionWidth)
                ),
                longAttentionIndex: longIndex
            ))
        }
        self.layers = layers

        if options.expertStreaming, maxGateUpExpertBytes + maxDownExpertBytes > 0 {
            // Slot cache first: when the budget yields at least one slot per
            // active-expert rank the slots replace the staging triples (they
            // are the staging, plus persistence across tokens). Allocation
            // stops quietly at the first failure — whatever fits still works,
            // and below the minimum the plain staging path takes over.
            let slotBytes = 2 * maxGateUpExpertBytes + maxDownExpertBytes
            let requested = options.expertCacheSlots.map { max(0, $0) }
                ?? (options.expertCacheMB > 0
                    ? (options.expertCacheMB << 20) / slotBytes : 0)
            if requested >= activeExperts {
                var slots: [ExpertSlot] = []
                slots.reserveCapacity(requested)
                for _ in 0..<requested {
                    guard let gate = runtime.device.makeBuffer(
                            length: maxGateUpExpertBytes,
                            options: .storageModeShared),
                          let up = runtime.device.makeBuffer(
                            length: maxGateUpExpertBytes,
                            options: .storageModeShared),
                          let down = runtime.device.makeBuffer(
                            length: maxDownExpertBytes,
                            options: .storageModeShared) else { break }
                    slots.append(ExpertSlot(gate: gate, up: up, down: down))
                }
                if slots.count >= activeExperts {
                    self.expertSlots = slots
                    self.expertSlotIndex = Dictionary(
                        minimumCapacity: slots.count)
                    let streamedLayerIndices = layers.compactMap {
                        layer -> Int? in
                        guard case .moe(
                            _, _, _, let routed, _, _, _, _, _
                        ) = layer.ffn,
                              case .streamed = routed else {
                            return nil
                        }
                        return layer.index
                    }
                    let minimumPartitioned = activeExperts
                        * streamedLayerIndices.count
                    if Self.partitionExpertCache,
                       !streamedLayerIndices.isEmpty,
                       slots.count >= minimumPartitioned {
                        let base = slots.count
                            / streamedLayerIndices.count
                        let extra = slots.count
                            % streamedLayerIndices.count
                        var start = 0
                        for (rank, layerIndex)
                            in streamedLayerIndices.enumerated() {
                            let count = base + (rank < extra ? 1 : 0)
                            self.expertLayerSlotRanges[layerIndex] =
                                start..<(start + count)
                            start += count
                        }
                    }
                }
            }
            if expertSlots.isEmpty {
                var staging: [(gate: MTLBuffer, up: MTLBuffer,
                               down: MTLBuffer)] = []
                staging.reserveCapacity(activeExperts)
                for _ in 0..<activeExperts {
                    guard let gate = runtime.device.makeBuffer(
                            length: maxGateUpExpertBytes,
                            options: .storageModeShared),
                          let up = runtime.device.makeBuffer(
                            length: maxGateUpExpertBytes,
                            options: .storageModeShared),
                          let down = runtime.device.makeBuffer(
                            length: maxDownExpertBytes,
                            options: .storageModeShared) else {
                        throw LagunaResidentModelError.bufferAllocation(
                            "expert streaming staging")
                    }
                    staging.append((gate, up, down))
                }
                self.expertStaging = staging
            }
            if options.expertPread {
                self.streamFD = model.uncachedFD() ?? -1
            }
            if options.metalIO {
                self.metalIO = Self.makeMetalIO(
                    runtime: runtime, path: path)
            }
            self.expertEvent = runtime.device.makeSharedEvent()
        }
        if lockResident {
            _ = mlock(outputNorm.contents(), outputNorm.length)
            _ = mlock(outputHead.contents(), outputHead.length)
            for slot in expertSlots {
                _ = mlock(slot.gate.contents(), slot.gate.length)
                _ = mlock(slot.up.contents(), slot.up.length)
                _ = mlock(slot.down.contents(), slot.down.length)
            }
            for staging in expertStaging {
                _ = mlock(staging.gate.contents(), staging.gate.length)
                _ = mlock(staging.up.contents(), staging.up.length)
                _ = mlock(staging.down.contents(), staging.down.length)
            }
        }
    }

    // MARK: Public API

    public func resetProfile() {
        profile = DecodeProfile()
        gpuAttnS = 0; gpuExpertsS = 0; gpuDenseS = 0; gpuHeadS = 0
        indexedAttentionDispatches = 0
    }

    /// Same shape as the GLM report: the shared per-phase table plus one
    /// line splitting REAL GPU execution from host overhead (encode, the
    /// two syncs per MoE layer, the router readback, staging reads).
    public func profileReport(title: String = "Profilo decode") -> String {
        var report = profile.report(title: title)
        if profile.forwards > 0 {
            let f = Double(profile.forwards)
            func ms(_ s: Double) -> String {
                String(format: "%.0f", s / f * 1000)
            }
            report += "\n  gpu          route/attn \(ms(gpuAttnS))"
                + " · experts \(ms(gpuExpertsS))"
                + " · layer(alt) \(ms(gpuDenseS))"
                + " · head \(ms(gpuHeadS)) ms/token — il resto è host"
                + " (encode/sync/readback router"
                + (isExpertStreaming ? "/gather)" : ")")
            if indexedAttentionDispatches > 0 {
                report += "\n  long-attn    \(indexedAttentionDispatches)"
                    + " dispatch indicizzate"
            }
        }
        return report
    }

    public func resetContext(releaseExcessKV: Bool = false) {
        try? flushPendingExpertCommands(wait: true)
        position = 0
        if releaseExcessKV {
            trimFullCacheCapacity()
        }
    }

    /// Return the full-attention layers to the small initial allocation.
    /// The live row count is already zero, so no copy is needed. This matters
    /// after leaving a long conversation: retaining a formerly-grown 32k KV
    /// keeps roughly 1.5 GiB wired and recreates the same VM pressure that
    /// lazy allocation is meant to avoid.
    private func trimFullCacheCapacity() {
        for index in layers.indices
        where !shape.isSlidingWindowLayer(layers[index].index)
            && layers[index].cache.capacity > initialFullCacheCapacity {
            guard let compact = try? runtime.lagunaKVCache(
                capacity: initialFullCacheCapacity,
                rowWidth: layers[index].cache.rowWidth
            ) else { continue }
            let spec = layers[index].spec
            layers[index].cache = compact
            if layers[index].longAttentionIndex != nil {
                layers[index].longAttentionIndex =
                    try? Self.makeLongAttentionIndex(
                        runtime: runtime,
                        capacity: initialFullCacheCapacity,
                        rowWidth: compact.rowWidth,
                        blockSize: longAttentionBlockSize)
            }
            layers[index].spec = LagunaAttentionSpec(
                headCount: spec.headCount,
                kvHeadCount: spec.kvHeadCount,
                headDim: spec.headDim,
                rotationDims: spec.rotationDims,
                cacheCapacity: initialFullCacheCapacity,
                ropeFrequencyBase: spec.ropeFrequencyBase,
                ropeFrequencyScale: spec.ropeFrequencyScale,
                extrapolationFactor: spec.extrapolationFactor,
                attentionFactor: spec.attentionFactor,
                betaFast: spec.betaFast,
                betaSlow: spec.betaSlow,
                ropeOriginalContext: spec.ropeOriginalContext,
                rmsEpsilon: spec.rmsEpsilon)
        }
    }

    /// Grow only the 12 full-attention caches. Rows are still contiguous at
    /// every growth point (growth happens before the old capacity can wrap),
    /// so copying the existing F16 prefix preserves exact cache semantics.
    private func ensureFullCacheCapacity(requiredRows: Int) throws {
        guard requiredRows > initialFullCacheCapacity else { return }
        var target = initialFullCacheCapacity
        while target < requiredRows {
            target = min(cacheCapacity, max(target * 2, requiredRows))
        }
        for index in layers.indices
        where !shape.isSlidingWindowLayer(layers[index].index)
            && layers[index].cache.capacity < target {
            let old = layers[index].cache
            let grown = try runtime.lagunaKVCache(
                capacity: target, rowWidth: old.rowWidth)
            memcpy(grown.keys.contents(), old.keys.contents(),
                   old.keys.length)
            memcpy(grown.values.contents(), old.values.contents(),
                   old.values.length)
            let spec = layers[index].spec
            layers[index].cache = grown
            if let oldIndex = layers[index].longAttentionIndex {
                let grownIndex = try Self.makeLongAttentionIndex(
                    runtime: runtime, capacity: target,
                    rowWidth: oldIndex.rowWidth,
                    blockSize: oldIndex.blockSize)
                memcpy(grownIndex.keys.contents(), oldIndex.keys.contents(),
                       oldIndex.keys.length)
                layers[index].longAttentionIndex = grownIndex
            }
            layers[index].spec = LagunaAttentionSpec(
                headCount: spec.headCount,
                kvHeadCount: spec.kvHeadCount,
                headDim: spec.headDim,
                rotationDims: spec.rotationDims,
                cacheCapacity: target,
                ropeFrequencyBase: spec.ropeFrequencyBase,
                ropeFrequencyScale: spec.ropeFrequencyScale,
                extrapolationFactor: spec.extrapolationFactor,
                attentionFactor: spec.attentionFactor,
                betaFast: spec.betaFast,
                betaSlow: spec.betaSlow,
                ropeOriginalContext: spec.ropeOriginalContext,
                rmsEpsilon: spec.rmsEpsilon)
        }
    }

    private func ensureLongAttentionScratch(
        blockCapacity: Int
    ) throws {
        guard longAttentionIndexEnabled,
              blockCapacity > longAttentionScratchBlocks else { return }
        let heads = Int(shape.nHeadFull)
        let scoreBytes = heads * blockCapacity
            * MemoryLayout<Float>.stride
        guard let scores = runtime.device.makeBuffer(
                  length: scoreBytes, options: .storageModeShared
              ) else {
            throw LagunaResidentModelError.bufferAllocation(
                "long-attention scores")
        }
        if longAttentionSelected == nil {
            let selectedBytes = heads * longAttentionTopBlocks
                * MemoryLayout<Int32>.stride
            guard let selected = runtime.device.makeBuffer(
                length: selectedBytes, options: .storageModeShared
            ) else {
                throw LagunaResidentModelError.bufferAllocation(
                    "long-attention selected blocks")
            }
            longAttentionSelected = selected
        }
        longAttentionScores = scores
        longAttentionScratchBlocks = blockCapacity
    }

    private func usesIndexedLongAttention(
        position: Int, index: LongAttentionIndex
    ) -> Bool {
        let visible = position + 1
        guard longAttentionIndexEnabled,
              visible >= longAttentionThreshold else { return false }
        let recentStart = max(0, visible - longAttentionRecentTokens)
        let selectableBlocks = recentStart / index.blockSize
        return selectableBlocks > longAttentionTopBlocks
    }

    // MARK: - Disk KV checkpoints

    /// Engine-native checkpoint identity. On disk, little-endian "LKV1".
    public static let kvMagic: UInt32 = 0x3156_4B4C
    private static let kvHeaderBytes = 40

    /// Persist the live F16 cache without first materializing a model-sized
    /// `Data`: header, token ids and each Metal plane are streamed directly
    /// through an F_NOCACHE descriptor. Full-attention layers store
    /// `position` rows; sliding layers store their physical ring (up to 512
    /// rows), preserving the exact `position % capacity` mapping on restore.
    public func saveKVCheckpoint(to url: URL, tokens: [Int32]) throws {
        try flushPendingExpertCommands(wait: true)
        guard position > 0, tokens.count == position else {
            throw LagunaResidentModelError.kvCheckpoint(
                "i token (\(tokens.count)) non combaciano con la position "
                + "(\(position))")
        }

        var header = Data()
        func u32(_ value: UInt32) {
            withUnsafeBytes(of: value.littleEndian) {
                header.append(contentsOf: $0)
            }
        }
        func u64(_ value: UInt64) {
            withUnsafeBytes(of: value.littleEndian) {
                header.append(contentsOf: $0)
            }
        }
        let rowWidth = layers.first?.cache.rowWidth ?? 0
        let slidingCapacity = min(Int(shape.nSWA), cacheCapacity)
        let slidingLayers = layers.reduce(into: 0) {
            if shape.isSlidingWindowLayer($1.index) { $0 += 1 }
        }
        u32(Self.kvMagic); u32(1)
        u64(model.size)
        u32(UInt32(layers.count)); u32(UInt32(position))
        u32(UInt32(rowWidth)); u32(UInt32(slidingCapacity))
        u32(UInt32(slidingLayers)); u32(0)
        precondition(header.count == Self.kvHeaderBytes)

        let directory = url.deletingLastPathComponent()
        try FileManager.default.createDirectory(
            at: directory, withIntermediateDirectories: true)
        let part = url.appendingPathExtension("part")
        let fd = open(part.path, O_CREAT | O_TRUNC | O_WRONLY, 0o644)
        guard fd >= 0 else {
            throw LagunaResidentModelError.kvCheckpoint(
                String(cString: strerror(errno)))
        }
        _ = fcntl(fd, F_NOCACHE, 1)
        var fdOpen = true
        defer {
            if fdOpen { close(fd) }
            try? FileManager.default.removeItem(at: part)
        }
        func writeAll(_ pointer: UnsafeRawPointer, _ byteCount: Int) throws {
            var offset = 0
            while offset < byteCount {
                let written = Darwin.write(
                    fd, pointer.advanced(by: offset), byteCount - offset)
                guard written > 0 else {
                    throw LagunaResidentModelError.kvCheckpoint(
                        String(cString: strerror(errno)))
                }
                offset += written
            }
        }
        try header.withUnsafeBytes {
            try writeAll($0.baseAddress!, $0.count)
        }
        try tokens.withUnsafeBytes {
            try writeAll($0.baseAddress!, $0.count)
        }
        for layer in layers {
            let rows = shape.isSlidingWindowLayer(layer.index)
                ? min(position, layer.cache.capacity) : position
            let bytes = rows * layer.cache.rowBytes
            try writeAll(layer.cache.keys.contents(), bytes)
            try writeAll(layer.cache.values.contents(), bytes)
        }
        guard fsync(fd) == 0 else {
            throw LagunaResidentModelError.kvCheckpoint(
                String(cString: strerror(errno)))
        }
        close(fd)
        fdOpen = false
        guard rename(part.path, url.path) == 0 else {
            throw LagunaResidentModelError.kvCheckpoint(
                String(cString: strerror(errno)))
        }
    }

    /// Header-only validation used before restore. A checkpoint belongs to
    /// this exact GGUF and loaded layer geometry; malformed/foreign files
    /// are ignored by returning nil.
    public func peekKVCheckpoint(at url: URL) -> [Int32]? {
        guard let handle = FileHandle(forReadingAtPath: url.path) else {
            return nil
        }
        defer { try? handle.close() }
        guard let header = try? handle.read(
                  upToCount: Self.kvHeaderBytes),
              header.count == Self.kvHeaderBytes else { return nil }
        func u32(_ offset: Int) -> UInt32 {
            UInt32(littleEndian: header.withUnsafeBytes {
                $0.loadUnaligned(fromByteOffset: offset, as: UInt32.self)
            })
        }
        func u64(_ offset: Int) -> UInt64 {
            UInt64(littleEndian: header.withUnsafeBytes {
                $0.loadUnaligned(fromByteOffset: offset, as: UInt64.self)
            })
        }
        let rowWidth = layers.first?.cache.rowWidth ?? 0
        let slidingCapacity = min(Int(shape.nSWA), cacheCapacity)
        let slidingLayers = layers.reduce(into: 0) {
            if shape.isSlidingWindowLayer($1.index) { $0 += 1 }
        }
        guard u32(0) == Self.kvMagic, u32(4) == 1,
              u64(8) == model.size,
              u32(16) == UInt32(layers.count),
              u32(24) == UInt32(rowWidth),
              u32(28) == UInt32(slidingCapacity),
              u32(32) == UInt32(slidingLayers) else { return nil }
        let count = Int(u32(20))
        guard count > 0, count <= cacheCapacity,
              let tokenData = try? handle.read(upToCount: count * 4),
              tokenData.count == count * 4 else { return nil }
        return tokenData.withUnsafeBytes {
            Array($0.bindMemory(to: Int32.self))
        }
    }

    /// Restore the native cache planes directly into Metal shared memory.
    /// `position` is published only after every layer has been read, so a
    /// truncated file cannot turn partial rows into live state.
    @discardableResult
    public func restoreKVCheckpoint(from url: URL) throws -> [Int32] {
        try flushPendingExpertCommands(wait: true)
        guard let tokens = peekKVCheckpoint(at: url) else {
            throw LagunaResidentModelError.kvCheckpoint(
                "checkpoint assente, incompatibile o estraneo")
        }
        let count = tokens.count
        try ensureFullCacheCapacity(requiredRows: count)
        let fd = open(url.path, O_RDONLY)
        guard fd >= 0 else {
            throw LagunaResidentModelError.kvCheckpoint(
                String(cString: strerror(errno)))
        }
        defer { close(fd) }
        _ = fcntl(fd, F_NOCACHE, 1)
        let payloadOffset = Self.kvHeaderBytes + count * 4
        guard lseek(fd, off_t(payloadOffset), SEEK_SET)
                == off_t(payloadOffset) else {
            throw LagunaResidentModelError.kvCheckpoint(
                String(cString: strerror(errno)))
        }
        func readAll(_ pointer: UnsafeMutableRawPointer,
                     _ byteCount: Int) throws {
            var offset = 0
            while offset < byteCount {
                let readCount = Darwin.read(
                    fd, pointer.advanced(by: offset), byteCount - offset)
                guard readCount > 0 else {
                    throw LagunaResidentModelError.kvCheckpoint(
                        readCount == 0 ? "checkpoint troncato"
                        : String(cString: strerror(errno)))
                }
                offset += readCount
            }
        }
        for layer in layers {
            let rows = shape.isSlidingWindowLayer(layer.index)
                ? min(count, layer.cache.capacity) : count
            let bytes = rows * layer.cache.rowBytes
            try readAll(layer.cache.keys.contents(), bytes)
            try readAll(layer.cache.values.contents(), bytes)
        }
        try rebuildLongAttentionIndices(rowCount: count)
        position = count
        return tokens
    }

    private func rebuildLongAttentionIndices(rowCount: Int) throws {
        guard longAttentionIndexEnabled, rowCount > 0 else { return }
        guard let buffer = runtime.queue.makeCommandBuffer(),
              let encoder = buffer.makeComputeCommandEncoder() else {
            throw LagunaResidentModelError.bufferAllocation(
                "long-attention index rebuild")
        }
        let commands = Commands(buffer: buffer, encoder: encoder)
        for layer in layers {
            if let index = layer.longAttentionIndex {
                try encodeLongAttentionIndexUpdate(
                    commands, cache: layer.cache, index: index,
                    position: 0, count: rowCount)
            }
        }
        encoder.endEncoding()
        buffer.commit()
        buffer.waitUntilCompleted()
        if let error = buffer.error { throw error }
    }

    /// Dequantize one Q8_0 embedding row on the CPU (rows are read per token,
    /// exactly like the reference engine; the table is never resident).
    public func embeddingRow(_ token: Int32) throws -> [Float] {
        try Self.dequantizeQ8Row(
            base: model.mapBase + Int(embeddingDescriptor.absOffset),
            row: Int(token),
            rowCount: Int(shape.nVocab),
            width: Int(shape.nEmbd)
        )
    }

    private func forwardNextLogits(
        _ token: Int32
    ) throws -> UnsafeBufferPointer<Float> {
        guard token >= 0, Int(token) < Int(shape.nVocab) else {
            throw LagunaResidentModelError.invalidToken(token)
        }
        guard position < cacheCapacity else {
            throw LagunaResidentModelError.contextFull(capacity: cacheCapacity)
        }
        try ensureFullCacheCapacity(requiredRows: position + 1)

        let embedStart = Date()
        let embedding = try embeddingRow(token)
        embedding.withUnsafeBytes {
            _ = memcpy(hidden.contents(), $0.baseAddress!, $0.count)
        }
        profile.embedS += Date().timeIntervalSince(embedStart)

        for layer in layers {
            try forward(layer: layer, position: position)
        }

        let headStart = Date()
        try encodeOutputHead()
        position += 1

        let count = Int(shape.nVocab)
        profile.headS += Date().timeIntervalSince(headStart)
        profile.forwards += 1
        let pointer = logits.contents().bindMemory(
            to: Float.self, capacity: count)
        return UnsafeBufferPointer(start: pointer, count: count)
    }

    public func forwardNext(_ token: Int32) throws -> [Float] {
        Array(try forwardNextLogits(token))
    }

    public func forwardNextGreedy(_ token: Int32) throws -> Int32 {
        let logits = try forwardNextLogits(token)
        var best = logits[0]
        var bestIndex: Int32 = 0
        for index in 1..<logits.count where logits[index] > best {
            best = logits[index]
            bestIndex = Int32(index)
        }
        return bestIndex
    }

    /// Consume one token and sample directly from the shared logits buffer.
    /// This avoids materializing a ~392 KiB `[Float]` on every chat decode
    /// step (unless repetition penalty explicitly requires a mutable copy).
    public func forwardNextSampled(
        _ token: Int32,
        temperature: Float,
        topK: Int,
        topP: Float,
        minP: Float,
        repetitionPenalty: Float,
        recentTokens: ArraySlice<Int32>,
        rng: inout UInt64
    ) throws -> Int32? {
        let logits = try forwardNextLogits(token)
        return LagunaSampler.sample(
            logits: logits,
            temperature: temperature,
            topK: topK,
            topP: topP,
            minP: minP,
            repetitionPenalty: repetitionPenalty,
            recentTokens: recentTokens,
            rng: &rng)
    }

    /// Prefill; returns the logits of the last prompt token. Multi-token
    /// prompts run LAYER-MAJOR in chunks: phase A (attention half + router)
    /// of every token in the chunk goes into ONE command buffer per layer —
    /// the causal order is the encode order, so numerics match the
    /// token-by-token path by construction — followed by ONE router
    /// readback, ONE union gather over the slot cache and ONE expert batch
    /// per layer. Two syncs per LAYER instead of two per layer×token, and
    /// each streamed expert slab is read once per layer for the whole
    /// chunk instead of once per token. Falls back to the sequential
    /// decode path when the streamed slot cache cannot hold a layer union.
    /// `onLayerProgress(done, total)` fires per swept layer (all chunks).
    public func prefill(_ tokens: [Int32],
                        onLayerProgress: ((Int, Int) -> Void)? = nil)
        throws -> [Float] {
        try forwardPrompt(tokens, collectLogits: false,
                          onLayerProgress: onLayerProgress).last ?? []
    }

    /// Layer-major forward of a window returning the logits of EVERY
    /// position (the Correctness benchmark scores at prefill speed).
    public func forwardBatch(_ tokens: [Int32]) throws -> [[Float]] {
        try forwardPrompt(tokens, collectLogits: true, onLayerProgress: nil)
    }

    private func forwardPrompt(_ tokens: [Int32], collectLogits: Bool,
                               onLayerProgress: ((Int, Int) -> Void)?)
        throws -> [[Float]] {
        guard !tokens.isEmpty else { return [] }
        guard position + tokens.count <= cacheCapacity else {
            throw LagunaResidentModelError.contextFull(
                capacity: cacheCapacity)
        }
        try ensureFullCacheCapacity(
            requiredRows: position + tokens.count)
        let chunkSize = layerMajorChunkSize()
        guard tokens.count > 1, chunkSize > 1 else {
            var out: [[Float]] = []
            for token in tokens {
                try Task.checkCancellation()
                let logits = try forwardNext(token)
                if collectLogits { out.append(logits) }
                else { out = [logits] }
            }
            return out
        }
        let planes = try reusablePrefillPlanes(
            minimumCapacity: min(tokens.count, chunkSize))
        var results: [[Float]] = []
        var done = 0
        let totalLayerSteps = layers.count
            * ((tokens.count + chunkSize - 1) / chunkSize)
        var layerSteps = 0
        while done < tokens.count {
            let chunk = Array(tokens[done ..< min(done + chunkSize,
                                                  tokens.count)])
            let lastPlane = try sweepChunk(chunk, planes: planes) { _ in
                layerSteps += 1
                onLayerProgress?(layerSteps, totalLayerSteps)
            }
            done += chunk.count
            if collectLogits {
                for i in 0..<chunk.count {
                    results.append(try headLogits(
                        plane: lastPlane,
                        offset: i * planes.embdBytes))
                }
            } else if done == tokens.count {
                results = [try headLogits(
                    plane: lastPlane,
                    offset: (chunk.count - 1) * planes.embdBytes)]
            }
        }
        if !expertLayerSlotRanges.isEmpty {
            // Layer-major prefill needs a large per-layer union and therefore
            // borrows the whole pool. Decode starts from clean ownership so
            // each layer can retain its previous top-N in its own small range.
            expertSlotIndex.removeAll(keepingCapacity: true)
            for index in expertSlots.indices {
                expertSlots[index].key = -1
                expertSlots[index].lastUse = 0
            }
            expertClock = 0
        }
        return results
    }

    /// Chunk width the layer-major sweep can afford: bounded by the
    /// activation planes (fixed cap) and, while streaming, by the slot
    /// cache — one layer's expert union must fit alongside safety margin,
    /// or eviction would tear slabs out from under the batch. 1 = fall
    /// back to the sequential path.
    private func layerMajorChunkSize() -> Int {
        let cap = prefillChunk
        guard isExpertStreaming else { return cap }
        guard !expertSlots.isEmpty else { return 1 }
        let used = activeExperts
        let spare = expertSlots.count - used
        if spare >= LagunaRouterReference.expertCount { return cap }
        return max(1, min(cap, spare / used))
    }

    private struct PrefillPlanes {
        let capacity: Int
        let embdBytes: Int
        let kvBytes: Int
        let midBytes: Int
        let routerLogitsBytes: Int
        let routerPickBytes: Int
        var hidden: MTLBuffer
        var hiddenNext: MTLBuffer
        let normed: MTLBuffer
        let queryRows: MTLBuffer
        let keyRows: MTLBuffer
        let valueRows: MTLBuffer
        let gateRows: MTLBuffer
        let heads: MTLBuffer
        let attnOut: MTLBuffer
        let afterAttn: MTLBuffer
        let ffnNormed: MTLBuffer
        let ffnMid: MTLBuffer
        let ffnUp: MTLBuffer
        let ffnOut: MTLBuffer
        let routerLogits: MTLBuffer
        let routerSelected: MTLBuffer
        let routerWeights: MTLBuffer
        let routerProbs: MTLBuffer
        let stagedKeys: MTLBuffer
        let stagedValues: MTLBuffer
    }

    private func makePrefillPlanes(capacity: Int) throws -> PrefillPlanes {
        let embd = Int(shape.nEmbd)
        let maxQueryWidth = Int(shape.nHead) * Int(shape.nHeadDim)
        let kvWidth = Int(shape.keyValueProjectionWidth)
        let used = activeExperts
        let experts = LagunaRouterReference.expertCount
        func plane(_ width: Int, _ what: String) throws -> MTLBuffer {
            guard let buffer = runtime.device.makeBuffer(
                length: capacity * width * MemoryLayout<Float>.stride,
                options: .storageModeShared
            ) else {
                throw LagunaResidentModelError.bufferAllocation(what)
            }
            return buffer
        }
        func halfPlane(_ width: Int, _ what: String) throws -> MTLBuffer {
            guard let buffer = runtime.device.makeBuffer(
                length: capacity * width * MemoryLayout<UInt16>.stride,
                options: .storageModeShared
            ) else {
                throw LagunaResidentModelError.bufferAllocation(what)
            }
            return buffer
        }
        return PrefillPlanes(
            capacity: capacity,
            embdBytes: embd * 4,
            kvBytes: kvWidth * 4,
            midBytes: Int(shape.nFFDense) * 4,
            routerLogitsBytes: experts * 4,
            routerPickBytes: used * 4,
            hidden: try plane(embd, "prefill hidden plane"),
            hiddenNext: try plane(embd, "prefill next-hidden plane"),
            normed: try plane(embd, "prefill norm plane"),
            queryRows: try plane(maxQueryWidth, "prefill query plane"),
            keyRows: try plane(kvWidth, "prefill key plane"),
            valueRows: try plane(kvWidth, "prefill value plane"),
            gateRows: try plane(Int(shape.nHead), "prefill gate plane"),
            heads: try plane(maxQueryWidth, "prefill heads plane"),
            attnOut: try plane(embd, "prefill attention-out plane"),
            afterAttn: try plane(embd, "prefill post-attention plane"),
            ffnNormed: try plane(embd, "prefill FFN-norm plane"),
            ffnMid: try plane(Int(shape.nFFDense), "prefill FFN-mid plane"),
            ffnUp: try plane(Int(shape.nFFDense), "prefill FFN-up plane"),
            ffnOut: try plane(embd, "prefill FFN-out plane"),
            routerLogits: try plane(experts, "prefill router logits plane"),
            routerSelected: try plane(used, "prefill router selection plane"),
            routerWeights: try plane(used, "prefill router weights plane"),
            routerProbs: try plane(experts, "prefill router probs plane"),
            stagedKeys: try halfPlane(kvWidth, "prefill staged keys"),
            stagedValues: try halfPlane(kvWidth, "prefill staged values")
        )
    }

    private func reusablePrefillPlanes(
        minimumCapacity: Int
    ) throws -> PrefillPlanes {
        if let cachedPrefillPlanes,
           cachedPrefillPlanes.capacity >= minimumCapacity {
            return cachedPrefillPlanes
        }
        let planes = try makePrefillPlanes(capacity: minimumCapacity)
        cachedPrefillPlanes = planes
        return planes
    }

    /// One layer-major sweep of `chunk` through every layer. Returns the
    /// plane holding the chunk's final hidden states (after the swaps).
    private func sweepChunk(_ chunk: [Int32], planes: PrefillPlanes,
                            onLayer: (Int) -> Void) throws -> MTLBuffer {
        var planes = planes
        let n = chunk.count
        let embd = Int(shape.nEmbd)
        let used = activeExperts

        let embedStart = Date()
        for (i, token) in chunk.enumerated() {
            let row = try embeddingRow(token)
            row.withUnsafeBytes {
                _ = memcpy(planes.hidden.contents() + i * planes.embdBytes,
                           $0.baseAddress!, $0.count)
            }
        }
        profile.embedS += Date().timeIntervalSince(embedStart)

        let p0 = position
        for layer in layers {
            try Task.checkCancellation()
            let spec = layer.spec
            let queryWidth = spec.queryWidth
            let kvWidth = spec.keyValueWidth
            // The batch QK/RoPE and attention kernels use tightly packed
            // token rows. Full-attention layers have fewer query/gate heads
            // than SWA layers, so the maximum-width plane stride is invalid
            // for them even though the backing buffers are large enough.
            let queryRowBytes = queryWidth * MemoryLayout<Float>.stride
            let gateRowBytes = spec.headCount * MemoryLayout<Float>.stride
            let denseMM = Self.batchedPrefillAttention
                && Self.batchedPrefillDenseMM
                && n >= Self.prefillDenseMMMinimumTokens
            if !Self.batchedPrefillAttention,
               let index = layer.longAttentionIndex,
               usesIndexedLongAttention(
                   position: p0 + n - 1, index: index) {
                try ensureLongAttentionScratch(
                    blockCapacity: index.blockCapacity)
            }
            let phaseStart = Date()
            let commands = try beginCommands()
            if denseMM {
                try encodeRMSNormRows(
                    commands, input: planes.hidden,
                    weight: layer.attnNorm, output: planes.normed,
                    width: embd, count: n)
            } else {
                for i in 0..<n {
                    try encodeRMSNorm(commands, input: planes.hidden,
                                      inputOffset: i * planes.embdBytes,
                                      weight: layer.attnNorm,
                                      output: planes.normed,
                                      outputOffset: i * planes.embdBytes,
                                      width: embd)
                }
            }
            if denseMM {
                try encodeQ8MatmulRows(
                    commands, weights: layer.query,
                    input: planes.normed, output: planes.queryRows,
                    inputWidth: embd, outputWidth: queryWidth, count: n)
                try encodeQ8MatmulRows(
                    commands, weights: layer.key,
                    input: planes.normed, output: planes.keyRows,
                    inputWidth: embd, outputWidth: kvWidth, count: n)
                try encodeQ8MatmulRows(
                    commands, weights: layer.value,
                    input: planes.normed, output: planes.valueRows,
                    inputWidth: embd, outputWidth: kvWidth, count: n)
                try encodeQ8MatmulRows(
                    commands, weights: layer.gate,
                    input: planes.normed, output: planes.gateRows,
                    inputWidth: embd, outputWidth: spec.headCount, count: n)
            } else {
                for i in 0..<n {
                    try encodeMatvecPair(
                        commands, x: planes.normed,
                        xOffset: i * planes.embdBytes,
                        typeA: Self.q8Type, rowsA: queryWidth,
                        weightsA: layer.query,
                        typeB: Self.q8Type, rowsB: kvWidth,
                        weightsB: layer.key,
                        inputWidth: embd,
                        outA: planes.queryRows,
                        outAOffset: i * queryRowBytes,
                        outB: planes.keyRows,
                        outBOffset: i * planes.kvBytes)
                    try encodeMatvecPair(
                        commands, x: planes.normed,
                        xOffset: i * planes.embdBytes,
                        typeA: Self.q8Type, rowsA: kvWidth,
                        weightsA: layer.value,
                        typeB: Self.q8Type,
                        rowsB: spec.headCount,
                        weightsB: layer.gate,
                        inputWidth: embd,
                        outA: planes.valueRows,
                        outAOffset: i * planes.kvBytes,
                        outB: planes.gateRows,
                        outBOffset: i * gateRowBytes)
                }
            }
            if Self.batchedPrefillAttention {
                try encodeQKNormRopeRows(
                    commands, spec: spec, position: p0, count: n,
                    queryNorm: layer.queryNorm, keyNorm: layer.keyNorm,
                    query: planes.queryRows, key: planes.keyRows)
                try encodePrefillAttention(
                    commands, spec: spec, cache: layer.cache,
                    longIndex: layer.longAttentionIndex,
                    position: p0, count: n,
                    query: planes.queryRows, gate: planes.gateRows,
                    key: planes.keyRows, value: planes.valueRows,
                    stagedKey: planes.stagedKeys,
                    stagedValue: planes.stagedValues,
                    out: planes.heads)
            } else {
                for i in 0..<n {
                    let pos = p0 + i
                    try encodeQKNormRope(
                        commands, spec: spec, position: pos,
                        queryNorm: layer.queryNorm,
                        keyNorm: layer.keyNorm,
                        query: planes.queryRows,
                        queryOffset: i * queryRowBytes,
                        key: planes.keyRows,
                        keyOffset: i * planes.kvBytes)
                    try encodeStoreKV(
                        commands, cache: layer.cache, position: pos,
                        key: planes.keyRows,
                        keyOffset: i * planes.kvBytes,
                        value: planes.valueRows,
                        valueOffset: i * planes.kvBytes)
                    if let longIndex = layer.longAttentionIndex {
                        try encodeLongAttentionIndexUpdate(
                            commands, cache: layer.cache,
                            index: longIndex, position: pos, count: 1)
                    }
                    try encodeAttention(
                        commands, spec: spec, cache: layer.cache,
                        longIndex: layer.longAttentionIndex,
                        position: pos,
                        query: planes.queryRows,
                        queryOffset: i * queryRowBytes,
                        gate: planes.gateRows,
                        gateOffset: i * gateRowBytes,
                        out: planes.heads,
                        outOffset: i * queryRowBytes)
                }
            }

            if denseMM {
                try encodeQ8MatmulRows(
                    commands, weights: layer.attnOutput,
                    input: planes.heads, output: planes.attnOut,
                    inputWidth: queryWidth, outputWidth: embd, count: n)
                try encodeAdd(
                    commands, a: planes.hidden, b: planes.attnOut,
                    out: planes.afterAttn, count: n * embd)
            } else {
                for i in 0..<n {
                    try encodeQuantMatvec(
                        commands, type: Self.q8Type,
                        rows: embd, inputWidth: queryWidth,
                        weights: layer.attnOutput,
                        input: planes.heads,
                        inputOffset: i * queryRowBytes,
                        output: planes.attnOut,
                        outputOffset: i * planes.embdBytes,
                        accumulate: false)
                    try encodeAdd(
                        commands, a: planes.hidden,
                        aOffset: i * planes.embdBytes,
                        b: planes.attnOut,
                        bOffset: i * planes.embdBytes,
                        out: planes.afterAttn,
                        outOffset: i * planes.embdBytes,
                        count: embd)
                }
            }
            if denseMM {
                try encodeRMSNormRows(
                    commands, input: planes.afterAttn,
                    weight: layer.ffnNorm, output: planes.ffnNormed,
                    width: embd, count: n)
            } else {
                for i in 0..<n {
                    try encodeRMSNorm(commands, input: planes.afterAttn,
                                      inputOffset: i * planes.embdBytes,
                                      weight: layer.ffnNorm,
                                      output: planes.ffnNormed,
                                      outputOffset: i * planes.embdBytes,
                                      width: embd)
                }
            }
            if case .moe(_, let routerRows, let routerBias,
                         _, _, _, _, _, _) = layer.ffn {
                if denseMM {
                    try encodeF32MatvecRows(
                        commands, rows: routerRows,
                        rowCount: LagunaRouterReference.expertCount,
                        inputWidth: embd,
                        input: planes.ffnNormed,
                        output: planes.routerLogits,
                        count: n)
                    try encodeRouterSelect(
                        commands, bias: routerBias,
                        logits: planes.routerLogits, logitsOffset: 0,
                        selected: planes.routerSelected, selectedOffset: 0,
                        weights: planes.routerWeights, weightsOffset: 0,
                        probs: planes.routerProbs, probsOffset: 0,
                        count: n)
                } else {
                    for i in 0..<n {
                        try encodeF32Matvec(
                            commands, rows: routerRows,
                            rowCount: LagunaRouterReference.expertCount,
                            inputWidth: embd,
                            x: planes.ffnNormed,
                            xOffset: i * planes.embdBytes,
                            out: planes.routerLogits,
                            outOffset: i * planes.routerLogitsBytes)
                        try encodeRouterSelect(
                            commands, bias: routerBias,
                            logits: planes.routerLogits,
                            logitsOffset: i * planes.routerLogitsBytes,
                            selected: planes.routerSelected,
                            selectedOffset: i * planes.routerPickBytes,
                            weights: planes.routerWeights,
                            weightsOffset: i * planes.routerPickBytes,
                            probs: planes.routerProbs,
                            probsOffset: i * planes.routerLogitsBytes)
                    }
                }
            }

            switch layer.ffn {
            case .dense(let gate, let up, let down):
                let intermediate = Int(shape.nFFDense)
                if denseMM {
                    try encodeQ8MatmulRows(
                        commands, weights: gate,
                        input: planes.ffnNormed, output: planes.ffnMid,
                        inputWidth: embd, outputWidth: intermediate,
                        count: n)
                    try encodeQ8MatmulRows(
                        commands, weights: up,
                        input: planes.ffnNormed, output: planes.ffnUp,
                        inputWidth: embd, outputWidth: intermediate,
                        count: n)
                    try encodeSwiGLURows(
                        commands, gate: planes.ffnMid, up: planes.ffnUp,
                        output: planes.ffnMid, width: intermediate,
                        count: n)
                    try encodeQ8MatmulRows(
                        commands, weights: down,
                        input: planes.ffnMid, output: planes.ffnOut,
                        inputWidth: intermediate, outputWidth: embd,
                        count: n)
                    try encodeAdd(
                        commands, a: planes.afterAttn, b: planes.ffnOut,
                        out: planes.hiddenNext, count: n * embd)
                } else {
                    for i in 0..<n {
                        try encodePairSwiGLU(commands, type: Self.q8Type,
                                             rows: intermediate,
                                             inputWidth: embd,
                                             gateRows: gate, upRows: up,
                                             input: planes.ffnNormed,
                                             inputOffset: i * planes.embdBytes,
                                             routeWeight: 1,
                                             mid: planes.ffnMid,
                                             midOffset: i * planes.midBytes)
                        try encodeQuantMatvec(commands, type: Self.q8Type,
                                              rows: embd,
                                              inputWidth: intermediate,
                                              weights: down,
                                              input: planes.ffnMid,
                                              inputOffset: i * planes.midBytes,
                                              output: planes.ffnOut,
                                              outputOffset: i * planes.embdBytes,
                                              accumulate: false)
                        try encodeAdd(commands, a: planes.afterAttn,
                                      aOffset: i * planes.embdBytes,
                                      b: planes.ffnOut,
                                      bOffset: i * planes.embdBytes,
                                      out: planes.hiddenNext,
                                      outOffset: i * planes.embdBytes,
                                      count: embd)
                    }
                }
                try endCommands(commands, phase: .dense)
                profile.layerOtherS += Date().timeIntervalSince(phaseStart)

            case .moe(let routedType, _, _, let routed,
                      let gateUpExpertBytes, let downExpertBytes,
                      let sharedGate, let sharedUp, let sharedDown):
                try endCommands(commands, phase: .attention)
                let selPointer = planes.routerSelected.contents()
                    .bindMemory(to: Int32.self, capacity: n * used)
                let weightPointer = planes.routerWeights.contents()
                    .bindMemory(to: Float.self, capacity: n * used)
                profile.routeS += Date().timeIntervalSince(phaseStart)

                try encodeChunkExperts(
                    layer: layer, planes: planes, count: n,
                    routedType: routedType, routed: routed,
                    gateUpExpertBytes: gateUpExpertBytes,
                    downExpertBytes: downExpertBytes,
                    sharedGate: sharedGate, sharedUp: sharedUp,
                    sharedDown: sharedDown,
                    selected: selPointer, routeWeights: weightPointer)
            }
            profile.layers += n
            swap(&planes.hidden, &planes.hiddenNext)
            onLayer(layer.index)
        }
        position += n
        profile.forwards += n
        return planes.hidden
    }

    /// Phase B of one layer for a whole chunk: one LRU tick unions the
    /// selections over the slot cache (`layerMajorChunkSize` guarantees the
    /// union fits), each missing slab streams ONCE for the chunk, and every
    /// token's routed ranks are encoded IN RANK ORDER (same accumulate
    /// rounding as decode — the C-parity audit holds) behind a per-token
    /// event wait on the worst miss it depends on. Tokens are encoded in
    /// readiness order so early arrivals compute under the I/O of later
    /// slabs instead of stalling behind them.
    private func encodeChunkExperts(
        layer: Layer, planes: PrefillPlanes, count n: Int,
        routedType: UInt32, routed: RoutedWeights,
        gateUpExpertBytes: Int, downExpertBytes: Int,
        sharedGate: MTLBuffer, sharedUp: MTLBuffer, sharedDown: MTLBuffer,
        selected: UnsafePointer<Int32>,
        routeWeights: UnsafePointer<Float>
    ) throws {
        let used = activeExperts
        let embd = Int(shape.nEmbd)
        let expertWidth = Int(shape.nFFExpert)
        let readParts = preadSplit
        let denseMM = Self.batchedPrefillAttention
            && Self.batchedPrefillDenseMM
            && n >= Self.prefillDenseMMMinimumTokens
        let gatherStart = Date()

        var rankSlot = [Int](repeating: -1, count: n * used)
        var rankReady = [Int](repeating: 0, count: n * used)
        var requiredMiss = [Int](repeating: 0, count: n)
        var missReads: [[ExpertRead]] = []
        var isStreamed = false
        if case .streamed(let gateBase, let upBase,
                          let downBase) = routed {
            isStreamed = true
            let slabBytes = 2 * gateUpExpertBytes + downExpertBytes
            expertClock += 1
            var missIndexOfKey: [Int: Int] = [:]
            for i in 0..<n {
                var worst = 0
                for r in 0..<used {
                    let e = Int(selected[i * used + r])
                    let key = layer.index << 16 | e
                    if let slot = expertSlotIndex[key] {
                        expertSlots[slot].lastUse = expertClock
                        rankSlot[i * used + r] = slot
                        if let miss = missIndexOfKey[key] {
                            // Queued by an earlier token this tick: same
                            // dependency, no extra I/O and no hit credit.
                            worst = max(worst, miss + 1)
                            rankReady[i * used + r] = miss + 1
                        } else {
                            profile.expertHits += 1
                            profile.expertHitBytes += slabBytes
                        }
                        continue
                    }
                    var victim = -1
                    var oldest = Int.max
                    for index in expertSlots.indices
                    where expertSlots[index].lastUse < expertClock
                        && expertSlots[index].lastUse < oldest {
                        victim = index
                        oldest = expertSlots[index].lastUse
                    }
                    let evicted = expertSlots[victim].key
                    if evicted >= 0 {
                        expertSlotIndex.removeValue(forKey: evicted)
                    }
                    expertSlots[victim].key = key
                    expertSlots[victim].lastUse = expertClock
                    expertSlotIndex[key] = victim
                    rankSlot[i * used + r] = victim
                    let slot = expertSlots[victim]
                    let missIndex = missReads.count
                    missIndexOfKey[key] = missIndex
                    rankReady[i * used + r] = missIndex + 1
                    missReads.append([
                        ExpertRead(
                            buffer: slot.gate, bufferOffset: 0,
                            fileOffset: gateBase + e * gateUpExpertBytes,
                            bytes: gateUpExpertBytes),
                        ExpertRead(
                            buffer: slot.up, bufferOffset: 0,
                            fileOffset: upBase + e * gateUpExpertBytes,
                            bytes: gateUpExpertBytes),
                        ExpertRead(
                            buffer: slot.down, bufferOffset: 0,
                            fileOffset: downBase + e * downExpertBytes,
                            bytes: downExpertBytes),
                    ])
                    worst = max(worst, missIndex + 1)
                    profile.expertMisses += 1
                    profile.expertMissBytes += slabBytes
                }
                requiredMiss[i] = worst
            }
            profile.gatherBytes += missReads.count * slabBytes
        }

        // Miss I/O: concurrent F_NOCACHE preads with the rank-ordered
        // frontier signaler when possible, synchronous mmap copies as the
        // fallback (no events — everything is in RAM before the encode).
        let failureLock = NSLock()
        nonisolated(unsafe) var failed = false
        nonisolated(unsafe) var ioDone = gatherStart
        let ioGroup = DispatchGroup()
        var eventBase: UInt64 = 0
        let allMissReads = missReads.flatMap { $0 }
        let loadedByMetalIO = isStreamed && !allMissReads.isEmpty
            && loadExpertsMetalIO(allMissReads)
        if loadedByMetalIO { ioDone = Date() }
        let overlapped = isStreamed && !loadedByMetalIO && streamFD >= 0
            && expertEvent != nil && !missReads.isEmpty
        if isStreamed, !missReads.isEmpty {
            if loadedByMetalIO {
                // Direct SSD → MTLBuffer command already completed.
            } else if overlapped, let event = expertEvent {
                let fd = streamFD
                var groups: [DispatchGroup] = []
                groups.reserveCapacity(missReads.count)
                for reads in missReads {
                    let group = DispatchGroup()
                    groups.append(group)
                    for read in reads {
                        group.enter()
                        nonisolated(unsafe) let job = read
                        DispatchQueue.global(qos: .userInitiated).async {
                            if !GGUFWeights.preadFullSplit(
                                fd, into: job.destination, bytes: job.bytes,
                                offset: job.fileOffset, parts: readParts
                            ) {
                                failureLock.lock()
                                failed = true
                                failureLock.unlock()
                            }
                            group.leave()
                        }
                    }
                }
                eventBase = expertEventBase
                expertEventBase += UInt64(missReads.count)
                let base = eventBase
                ioGroup.enter()
                nonisolated(unsafe) let sharedEvent = event
                nonisolated(unsafe) let missGroups = groups
                DispatchQueue.global(qos: .userInitiated).async {
                    for (index, group) in missGroups.enumerated() {
                        group.wait()
                        sharedEvent.signaledValue = base + UInt64(index) + 1
                    }
                    ioDone = Date()
                    ioGroup.leave()
                }
            } else if streamFD >= 0 {
                let fd = streamFD
                nonisolated(unsafe) let jobs = allMissReads
                DispatchQueue.concurrentPerform(iterations: jobs.count) { i in
                    let job = jobs[i]
                    if !GGUFWeights.preadFullSplit(
                        fd, into: job.destination, bytes: job.bytes,
                        offset: job.fileOffset, parts: readParts
                    ) {
                        failureLock.lock()
                        failed = true
                        failureLock.unlock()
                    }
                }
                ioDone = Date()
            } else {
                if willNeedExperts {
                    GGUFModel.prefetch(
                        base: model.mapBase,
                        ranges: allMissReads.map {
                            (UInt64($0.fileOffset), UInt64($0.bytes))
                        })
                }
                for reads in missReads {
                    for job in reads {
                        memcpy(job.destination,
                               model.mapBase + job.fileOffset,
                               job.bytes)
                    }
                }
                ioDone = Date()
            }
            if failed, !overlapped {
                throw LagunaResidentModelError.expertRead(
                    "pread of a routed expert slab failed on layer "
                    + "\(layer.index)")
            }
        }

        let order = overlapped
            ? (0..<n).sorted {
                requiredMiss[$0] != requiredMiss[$1]
                    ? requiredMiss[$0] < requiredMiss[$1] : $0 < $1
            }
            : Array(0..<n)
        let sharedOverlapped = overlapped
            && Self.sharedExpertIOOverlap
        guard let buffer = runtime.queue.makeCommandBuffer() else {
            throw LagunaResidentModelError.bufferAllocation("command buffer")
        }
        if sharedOverlapped {
            guard let encoder = buffer.makeComputeCommandEncoder() else {
                throw LagunaResidentModelError.bufferAllocation(
                    "prefill shared-expert overlap encoder")
            }
            let commands = Commands(buffer: buffer, encoder: encoder)
            if denseMM {
                try encodeChunkSharedOnlyBatched(
                    commands, planes: planes, count: n,
                    sharedGate: sharedGate, sharedUp: sharedUp,
                    sharedDown: sharedDown)
            } else {
                for token in 0..<n {
                    try encodeChunkSharedOnly(
                        commands, planes: planes, token: token,
                        sharedGate: sharedGate, sharedUp: sharedUp,
                        sharedDown: sharedDown)
                }
            }
            encoder.endEncoding()
        }
        if Self.batchedPrefillMoE {
            try encodeChunkExpertsBatched(
                into: buffer,
                planes: planes,
                count: n,
                routedType: routedType,
                routed: routed,
                gateUpExpertBytes: gateUpExpertBytes,
                downExpertBytes: downExpertBytes,
                sharedGate: sharedGate,
                sharedUp: sharedUp,
                sharedDown: sharedDown,
                selected: selected,
                routeWeights: routeWeights,
                rankSlot: rankSlot,
                rankReady: rankReady,
                overlapped: overlapped,
                eventBase: eventBase)
        } else {
          for i in order {
            if overlapped, requiredMiss[i] > 0, let event = expertEvent {
                buffer.encodeWaitForEvent(
                    event, value: eventBase + UInt64(requiredMiss[i]))
            }
            guard let encoder = buffer.makeComputeCommandEncoder() else {
                throw LagunaResidentModelError.bufferAllocation(
                    "command encoder")
            }
            let commands = Commands(buffer: buffer, encoder: encoder)
            for r in 0..<used {
                let gate: MTLBuffer, up: MTLBuffer, down: MTLBuffer
                let gateUpOffset: Int, downOffset: Int
                switch routed {
                case .resident(let residentGate, let residentUp,
                               let residentDown):
                    let e = Int(selected[i * used + r])
                    gate = residentGate; up = residentUp
                    down = residentDown
                    gateUpOffset = e * gateUpExpertBytes
                    downOffset = e * downExpertBytes
                case .streamed:
                    let slot = expertSlots[rankSlot[i * used + r]]
                    gate = slot.gate; up = slot.up; down = slot.down
                    gateUpOffset = 0
                    downOffset = 0
                }
                try encodePairSwiGLU(
                    commands, type: routedType, rows: expertWidth,
                    inputWidth: embd,
                    gateRows: gate, gateOffset: gateUpOffset,
                    upRows: up, upOffset: gateUpOffset,
                    input: planes.ffnNormed,
                    inputOffset: i * planes.embdBytes,
                    routeWeight: routeWeights[i * used + r],
                    mid: planes.ffnMid, midOffset: i * planes.midBytes
                )
                try encodeQuantMatvec(
                    commands, type: routedType, rows: embd,
                    inputWidth: expertWidth,
                    weights: down, weightsOffset: downOffset,
                    input: planes.ffnMid,
                    inputOffset: i * planes.midBytes,
                    output: planes.ffnOut,
                    outputOffset: i * planes.embdBytes,
                    accumulate: r != 0
                )
            }
            if !denseMM, sharedOverlapped {
                try encodeChunkOverlappedTail(
                    commands, planes: planes, token: i)
            } else if !denseMM {
                try encodeChunkSharedTail(
                    commands, planes: planes, token: i,
                    sharedGate: sharedGate, sharedUp: sharedUp,
                    sharedDown: sharedDown)
            }
            encoder.endEncoding()
          }
          if denseMM {
            guard let encoder = buffer.makeComputeCommandEncoder() else {
                throw LagunaResidentModelError.bufferAllocation(
                    "shared-expert batch encoder")
            }
            let commands = Commands(buffer: buffer, encoder: encoder)
            if sharedOverlapped {
                try encodeChunkOverlappedTailBatched(
                    commands, planes: planes, count: n)
            } else {
                try encodeChunkSharedTailBatched(
                    commands, planes: planes, count: n,
                    sharedGate: sharedGate, sharedUp: sharedUp,
                    sharedDown: sharedDown)
            }
            encoder.endEncoding()
          }
        }
        buffer.commit()
        buffer.waitUntilCompleted()
        if overlapped { ioGroup.wait() }
        if failed {
            throw LagunaResidentModelError.expertRead(
                "pread of a routed expert slab failed on layer "
                + "\(layer.index)")
        }
        if let error = buffer.error { throw error }
        let gpu = buffer.gpuEndTime - buffer.gpuStartTime
        if gpu.isFinite, gpu > 0 { gpuExpertsS += gpu }
        // Same wall split as the decode overlap: up to the last read is
        // gather (compute already running under it), the tail is experts.
        let gpuDone = Date()
        let ioEnd = min(max(ioDone, gatherStart), gpuDone)
        profile.gatherS += ioEnd.timeIntervalSince(gatherStart)
        profile.expertsS += gpuDone.timeIntervalSince(ioEnd)
    }

    private func encodeChunkSharedTail(
        _ commands: Commands,
        planes: PrefillPlanes,
        token: Int,
        sharedGate: MTLBuffer,
        sharedUp: MTLBuffer,
        sharedDown: MTLBuffer
    ) throws {
        let embd = Int(shape.nEmbd)
        let sharedWidth = Int(shape.nFFShared)
        let embdOffset = token * planes.embdBytes
        try encodeAdd(
            commands,
            a: planes.afterAttn, aOffset: embdOffset,
            b: planes.ffnOut, bOffset: embdOffset,
            out: planes.hiddenNext, outOffset: embdOffset,
            count: embd)
        try encodePairSwiGLU(
            commands, type: Self.q8Type,
            rows: sharedWidth, inputWidth: embd,
            gateRows: sharedGate, upRows: sharedUp,
            input: planes.ffnNormed, inputOffset: embdOffset,
            routeWeight: 1,
            mid: planes.ffnMid, midOffset: token * planes.midBytes)
        try encodeQuantMatvec(
            commands, type: Self.q8Type,
            rows: embd, inputWidth: sharedWidth,
            weights: sharedDown,
            input: planes.ffnMid,
            inputOffset: token * planes.midBytes,
            output: planes.ffnOut, outputOffset: embdOffset,
            accumulate: false)
        try encodeAdd(
            commands,
            a: planes.hiddenNext, aOffset: embdOffset,
            b: planes.ffnOut, bOffset: embdOffset,
            out: planes.hiddenNext, outOffset: embdOffset,
            count: embd)
    }

    private func encodeChunkSharedOnly(
        _ commands: Commands,
        planes: PrefillPlanes,
        token: Int,
        sharedGate: MTLBuffer,
        sharedUp: MTLBuffer,
        sharedDown: MTLBuffer
    ) throws {
        let embd = Int(shape.nEmbd)
        let sharedWidth = Int(shape.nFFShared)
        let embdOffset = token * planes.embdBytes
        try encodePairSwiGLU(
            commands, type: Self.q8Type,
            rows: sharedWidth, inputWidth: embd,
            gateRows: sharedGate, upRows: sharedUp,
            input: planes.ffnNormed, inputOffset: embdOffset,
            routeWeight: 1,
            mid: planes.ffnMid, midOffset: token * planes.midBytes)
        try encodeQuantMatvec(
            commands, type: Self.q8Type,
            rows: embd, inputWidth: sharedWidth,
            weights: sharedDown,
            input: planes.ffnMid,
            inputOffset: token * planes.midBytes,
            output: planes.hiddenNext, outputOffset: embdOffset,
            accumulate: false)
    }

    private func encodeChunkOverlappedTail(
        _ commands: Commands,
        planes: PrefillPlanes,
        token: Int
    ) throws {
        let embd = Int(shape.nEmbd)
        let embdOffset = token * planes.embdBytes
        let temporaryOffset = token * planes.midBytes
        try encodeAdd(
            commands,
            a: planes.afterAttn, aOffset: embdOffset,
            b: planes.ffnOut, bOffset: embdOffset,
            out: planes.ffnMid, outOffset: temporaryOffset,
            count: embd)
        try encodeAdd(
            commands,
            a: planes.ffnMid, aOffset: temporaryOffset,
            b: planes.hiddenNext, bOffset: embdOffset,
            out: planes.hiddenNext, outOffset: embdOffset,
            count: embd)
    }

    private func encodeChunkSharedTailBatched(
        _ commands: Commands,
        planes: PrefillPlanes,
        count: Int,
        sharedGate: MTLBuffer,
        sharedUp: MTLBuffer,
        sharedDown: MTLBuffer
    ) throws {
        let embd = Int(shape.nEmbd)
        let sharedWidth = Int(shape.nFFShared)
        try encodeAdd(
            commands, a: planes.afterAttn, b: planes.ffnOut,
            out: planes.hiddenNext, count: count * embd)
        try encodeQ8MatmulRows(
            commands, weights: sharedGate,
            input: planes.ffnNormed, output: planes.ffnMid,
            inputWidth: embd, outputWidth: sharedWidth, count: count)
        try encodeQ8MatmulRows(
            commands, weights: sharedUp,
            input: planes.ffnNormed, output: planes.ffnUp,
            inputWidth: embd, outputWidth: sharedWidth, count: count)
        try encodeSwiGLURows(
            commands, gate: planes.ffnMid, up: planes.ffnUp,
            output: planes.ffnMid, width: sharedWidth, count: count)
        try encodeQ8MatmulRows(
            commands, weights: sharedDown,
            input: planes.ffnMid, output: planes.ffnOut,
            inputWidth: sharedWidth, outputWidth: embd, count: count)
        try encodeAdd(
            commands, a: planes.hiddenNext, b: planes.ffnOut,
            out: planes.hiddenNext, count: count * embd)
    }

    private func encodeChunkSharedOnlyBatched(
        _ commands: Commands,
        planes: PrefillPlanes,
        count: Int,
        sharedGate: MTLBuffer,
        sharedUp: MTLBuffer,
        sharedDown: MTLBuffer
    ) throws {
        let embd = Int(shape.nEmbd)
        let sharedWidth = Int(shape.nFFShared)
        try encodeQ8MatmulRows(
            commands, weights: sharedGate,
            input: planes.ffnNormed, output: planes.ffnMid,
            inputWidth: embd, outputWidth: sharedWidth, count: count)
        try encodeQ8MatmulRows(
            commands, weights: sharedUp,
            input: planes.ffnNormed, output: planes.ffnUp,
            inputWidth: embd, outputWidth: sharedWidth, count: count)
        try encodeSwiGLURows(
            commands, gate: planes.ffnMid, up: planes.ffnUp,
            output: planes.ffnMid, width: sharedWidth, count: count)
        try encodeQ8MatmulRows(
            commands, weights: sharedDown,
            input: planes.ffnMid, output: planes.hiddenNext,
            inputWidth: sharedWidth, outputWidth: embd, count: count)
    }

    private func encodeChunkOverlappedTailBatched(
        _ commands: Commands,
        planes: PrefillPlanes,
        count: Int
    ) throws {
        // The first embedding-width prefix of every FFN-mid row is dead
        // after routed compute and serves as the exact residual temporary.
        for token in 0..<count {
            try encodeChunkOverlappedTail(
                commands, planes: planes, token: token)
        }
    }

    /// Expert-major phase B: applications are grouped by the selected expert,
    /// then the token loop runs inside two tiled kernels. Contributions are
    /// reduced per token in router-rank order, preserving the legacy
    /// accumulation association.
    private func encodeChunkExpertsBatched(
        into buffer: MTLCommandBuffer,
        planes: PrefillPlanes,
        count n: Int,
        routedType: UInt32,
        routed: RoutedWeights,
        gateUpExpertBytes: Int,
        downExpertBytes: Int,
        sharedGate: MTLBuffer,
        sharedUp: MTLBuffer,
        sharedDown: MTLBuffer,
        selected: UnsafePointer<Int32>,
        routeWeights: UnsafePointer<Float>,
        rankSlot: [Int],
        rankReady: [Int],
        overlapped: Bool,
        eventBase: UInt64
    ) throws {
        struct Application {
            let token: Int
            let rank: Int
            let expert: Int
            let slot: Int
            let weight: Float
            let ready: Int
        }
        struct Entry {
            let expert: Int
            let slot: Int
            let start: Int
            let count: Int
            let ready: Int
        }

        let used = activeExperts
        let embd = Int(shape.nEmbd)
        let expertWidth = Int(shape.nFFExpert)
        var grouped: [Int: [Application]] = [:]
        grouped.reserveCapacity(min(n * used,
                                    LagunaRouterReference.expertCount))
        for token in 0..<n {
            for rank in 0..<used {
                let index = token * used + rank
                let expert = Int(selected[index])
                let slot = rankSlot[index]
                let groupKey: Int
                switch routed {
                case .resident:
                    groupKey = expert
                case .streamed:
                    groupKey = slot
                }
                grouped[groupKey, default: []].append(Application(
                    token: token,
                    rank: rank,
                    expert: expert,
                    slot: slot,
                    weight: routeWeights[index],
                    ready: rankReady[index]))
            }
        }

        let orderedGroups = grouped.values.sorted {
            let aReady = $0.map(\.ready).max() ?? 0
            let bReady = $1.map(\.ready).max() ?? 0
            if aReady != bReady { return aReady < bReady }
            if $0[0].expert != $1[0].expert {
                return $0[0].expert < $1[0].expert
            }
            return $0[0].slot < $1[0].slot
        }
        var appTokens: [UInt32] = []
        var appWeights: [Float] = []
        var appIndexByTokenRank = [Int](repeating: -1, count: n * used)
        var entries: [Entry] = []
        appTokens.reserveCapacity(n * used)
        appWeights.reserveCapacity(n * used)
        entries.reserveCapacity(orderedGroups.count)
        for unsorted in orderedGroups {
            let applications = unsorted.sorted {
                $0.token != $1.token
                    ? $0.token < $1.token : $0.rank < $1.rank
            }
            let start = appTokens.count
            for application in applications {
                let appIndex = appTokens.count
                appTokens.append(UInt32(application.token))
                appWeights.append(application.weight)
                appIndexByTokenRank[
                    application.token * used + application.rank] = appIndex
            }
            entries.append(Entry(
                expert: applications[0].expert,
                slot: applications[0].slot,
                start: start,
                count: applications.count,
                ready: applications.map(\.ready).max() ?? 0))
        }

        var tokenMeta: [UInt32] = []
        var tokenApps: [UInt32] = []
        tokenMeta.reserveCapacity(n * 3)
        tokenApps.reserveCapacity(n * used)
        for token in 0..<n {
            tokenMeta.append(UInt32(token))
            tokenMeta.append(UInt32(tokenApps.count))
            tokenMeta.append(UInt32(used))
            for rank in 0..<used {
                let app = appIndexByTokenRank[token * used + rank]
                precondition(app >= 0)
                tokenApps.append(UInt32(app))
            }
        }

        func upload<T>(_ values: [T], name: String) throws -> MTLBuffer {
            let bytes = values.count * MemoryLayout<T>.stride
            let buffer = try prefillMoEScratch.buffer(
                device: runtime.device, name: name,
                minimumBytes: max(bytes, 1))
            values.withUnsafeBytes {
                if let base = $0.baseAddress, !$0.isEmpty {
                    _ = memcpy(buffer.contents(), base, $0.count)
                }
            }
            return buffer
        }
        let appTokensBuffer = try upload(appTokens, name: "app tokens")
        let appWeightsBuffer = try upload(appWeights, name: "app weights")
        let tokenMetaBuffer = try upload(tokenMeta, name: "token metadata")
        let tokenAppsBuffer = try upload(tokenApps, name: "token applications")
        let mids = try prefillMoEScratch.buffer(
            device: runtime.device, name: "mids",
            minimumBytes: n * used * expertWidth
                * MemoryLayout<Float>.stride)
        let contribs = try prefillMoEScratch.buffer(
            device: runtime.device, name: "contributions",
            minimumBytes: n * used * embd
                * MemoryLayout<Float>.stride)
        memset(planes.ffnOut.contents(), 0,
               n * embd * MemoryLayout<Float>.stride)

        var encoder: MTLComputeCommandEncoder?
        func beginEncoder() throws -> MTLComputeCommandEncoder {
            if let encoder { return encoder }
            guard let created = buffer.makeComputeCommandEncoder() else {
                throw LagunaResidentModelError.bufferAllocation(
                    "prefill MoE command encoder")
            }
            encoder = created
            return created
        }
        func closeEncoder() {
            encoder?.endEncoding()
            encoder = nil
        }

        for entry in entries {
            if overlapped, entry.ready > 0, let event = expertEvent {
                closeEncoder()
                buffer.encodeWaitForEvent(
                    event, value: eventBase + UInt64(entry.ready))
            }
            let encoder = try beginEncoder()
            let commands = Commands(buffer: buffer, encoder: encoder)
            let gate: MTLBuffer
            let up: MTLBuffer
            let down: MTLBuffer
            let gateUpOffset: Int
            let downOffset: Int
            switch routed {
            case .resident(let residentGate, let residentUp,
                           let residentDown):
                gate = residentGate
                up = residentUp
                down = residentDown
                gateUpOffset = entry.expert * gateUpExpertBytes
                downOffset = entry.expert * downExpertBytes
            case .streamed:
                let slot = expertSlots[entry.slot]
                gate = slot.gate
                up = slot.up
                down = slot.down
                gateUpOffset = 0
                downOffset = 0
            }
            try encodePrefillExpertApplications(
                commands,
                routedType: routedType,
                start: entry.start,
                count: entry.count,
                embd: embd,
                expertWidth: expertWidth,
                appTokens: appTokensBuffer,
                appWeights: appWeightsBuffer,
                inputs: planes.ffnNormed,
                gate: gate,
                gateOffset: gateUpOffset,
                up: up,
                upOffset: gateUpOffset,
                down: down,
                downOffset: downOffset,
                mids: mids,
                contribs: contribs)
        }

        let tailEncoder = try beginEncoder()
        let commands = Commands(buffer: buffer, encoder: tailEncoder)
        let reduce = try runtime.pipeline("kernel_glm52_moe_prefill_reduce")
        tailEncoder.setComputePipelineState(reduce)
        setArguments(tailEncoder, [
            routedType, UInt32(expertWidth), UInt32(embd), 0,
            0, 0, routedType, UInt32(n),
        ])
        tailEncoder.setBuffer(tokenMetaBuffer, offset: 0, index: 1)
        tailEncoder.setBuffer(tokenAppsBuffer, offset: 0, index: 2)
        tailEncoder.setBuffer(contribs, offset: 0, index: 3)
        tailEncoder.setBuffer(planes.ffnOut, offset: 0, index: 4)
        tailEncoder.dispatchThreadgroups(
            MTLSize(width: (embd + 255) / 256, height: n, depth: 1),
            threadsPerThreadgroup: MTLSize(width: 256, height: 1, depth: 1))

        let sharedOverlapped = overlapped
            && Self.sharedExpertIOOverlap
        if sharedOverlapped,
           Self.batchedPrefillAttention
            && Self.batchedPrefillDenseMM
            && n >= Self.prefillDenseMMMinimumTokens {
            try encodeChunkOverlappedTailBatched(
                commands, planes: planes, count: n)
        } else if sharedOverlapped {
            for token in 0..<n {
                try encodeChunkOverlappedTail(
                    commands, planes: planes, token: token)
            }
        } else if Self.batchedPrefillAttention
            && Self.batchedPrefillDenseMM
            && n >= Self.prefillDenseMMMinimumTokens {
            try encodeChunkSharedTailBatched(
                commands, planes: planes, count: n,
                sharedGate: sharedGate, sharedUp: sharedUp,
                sharedDown: sharedDown)
        } else {
            for token in 0..<n {
                try encodeChunkSharedTail(
                    commands, planes: planes, token: token,
                    sharedGate: sharedGate, sharedUp: sharedUp,
                    sharedDown: sharedDown)
            }
        }
        closeEncoder()
    }

    private func encodePrefillExpertApplications(
        _ commands: Commands,
        routedType: UInt32,
        start: Int,
        count: Int,
        embd: Int,
        expertWidth: Int,
        appTokens: MTLBuffer,
        appWeights: MTLBuffer,
        inputs: MTLBuffer,
        gate: MTLBuffer,
        gateOffset: Int,
        up: MTLBuffer,
        upOffset: Int,
        down: MTLBuffer,
        downOffset: Int,
        mids: MTLBuffer,
        contribs: MTLBuffer
    ) throws {
        let arguments = [
            routedType, UInt32(expertWidth), UInt32(embd),
            UInt32(start), UInt32(count), 0, 0, 0,
        ]
        let encoder = commands.encoder
        let swiglu = try runtime.pipeline(
            "kernel_laguna_moe_prefill_swiglu_sg")
        encoder.setComputePipelineState(swiglu)
        setArguments(encoder, arguments)
        encoder.setBuffer(appTokens, offset: 0, index: 1)
        encoder.setBuffer(appWeights, offset: 0, index: 2)
        encoder.setBuffer(inputs, offset: 0, index: 3)
        encoder.setBuffer(gate, offset: gateOffset, index: 4)
        encoder.setBuffer(up, offset: upOffset, index: 5)
        encoder.setBuffer(mids, offset: 0, index: 6)
        encoder.dispatchThreadgroups(
            rowGroups(expertWidth),
            threadsPerThreadgroup: simdgroupThreads)

        let downPipeline = try runtime.pipeline(
            "kernel_laguna_moe_prefill_down_sg")
        encoder.setComputePipelineState(downPipeline)
        setArguments(encoder, arguments)
        encoder.setBuffer(mids, offset: 0, index: 1)
        encoder.setBuffer(down, offset: downOffset, index: 2)
        encoder.setBuffer(contribs, offset: 0, index: 3)
        encoder.dispatchThreadgroups(
            rowGroups(embd),
            threadsPerThreadgroup: simdgroupThreads)
    }

    private func headLogits(plane: MTLBuffer, offset: Int) throws -> [Float] {
        let headStart = Date()
        let commands = try beginCommands()
        try encodeRMSNorm(commands, input: plane, inputOffset: offset,
                          weight: outputNorm, output: normed,
                          width: Int(shape.nEmbd))
        try encodeQuantMatvec(commands, type: Self.q8Type,
                              rows: Int(shape.nVocab),
                              inputWidth: Int(shape.nEmbd),
                              weights: outputHead, input: normed,
                              output: logits, accumulate: false)
        try endCommands(commands, phase: .head)
        let count = Int(shape.nVocab)
        let pointer = logits.contents().bindMemory(to: Float.self,
                                                   capacity: count)
        let result = Array(UnsafeBufferPointer(start: pointer, count: count))
        profile.headS += Date().timeIntervalSince(headStart)
        return result
    }

    public func generateGreedy(prompt: [Int32], maxNewTokens: Int,
                               endTokens: Set<Int32> = []) throws -> [Int32] {
        guard !prompt.isEmpty else { return [] }
        let logits = try prefill(prompt)
        guard !logits.isEmpty else { return [] }
        var best = logits[0]
        var bestIndex: Int32 = 0
        for index in 1..<logits.count where logits[index] > best {
            best = logits[index]
            bestIndex = Int32(index)
        }
        var output: [Int32] = []
        for _ in 0..<maxNewTokens {
            if endTokens.contains(bestIndex) { break }
            output.append(bestIndex)
            bestIndex = try forwardNextGreedy(bestIndex)
        }
        return output
    }

    // MARK: Layer forward

    private func forward(layer: Layer, position: Int) throws {
        let embd = Int(shape.nEmbd)
        let spec = layer.spec
        let queryWidth = spec.queryWidth
        let kvWidth = spec.keyValueWidth
        if let index = layer.longAttentionIndex,
           usesIndexedLongAttention(position: position, index: index) {
            try ensureLongAttentionScratch(
                blockCapacity: index.blockCapacity)
        }

        let phaseStart = Date()
        var commands = try beginCommands()

        // Attention half: norm, projections, per-head norm/RoPE, KV store,
        // gated attention, output projection and residual.
        try encodeRMSNorm(commands, input: hidden, weight: layer.attnNorm,
                          output: normed, width: embd)
        try encodeMatvecPair(commands, x: normed,
                             typeA: Self.q8Type, rowsA: queryWidth, weightsA: layer.query,
                             typeB: Self.q8Type, rowsB: kvWidth, weightsB: layer.key,
                             inputWidth: embd, outA: queryRows, outB: keyRows)
        try encodeMatvecPair(commands, x: normed,
                             typeA: Self.q8Type, rowsA: kvWidth, weightsA: layer.value,
                             typeB: Self.q8Type, rowsB: spec.headCount, weightsB: layer.gate,
                             inputWidth: embd, outA: valueRows, outB: gateRows)
        try encodeQKNormRope(commands, spec: spec, position: position,
                             queryNorm: layer.queryNorm, keyNorm: layer.keyNorm,
                             query: queryRows, queryOffset: 0,
                             key: keyRows, keyOffset: 0)
        try encodeStoreKV(commands, cache: layer.cache, position: position,
                          key: keyRows, keyOffset: 0,
                          value: valueRows, valueOffset: 0)
        if let longIndex = layer.longAttentionIndex {
            try encodeLongAttentionIndexUpdate(
                commands, cache: layer.cache, index: longIndex,
                position: position, count: 1)
        }
        try encodeAttention(commands, spec: spec, cache: layer.cache,
                            longIndex: layer.longAttentionIndex,
                            position: position,
                            query: queryRows, queryOffset: 0,
                            gate: gateRows, gateOffset: 0,
                            out: heads, outOffset: 0)
        try encodeQuantMatvec(commands, type: Self.q8Type, rows: embd,
                              inputWidth: queryWidth, weights: layer.attnOutput,
                              input: heads, output: attnOut, accumulate: false)
        try encodeAdd(commands, a: hidden, b: attnOut, out: afterAttn,
                      count: embd)
        try encodeRMSNorm(commands, input: afterAttn, weight: layer.ffnNorm,
                          output: ffnNormed, width: embd)

        switch layer.ffn {
        case .dense(let gate, let up, let down):
            let intermediate = Int(shape.nFFDense)
            try encodePairSwiGLU(commands, type: Self.q8Type, rows: intermediate,
                                 inputWidth: embd, gateRows: gate, upRows: up,
                                 input: ffnNormed, routeWeight: 1, mid: ffnMid)
            try encodeQuantMatvec(commands, type: Self.q8Type, rows: embd,
                                  inputWidth: intermediate, weights: down,
                                  input: ffnMid, output: ffnOut,
                                  accumulate: false)
            try encodeAdd(commands, a: afterAttn, b: ffnOut, out: hiddenNext,
                          count: embd)
            try endCommands(commands, phase: .dense)
            profile.layerOtherS += Date().timeIntervalSince(phaseStart)

        case .moe(let routedType, let routerRows, let routerBias,
                  let routed, let gateUpExpertBytes, let downExpertBytes,
                  let sharedGate, let sharedUp, let sharedDown):
            // Phase A ends with the router: selection is read back on the
            // host to address the expert slabs (GLM chained-decode pattern).
            try encodeF32Matvec(commands, rows: routerRows,
                                rowCount: LagunaRouterReference.expertCount,
                                inputWidth: embd, x: ffnNormed,
                                out: routerLogits)
            try encodeRouterSelect(commands, bias: routerBias,
                                   logits: routerLogits, logitsOffset: 0,
                                   selected: routerSelected,
                                   selectedOffset: 0,
                                   weights: routerWeights, weightsOffset: 0,
                                   probs: routerProbs, probsOffset: 0)
            try endCommands(commands, phase: .attention)

            let used = activeExperts
            let selectedPointer = routerSelected.contents()
                .bindMemory(to: Int32.self, capacity: used)
            let weightPointer = routerWeights.contents()
                .bindMemory(to: Float.self, capacity: used)
            let selected = Array(UnsafeBufferPointer(start: selectedPointer,
                                                     count: used))
            let routeWeights = Array(UnsafeBufferPointer(start: weightPointer,
                                                         count: used))
            let readParts = preadSplit
            profile.routeS += Date().timeIntervalSince(phaseStart)

            // Streaming: read the selected slabs into GPU-visible buffers.
            // The file bytes are identical to what the resident path uploads
            // at load — only the storage (and this per-token I/O, counted as
            // gather) differs. With the slot cache a hit binds the cached
            // slab with no I/O and no copy; misses (or, without cache, all
            // active ranks) are disjoint slabs, so one concurrent F_NOCACHE
            // pread each keeps the NVMe queue deep instead of faulting the
            // mmap at queue depth 1.
            var streamedRankSlots: [Int] = []
            var rankReads: [[ExpertRead]] = []
            var gatherStart = Date()
            if case .streamed(let gateBase, let upBase,
                              let downBase) = routed {
                gatherStart = Date()
                let slabBytes = 2 * gateUpExpertBytes + downExpertBytes
                rankReads = Array(repeating: [], count: selected.count)
                if expertSlots.isEmpty {
                    for (rank, expert) in selected.enumerated() {
                        let e = Int(expert)
                        let staging = expertStaging[rank]
                        rankReads[rank] = [
                            ExpertRead(
                                buffer: staging.gate, bufferOffset: 0,
                                fileOffset: gateBase
                                    + e * gateUpExpertBytes,
                                bytes: gateUpExpertBytes),
                            ExpertRead(
                                buffer: staging.up, bufferOffset: 0,
                                fileOffset: upBase + e * gateUpExpertBytes,
                                bytes: gateUpExpertBytes),
                            ExpertRead(
                                buffer: staging.down, bufferOffset: 0,
                                fileOffset: downBase + e * downExpertBytes,
                                bytes: downExpertBytes),
                        ]
                    }
                    profile.gatherBytes += selected.count * slabBytes
                } else {
                    // One clock tick per (layer, token): slots touched in
                    // this tick — hit or freshly assigned — are never chosen
                    // as victims, so active ranks always land in distinct
                    // slots.
                    expertClock += 1
                    streamedRankSlots = [Int](repeating: -1,
                                              count: selected.count)
                    for (rank, expert) in selected.enumerated() {
                        let key = layer.index << 16 | Int(expert)
                        if let hit = expertSlotIndex[key] {
                            expertSlots[hit].lastUse = expertClock
                            streamedRankSlots[rank] = hit
                            profile.expertHits += 1
                            profile.expertHitBytes += slabBytes
                            continue
                        }
                        var victim = -1
                        var oldest = Int.max
                        let candidates = expertLayerSlotRanges[layer.index]
                            ?? 0..<expertSlots.count
                        for index in candidates
                        where expertSlots[index].lastUse < expertClock
                            && expertSlots[index].lastUse < oldest {
                            victim = index
                            oldest = expertSlots[index].lastUse
                        }
                        let evicted = expertSlots[victim].key
                        if evicted >= 0 {
                            expertSlotIndex.removeValue(forKey: evicted)
                        }
                        expertSlots[victim].key = key
                        expertSlots[victim].lastUse = expertClock
                        expertSlotIndex[key] = victim
                        streamedRankSlots[rank] = victim
                        let e = Int(expert)
                        let slot = expertSlots[victim]
                        rankReads[rank] = [
                            ExpertRead(
                                buffer: slot.gate, bufferOffset: 0,
                                fileOffset: gateBase
                                    + e * gateUpExpertBytes,
                                bytes: gateUpExpertBytes),
                            ExpertRead(
                                buffer: slot.up, bufferOffset: 0,
                                fileOffset: upBase + e * gateUpExpertBytes,
                                bytes: gateUpExpertBytes),
                            ExpertRead(
                                buffer: slot.down, bufferOffset: 0,
                                fileOffset: downBase + e * downExpertBytes,
                                bytes: downExpertBytes),
                        ]
                        profile.expertMisses += 1
                        profile.expertMissBytes += slabBytes
                    }
                    profile.gatherBytes += rankReads.reduce(0) {
                        $0 + ($1.isEmpty ? 0 : slabBytes)
                    }
                }
            }

            let expertWidth = Int(shape.nFFExpert)
            func encodeRoutedRank(into commands: Commands,
                                  rank: Int) throws {
                let e = Int(selected[rank])
                let gate: MTLBuffer, up: MTLBuffer, down: MTLBuffer
                let gateUpOffset: Int, downOffset: Int
                switch routed {
                case .resident(let residentGate, let residentUp,
                               let residentDown):
                    gate = residentGate; up = residentUp; down = residentDown
                    gateUpOffset = e * gateUpExpertBytes
                    downOffset = e * downExpertBytes
                case .streamed:
                    if expertSlots.isEmpty {
                        let staging = expertStaging[rank]
                        gate = staging.gate; up = staging.up
                        down = staging.down
                    } else {
                        let slot = expertSlots[streamedRankSlots[rank]]
                        gate = slot.gate; up = slot.up; down = slot.down
                    }
                    gateUpOffset = 0
                    downOffset = 0
                }
                try encodePairSwiGLU(
                    commands, type: routedType, rows: expertWidth,
                    inputWidth: embd,
                    gateRows: gate, gateOffset: gateUpOffset,
                    upRows: up, upOffset: gateUpOffset,
                    input: ffnNormed, routeWeight: routeWeights[rank],
                    mid: ffnMid
                )
                try encodeQuantMatvec(
                    commands, type: routedType, rows: embd,
                    inputWidth: expertWidth,
                    weights: down, weightsOffset: downOffset,
                    input: ffnMid, output: ffnOut, accumulate: rank != 0
                )
            }
            // Residual in the upstream association: `add3(after_attn,
            // routed, shared)` evaluates `(after_attn + routed) + shared`,
            // and the CPU oracle does the same. Folding the shared expert
            // into the routed accumulator first changes the rounding, so
            // the routed sum is closed here and the shared expert lands in
            // a second add (C-parity audit, MoE §6).
            func encodeMoeTail(into commands: Commands) throws {
                try encodeAdd(commands, a: afterAttn, b: ffnOut,
                              out: hiddenNext, count: embd)
                let sharedWidth = Int(shape.nFFShared)
                try encodePairSwiGLU(commands, type: Self.q8Type,
                                     rows: sharedWidth, inputWidth: embd,
                                     gateRows: sharedGate, upRows: sharedUp,
                                     input: ffnNormed, routeWeight: 1,
                                     mid: ffnMid)
                try encodeQuantMatvec(commands, type: Self.q8Type, rows: embd,
                                      inputWidth: sharedWidth,
                                      weights: sharedDown, input: ffnMid,
                                      output: ffnOut, accumulate: false)
                try encodeAdd(commands, a: hiddenNext, b: ffnOut,
                              out: hiddenNext, count: embd)
            }
            // DeepSeek-style I/O overlap. Unlike the historical tail above,
            // this independent half can run while rank 0 is still arriving
            // from SSD. `hiddenNext` is otherwise dead until the layer closes.
            func encodeSharedExpert(into commands: Commands) throws {
                let sharedWidth = Int(shape.nFFShared)
                try encodePairSwiGLU(commands, type: Self.q8Type,
                                     rows: sharedWidth, inputWidth: embd,
                                     gateRows: sharedGate, upRows: sharedUp,
                                     input: ffnNormed, routeWeight: 1,
                                     mid: ffnMid)
                try encodeQuantMatvec(commands, type: Self.q8Type, rows: embd,
                                      inputWidth: sharedWidth,
                                      weights: sharedDown, input: ffnMid,
                                      output: hiddenNext, accumulate: false)
            }
            func encodeOverlappedMoeTail(
                into commands: Commands
            ) throws {
                // ffnMid is no longer live after the last routed down
                // projection and is wide enough for one embedding row.
                try encodeAdd(commands, a: afterAttn, b: ffnOut,
                              out: ffnMid, count: embd)
                try encodeAdd(commands, a: ffnMid, b: hiddenNext,
                              out: hiddenNext, count: embd)
            }

            let allRankReads = rankReads.flatMap { $0 }
            let loadedByMetalIO: Bool
            if case .streamed = routed, !allRankReads.isEmpty {
                loadedByMetalIO = loadExpertsMetalIO(allRankReads)
            } else {
                loadedByMetalIO = false
            }

            if case .streamed = routed, !loadedByMetalIO, streamFD >= 0,
               let event = expertEvent {
                // Overlapped expert batch: the misses stream in the
                // background while ONE command buffer carries every rank,
                // each behind `encodeWaitForEvent(base+rank+1)`; a signaler
                // thread advances the event as ranks complete IN RANK ORDER,
                // so the GPU crunches early ranks under the I/O of later
                // ones without changing the accumulate order (same rounding,
                // C-parity audit above holds).
                let fd = streamFD
                let failureLock = NSLock()
                nonisolated(unsafe) var failed = false
                var groups: [DispatchGroup] = []
                groups.reserveCapacity(rankReads.count)
                for reads in rankReads {
                    let group = DispatchGroup()
                    groups.append(group)
                    for read in reads {
                        group.enter()
                        nonisolated(unsafe) let job = read
                        DispatchQueue.global(qos: .userInitiated).async {
                            if !GGUFWeights.preadFullSplit(
                                fd, into: job.destination, bytes: job.bytes,
                                offset: job.fileOffset, parts: readParts
                            ) {
                                failureLock.lock()
                                failed = true
                                failureLock.unlock()
                            }
                            group.leave()
                        }
                    }
                }
                let base = expertEventBase
                expertEventBase += UInt64(selected.count)
                nonisolated(unsafe) var ioDone = Date()
                let ioGroup = DispatchGroup()
                ioGroup.enter()
                nonisolated(unsafe) let sharedEvent = event
                nonisolated(unsafe) let rankGroups = groups
                DispatchQueue.global(qos: .userInitiated).async {
                    for (rank, group) in rankGroups.enumerated() {
                        group.wait()
                        sharedEvent.signaledValue = base + UInt64(rank) + 1
                    }
                    ioDone = Date()
                    ioGroup.leave()
                }

                guard let buffer = runtime.queue.makeCommandBuffer() else {
                    throw LagunaResidentModelError.bufferAllocation(
                        "command buffer")
                }
                if Self.sharedExpertIOOverlap {
                    guard let sharedEncoder =
                            buffer.makeComputeCommandEncoder() else {
                        throw LagunaResidentModelError.bufferAllocation(
                            "shared-expert overlap encoder")
                    }
                    try encodeSharedExpert(
                        into: Commands(buffer: buffer,
                                       encoder: sharedEncoder))
                    sharedEncoder.endEncoding()
                }
                for rank in 0..<selected.count {
                    buffer.encodeWaitForEvent(
                        event, value: base + UInt64(rank) + 1)
                    guard let encoder = buffer.makeComputeCommandEncoder()
                    else {
                        throw LagunaResidentModelError.bufferAllocation(
                            "command encoder")
                    }
                    let commands = Commands(buffer: buffer, encoder: encoder)
                    try encodeRoutedRank(into: commands, rank: rank)
                    encoder.endEncoding()
                }
                guard let tailEncoder = buffer.makeComputeCommandEncoder()
                else {
                    throw LagunaResidentModelError.bufferAllocation(
                        "command encoder")
                }
                let tailCommands = Commands(buffer: buffer,
                                            encoder: tailEncoder)
                if Self.sharedExpertIOOverlap {
                    try encodeOverlappedMoeTail(into: tailCommands)
                } else {
                    try encodeMoeTail(into: tailCommands)
                }
                tailEncoder.endEncoding()
                if Self.chainedDecode {
                    precondition(pendingExpertCommands == nil)
                    let stamp = CompletionStamp()
                    buffer.addCompletedHandler { _ in stamp.mark() }
                    buffer.commit()
                    // The CPU needs no expert result here. Wait only for the
                    // reads so failures are reported and the staging jobs no
                    // longer mutate their buffers; GPU completion is carried
                    // into the following layer's trunk wait.
                    ioGroup.wait()
                    if failed {
                        buffer.waitUntilCompleted()
                        throw LagunaResidentModelError.expertRead(
                            "pread of a routed expert slab failed on layer "
                            + "\(layer.index)")
                    }
                    let ioEnd = max(ioDone, gatherStart)
                    profile.gatherS += ioEnd.timeIntervalSince(gatherStart)
                    pendingExpertCommands = PendingExpertCommands(
                        buffer: buffer, wallStart: ioEnd, completed: stamp)
                } else {
                    buffer.commit()
                    buffer.waitUntilCompleted()
                    ioGroup.wait()
                    if failed {
                        throw LagunaResidentModelError.expertRead(
                            "pread of a routed expert slab failed on layer "
                            + "\(layer.index)")
                    }
                    if let error = buffer.error { throw error }
                    // gpuEndTime−gpuStartTime includes the in-buffer event
                    // stalls, so the "REAL GPU" experts figure is an upper
                    // bound while streaming overlapped.
                    let gpu = buffer.gpuEndTime - buffer.gpuStartTime
                    if gpu.isFinite, gpu > 0 { gpuExpertsS += gpu }
                    // Wall split without double counting: up to the last pread
                    // is gather (the GPU is already crunching under it), the
                    // compute tail after it is experts.
                    let gpuDone = Date()
                    let ioEnd = min(max(ioDone, gatherStart), gpuDone)
                    profile.gatherS += ioEnd.timeIntervalSince(gatherStart)
                    profile.expertsS += gpuDone.timeIntervalSince(ioEnd)
                }
            } else {
                // Synchronous fallback: fill every slab, then encode the
                // whole batch (resident experts always land here — no
                // gather at all).
                if case .streamed = routed {
                    let reads = allRankReads
                    if loadedByMetalIO {
                        // Direct SSD → MTLBuffer command already completed.
                    } else if streamFD >= 0 {
                        let fd = streamFD
                        let failureLock = NSLock()
                        nonisolated(unsafe) var failed = false
                        nonisolated(unsafe) let jobs = reads
                        DispatchQueue.concurrentPerform(
                            iterations: jobs.count
                        ) { i in
                            let job = jobs[i]
                            if !GGUFWeights.preadFullSplit(
                                fd, into: job.destination, bytes: job.bytes,
                                offset: job.fileOffset, parts: readParts
                            ) {
                                failureLock.lock()
                                failed = true
                                failureLock.unlock()
                            }
                        }
                        if failed {
                            throw LagunaResidentModelError.expertRead(
                                "pread of a routed expert slab failed on "
                                + "layer \(layer.index)")
                        }
                    } else {
                        if willNeedExperts {
                            GGUFModel.prefetch(
                                base: model.mapBase,
                                ranges: reads.map {
                                    (UInt64($0.fileOffset),
                                     UInt64($0.bytes))
                                })
                        }
                        for job in reads {
                            memcpy(job.destination,
                                   model.mapBase + job.fileOffset,
                                   job.bytes)
                        }
                    }
                    profile.gatherS += Date().timeIntervalSince(gatherStart)
                }
                let expertsStart = Date()
                commands = try beginCommands()
                for rank in 0..<selected.count {
                    try encodeRoutedRank(into: commands, rank: rank)
                }
                try encodeMoeTail(into: commands)
                if Self.chainedDecode {
                    commands.encoder.endEncoding()
                    try submitPendingExpertCommands(
                        commands.buffer, wallStart: expertsStart)
                } else {
                    try endCommands(commands, phase: .experts)
                    profile.expertsS += Date().timeIntervalSince(expertsStart)
                }
            }
        }

        profile.layers += 1
        swap(&hidden, &hiddenNext)
    }

    private func encodeOutputHead() throws {
        let commands = try beginCommands()
        try encodeRMSNorm(commands, input: hidden, weight: outputNorm,
                          output: normed, width: Int(shape.nEmbd))
        try encodeQuantMatvec(commands, type: Self.q8Type,
                              rows: Int(shape.nVocab),
                              inputWidth: Int(shape.nEmbd),
                              weights: outputHead, input: normed,
                              output: logits, accumulate: false)
        try endCommands(commands, phase: .head)
    }

    // MARK: Command encoding

    private func loadExpertsMetalIO(_ reads: [ExpertRead]) -> Bool {
        guard let io = metalIO, !reads.isEmpty else { return false }
        let commandBuffer = io.queue.makeCommandBuffer()
        commandBuffer.label = "Laguna streamed experts × \(reads.count)"
        for read in reads {
            guard read.bufferOffset >= 0, read.bytes > 0,
                  read.bufferOffset + read.bytes <= read.buffer.length else {
                metalIO = nil
                return false
            }
            commandBuffer.load(
                read.buffer, offset: read.bufferOffset, size: read.bytes,
                sourceHandle: io.handle,
                sourceHandleOffset: read.fileOffset)
        }
        commandBuffer.commit()
        commandBuffer.waitUntilCompleted()
        guard commandBuffer.status == .complete else {
            metalIO = nil
            return false
        }
        return true
    }

    private struct Commands {
        let buffer: MTLCommandBuffer
        let encoder: MTLComputeCommandEncoder
    }

    private func beginCommands() throws -> Commands {
        guard let buffer = runtime.queue.makeCommandBuffer(),
              let encoder = buffer.makeComputeCommandEncoder() else {
            throw LagunaResidentModelError.bufferAllocation("command buffer")
        }
        return Commands(buffer: buffer, encoder: encoder)
    }

    private enum GpuPhase { case attention, experts, dense, head }

    private func endCommands(_ commands: Commands,
                             phase: GpuPhase) throws {
        commands.encoder.endEncoding()
        commands.buffer.commit()
        commands.buffer.waitUntilCompleted()
        // The queue is ordered: completion of this buffer also completes the
        // deferred expert tail submitted immediately before it.
        try flushPendingExpertCommands(wait: false)
        if let error = commands.buffer.error { throw error }
        let gpu = commands.buffer.gpuEndTime - commands.buffer.gpuStartTime
        guard gpu.isFinite, gpu > 0 else { return }
        switch phase {
        case .attention: gpuAttnS += gpu
        case .experts: gpuExpertsS += gpu
        case .dense: gpuDenseS += gpu
        case .head: gpuHeadS += gpu
        }
    }

    private func submitPendingExpertCommands(
        _ buffer: MTLCommandBuffer,
        wallStart: Date
    ) throws {
        precondition(
            pendingExpertCommands == nil,
            "the next trunk must flush the previous Laguna expert tail")
        let stamp = CompletionStamp()
        buffer.addCompletedHandler { _ in stamp.mark() }
        buffer.commit()
        pendingExpertCommands = PendingExpertCommands(
            buffer: buffer, wallStart: wallStart, completed: stamp)
    }

    private func flushPendingExpertCommands(wait: Bool) throws {
        guard let pending = pendingExpertCommands else { return }
        if wait { pending.buffer.waitUntilCompleted() }
        // `wait == false` is used only after waiting on a later buffer from
        // the same in-order queue; the pending one is therefore complete.
        if pending.buffer.status != .completed
            && pending.buffer.status != .error {
            pending.buffer.waitUntilCompleted()
        }
        pendingExpertCommands = nil
        if let error = pending.buffer.error { throw error }
        let completed = pending.completed.read() ?? Date()
        profile.expertsS += max(
            0, completed.timeIntervalSince(pending.wallStart))
        let gpu = pending.buffer.gpuEndTime - pending.buffer.gpuStartTime
        if gpu.isFinite, gpu > 0 { gpuExpertsS += gpu }
    }

    private func setArguments(_ encoder: MTLComputeCommandEncoder,
                              _ words: [UInt32]) {
        words.withUnsafeBytes {
            encoder.setBytes($0.baseAddress!, length: $0.count, index: 0)
        }
    }

    private func rowGroups(_ rows: Int) -> MTLSize {
        MTLSize(width: (rows + simdgroupsPerThreadgroup - 1)
                    / simdgroupsPerThreadgroup,
                height: 1, depth: 1)
    }

    private var simdgroupThreads: MTLSize {
        MTLSize(width: 32, height: simdgroupsPerThreadgroup, depth: 1)
    }

    private func encodeRMSNorm(_ commands: Commands, input: MTLBuffer,
                               inputOffset: Int = 0,
                               weight: MTLBuffer, output: MTLBuffer,
                               outputOffset: Int = 0,
                               width: Int) throws {
        let pipeline = try runtime.pipeline("kernel_glm52_rms_norm_f32")
        let encoder = commands.encoder
        encoder.setComputePipelineState(pipeline)
        setArguments(encoder, [UInt32(width), shape.rmsEpsilon.bitPattern, 0, 0])
        encoder.setBuffer(input, offset: inputOffset, index: 1)
        encoder.setBuffer(weight, offset: 0, index: 2)
        encoder.setBuffer(output, offset: outputOffset, index: 3)
        encoder.setThreadgroupMemoryLength(256 * MemoryLayout<Float>.stride,
                                           index: 0)
        encoder.dispatchThreadgroups(
            MTLSize(width: 1, height: 1, depth: 1),
            threadsPerThreadgroup: MTLSize(width: 256, height: 1, depth: 1)
        )
    }

    private func encodeRMSNormRows(
        _ commands: Commands, input: MTLBuffer,
        weight: MTLBuffer, output: MTLBuffer,
        width: Int, count: Int
    ) throws {
        let pipeline = try runtime.pipeline("kernel_glm52_rms_norm_f32")
        let encoder = commands.encoder
        encoder.setComputePipelineState(pipeline)
        setArguments(
            encoder,
            [UInt32(width), shape.rmsEpsilon.bitPattern, 0, 0])
        encoder.setBuffer(input, offset: 0, index: 1)
        encoder.setBuffer(weight, offset: 0, index: 2)
        encoder.setBuffer(output, offset: 0, index: 3)
        encoder.setThreadgroupMemoryLength(
            256 * MemoryLayout<Float>.stride, index: 0)
        encoder.dispatchThreadgroups(
            MTLSize(width: count, height: 1, depth: 1),
            threadsPerThreadgroup: MTLSize(width: 256, height: 1, depth: 1)
        )
    }

    private func encodeMatvecPair(_ commands: Commands, x: MTLBuffer,
                                  xOffset: Int = 0,
                                  typeA: UInt32, rowsA: Int, weightsA: MTLBuffer,
                                  typeB: UInt32, rowsB: Int, weightsB: MTLBuffer,
                                  inputWidth: Int,
                                  outA: MTLBuffer, outAOffset: Int = 0,
                                  outB: MTLBuffer, outBOffset: Int = 0) throws {
        let pipeline = try runtime.pipeline("kernel_glm52_matvec_pair_sg")
        let encoder = commands.encoder
        encoder.setComputePipelineState(pipeline)
        setArguments(encoder, [typeA, UInt32(rowsA), typeB, UInt32(rowsB),
                               UInt32(inputWidth), 0, 0, 0])
        encoder.setBuffer(x, offset: xOffset, index: 1)
        encoder.setBuffer(weightsA, offset: 0, index: 2)
        encoder.setBuffer(weightsB, offset: 0, index: 3)
        encoder.setBuffer(outA, offset: outAOffset, index: 4)
        encoder.setBuffer(outB, offset: outBOffset, index: 5)
        encoder.dispatchThreadgroups(rowGroups(rowsA + rowsB),
                                     threadsPerThreadgroup: simdgroupThreads)
    }

    private func encodePairSwiGLU(_ commands: Commands, type: UInt32, rows: Int,
                                  inputWidth: Int,
                                  gateRows: MTLBuffer, gateOffset: Int = 0,
                                  upRows: MTLBuffer, upOffset: Int = 0,
                                  input: MTLBuffer, inputOffset: Int = 0,
                                  routeWeight: Float,
                                  mid: MTLBuffer, midOffset: Int = 0) throws {
        let pipeline = try runtime.pipeline("kernel_glm52_moe_pair_swiglu_sg")
        let encoder = commands.encoder
        encoder.setComputePipelineState(pipeline)
        setArguments(encoder, [type, UInt32(rows), UInt32(inputWidth),
                               routeWeight.bitPattern])
        encoder.setBuffer(input, offset: inputOffset, index: 1)
        encoder.setBuffer(gateRows, offset: gateOffset, index: 2)
        encoder.setBuffer(upRows, offset: upOffset, index: 3)
        encoder.setBuffer(mid, offset: midOffset, index: 4)
        encoder.dispatchThreadgroups(rowGroups(rows),
                                     threadsPerThreadgroup: simdgroupThreads)
    }

    private func encodeQuantMatvec(_ commands: Commands, type: UInt32, rows: Int,
                                   inputWidth: Int,
                                   weights: MTLBuffer, weightsOffset: Int = 0,
                                   input: MTLBuffer, inputOffset: Int = 0,
                                   output: MTLBuffer, outputOffset: Int = 0,
                                   accumulate: Bool) throws {
        let name = accumulate
            ? "kernel_glm52_moe_down_acc_sg"
            : "kernel_glm52_moe_down_sg"
        let pipeline = try runtime.pipeline(name)
        let encoder = commands.encoder
        encoder.setComputePipelineState(pipeline)
        setArguments(encoder, [type, UInt32(rows), UInt32(inputWidth), 0])
        encoder.setBuffer(input, offset: inputOffset, index: 1)
        encoder.setBuffer(weights, offset: weightsOffset, index: 2)
        encoder.setBuffer(output, offset: outputOffset, index: 3)
        encoder.dispatchThreadgroups(rowGroups(rows),
                                     threadsPerThreadgroup: simdgroupThreads)
    }

    /// Multi-token Q8_0 projection used only by the opt-in dense-MM prefill.
    /// Input and output are tightly packed token-major matrices.
    private func encodeQ8MatmulRows(
        _ commands: Commands,
        weights: MTLBuffer,
        input: MTLBuffer,
        output: MTLBuffer,
        inputWidth: Int,
        outputWidth: Int,
        count: Int
    ) throws {
        precondition(inputWidth % 32 == 0)
        let rowBytes = (inputWidth / 32) * 34
        let arguments = MetalRuntime.mulMMArgs(
            inDim: inputWidth, outDim: outputWidth, nTok: count,
            rowBytes: UInt64(rowBytes))
        let boundedOutput = outputWidth % 64 != 0 || count % 32 != 0
        let pipeline = try runtime.mulMMPipeline(
            "kernel_mul_mm_q8_0_f32",
            bcInp: false, bcOut: boundedOutput)
        let encoder = commands.encoder
        encoder.setComputePipelineState(pipeline)
        arguments.withUnsafeBytes {
            encoder.setBytes($0.baseAddress!, length: $0.count, index: 0)
        }
        encoder.setBuffer(weights, offset: 0, index: 1)
        encoder.setBuffer(input, offset: 0, index: 2)
        encoder.setBuffer(output, offset: 0, index: 3)
        encoder.setThreadgroupMemoryLength(
            boundedOutput ? 8_192 : 6_144, index: 0)
        encoder.dispatchThreadgroups(
            MTLSize(width: (count + 31) / 32,
                    height: (outputWidth + 63) / 64, depth: 1),
            threadsPerThreadgroup: MTLSize(width: 128, height: 1, depth: 1))
    }

    private func encodeSwiGLURows(
        _ commands: Commands,
        gate: MTLBuffer,
        up: MTLBuffer,
        output: MTLBuffer,
        width: Int,
        count: Int
    ) throws {
        let arguments = MetalRuntime.gluArgs(
            n: width, alpha: 1, limit: 0)
        let pipeline = try runtime.pipeline("kernel_swiglu_f32")
        let encoder = commands.encoder
        encoder.setComputePipelineState(pipeline)
        arguments.withUnsafeBytes {
            encoder.setBytes($0.baseAddress!, length: $0.count, index: 0)
        }
        encoder.setBuffer(gate, offset: 0, index: 1)
        encoder.setBuffer(up, offset: 0, index: 2)
        encoder.setBuffer(output, offset: 0, index: 3)
        let threads = max(
            1, min(pipeline.maxTotalThreadsPerThreadgroup,
                   max(1, width / 2)))
        encoder.dispatchThreadgroups(
            MTLSize(width: count, height: 1, depth: 1),
            threadsPerThreadgroup: MTLSize(
                width: threads, height: 1, depth: 1))
    }

    private func encodeF32Matvec(_ commands: Commands, rows: MTLBuffer,
                                 rowCount: Int, inputWidth: Int,
                                 x: MTLBuffer, xOffset: Int = 0,
                                 out: MTLBuffer, outOffset: Int = 0) throws {
        let pipeline = try runtime.pipeline("kernel_glm52_matvec_f32")
        let encoder = commands.encoder
        encoder.setComputePipelineState(pipeline)
        setArguments(encoder, [UInt32(rowCount), UInt32(inputWidth), 0, 0])
        encoder.setBuffer(rows, offset: 0, index: 1)
        encoder.setBuffer(x, offset: xOffset, index: 2)
        encoder.setBuffer(out, offset: outOffset, index: 3)
        let threads = 256
        encoder.dispatchThreadgroups(
            MTLSize(width: (rowCount + threads - 1) / threads, height: 1, depth: 1),
            threadsPerThreadgroup: MTLSize(width: threads, height: 1, depth: 1)
        )
    }

    private func encodeF32MatvecRows(
        _ commands: Commands,
        rows: MTLBuffer,
        rowCount: Int,
        inputWidth: Int,
        input: MTLBuffer,
        output: MTLBuffer,
        count: Int
    ) throws {
        let pipeline = try runtime.pipeline("kernel_glm52_matvec_f32")
        let encoder = commands.encoder
        encoder.setComputePipelineState(pipeline)
        setArguments(encoder, [
            UInt32(rowCount), UInt32(inputWidth), UInt32(count), 0,
        ])
        encoder.setBuffer(rows, offset: 0, index: 1)
        encoder.setBuffer(input, offset: 0, index: 2)
        encoder.setBuffer(output, offset: 0, index: 3)
        let threads = 256
        let totalRows = rowCount * count
        encoder.dispatchThreadgroups(
            MTLSize(width: (totalRows + threads - 1) / threads,
                    height: 1, depth: 1),
            threadsPerThreadgroup: MTLSize(
                width: threads, height: 1, depth: 1))
    }

    private func encodeAdd(_ commands: Commands, a: MTLBuffer, aOffset: Int = 0,
                           b: MTLBuffer, bOffset: Int = 0,
                           out: MTLBuffer, outOffset: Int = 0,
                           count: Int) throws {
        let pipeline = try runtime.pipeline("kernel_glm52_add_f32")
        let encoder = commands.encoder
        encoder.setComputePipelineState(pipeline)
        setArguments(encoder, [UInt32(count), 0, 0, 0])
        encoder.setBuffer(a, offset: aOffset, index: 1)
        encoder.setBuffer(b, offset: bOffset, index: 2)
        encoder.setBuffer(out, offset: outOffset, index: 3)
        let threads = 256
        encoder.dispatchThreadgroups(
            MTLSize(width: (count + threads - 1) / threads, height: 1, depth: 1),
            threadsPerThreadgroup: MTLSize(width: threads, height: 1, depth: 1)
        )
    }

    private func encodeRouterSelect(_ commands: Commands,
                                    bias: MTLBuffer,
                                    logits: MTLBuffer, logitsOffset: Int,
                                    selected: MTLBuffer, selectedOffset: Int,
                                    weights: MTLBuffer, weightsOffset: Int,
                                    probs: MTLBuffer,
                                    probsOffset: Int,
                                    count: Int = 1) throws {
        let pipeline = try runtime.pipeline("kernel_glm52_router_select")
        let encoder = commands.encoder
        encoder.setComputePipelineState(pipeline)
        setArguments(encoder, [
            UInt32(LagunaRouterReference.expertCount),
            UInt32(activeExperts),
            LagunaRouterReference.expertWeightScale.bitPattern,
            0,
        ])
        encoder.setBuffer(logits, offset: logitsOffset, index: 1)
        encoder.setBuffer(bias, offset: 0, index: 2)
        encoder.setBuffer(selected, offset: selectedOffset, index: 3)
        encoder.setBuffer(weights, offset: weightsOffset, index: 4)
        encoder.setBuffer(probs, offset: probsOffset, index: 5)
        // 256 selection scores + 256 int32 indices.
        encoder.setThreadgroupMemoryLength(
            256 * (MemoryLayout<Float>.stride + MemoryLayout<Int32>.stride),
            index: 0
        )
        encoder.dispatchThreadgroups(
            MTLSize(width: count, height: 1, depth: 1),
            threadsPerThreadgroup: MTLSize(width: 256, height: 1, depth: 1)
        )
    }

    private func encodeQKNormRope(_ commands: Commands,
                                  spec: LagunaAttentionSpec,
                                  position: Int,
                                  queryNorm: MTLBuffer,
                                  keyNorm: MTLBuffer,
                                  query: MTLBuffer, queryOffset: Int,
                                  key: MTLBuffer, keyOffset: Int) throws {
        let pipeline = try runtime.pipeline("kernel_laguna_qk_head_rms_norm_rope_neox")
        let encoder = commands.encoder
        encoder.setComputePipelineState(pipeline)
        let combinedHeads = spec.headCount + spec.kvHeadCount
        var arguments = [UInt32](repeating: 0, count: 14)
        arguments[0] = 1
        arguments[1] = UInt32(combinedHeads)
        arguments[2] = UInt32(spec.headDim)
        arguments[3] = UInt32(spec.rotationDims)
        arguments[4] = UInt32(position)
        arguments[5] = UInt32(spec.ropeOriginalContext)
        arguments[6] = spec.rmsEpsilon.bitPattern
        arguments[7] = spec.ropeFrequencyBase.bitPattern
        arguments[8] = spec.ropeFrequencyScale.bitPattern
        arguments[9] = spec.extrapolationFactor.bitPattern
        arguments[10] = spec.attentionFactor.bitPattern
        arguments[11] = spec.betaFast.bitPattern
        arguments[12] = spec.betaSlow.bitPattern
        setArguments(encoder, arguments)
        encoder.setBuffer(query, offset: queryOffset, index: 1)
        encoder.setBuffer(key, offset: keyOffset, index: 2)
        encoder.setBuffer(queryNorm, offset: 0, index: 3)
        encoder.setBuffer(keyNorm, offset: 0, index: 4)
        var queryHeads = UInt32(spec.headCount)
        encoder.setBytes(&queryHeads, length: MemoryLayout<UInt32>.stride,
                         index: 5)
        encoder.setThreadgroupMemoryLength(128 * MemoryLayout<Float>.stride,
                                           index: 0)
        encoder.dispatchThreadgroups(
            MTLSize(width: combinedHeads, height: 1, depth: 1),
            threadsPerThreadgroup: MTLSize(width: 128, height: 1, depth: 1)
        )
    }

    private func encodeQKNormRopeRows(
        _ commands: Commands,
        spec: LagunaAttentionSpec,
        position: Int,
        count: Int,
        queryNorm: MTLBuffer,
        keyNorm: MTLBuffer,
        query: MTLBuffer,
        key: MTLBuffer
    ) throws {
        let pipeline = try runtime.pipeline(
            "kernel_laguna_qk_head_rms_norm_rope_neox")
        let encoder = commands.encoder
        encoder.setComputePipelineState(pipeline)
        let combinedHeads = spec.headCount + spec.kvHeadCount
        var arguments = [UInt32](repeating: 0, count: 14)
        arguments[0] = UInt32(count)
        arguments[1] = UInt32(combinedHeads)
        arguments[2] = UInt32(spec.headDim)
        arguments[3] = UInt32(spec.rotationDims)
        arguments[4] = UInt32(position)
        arguments[5] = UInt32(spec.ropeOriginalContext)
        arguments[6] = spec.rmsEpsilon.bitPattern
        arguments[7] = spec.ropeFrequencyBase.bitPattern
        arguments[8] = spec.ropeFrequencyScale.bitPattern
        arguments[9] = spec.extrapolationFactor.bitPattern
        arguments[10] = spec.attentionFactor.bitPattern
        arguments[11] = spec.betaFast.bitPattern
        arguments[12] = spec.betaSlow.bitPattern
        setArguments(encoder, arguments)
        encoder.setBuffer(query, offset: 0, index: 1)
        encoder.setBuffer(key, offset: 0, index: 2)
        encoder.setBuffer(queryNorm, offset: 0, index: 3)
        encoder.setBuffer(keyNorm, offset: 0, index: 4)
        var queryHeads = UInt32(spec.headCount)
        encoder.setBytes(&queryHeads, length: MemoryLayout<UInt32>.stride,
                         index: 5)
        encoder.setThreadgroupMemoryLength(128 * MemoryLayout<Float>.stride,
                                           index: 0)
        encoder.dispatchThreadgroups(
            MTLSize(width: combinedHeads, height: count, depth: 1),
            threadsPerThreadgroup: MTLSize(width: 128, height: 1, depth: 1)
        )
    }

    /// Convert the whole chunk's K/V rows to F16, attend against the
    /// untouched ring plus the causal staged prefix, then commit the staged
    /// rows. This replaces three dispatches per token with four dispatches
    /// per layer and remains correct when an SWA chunk crosses the ring wrap.
    private func encodePrefillAttention(
        _ commands: Commands,
        spec: LagunaAttentionSpec,
        cache: LagunaMetalKVCache,
        longIndex: LongAttentionIndex?,
        position: Int,
        count: Int,
        query: MTLBuffer,
        gate: MTLBuffer,
        key: MTLBuffer,
        value: MTLBuffer,
        stagedKey: MTLBuffer,
        stagedValue: MTLBuffer,
        out: MTLBuffer
    ) throws {
        var arguments = [UInt32](repeating: 0, count: 8)
        arguments[0] = UInt32(count)
        arguments[1] = UInt32(position)
        arguments[2] = UInt32(cache.capacity)
        arguments[3] = UInt32(spec.headCount)
        arguments[4] = UInt32(spec.kvHeadCount)
        arguments[5] = UInt32(spec.headDim)
        arguments[6] = (1 / Float(spec.headDim).squareRoot()).bitPattern

        let encoder = commands.encoder
        let stage = try runtime.pipeline("kernel_laguna_stage_kv_f16")
        encoder.setComputePipelineState(stage)
        setArguments(encoder, arguments)
        encoder.setBuffer(key, offset: 0, index: 1)
        encoder.setBuffer(value, offset: 0, index: 2)
        encoder.setBuffer(stagedKey, offset: 0, index: 3)
        encoder.setBuffer(stagedValue, offset: 0, index: 4)
        let values = count * spec.keyValueWidth
        let stageThreads = min(256, stage.maxTotalThreadsPerThreadgroup)
        encoder.dispatchThreadgroups(
            MTLSize(width: (values + stageThreads - 1) / stageThreads,
                    height: 1, depth: 1),
            threadsPerThreadgroup: MTLSize(
                width: stageThreads, height: 1, depth: 1)
        )

        let headsPerKV = spec.headCount / spec.kvHeadCount
        let attentionName: String
        let headGroups: Int
        switch headsPerKV {
        case 6:
            attentionName = "kernel_laguna_attention_prefill_gqa6_f16"
            headGroups = spec.headCount / 6
        case 3:
            attentionName = "kernel_laguna_attention_prefill_gqa3_f16"
            headGroups = spec.headCount / 3
        default:
            attentionName = "kernel_laguna_attention_prefill_gqa_f16"
            headGroups = spec.headCount
        }
        let attention = try runtime.pipeline(attentionName)
        encoder.setComputePipelineState(attention)
        setArguments(encoder, arguments)
        encoder.setBuffer(query, offset: 0, index: 1)
        encoder.setBuffer(gate, offset: 0, index: 2)
        encoder.setBuffer(cache.keys, offset: 0, index: 3)
        encoder.setBuffer(cache.values, offset: 0, index: 4)
        encoder.setBuffer(stagedKey, offset: 0, index: 5)
        encoder.setBuffer(stagedValue, offset: 0, index: 6)
        encoder.setBuffer(out, offset: 0, index: 7)
        encoder.dispatchThreadgroups(
            MTLSize(width: headGroups, height: count, depth: 1),
            threadsPerThreadgroup: MTLSize(width: 32, height: 1, depth: 1)
        )

        let commit = try runtime.pipeline("kernel_laguna_commit_kv_f16")
        encoder.setComputePipelineState(commit)
        setArguments(encoder, arguments)
        encoder.setBuffer(stagedKey, offset: 0, index: 1)
        encoder.setBuffer(stagedValue, offset: 0, index: 2)
        encoder.setBuffer(cache.keys, offset: 0, index: 3)
        encoder.setBuffer(cache.values, offset: 0, index: 4)
        let commitThreads = min(256, commit.maxTotalThreadsPerThreadgroup)
        encoder.dispatchThreadgroups(
            MTLSize(width: (values + commitThreads - 1) / commitThreads,
                    height: 1, depth: 1),
            threadsPerThreadgroup: MTLSize(
                width: commitThreads, height: 1, depth: 1)
        )
        if let longIndex {
            try encodeLongAttentionIndexUpdate(
                commands, cache: cache, index: longIndex,
                position: position, count: count)
        }
    }

    /// Emit one centroid row for every newly-completed raw-key block. The
    /// summaries are only a search index: selected attention still reads the
    /// original F16 K/V rows, avoiding an additional approximation.
    private func encodeLongAttentionIndexUpdate(
        _ commands: Commands,
        cache: LagunaMetalKVCache,
        index: LongAttentionIndex,
        position: Int,
        count: Int
    ) throws {
        let oldComplete = position / index.blockSize
        let newComplete = (position + count) / index.blockSize
        guard newComplete > oldComplete else { return }
        let blockCount = newComplete - oldComplete
        precondition(newComplete <= index.blockCapacity)
        let pipeline = try runtime.pipeline(
            "kernel_laguna_long_index_compress_f16")
        let encoder = commands.encoder
        encoder.setComputePipelineState(pipeline)
        setArguments(encoder, [
            UInt32(cache.rowWidth),
            UInt32(index.blockSize),
            UInt32(oldComplete),
            UInt32(blockCount),
            UInt32(index.blockCapacity),
            0, 0, 0,
        ])
        encoder.setBuffer(cache.keys, offset: 0, index: 1)
        encoder.setBuffer(index.keys, offset: 0, index: 2)
        let total = blockCount * cache.rowWidth
        let threads = min(
            256, pipeline.maxTotalThreadsPerThreadgroup)
        encoder.dispatchThreadgroups(
            MTLSize(width: (total + threads - 1) / threads,
                    height: 1, depth: 1),
            threadsPerThreadgroup: MTLSize(
                width: threads, height: 1, depth: 1))
    }

    private func encodeStoreKV(_ commands: Commands,
                               cache: LagunaMetalKVCache,
                               position: Int,
                               key: MTLBuffer, keyOffset: Int,
                               value: MTLBuffer, valueOffset: Int) throws {
        let pipeline = try runtime.pipeline("kernel_laguna_store_kv_f16")
        let encoder = commands.encoder
        encoder.setComputePipelineState(pipeline)
        setArguments(encoder, [
            UInt32(cache.capacity),
            UInt32(position % cache.capacity),
            1,
            UInt32(cache.rowWidth),
        ])
        encoder.setBuffer(key, offset: keyOffset, index: 1)
        encoder.setBuffer(value, offset: valueOffset, index: 2)
        encoder.setBuffer(cache.keys, offset: 0, index: 3)
        encoder.setBuffer(cache.values, offset: 0, index: 4)
        let threads = 256
        encoder.dispatchThreadgroups(
            MTLSize(width: (cache.rowWidth + threads - 1) / threads,
                    height: 1, depth: 1),
            threadsPerThreadgroup: MTLSize(width: threads, height: 1, depth: 1)
        )
    }

    private func encodeAttention(_ commands: Commands,
                                 spec: LagunaAttentionSpec,
                                 cache: LagunaMetalKVCache,
                                 longIndex: LongAttentionIndex?,
                                 position: Int,
                                 query: MTLBuffer, queryOffset: Int,
                                 gate: MTLBuffer, gateOffset: Int,
                                 out: MTLBuffer, outOffset: Int) throws {
        let keyCount = min(position + 1, cache.capacity)
        let keyStart = position + 1 - keyCount
        if let longIndex,
           usesIndexedLongAttention(
               position: position, index: longIndex) {
            indexedAttentionDispatches += 1
            try encodeIndexedLongAttention(
                commands, spec: spec, cache: cache,
                index: longIndex, position: position,
                query: query, queryOffset: queryOffset,
                gate: gate, gateOffset: gateOffset,
                out: out, outOffset: outOffset)
            return
        }
        let headsPerKV = spec.headCount / max(1, spec.kvHeadCount)
        if Self.decodeSplitK,
           spec.headCount == Int(shape.nHeadFull),
           spec.headDim == 128,
           spec.headCount % 3 == 0,
           headsPerKV % 3 == 0,
           keyStart == 0,
           keyCount >= Self.decodeSplitKMinimumKeys {
            try encodeAttentionSplitK(
                commands, spec: spec, cache: cache, keyCount: keyCount,
                query: query, queryOffset: queryOffset,
                gate: gate, gateOffset: gateOffset,
                out: out, outOffset: outOffset)
            return
        }

        let pipeline = try runtime.pipeline("kernel_laguna_attention_decode_gqa_f16")
        let encoder = commands.encoder
        encoder.setComputePipelineState(pipeline)
        var arguments = [UInt32](repeating: 0, count: 8)
        arguments[0] = UInt32(spec.headCount)
        arguments[1] = UInt32(spec.kvHeadCount)
        arguments[2] = UInt32(spec.headDim)
        arguments[3] = UInt32(cache.capacity)
        arguments[4] = UInt32(keyStart)
        arguments[5] = UInt32(keyCount)
        arguments[6] = (1 / Float(spec.headDim).squareRoot()).bitPattern
        setArguments(encoder, arguments)
        encoder.setBuffer(query, offset: queryOffset, index: 1)
        encoder.setBuffer(gate, offset: gateOffset, index: 2)
        encoder.setBuffer(cache.keys, offset: 0, index: 3)
        encoder.setBuffer(cache.values, offset: 0, index: 4)
        encoder.setBuffer(out, offset: outOffset, index: 5)
        encoder.setThreadgroupMemoryLength(
            (16 + 8 * spec.headDim) * MemoryLayout<Float>.stride, index: 0
        )
        encoder.dispatchThreadgroups(
            MTLSize(width: spec.headCount, height: 1, depth: 1),
            threadsPerThreadgroup: MTLSize(width: 256, height: 1, depth: 1)
        )
    }

    private func encodeIndexedLongAttention(
        _ commands: Commands,
        spec: LagunaAttentionSpec,
        cache: LagunaMetalKVCache,
        index: LongAttentionIndex,
        position: Int,
        query: MTLBuffer,
        queryOffset: Int,
        gate: MTLBuffer,
        gateOffset: Int,
        out: MTLBuffer,
        outOffset: Int
    ) throws {
        guard let scores = longAttentionScores,
              let selected = longAttentionSelected,
              longAttentionScratchBlocks >= index.blockCapacity else {
            throw LagunaResidentModelError.bufferAllocation(
                "long-attention scratch not prepared")
        }
        let visible = position + 1
        let recentStart = max(0, visible - longAttentionRecentTokens)
        let blockCount = recentStart / index.blockSize
        let rawStart = blockCount * index.blockSize
        let rawCount = visible - rawStart
        let keep = min(longAttentionTopBlocks, blockCount)
        precondition(keep > 0 && blockCount > keep)

        var arguments = [UInt32](repeating: 0, count: 12)
        arguments[0] = UInt32(spec.headCount)
        arguments[1] = UInt32(spec.kvHeadCount)
        arguments[2] = UInt32(spec.headDim)
        arguments[3] = UInt32(cache.capacity)
        arguments[4] = UInt32(index.blockSize)
        arguments[5] = UInt32(blockCount)
        arguments[6] = UInt32(keep)
        arguments[7] = UInt32(rawStart)
        arguments[8] = UInt32(rawCount)
        arguments[9] = UInt32(index.blockCapacity)
        arguments[10] =
            (1 / Float(spec.headDim).squareRoot()).bitPattern

        let encoder = commands.encoder
        let score = try runtime.pipeline(
            "kernel_laguna_long_index_scores_f16")
        encoder.setComputePipelineState(score)
        setArguments(encoder, arguments)
        encoder.setBuffer(query, offset: queryOffset, index: 1)
        encoder.setBuffer(index.keys, offset: 0, index: 2)
        encoder.setBuffer(scores, offset: 0, index: 3)
        encoder.dispatchThreadgroups(
            MTLSize(width: blockCount, height: spec.headCount, depth: 1),
            threadsPerThreadgroup: MTLSize(width: 32, height: 1, depth: 1))

        let topK = try runtime.pipeline(
            "kernel_laguna_long_index_topk")
        encoder.setComputePipelineState(topK)
        setArguments(encoder, arguments)
        encoder.setBuffer(scores, offset: 0, index: 1)
        encoder.setBuffer(selected, offset: 0, index: 2)
        encoder.setThreadgroupMemoryLength(
            max(1, keep) * MemoryLayout<Int32>.stride, index: 0)
        encoder.dispatchThreadgroups(
            MTLSize(width: spec.headCount, height: 1, depth: 1),
            threadsPerThreadgroup: MTLSize(width: 32, height: 1, depth: 1))

        let attention = try runtime.pipeline(
            "kernel_laguna_attention_decode_indexed_f16")
        encoder.setComputePipelineState(attention)
        setArguments(encoder, arguments)
        encoder.setBuffer(query, offset: queryOffset, index: 1)
        encoder.setBuffer(gate, offset: gateOffset, index: 2)
        encoder.setBuffer(cache.keys, offset: 0, index: 3)
        encoder.setBuffer(cache.values, offset: 0, index: 4)
        encoder.setBuffer(selected, offset: 0, index: 5)
        encoder.setBuffer(out, offset: outOffset, index: 6)
        encoder.dispatchThreadgroups(
            MTLSize(width: spec.headCount, height: 1, depth: 1),
            threadsPerThreadgroup: MTLSize(width: 32, height: 1, depth: 1))
    }

    private func splitAttentionReducePipeline(
        workgroups: Int
    ) throws -> MTLComputePipelineState {
        let key = "laguna_split_reduce_dv=128_nwg=\(workgroups)"
        if let pipeline = runtime.mulMVPipelineCache[key] {
            return pipeline
        }
        let constants = MTLFunctionConstantValues()
        var dv = Int32(128)
        var nwg = Int32(workgroups)
        constants.setConstantValue(&dv, type: .int, index: 500)
        constants.setConstantValue(&nwg, type: .int, index: 501)
        let function = try runtime.library.makeFunction(
            name: "kernel_laguna_flash_attn_reduce_gate_f32",
            constantValues: constants)
        let pipeline = try runtime.device.makeComputePipelineState(
            function: function)
        runtime.mulMVPipelineCache[key] = pipeline
        return pipeline
    }

    private func encodeAttentionSplitK(
        _ commands: Commands,
        spec: LagunaAttentionSpec,
        cache: LagunaMetalKVCache,
        keyCount: Int,
        query: MTLBuffer,
        queryOffset: Int,
        gate: MTLBuffer,
        gateOffset: Int,
        out: MTLBuffer,
        outOffset: Int
    ) throws {
        let split = try runtime.pipeline(
            "kernel_laguna_attention_decode_gqa3_split_f16")
        let maximum = min(
            Self.decodeSplitKMaximumWorkgroups,
            runtime.device.maxThreadsPerThreadgroup.width / 32)
        // The imported reducer uses all 32 SIMD lanes to read partial stats;
        // NWG therefore has to be 32 (the upstream global-layer contract).
        // A smaller adaptive value makes lanes NWG..<32 read the next row.
        guard maximum == Self.decodeSplitKMaximumWorkgroups else {
            throw MetalError.unsupported(
                "Laguna split-K decode requires 1024-thread reduction")
        }
        let workgroups = Self.decodeSplitKMaximumWorkgroups
        let reduce = try splitAttentionReducePipeline(workgroups: workgroups)
        let simdgroups = 1

        let encoder = commands.encoder
        encoder.setComputePipelineState(split)
        var arguments = [UInt32](repeating: 0, count: 8)
        arguments[0] = UInt32(spec.headCount)
        arguments[1] = UInt32(spec.kvHeadCount)
        arguments[2] = UInt32(spec.headDim)
        arguments[3] = UInt32(keyCount)
        arguments[4] = 1
        arguments[5] = UInt32(simdgroups)
        arguments[6] = UInt32(workgroups)
        arguments[7] = (1 / Float(spec.headDim).squareRoot()).bitPattern
        setArguments(encoder, arguments)
        encoder.setBuffer(query, offset: queryOffset, index: 1)
        encoder.setBuffer(cache.keys, offset: 0, index: 2)
        encoder.setBuffer(cache.values, offset: 0, index: 3)
        encoder.setBuffer(splitAttentionTmp, offset: 0, index: 4)
        let scratchFloats = 3 * simdgroups * (2 + spec.headDim)
        encoder.setThreadgroupMemoryLength(
            scratchFloats * MemoryLayout<Float>.stride, index: 0)
        encoder.dispatchThreadgroups(
            MTLSize(width: spec.headCount / 3, height: 1,
                    depth: workgroups),
            threadsPerThreadgroup: MTLSize(
                width: 32, height: simdgroups, depth: 1))

        encoder.setComputePipelineState(reduce)
        var rows = Int32(spec.headCount)
        encoder.setBytes(&rows, length: MemoryLayout<Int32>.stride, index: 0)
        encoder.setBuffer(splitAttentionTmp, offset: 0, index: 1)
        encoder.setBuffer(out, offset: outOffset, index: 2)
        encoder.setBuffer(gate, offset: gateOffset, index: 3)
        encoder.dispatchThreadgroups(
            MTLSize(width: spec.headCount, height: 1, depth: 1),
            threadsPerThreadgroup: MTLSize(
                width: 32 * workgroups, height: 1, depth: 1))
    }

    // MARK: CPU helpers

    /// Q8_0 row dequantization (32-element blocks: F16 scale + 32 int8).
    static func dequantizeQ8Row(base: UnsafeRawPointer, row: Int,
                                rowCount: Int, width: Int) throws -> [Float] {
        guard row >= 0, row < rowCount, width % 32 == 0 else {
            throw LagunaResidentModelError.invalidToken(Int32(clamping: row))
        }
        let blockBytes = 34
        let blocks = width / 32
        let rowBytes = blocks * blockBytes
        let rowBase = base + row * rowBytes
        var out = [Float](repeating: 0, count: width)
        for block in 0..<blocks {
            let blockBase = rowBase + block * blockBytes
            let scaleBits = blockBase.loadUnaligned(as: UInt16.self)
            let scale = Half.float(scaleBits)
            let quants = (blockBase + 2).assumingMemoryBound(to: Int8.self)
            for i in 0..<32 {
                out[block * 32 + i] = scale * Float(quants[i])
            }
        }
        return out
    }
}
