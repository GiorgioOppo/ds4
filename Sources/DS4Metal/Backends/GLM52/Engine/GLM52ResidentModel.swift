import DS4Core
import Foundation
import Metal

// The real engine over the real GGUF: loads the validated weight map into
// the resident decode graph — attention, dense/shared FFN and output head
// uploaded once per layer, routed experts streamed per token through the
// slot cache — and drives prefill plus greedy decode with the growing
// compact caches. This is roadmap wiring, not enablement: BackendSelector
// still refuses `glm-dsa` until the real-GGUF logits parity gate passes.
//
// Loading is scoped on purpose: `layerCount` may truncate the stack for
// smoke tests and partial validation. Routed experts require a type with a
// validated kernel (Q8_0, the K-quants and IQ2_XXS — the published GGUF's
// routed format); anything else is refused at load.

public struct GLM52ResidentModelOptions: Sendable {
    /// Layers to load from the front of the stack; nil loads every
    /// autoregressive layer (78 — the nextn block is never executed).
    public var layerCount: Int?
    /// Compact-cache capacity in tokens (per layer).
    public var cacheCapacity: Int
    /// Expert slots per sparse layer's streaming cache.
    public var expertSlotCount: Int
    /// Layers kept fully resident from the front; the remaining SPARSE
    /// layers stream their big tensors from SSD per token (double-buffered
    /// prefetch). nil keeps every loaded layer resident. Must cover at
    /// least the three leading dense layers.
    public var residentLayerCount: Int?
    /// Cap on routed experts executed per token (rank order, weights
    /// untouched): less expert I/O, lower quality. nil runs the full top-8.
    public var activeExperts: Int?
    /// Staging slots of the layer streamer (min 2). With N slots, N-1
    /// prefetch fills run CONCURRENTLY while one slot computes — deeper SSD
    /// queue on the dominant stream at ~250 MiB of RAM per extra slot.
    public var streamSlotCount: Int

    /// RAM-adaptive resident budget: half the physical memory minus a 6 GiB
    /// reserve (output head, dense layers, caches, staging, OS), at ~230 MiB
    /// per resident sparse layer. Floor: the three dense layers. Every
    /// resident layer removes ~230 MiB of SSD reads from EVERY decoded
    /// token, so this is the single biggest tok/s lever on streaming.
    public static func adaptiveResidentLayerCount(
        totalLayers: Int = 78) -> Int {
        let physical = Double(ProcessInfo.processInfo.physicalMemory)
        let budget = physical * 0.5 - 6.0 * 1_073_741_824
        let perLayer = 230.0 * 1_048_576
        let extra = budget > 0 ? Int(budget / perLayer) : 0
        return max(3, min(totalLayers, 3 + extra))
    }

    public init(layerCount: Int? = nil,
                cacheCapacity: Int = 4_096,
                expertSlotCount: Int = 16,
                residentLayerCount: Int? = nil,
                activeExperts: Int? = nil,
                streamSlotCount: Int = 3) {
        self.layerCount = layerCount
        self.cacheCapacity = cacheCapacity
        self.expertSlotCount = expertSlotCount
        self.residentLayerCount = residentLayerCount
        self.activeExperts = activeExperts
        self.streamSlotCount = max(2, streamSlotCount)
    }
}

public final class GLM52ResidentModel {
    public let configuration: GLM52Configuration
    public let loadedLayerCount: Int
    public private(set) var position = 0

    private struct StreamedLayer {
        let tensors: GLM52StreamedLayerTensors
        /// Sum of the streamed big-tensor bytes — the per-token SSD cost of
        /// this layer, precomputed for the telemetry counters.
        let bigTensorBytes: Int
        let attnNorm: MTLBuffer
        let qANorm: MTLBuffer
        let kvANorm: MTLBuffer
        let ffnNorm: MTLBuffer
        let indexerKeyNorm: MTLBuffer?
        let indexerKeyNormBias: MTLBuffer?
        let proj: [Float]?
        let routerRows: [Float]
        let routerBias: [Float]
        let provider: GLM52StreamedExpertProvider
        let caches: GLM52ResidentDecodeCaches

        /// Kernel weight types straight from the descriptors — Q8_0 on the
        /// GGUF path, the requantized types when a Q4_K sidecar serves the
        /// layer.
        var weightTypes: GLM52StreamedWeightTypes {
            var types = GLM52StreamedWeightTypes()
            types.qA = tensors.qA.type
            types.qB = tensors.qB.type
            types.kvA = tensors.kvA.type
            types.attnOutput = tensors.attnOutput.type
            if let key = tensors.indexerKey { types.indexerKey = key.type }
            if let queryB = tensors.indexerQueryB {
                types.indexerQueryB = queryB.type
            }
            types.sharedGateUp = tensors.sharedGate.type
            types.sharedDown = tensors.sharedDown.type
            return types
        }
    }

    private let runtime: MetalRuntime
    private let reader: GLM52PayloadReader
    private let embedding: GLM52WeightDescriptor
    private let embeddingRowBytes: Int
    private let stack: [GLM52ResidentStackLayer]
    private let streamedLayers: [StreamedLayer]
    private let streamer: GLM52LayerStreamer?
    private let head: GLM52ResidentOutputHead
    private let providers: [Int: GLM52StreamedExpertProvider]
    private let activeExperts: Int?
    private let scratch: GLM52DecodeScratch
    private let geometry = GLM52DecodeGeometry.v5_2
    private let vocabulary: Int
    private let embeddingWidth: Int
    private let counters: GLM52StreamingCounters
    /// Prefetches kept in flight on the layer streamer (slots - 1).
    private let prefetchDepth: Int
    /// Keyed LRU arena the staged fetches resolve into (nil when no sparse
    /// layer streams experts).
    private var arena: GLM52ExpertArena?
    /// Per-layer staged zero-copy expert fetch (arena-backed, concurrent
    /// reads); layers whose record layout cannot be offset-bound are simply
    /// absent and keep the copying provider path.
    private let stagedFetch:
        [Int: ([UInt32]) throws -> GLM52StagedExpertSelection]
    /// Last token's routed selection per layer — the speculative warm-up's
    /// guess. DS4_GLM_SPEC_EXPERTS=0 turns speculation off.
    private var lastRouted: [Int: [UInt32]] = [:]
    private let speculationEnabled = ProcessInfo.processInfo
        .environment["DS4_GLM_SPEC_EXPERTS"] != "0"
    private let speculationQueue = DispatchQueue(
        label: "glm52.expert.speculation", qos: .userInitiated,
        attributes: .concurrent)

    public init(runtime: MetalRuntime,
                path: String,
                options: GLM52ResidentModelOptions = .init()) throws {
        let model = try GGUFModel(path: path, metalMapping: false,
                                  prefetchCPU: false)
        let configuration = try GLM52Configuration(model: model)
        try GLM52TensorSchema.validate(model: model)
        let map = try GLM52WeightMap(model: model)
        let reader = try GLM52PayloadReader(path: path, weightMap: map)

        self.runtime = runtime
        self.configuration = configuration
        self.reader = reader
        let shape = configuration.shape
        let inferenceLayers = Int(shape.inferenceLayerCount)
        let count = options.layerCount ?? inferenceLayers
        guard count >= 1, count <= inferenceLayers else {
            throw MetalError.unsupported(
                "GLM 5.2 engine layer count \(count) is outside "
                + "1...\(inferenceLayers)")
        }
        vocabulary = Int(shape.nVocab)
        embeddingWidth = Int(shape.nEmbd)
        embedding = try map.global(.tokenEmbedding)
        guard embedding.type == GLM52TensorSchema.q8_0 else {
            throw MetalError.unsupported(
                "GLM 5.2 engine expects a Q8_0 token embedding")
        }
        embeddingRowBytes = MetalRuntime.glm52Q8RowBytes(embeddingWidth)
        loadedLayerCount = count
        activeExperts = options.activeExperts
        let residentCount = min(count, options.residentLayerCount ?? count)
        guard residentCount >= min(count, Int(shape.nLeadingDense)) else {
            throw MetalError.unsupported(
                "GLM 5.2 streaming requires the \(shape.nLeadingDense) "
                + "leading dense layers to stay resident")
        }

        func f32(_ descriptor: GLM52WeightDescriptor) throws -> [Float] {
            guard descriptor.type == GLM52TensorSchema.f32 else {
                throw MetalError.unsupported(
                    "\(descriptor.name) must be F32 for the engine")
            }
            let raw = try reader.bytes(of: descriptor)
            return raw.withUnsafeBytes { Array($0.bindMemory(to: Float.self)) }
        }

        LoadProgress.shared.set(0.02, "GLM: mappa pesi e schema")
        // Sidecar convention (like the DeepSeek .expbundle): the expert
        // bundles live beside the GGUF unless DS4_GLM_BUNDLE_DIR overrides.
        // Auto-discovered and identity-validated per layer; absence is
        // simply the plain GGUF path.
        let bundleDirectory = ProcessInfo.processInfo
            .environment["DS4_GLM_BUNDLE_DIR"] ?? (path + ".glm-experts")
        let geometry = GLM52DecodeGeometry.v5_2
        var layers: [GLM52ResidentStackLayer] = []
        var providers: [Int: GLM52StreamedExpertProvider] = [:]
        layers.reserveCapacity(residentCount)
        for index in 0..<residentCount {
            LoadProgress.shared.set(
                0.05 + 0.75 * Double(index) / Double(max(residentCount, 1)),
                "GLM: layer residente \(index + 1)/\(residentCount)")
            let attention = GLM52QuantizedDecodeAttention(
                attnNorm: try f32(map.layer(index, .attentionNorm)),
                qA: try reader.bytes(of: map.layer(index, .attentionQueryA)),
                qANorm: try f32(map.layer(index, .attentionQueryANorm)),
                qB: try reader.bytes(of: map.layer(index, .attentionQueryB)),
                kvA: try reader.bytes(of: map.layer(index, .attentionKVA)),
                kvANorm: try f32(map.layer(index, .attentionKVANorm)),
                keyB: try reader.bytes(of: map.layer(index, .attentionKeyB)),
                valueB: try reader.bytes(of: map.layer(index, .attentionValueB)),
                attnOutput: try reader.bytes(
                    of: map.layer(index, .attentionOutput)))
            let isFull = GLM52IndexSharePolicy.isFullIndexerLayer(
                index, shape: shape)
            let indexer: GLM52QuantizedDecodeIndexer? = isFull
                ? GLM52QuantizedDecodeIndexer(
                    key: try reader.bytes(of: map.layer(index, .indexerKey)),
                    keyNorm: try f32(map.layer(index, .indexerKeyNorm)),
                    keyNormBias: try f32(
                        map.layer(index, .indexerKeyNormBias)),
                    queryB: try reader.bytes(
                        of: map.layer(index, .indexerQueryB)),
                    proj: try f32(map.layer(index, .indexerProjection)))
                : nil
            let ffnNorm = try f32(map.layer(index, .feedForwardNorm))

            let quantizedFFN: GLM52QuantizedLayerFFN
            if index < Int(shape.nLeadingDense) {
                quantizedFFN = .dense(
                    gate: try reader.bytes(of: map.layer(index, .denseGate)),
                    up: try reader.bytes(of: map.layer(index, .denseUp)),
                    down: try reader.bytes(of: map.layer(index, .denseDown)))
            } else {
                // Resident layers stream their EXPERTS too (the routed bank
                // never fits) — and deserve the bundle exactly like the
                // streamed tail.
                let provider = try GLM52StreamedExpertProvider(
                    reader: reader, weightMap: map, layer: index,
                    slotCount: options.expertSlotCount,
                    bundleDirectory: bundleDirectory)
                providers[index] = provider
                quantizedFFN = .sparse(
                    routerRows: try f32(map.layer(index, .router)),
                    routerBias: try f32(map.layer(index, .routerBias)),
                    sharedGate: try reader.bytes(
                        of: map.layer(index, .sharedGate)),
                    sharedUp: try reader.bytes(of: map.layer(index, .sharedUp)),
                    sharedDown: try reader.bytes(
                        of: map.layer(index, .sharedDown)),
                    expertProvider: { [provider] in try provider.expert($0) })
            }

            layers.append(GLM52ResidentStackLayer(
                index: index,
                weights: try GLM52ResidentDecodeWeights(
                    runtime: runtime, geometry: geometry,
                    attention: attention, indexer: indexer),
                ffn: try GLM52ResidentFFN(
                    runtime: runtime, geometry: geometry,
                    ffnNorm: ffnNorm, ffn: quantizedFFN),
                caches: try GLM52ResidentDecodeCaches(
                    runtime: runtime, geometry: geometry,
                    capacity: options.cacheCapacity, fullIndexer: isFull)))
        }
        stack = layers

        // Streamed tail: sparse layers whose big tensors arrive per token
        // through the concurrent staging slots. Small per-layer state
        // (norms, router, proj, caches) stays resident. Layers with a valid
        // Q4_K sidecar (convenzione: accanto al GGUF, DS4_GLM_LAYERQ4_DIR
        // per spostarlo) stream the requantized tensors — about half the
        // bytes; the others stream Q8_0 from the GGUF.
        LoadProgress.shared.set(0.82, "GLM: stato streaming per layer")
        let layerQ4Directory = ProcessInfo.processInfo
            .environment["DS4_GLM_LAYERQ4_DIR"] ?? (path + ".glm-layers-q4")
        var sidecarReaders: [Int: GLM52PayloadReader] = [:]
        var streamed: [StreamedLayer] = []
        for index in residentCount..<count {
            let isFull = GLM52IndexSharePolicy.isFullIndexerLayer(
                index, shape: shape)
            let provider = try GLM52StreamedExpertProvider(
                reader: reader, weightMap: map, layer: index,
                slotCount: options.expertSlotCount,
                bundleDirectory: bundleDirectory)
            providers[index] = provider
            let ggufTensors = try GLM52StreamedLayerTensors(
                index: index, map: map, fullIndexer: isFull)
            let sidecar = try GLM52LayerQuantSidecar.open(
                directory: layerQ4Directory, layer: index,
                source: ggufTensors, sourceFileSize: reader.fileSize)
            if let sidecar { sidecarReaders[index] = sidecar.reader }
            let tensors = sidecar?.tensors ?? ggufTensors
            streamed.append(StreamedLayer(
                tensors: tensors,
                bigTensorBytes: tensors.all.reduce(0) {
                    $0 + Int($1.bytes)
                },
                attnNorm: try runtime.glm52GraphBuffer(
                    try f32(map.layer(index, .attentionNorm))),
                qANorm: try runtime.glm52GraphBuffer(
                    try f32(map.layer(index, .attentionQueryANorm))),
                kvANorm: try runtime.glm52GraphBuffer(
                    try f32(map.layer(index, .attentionKVANorm))),
                ffnNorm: try runtime.glm52GraphBuffer(
                    try f32(map.layer(index, .feedForwardNorm))),
                indexerKeyNorm: isFull ? try runtime.glm52GraphBuffer(
                    try f32(map.layer(index, .indexerKeyNorm))) : nil,
                indexerKeyNormBias: isFull ? try runtime.glm52GraphBuffer(
                    try f32(map.layer(index, .indexerKeyNormBias))) : nil,
                proj: isFull
                    ? try f32(map.layer(index, .indexerProjection)) : nil,
                routerRows: try f32(map.layer(index, .router)),
                routerBias: try f32(map.layer(index, .routerBias)),
                provider: provider,
                caches: try GLM52ResidentDecodeCaches(
                    runtime: runtime, geometry: geometry,
                    capacity: options.cacheCapacity, fullIndexer: isFull)))
        }
        streamedLayers = streamed
        if !sidecarReaders.isEmpty {
            FileHandle.standardError.write(Data(
                ("DS4 glm: sidecar Q4 layer attivo su \(sidecarReaders.count)"
                 + "/\(streamed.count) layer streamati\n").utf8))
        }
        // The template always sizes the indexer slots (every layer stores
        // indexer tensors in the schema, so the descriptors exist even for
        // IndexShare layers) and always uses the FULL Q8_0 GGUF sizes —
        // sidecar tensors are smaller and share the same slots.
        streamer = streamed.isEmpty ? nil : try GLM52LayerStreamer(
            runtime: runtime, reader: reader,
            template: try GLM52StreamedLayerTensors(
                index: streamed[0].tensors.index, map: map,
                fullIndexer: true),
            slotCount: options.streamSlotCount,
            sidecarReaders: sidecarReaders)
        prefetchDepth = max(1, options.streamSlotCount - 1)

        self.providers = providers
        scratch = try GLM52DecodeScratch(
            runtime: runtime, geometry: geometry,
            scoreCapacity: options.cacheCapacity)
        LoadProgress.shared.set(0.9, "GLM: output head")
        head = try GLM52ResidentOutputHead(
            runtime: runtime, geometry: geometry,
            outputNorm: try f32(map.global(.outputNorm)),
            outputHead: try reader.bytes(of: map.global(.output)),
            vocabularySize: vocabulary)

        // Staged zero-copy expert path over the keyed LRU ARENA: every
        // sparse layer resolves its selection to per-record offsets in one
        // shared buffer, reading only what is not already resident (repeat
        // selections across prefill tokens and speculative warm-ups become
        // zero-I/O hits). Installed only where the record layout binds with
        // 4-byte-aligned offsets; elsewhere the copying per-expert provider
        // path stays in charge.
        let counters = GLM52StreamingCounters()
        self.counters = counters
        var staged:
            [Int: ([UInt32]) throws -> GLM52StagedExpertSelection] = [:]
        if let maxRecord = providers.values.map(\.recordBytes).max(),
           maxRecord > 0 {
            let arenaSlots = ProcessInfo.processInfo
                .environment["DS4_GLM_EXPERT_ARENA"].flatMap(Int.init) ?? 24
            let arena = try GLM52ExpertArena(
                device: runtime.device, slotCount: arenaSlots,
                slotBytes: maxRecord)
            self.arena = arena
            for (index, provider) in providers
                where provider.bindableRecordLayout {
                staged[index] = { [arena, counters] ids in
                    let start = Date()
                    let offsets = try arena.stage(
                        layer: index, ids: ids,
                        recordBytes: provider.recordBytes) { id, slice in
                        try provider.readRecord(id, into: slice)
                    }
                    let stall = Date().timeIntervalSince(start)
                    counters.expertStallSeconds += stall
                    return GLM52StagedExpertSelection(
                        buffer: arena.buffer, recordOffsets: offsets,
                        gateBytes: provider.gateRecordBytes,
                        upBytes: provider.upRecordBytes,
                        downBytes: provider.downRecordBytes,
                        gateUpType: provider.gateUpType,
                        downType: provider.downType)
                }
            }
        } else {
            arena = nil
        }
        stagedFetch = staged
        for layer in stack {
            layer.ffn.stagedSelection = staged[layer.index]
        }
    }

    /// Kick a background warm-up of `layer`'s arena slots with the LAST
    /// token's routing for that layer (consecutive tokens reselect 25-40%
    /// of experts): the read overlaps the GPU attention instead of
    /// stalling the FFN. Best-effort by design.
    private func speculateExperts(layer index: Int) {
        guard speculationEnabled, let arena,
              stagedFetch[index] != nil,
              let ids = lastRouted[index],
              let provider = providers[index] else { return }
        let recordBytes = provider.recordBytes
        speculationQueue.async { [arena] in
            arena.speculate(layer: index, ids: ids,
                            recordBytes: recordBytes) { id, slice in
                try provider.readRecord(id, into: slice)
            }
        }
    }

    /// Forget the whole conversation: every layer cache back to zero rows,
    /// position to zero. The chat service uses this before re-prefilling a
    /// rendered conversation (no incremental KV suffix reuse yet).
    public func resetContext() {
        for layer in stack { layer.caches.reset() }
        for streamedLayer in streamedLayers { streamedLayer.caches.reset() }
        position = 0
    }

    /// One token's dequantized embedding row, read directly from the GGUF.
    public func embeddingRow(_ token: Int32) throws -> [Float] {
        guard token >= 0, Int(token) < vocabulary else {
            throw MetalError.unsupported(
                "GLM 5.2 token \(token) is outside 0..<\(vocabulary)")
        }
        let raw = try reader.bytes(
            of: embedding,
            byteOffset: UInt64(Int(token) * embeddingRowBytes),
            byteCount: UInt64(embeddingRowBytes))
        var row = [Float](repeating: 0, count: embeddingWidth)
        raw.withUnsafeBytes {
            Quantize.dequantQ8_0($0.baseAddress!, count: embeddingWidth,
                                 into: &row)
        }
        return row
    }

    /// Advance the model by one token; returns the full logits row. The
    /// resident prefix computes first; the streamed tail overlaps each
    /// layer's compute with the SSD prefetch of the next one. Routed experts
    /// arrive through the staged zero-copy path (one concurrent read burst
    /// per selection) — the old post-token lookahead is gone on purpose:
    /// with per-token traffic far beyond RAM the page cache cannot retain
    /// speculative reads until the next token, so they only stole SSD
    /// bandwidth from the real ones.
    public func forwardNext(_ token: Int32) throws -> [Float] {
        let embedded = try embeddingRow(token)
        scratch.loadHidden(embedded)
        var lastSelection: (source: Int, rows: [UInt32])?
        counters.tokens += 1

        if let streamer {
            for ahead in 0..<min(prefetchDepth, streamedLayers.count) {
                streamer.prefetch(streamedLayers[ahead].tensors)
            }
        }
        func reuse(for index: Int, isFull: Bool) throws -> [UInt32]? {
            if isFull { return nil }
            guard let source = GLM52IndexSharePolicy.selectionSourceLayer(
                      for: index),
                  let last = lastSelection, last.source == source else {
                throw MetalError.unsupported(
                    "GLM 5.2 IndexShare layer \(index) has no selection "
                    + "from its source layer")
            }
            return last.rows
        }

        for layer in stack {
            let isFull = GLM52IndexSharePolicy.isFullIndexerLayer(layer.index)
            speculateExperts(layer: layer.index)
            let result = try runtime.glm52ChainedDecodeLayer(
                weights: layer.weights, ffn: layer.ffn, caches: layer.caches,
                scratch: scratch,
                reusedSelection: try reuse(for: layer.index, isFull: isFull),
                position: position, activeExperts: activeExperts)
            if isFull { lastSelection = (layer.index, result.selection) }
            noteRouting(layer.index, result.routing)
        }
        if let first = streamedLayers.first {
            speculateExperts(layer: first.tensors.index)
        }
        for (rank, streamedLayer) in streamedLayers.enumerated() {
            let index = streamedLayer.tensors.index
            let waitStart = Date()
            let buffers = try streamer!.wait(for: index)
            counters.layerStallSeconds += Date().timeIntervalSince(waitStart)
            counters.layerBytes += UInt64(streamedLayer.bigTensorBytes)
            if rank + prefetchDepth < streamedLayers.count {
                streamer!.prefetch(
                    streamedLayers[rank + prefetchDepth].tensors)
            }
            if rank + 1 < streamedLayers.count {
                speculateExperts(
                    layer: streamedLayers[rank + 1].tensors.index)
            }
            let isFull = streamedLayer.proj != nil
            let weights = GLM52ResidentDecodeWeights(
                geometry: geometry,
                attnNorm: streamedLayer.attnNorm, qA: buffers.qA,
                qANorm: streamedLayer.qANorm, qB: buffers.qB,
                kvA: buffers.kvA, kvANorm: streamedLayer.kvANorm,
                keyB: buffers.keyB, valueB: buffers.valueB,
                attnOutput: buffers.attnOutput,
                indexer: isFull ? GLM52ResidentDecodeWeights.ResidentIndexer(
                    key: buffers.indexerKey,
                    keyNorm: streamedLayer.indexerKeyNorm!,
                    keyNormBias: streamedLayer.indexerKeyNormBias!,
                    queryB: buffers.indexerQueryB,
                    proj: streamedLayer.proj!) : nil,
                types: streamedLayer.weightTypes)
            let ffn = GLM52ResidentFFN(
                ffnNorm: streamedLayer.ffnNorm,
                kind: .sparse(
                    routerRows: streamedLayer.routerRows,
                    routerBias: streamedLayer.routerBias,
                    sharedGate: buffers.sharedGate,
                    sharedUp: buffers.sharedUp,
                    sharedDown: buffers.sharedDown,
                    expertProvider: { [provider = streamedLayer.provider] in
                        try provider.expert($0)
                    }))
            ffn.stagedSelection = stagedFetch[index]
            ffn.sharedWeightTypes = (streamedLayer.tensors.sharedGate.type,
                                     streamedLayer.tensors.sharedDown.type)
            let result = try runtime.glm52ChainedDecodeLayer(
                weights: weights, ffn: ffn, caches: streamedLayer.caches,
                scratch: scratch,
                reusedSelection: try reuse(for: index, isFull: isFull),
                position: position, activeExperts: activeExperts)
            if isFull { lastSelection = (index, result.selection) }
            noteRouting(index, result.routing)
        }
        position += 1
        return try runtime.glm52ResidentLogits(
            outputHead: head,
            hidden: scratch.readHidden(count: embeddingWidth))
    }

    /// Remember this token's routed selection (clamped to the experts the
    /// FFN actually runs) as the next token's speculative guess.
    private func noteRouting(_ index: Int, _ routing: GLM52RouterOutput?) {
        guard let routing else { return }
        let used = min(routing.selected.count,
                       max(1, activeExperts ?? routing.selected.count))
        lastRouted[index] = routing.selected.prefix(used)
            .map { UInt32(bitPattern: $0) }
    }

    /// Feed a whole prompt; returns the logits after the final prompt token
    /// (the distribution of the first generated token). LAYER-MAJOR batch:
    /// every layer's weights are visited ONCE for the whole prompt — with
    /// streaming that turns per-token weight reads into per-prompt reads —
    /// while each (layer, token) cell runs the exact same chained kernels in
    /// the exact same causal order as the token-by-token path, so the two
    /// are numerically identical by construction (the integration suite
    /// pins that equivalence on real weights).
    public func prefill(_ tokens: [Int32]) throws -> [Float] {
        guard !tokens.isEmpty else {
            throw MetalError.unsupported(
                "GLM 5.2 prefill requires at least one token")
        }
        if tokens.count == 1 { return try forwardNext(tokens[0]) }

        var hiddens: [[Float]] = try tokens.map { try embeddingRow($0) }
        let basePosition = position
        counters.tokens += tokens.count
        var lastSelections: [(source: Int, rows: [UInt32])?] =
            Array(repeating: nil, count: tokens.count)

        func sweep(index: Int, isFull: Bool,
                   weights: GLM52ResidentDecodeWeights,
                   ffn: GLM52ResidentFFN,
                   caches: GLM52ResidentDecodeCaches) throws {
            for t in 0..<tokens.count {
                scratch.loadHidden(hiddens[t])
                let reused: [UInt32]?
                if isFull {
                    reused = nil
                } else {
                    guard let source = GLM52IndexSharePolicy
                              .selectionSourceLayer(for: index),
                          let last = lastSelections[t],
                          last.source == source else {
                        throw MetalError.unsupported(
                            "GLM 5.2 IndexShare layer \(index) has no "
                            + "selection from its source layer")
                    }
                    reused = last.rows
                }
                let result = try runtime.glm52ChainedDecodeLayer(
                    weights: weights, ffn: ffn, caches: caches,
                    scratch: scratch, reusedSelection: reused,
                    position: basePosition + t,
                    activeExperts: activeExperts)
                if isFull {
                    lastSelections[t] = (index, result.selection)
                }
                noteRouting(index, result.routing)
                hiddens[t] = scratch.readHidden(count: embeddingWidth)
            }
        }

        if let streamer {
            for ahead in 0..<min(prefetchDepth, streamedLayers.count) {
                streamer.prefetch(streamedLayers[ahead].tensors)
            }
        }
        for layer in stack {
            try sweep(index: layer.index,
                      isFull: GLM52IndexSharePolicy.isFullIndexerLayer(
                          layer.index),
                      weights: layer.weights, ffn: layer.ffn,
                      caches: layer.caches)
        }
        for (rank, streamedLayer) in streamedLayers.enumerated() {
            let index = streamedLayer.tensors.index
            let waitStart = Date()
            let buffers = try streamer!.wait(for: index)
            counters.layerStallSeconds += Date().timeIntervalSince(waitStart)
            counters.layerBytes += UInt64(streamedLayer.bigTensorBytes)
            if rank + prefetchDepth < streamedLayers.count {
                streamer!.prefetch(
                    streamedLayers[rank + prefetchDepth].tensors)
            }
            let isFull = streamedLayer.proj != nil
            let weights = GLM52ResidentDecodeWeights(
                geometry: geometry,
                attnNorm: streamedLayer.attnNorm, qA: buffers.qA,
                qANorm: streamedLayer.qANorm, qB: buffers.qB,
                kvA: buffers.kvA, kvANorm: streamedLayer.kvANorm,
                keyB: buffers.keyB, valueB: buffers.valueB,
                attnOutput: buffers.attnOutput,
                indexer: isFull ? GLM52ResidentDecodeWeights.ResidentIndexer(
                    key: buffers.indexerKey,
                    keyNorm: streamedLayer.indexerKeyNorm!,
                    keyNormBias: streamedLayer.indexerKeyNormBias!,
                    queryB: buffers.indexerQueryB,
                    proj: streamedLayer.proj!) : nil,
                types: streamedLayer.weightTypes)
            let ffn = GLM52ResidentFFN(
                ffnNorm: streamedLayer.ffnNorm,
                kind: .sparse(
                    routerRows: streamedLayer.routerRows,
                    routerBias: streamedLayer.routerBias,
                    sharedGate: buffers.sharedGate,
                    sharedUp: buffers.sharedUp,
                    sharedDown: buffers.sharedDown,
                    expertProvider: { [provider = streamedLayer.provider] in
                        try provider.expert($0)
                    }))
            ffn.stagedSelection = stagedFetch[index]
            ffn.sharedWeightTypes = (streamedLayer.tensors.sharedGate.type,
                                     streamedLayer.tensors.sharedDown.type)
            try sweep(index: index, isFull: isFull, weights: weights,
                      ffn: ffn, caches: streamedLayer.caches)
        }
        position = basePosition + tokens.count
        return try runtime.glm52ResidentLogits(
            outputHead: head, hidden: hiddens[tokens.count - 1])
    }

    /// Prefill plus greedy decode. Returns only the generated tokens (the
    /// end token, when hit, is included).
    public func generateGreedy(prompt: [Int32],
                               maxNewTokens: Int,
                               endTokens: Set<Int32>) throws -> [Int32] {
        let logits = try prefill(prompt)
        return try GLM52GreedyDecoding.generate(
            logitsAfterPrompt: logits, maxNewTokens: maxNewTokens,
            endTokens: endTokens) { try self.forwardNext($0) }
    }

    // MARK: - Streaming telemetry

    /// Counters since the last reset. Prefill and decode both accumulate;
    /// reset between the phases to report them apart.
    public func streamingStats() -> GLM52StreamingStats {
        let arenaStats = arena?.statsSnapshot()
            ?? GLM52ExpertArena.Stats()
        return GLM52StreamingStats(
            tokens: counters.tokens,
            layerBytes: counters.layerBytes,
            layerStallSeconds: counters.layerStallSeconds,
            expertBytes: arenaStats.readBytes,
            expertStallSeconds: counters.expertStallSeconds,
            expertHitBytes: arenaStats.hitBytes,
            expertSpeculativeBytes: arenaStats.speculativeBytes)
    }

    public func resetStreamingStats() {
        counters.reset()
        arena?.resetStats()
    }

    /// One-line human report of where the token time goes on the SSD path.
    /// Layer "stallo" is the time decode WAITED on the double-buffered
    /// prefetch (0 = perfect overlap, its GB/s can legitimately exceed the
    /// SSD); expert stallo is fully synchronous, so its GB/s is the true
    /// effective read throughput of the staged expert path. "riuso" are
    /// bytes served from the arena without touching the SSD; "speculativi"
    /// were read in the background during GPU compute.
    public func streamingReport() -> String {
        let stats = streamingStats()
        guard stats.tokens > 0,
              stats.layerBytes + stats.expertBytes
                  + stats.expertHitBytes > 0 else {
            return "streaming: nessun byte streamato (stack residente)"
        }
        func gib(_ bytes: UInt64) -> Double {
            Double(bytes) / 1_073_741_824
        }
        func rate(_ bytes: UInt64, _ seconds: Double) -> Double {
            seconds > 0.001 ? Double(bytes) / 1e9 / seconds : 0
        }
        return String(
            format: "streaming: %d token · layer %.1f GiB "
                + "(%.2f GiB/token, stallo %.1fs, %.1f GB/s) · esperti "
                + "%.2f GiB letti (stallo %.1fs, %.1f GB/s) · riuso arena "
                + "%.2f GiB · speculativi %.2f GiB",
            stats.tokens, gib(stats.layerBytes),
            gib(stats.layerBytes) / Double(stats.tokens),
            stats.layerStallSeconds,
            rate(stats.layerBytes, stats.layerStallSeconds),
            gib(stats.expertBytes),
            stats.expertStallSeconds,
            rate(stats.expertBytes, stats.expertStallSeconds),
            gib(stats.expertHitBytes),
            gib(stats.expertSpeculativeBytes))
    }
}

/// Snapshot of the engine's streaming counters.
public struct GLM52StreamingStats: Sendable {
    public let tokens: Int
    public let layerBytes: UInt64
    public let layerStallSeconds: Double
    /// Expert bytes read from SSD by synchronous staging (misses).
    public let expertBytes: UInt64
    public let expertStallSeconds: Double
    /// Expert bytes served straight from the arena (no SSD touch).
    public let expertHitBytes: UInt64
    /// Expert bytes read in the background by the speculative warm-up.
    public let expertSpeculativeBytes: UInt64
}

/// Mutable accumulator behind the snapshot — touched only from the decode
/// thread (forwardNext/prefill and the staged fetch they invoke run there);
/// the expert BYTE counters live in the arena, which has its own lock.
final class GLM52StreamingCounters {
    var tokens = 0
    var layerBytes: UInt64 = 0
    var layerStallSeconds = 0.0
    var expertStallSeconds = 0.0

    func reset() {
        tokens = 0
        layerBytes = 0
        layerStallSeconds = 0
        expertStallSeconds = 0
    }
}
