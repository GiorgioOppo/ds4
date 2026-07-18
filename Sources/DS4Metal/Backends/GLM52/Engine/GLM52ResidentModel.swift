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

    public init(layerCount: Int? = nil,
                cacheCapacity: Int = 4_096,
                expertSlotCount: Int = 16,
                residentLayerCount: Int? = nil,
                activeExperts: Int? = nil) {
        self.layerCount = layerCount
        self.cacheCapacity = cacheCapacity
        self.expertSlotCount = expertSlotCount
        self.residentLayerCount = residentLayerCount
        self.activeExperts = activeExperts
    }
}

public final class GLM52ResidentModel {
    public let configuration: GLM52Configuration
    public let loadedLayerCount: Int
    public private(set) var position = 0

    private struct StreamedLayer {
        let tensors: GLM52StreamedLayerTensors
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
                let provider = try GLM52StreamedExpertProvider(
                    reader: reader, weightMap: map, layer: index,
                    slotCount: options.expertSlotCount)
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
        // through the double-buffered staging slots. Small per-layer state
        // (norms, router, proj, caches) stays resident.
        LoadProgress.shared.set(0.82, "GLM: stato streaming per layer")
        var streamed: [StreamedLayer] = []
        for index in residentCount..<count {
            let isFull = GLM52IndexSharePolicy.isFullIndexerLayer(
                index, shape: shape)
            let provider = try GLM52StreamedExpertProvider(
                reader: reader, weightMap: map, layer: index,
                slotCount: options.expertSlotCount,
                bundleDirectory: ProcessInfo.processInfo
                    .environment["DS4_GLM_BUNDLE_DIR"])
            providers[index] = provider
            streamed.append(StreamedLayer(
                tensors: try GLM52StreamedLayerTensors(
                    index: index, map: map, fullIndexer: isFull),
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
        // The template always sizes the indexer slots (every layer stores
        // indexer tensors in the schema, so the descriptors exist even for
        // IndexShare layers).
        streamer = streamed.isEmpty ? nil : try GLM52LayerStreamer(
            runtime: runtime, reader: reader,
            template: try GLM52StreamedLayerTensors(
                index: streamed[0].tensors.index, map: map,
                fullIndexer: true))

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
    /// layer's compute with the SSD prefetch of the next one, and after the
    /// step the expert lookahead warms the slot caches with this token's
    /// selections (consecutive tokens reselect often).
    public func forwardNext(_ token: Int32) throws -> [Float] {
        let embedded = try embeddingRow(token)
        scratch.loadHidden(embedded)
        var lastSelection: (source: Int, rows: [UInt32])?
        var routings: [Int: [Int32]] = [:]

        if let streamer, let first = streamedLayers.first {
            streamer.prefetch(first.tensors)
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
            let result = try runtime.glm52ChainedDecodeLayer(
                weights: layer.weights, ffn: layer.ffn, caches: layer.caches,
                scratch: scratch,
                reusedSelection: try reuse(for: layer.index, isFull: isFull),
                position: position, activeExperts: activeExperts)
            if isFull { lastSelection = (layer.index, result.selection) }
            if let routing = result.routing {
                routings[layer.index] = routing.selected
            }
        }
        for (rank, streamedLayer) in streamedLayers.enumerated() {
            let index = streamedLayer.tensors.index
            let buffers = try streamer!.wait(for: index)
            if rank + 1 < streamedLayers.count {
                streamer!.prefetch(streamedLayers[rank + 1].tensors)
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
                    proj: streamedLayer.proj!) : nil)
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
            let result = try runtime.glm52ChainedDecodeLayer(
                weights: weights, ffn: ffn, caches: streamedLayer.caches,
                scratch: scratch,
                reusedSelection: try reuse(for: index, isFull: isFull),
                position: position, activeExperts: activeExperts)
            if isFull { lastSelection = (index, result.selection) }
            if let routing = result.routing {
                routings[index] = routing.selected
            }
        }
        position += 1
        for (index, selected) in routings {
            providers[index]?.prefetch(selected.map { UInt32(bitPattern: $0) })
        }
        return try runtime.glm52ResidentLogits(
            outputHead: head,
            hidden: scratch.readHidden(count: embeddingWidth))
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
                hiddens[t] = scratch.readHidden(count: embeddingWidth)
            }
        }

        if let streamer, let first = streamedLayers.first {
            streamer.prefetch(first.tensors)
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
            let buffers = try streamer!.wait(for: index)
            if rank + 1 < streamedLayers.count {
                streamer!.prefetch(streamedLayers[rank + 1].tensors)
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
                    proj: streamedLayer.proj!) : nil)
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
}
