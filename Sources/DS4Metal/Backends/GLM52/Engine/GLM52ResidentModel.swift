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
// Loading is scoped on purpose: `layerCount` may truncate the stack (the
// leading dense layers are Q8_0 and run today; sparse layers additionally
// require a routed-expert type with a validated kernel — the published
// IQ2_XXS GGUF is refused at load until that kernel tranche lands).

public struct GLM52ResidentModelOptions: Sendable {
    /// Layers to load from the front of the stack; nil loads every
    /// autoregressive layer (78 — the nextn block is never executed).
    public var layerCount: Int?
    /// Compact-cache capacity in tokens (per layer).
    public var cacheCapacity: Int
    /// Expert slots per sparse layer's streaming cache.
    public var expertSlotCount: Int

    public init(layerCount: Int? = nil,
                cacheCapacity: Int = 4_096,
                expertSlotCount: Int = 16) {
        self.layerCount = layerCount
        self.cacheCapacity = cacheCapacity
        self.expertSlotCount = expertSlotCount
    }
}

public final class GLM52ResidentModel {
    public let configuration: GLM52Configuration
    public let loadedLayerCount: Int
    public private(set) var position = 0

    private let runtime: MetalRuntime
    private let reader: GLM52PayloadReader
    private let embedding: GLM52WeightDescriptor
    private let embeddingRowBytes: Int
    private let stack: [GLM52ResidentStackLayer]
    private let head: GLM52ResidentOutputHead
    private let providers: [GLM52StreamedExpertProvider]
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

        func f32(_ descriptor: GLM52WeightDescriptor) throws -> [Float] {
            guard descriptor.type == GLM52TensorSchema.f32 else {
                throw MetalError.unsupported(
                    "\(descriptor.name) must be F32 for the engine")
            }
            let raw = try reader.bytes(of: descriptor)
            return raw.withUnsafeBytes { Array($0.bindMemory(to: Float.self)) }
        }

        let geometry = GLM52DecodeGeometry.v5_2
        var layers: [GLM52ResidentStackLayer] = []
        var providers: [GLM52StreamedExpertProvider] = []
        layers.reserveCapacity(count)
        for index in 0..<count {
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
                providers.append(provider)
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
        self.providers = providers
        head = try GLM52ResidentOutputHead(
            runtime: runtime, geometry: geometry,
            outputNorm: try f32(map.global(.outputNorm)),
            outputHead: try reader.bytes(of: map.global(.output)),
            vocabularySize: vocabulary)
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

    /// Advance the model by one token; returns the full logits row.
    public func forwardNext(_ token: Int32) throws -> [Float] {
        let embedded = try embeddingRow(token)
        let result = try runtime.glm52ResidentDecodeForward(
            layers: stack, outputHead: head, embeddedToken: embedded,
            position: position)
        position += 1
        return result.logits
    }

    /// Feed a whole prompt token by token; returns the logits after the
    /// final prompt token (the distribution of the first generated token).
    public func prefill(_ tokens: [Int32]) throws -> [Float] {
        guard !tokens.isEmpty else {
            throw MetalError.unsupported(
                "GLM 5.2 prefill requires at least one token")
        }
        var logits: [Float] = []
        for token in tokens {
            logits = try forwardNext(token)
        }
        return logits
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
