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
    /// per resident sparse layer. Floor: the three dense layers.
    /// MISURATO (M1 Pro 16 GB): un budget PICCOLO (8 layer su 75) è
    /// controproducente — sotto pressione il sistema comprime/pagina quei
    /// "residenti" e ogni commit ne ripaga la residency al driver
    /// (~+750 ms/token di host), mentre lo stream li avrebbe serviti in
    /// overlap quasi gratis (3 → 11 residenti: 4.2 → 5.0 s/token). Sotto un
    /// quarto di copertura della coda streamata si resta quindi al floor
    /// dense: la residenza paga solo quando copre una quota sostanziale.
    public static func adaptiveResidentLayerCount(
        totalLayers: Int = 78) -> Int {
        let physical = Double(ProcessInfo.processInfo.physicalMemory)
        let budget = physical * 0.5 - 6.0 * 1_073_741_824
        let perLayer = 230.0 * 1_048_576
        let extra = budget > 0 ? Int(budget / perLayer) : 0
        guard extra >= (totalLayers - 3) / 4 else { return 3 }
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
        /// Buffer condivisi, upload una volta al load: il router fuso li
        /// legge su GPU, il fallback CPU dagli stessi byte.
        let routerRows: MTLBuffer
        let routerBias: MTLBuffer
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
    /// Profilo per-fase in stile DeepSeek: STESSA struttura e stesso report
    /// del decoder DeepSeek (DecodeProfile), alimentato dai confini di commit
    /// già sincroni del chained decode. Azzerato con resetStreamingStats().
    public private(set) var profile = DecodeProfile()
    /// Quota GPU delle fasi (dai timestamp dei commit): wall − gpu = host.
    private var gpuRouteS = 0.0
    private var gpuExpertsS = 0.0
    private var gpuDenseS = 0.0
    private var gpuHeadS = 0.0
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
    /// guess. OPT-IN (DS4_GLM_SPEC_EXPERTS): "1" specula l'INTERA selezione
    /// (storico — misurato net-zero su SSD saturo: ~40-46% hit ma il traffico
    /// extra ruba banda alle letture demand e fa stallare il prefetch layer);
    /// un valore N >= 2 specula solo i TOP-N esperti per peso di routing —
    /// i più stabili tra token consecutivi — dimezzando o più il traffico
    /// speculativo a parità della parte di hit che conta.
    private var lastRouted: [Int: [UInt32]] = [:]
    private let speculationTop: Int? = {
        guard let raw = ProcessInfo.processInfo
            .environment["DS4_GLM_SPEC_EXPERTS"], let n = Int(raw), n >= 1
        else { return nil }
        return n
    }()
    private var speculationEnabled: Bool { speculationTop != nil }
    /// Fusione dei commit nel decode (DS4_GLM_FUSE=0 per disattivare): FFN
    /// del layer N e trunk del layer N+1 in UN command buffer — una attesa
    /// sincrona per layer invece di due (~154 → ~78 commit/token). Esclusa
    /// quando la speculazione esperti è attiva: i suoi fill potrebbero
    /// sfrattare slot arena ancora referenziati dal FFN non committato.
    private lazy var fuseCommits: Bool = speculationTop == nil
        && ProcessInfo.processInfo.environment["DS4_GLM_FUSE"] != "0"
    private let speculationQueue = DispatchQueue(
        label: "glm52.expert.speculation", qos: .userInitiated,
        attributes: .concurrent)

    public init(runtime: MetalRuntime,
                path: String,
                options: GLM52ResidentModelOptions = .init()) throws {
        // Latch dei knob di dispatch per QUESTO load: un setenv della GUI
        // (o dell'auto-tune) prima del reload ha effetto qui.
        GLM52DispatchKnobs.refresh()
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
        let layerQ4Directory = ProcessInfo.processInfo
            .environment["DS4_GLM_LAYERQ4_DIR"] ?? (path + ".glm-layers-q4")
        // DS4_GLM_LAYERQ4=0: ignora i TENSORI Q4 del sidecar (lossy) e
        // streamma i layer Q8 dal GGUF; la sezione esperti unificata
        // (lossless) resta comunque in uso.
        let useQ4Tensors = ProcessInfo.processInfo
            .environment["DS4_GLM_LAYERQ4"] != "0"
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
                // never fits): the unified sidecar's embedded records win,
                // then the legacy bundle, then the plain GGUF.
                let routed = try map.routedExperts(layer: index)
                let sidecar = try GLM52LayerQuantSidecar.open(
                    directory: layerQ4Directory, layer: index,
                    source: try GLM52StreamedLayerTensors(
                        index: index, map: map, fullIndexer: isFull),
                    routed: routed,
                    expertCount: Int(shape.nExpert),
                    sourcePath: path,
                    sourceFileSize: reader.fileSize)
                let provider: GLM52StreamedExpertProvider
                if let view = sidecar?.expertView {
                    provider = try GLM52StreamedExpertProvider(
                        reader: reader, layer: index, weights: routed,
                        slotCount: options.expertSlotCount, bundle: view)
                } else {
                    provider = try GLM52StreamedExpertProvider(
                        reader: reader, weightMap: map, layer: index,
                        slotCount: options.expertSlotCount,
                        bundleDirectory: bundleDirectory)
                }
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
        var sidecarReaders: [Int: GLM52PayloadReader] = [:]
        var streamed: [StreamedLayer] = []
        var unifiedExpertLayers = 0
        for index in residentCount..<count {
            let isFull = GLM52IndexSharePolicy.isFullIndexerLayer(
                index, shape: shape)
            let ggufTensors = try GLM52StreamedLayerTensors(
                index: index, map: map, fullIndexer: isFull)
            let routed = try map.routedExperts(layer: index)
            let sidecar = try GLM52LayerQuantSidecar.open(
                directory: layerQ4Directory, layer: index,
                source: ggufTensors, routed: routed,
                expertCount: Int(shape.nExpert),
                sourcePath: path,
                sourceFileSize: reader.fileSize)
            let provider: GLM52StreamedExpertProvider
            if let view = sidecar?.expertView {
                unifiedExpertLayers += 1
                provider = try GLM52StreamedExpertProvider(
                    reader: reader, layer: index, weights: routed,
                    slotCount: options.expertSlotCount, bundle: view)
            } else {
                provider = try GLM52StreamedExpertProvider(
                    reader: reader, weightMap: map, layer: index,
                    slotCount: options.expertSlotCount,
                    bundleDirectory: bundleDirectory)
            }
            providers[index] = provider
            if let sidecar, useQ4Tensors {
                sidecarReaders[index] = sidecar.reader
            }
            let tensors = (useQ4Tensors ? sidecar?.tensors : nil)
                ?? ggufTensors
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
                routerRows: try runtime.glm52GraphBuffer(
                    try f32(map.layer(index, .router))),
                routerBias: try runtime.glm52GraphBuffer(
                    try f32(map.layer(index, .routerBias))),
                provider: provider,
                caches: try GLM52ResidentDecodeCaches(
                    runtime: runtime, geometry: geometry,
                    capacity: options.cacheCapacity, fullIndexer: isFull)))
        }
        streamedLayers = streamed
        if !sidecarReaders.isEmpty {
            DS4Log.info("glm", "sidecar Q4 layer attivo su "
                + "\(sidecarReaders.count)/\(streamed.count) layer "
                + "streamati (\(unifiedExpertLayers) con esperti unificati)"
                + " — dir: \(layerQ4Directory)")
        } else if !streamed.isEmpty {
            // Come il bundle DeepSeek: dire DOVE si è cercato risolve dal
            // solo log i misteri "il file c'è!" da path/nome sbagliato.
            DS4Log.info("glm", "nessun sidecar Q4 valido in: \(layerQ4Directory)"
                + " — layer Q8 dal GGUF")
        }
        // The template always sizes the indexer slots (every layer stores
        // indexer tensors in the schema, so the descriptors exist even for
        // IndexShare layers) and always uses the FULL Q8_0 GGUF sizes —
        // sidecar tensors are smaller and share the same slots.
        // Con la fusione dei commit serve uno slot in più e profondità
        // slots-2: il refill non deve MAI toccare lo slot il cui FFN è nel
        // carry non ancora committato (il fill CPU lo sovrascriverebbe
        // mentre la GPU lo legge).
        let speculating = ProcessInfo.processInfo
            .environment["DS4_GLM_SPEC_EXPERTS"].flatMap(Int.init)
            .map { $0 >= 1 } ?? false
        let fusing = !speculating && ProcessInfo.processInfo
            .environment["DS4_GLM_FUSE"] != "0"
        let slotCount = fusing
            ? max(4, options.streamSlotCount + 1) : options.streamSlotCount
        streamer = streamed.isEmpty ? nil : try GLM52LayerStreamer(
            runtime: runtime, reader: reader,
            template: try GLM52StreamedLayerTensors(
                index: streamed[0].tensors.index, map: map,
                fullIndexer: true),
            slotCount: slotCount,
            sidecarReaders: sidecarReaders)
        prefetchDepth = max(1, slotCount - (fusing ? 2 : 1))

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
        // Knob attivi nel log a OGNI init, come il motore DeepSeek: "l'app
        // vede davvero le env var?" deve essere rispondibile dal solo log.
        let knobs = ["DS4_GLM_MTLIO", "DS4_GLM_ACTIVE_EXPERTS",
                     "DS4_GLM_RESIDENT_LAYERS", "DS4_GLM_FUSE",
                     "DS4_GLM_READ_SPLIT", "DS4_GLM_SPEC_EXPERTS",
                     "DS4_GLM_SPEC_K", "DS4_GLM_EXPERT_ARENA",
                     "DS4_GLM_STREAM_SLOTS", "DS4_GLM_EXPERT_SLOTS",
                     "DS4_GLM_SG", "DS4_GLM_NSG", "DS4_GLM_NOCACHE",
                     "DS4_GLM_MLOCK", "DS4_GLM_MOE_BATCH",
                     "DS4_GLM_GPU_ROUTER", "DS4_GLM_LAYERQ4",
                     "DS4_GLM_USAGE_FILE"]
        let env = ProcessInfo.processInfo.environment
        DS4Log.info("glm", "knob " + knobs
            .map { "\($0)=\(env[$0] ?? "·")" }.joined(separator: "  "))
        DS4Log.info("glm", "config: residenti=\(residentCount)/\(count)  "
            + "fusione=\(fusing ? "on" : "off")  slot stream=\(slotCount)  "
            + "esperti attivi=\(activeExperts.map(String.init) ?? "8")")
        loadUsageProfile()
    }

    /// Kick a background warm-up of `layer`'s arena slots with the LAST
    /// token's routing for that layer (consecutive tokens reselect 25-40%
    /// of experts): the read overlaps the GPU attention instead of
    /// stalling the FFN. Best-effort by design.
    private func speculateExperts(layer index: Int) {
        guard let top = speculationTop, let arena,
              stagedFetch[index] != nil,
              let ids = lastRouted[index],
              let provider = providers[index] else { return }
        // lastRouted è in ordine di peso del router: con DS4_GLM_SPEC_EXPERTS
        // >= 2 il prefisso top-N è la parte della selezione più probabile
        // da rivedere al token successivo. "1" = selezione intera (storico).
        let guess = top >= 2 ? Array(ids.prefix(top)) : ids
        let recordBytes = provider.recordBytes
        speculationQueue.async { [arena] in
            arena.speculate(layer: index, ids: guess,
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
        // Anche lo streamer torna vergine: un passo abortito tra prefetch
        // e wait lascerebbe slot stantii nella FIFO.
        streamer?.reset()
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
    /// Corpo comune del forward di un token: embed, layer residenti e
    /// streamati, flush del carry — lascia lo hidden in scratch. I due
    /// wrapper sotto differiscono solo per il head (logits interi o argmax
    /// greedy sul device).
    private func advance(_ token: Int32) throws {
        let tEmbed = Date()
        let embedded = try embeddingRow(token)
        scratch.loadHidden(embedded)
        profile.embedS += Date().timeIntervalSince(tEmbed)
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

        let fuse = fuseCommits
        var carry: MTLCommandBuffer?
        for layer in stack {
            let isFull = GLM52IndexSharePolicy.isFullIndexerLayer(layer.index)
            speculateExperts(layer: layer.index)
            let result = try runtime.glm52ChainedDecodeLayer(
                weights: layer.weights, ffn: layer.ffn, caches: layer.caches,
                scratch: scratch,
                reusedSelection: try reuse(for: layer.index, isFull: isFull),
                position: position, activeExperts: activeExperts,
                fusedDecode: fuse, carry: &carry)
            accumulate(result.phases)
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
            let stall = Date().timeIntervalSince(waitStart)
            counters.layerStallSeconds += stall
            counters.layerBytes += UInt64(streamedLayer.bigTensorBytes)
            profile.layerStreamS += stall
            profile.layerStreamBytes += streamedLayer.bigTensorBytes
            // Il refill è sicuro anche col carry in volo: con la fusione la
            // profondità è slots-2, quindi lo slot bersaglio non è mai
            // quello referenziato dal FFN non ancora committato.
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
                position: position, activeExperts: activeExperts,
                fusedDecode: fuse, carry: &carry)
            accumulate(result.phases)
            if isFull { lastSelection = (index, result.selection) }
            noteRouting(index, result.routing)
        }
        // Flush del carry: il FFN dell'ultimo layer paga qui la sua unica
        // attesa, prima del head — attribuito alla fase experts.
        if let pending = carry {
            let tFlush = Date()
            let gpuFlush = GLM52GraphTelemetry.gpuSeconds
            try runtime.glm52GraphCommit(pending)
            profile.expertsS += Date().timeIntervalSince(tFlush)
            gpuExpertsS += GLM52GraphTelemetry.gpuSeconds - gpuFlush
            carry = nil
        }
        position += 1
    }

    /// Advance the model by one token; returns the full logits row.
    public func forwardNext(_ token: Int32) throws -> [Float] {
        try advance(token)
        let tHead = Date()
        let gpuHead = GLM52GraphTelemetry.gpuSeconds
        defer {
            profile.headS += Date().timeIntervalSince(tHead)
            gpuHeadS += GLM52GraphTelemetry.gpuSeconds - gpuHead
            profile.forwards += 1
        }
        return try runtime.glm52ResidentLogits(
            outputHead: head,
            hidden: scratch.readHidden(count: embeddingWidth))
    }

    /// Variante GREEDY: stesso forward, ma il head fa norm + matvec +
    /// argmax SUL DEVICE — readback di 4 byte invece dei ~600 KB di
    /// logits. Pareggi all'indice più basso, come l'argmax CPU: il decode
    /// greedy resta deterministico e identico.
    public func forwardNextGreedy(_ token: Int32) throws -> Int32 {
        try advance(token)
        let tHead = Date()
        let gpuHead = GLM52GraphTelemetry.gpuSeconds
        defer {
            profile.headS += Date().timeIntervalSince(tHead)
            gpuHeadS += GLM52GraphTelemetry.gpuSeconds - gpuHead
            profile.forwards += 1
        }
        return try runtime.glm52ResidentGreedyToken(
            outputHead: head,
            hidden: scratch.readHidden(count: embeddingWidth))
    }

    /// Somma le fasi di un layer chained nel profilo stile DeepSeek: i layer
    /// dense (commit unico, non splittabile) finiscono in "layer (alt)".
    private func accumulate(_ phases: GLM52LayerPhases) {
        profile.routeS += phases.routeS
        profile.gatherS += phases.gatherS
        profile.expertsS += phases.expertsS
        profile.layerOtherS += phases.denseS
        profile.layers += 1
        gpuRouteS += phases.routeGpuS
        gpuExpertsS += phases.expertsGpuS
        gpuDenseS += phases.denseGpuS
    }

    /// Remember this token's routed selection (clamped to the experts the
    /// FFN actually runs) as the next token's speculative guess, and feed
    /// the persistent usage profile.
    private func noteRouting(_ index: Int, _ routing: GLM52RouterOutput?) {
        guard let routing else { return }
        let used = min(routing.selected.count,
                       max(1, activeExperts ?? routing.selected.count))
        let selected = routing.selected.prefix(used)
            .map { UInt32(bitPattern: $0) }
        lastRouted[index] = selected
        var counts = usageCounts[index] ?? [UInt32](
            repeating: 0, count: GLM52RouterReference.expertCount)
        for id in selected where Int(id) < counts.count {
            counts[Int(id)] += 1
        }
        usageCounts[index] = counts
        usageDirty = true
    }

    // MARK: - Persistent expert-usage profile

    private var usageCounts: [Int: [UInt32]] = [:]
    private var usageDirty = false

    private var usageProfilePath: String {
        ProcessInfo.processInfo.environment["DS4_GLM_USAGE_FILE"]
            ?? (reader.path + ".glm-usage.json")
    }

    /// Persist the per-layer expert selection counts (the GLM usage
    /// imatrix): future loads start with the routing history instead of a
    /// blank slate. "off" disables.
    public func saveUsageProfile() {
        let path = usageProfilePath
        guard path != "off", usageDirty else { return }
        var object: [String: [UInt32]] = [:]
        for (layer, counts) in usageCounts {
            object[String(layer)] = counts
        }
        if let data = try? JSONEncoder().encode(object) {
            try? data.write(to: URL(fileURLWithPath: path))
            usageDirty = false
        }
    }

    private func loadUsageProfile() {
        let path = usageProfilePath
        guard path != "off",
              let data = FileManager.default.contents(atPath: path),
              let object = try? JSONDecoder().decode(
                  [String: [UInt32]].self, from: data) else { return }
        for (key, counts) in object {
            if let layer = Int(key) { usageCounts[layer] = counts }
        }
        let routes = usageCounts.values
            .reduce(UInt64(0)) { total, counts in
                total + counts.reduce(UInt64(0)) { $0 + UInt64($1) }
            }
        DS4Log.info("glm", "usage profile caricato (\(routes) route)")
    }

    // MARK: - Disk KV checkpoints

    /// Public: the disk-KV store scans entry headers in this wire format.
    public static let kvMagic: UInt32 = 0x3156_4B47   // "GKV1"

    private var allCaches: [GLM52ResidentDecodeCaches] {
        stack.map(\.caches) + streamedLayers.map(\.caches)
    }

    /// Persist the LIVE caches plus the tokens they hold (~96 KB/token).
    /// Atomic: `.part` then rename; identity = GGUF size + layer count.
    public func saveKVCheckpoint(to url: URL, tokens: [Int32]) throws {
        guard tokens.count == position, position > 0 else {
            throw MetalError.unsupported(
                "GLM 5.2 disk-KV: i token (\(tokens.count)) non combaciano "
                + "con la position (\(position))")
        }
        var data = Data()
        func u32(_ value: UInt32) {
            withUnsafeBytes(of: value.littleEndian) { data.append(contentsOf: $0) }
        }
        func u64(_ value: UInt64) {
            withUnsafeBytes(of: value.littleEndian) { data.append(contentsOf: $0) }
        }
        u32(Self.kvMagic); u32(1)
        u64(reader.fileSize)
        u32(UInt32(loadedLayerCount)); u32(UInt32(position))
        tokens.withUnsafeBufferPointer {
            data.append(Data(buffer: $0))
        }
        for caches in allCaches {
            let snapshot = caches.checkpointData()
            data.append(snapshot.compact)
            if let indexer = snapshot.indexer { data.append(indexer) }
        }
        let directory = url.deletingLastPathComponent()
        try FileManager.default.createDirectory(
            at: directory, withIntermediateDirectories: true)
        let part = url.appendingPathExtension("part")
        try data.write(to: part)
        _ = try FileManager.default.replaceItemAt(url, withItemAt: part)
    }

    /// Header-only peek: the tokens a checkpoint holds — nil when absent,
    /// foreign (different GGUF/layer count) or malformed.
    public func peekKVCheckpoint(at url: URL) -> [Int32]? {
        guard let handle = FileHandle(forReadingAtPath: url.path) else {
            return nil
        }
        defer { try? handle.close() }
        guard let header = try? handle.read(upToCount: 24),
              header.count == 24 else { return nil }
        func u32(_ offset: Int) -> UInt32 {
            header.withUnsafeBytes {
                $0.loadUnaligned(fromByteOffset: offset, as: UInt32.self)
            }
        }
        func u64(_ offset: Int) -> UInt64 {
            header.withUnsafeBytes {
                $0.loadUnaligned(fromByteOffset: offset, as: UInt64.self)
            }
        }
        guard u32(0) == Self.kvMagic, u32(4) == 1,
              u64(8) == reader.fileSize,
              u32(16) == UInt32(loadedLayerCount) else { return nil }
        let count = Int(u32(20))
        guard count > 0, count <= 1_000_000,
              let tokenData = try? handle.read(upToCount: count * 4),
              tokenData.count == count * 4 else { return nil }
        return tokenData.withUnsafeBytes {
            Array($0.bindMemory(to: Int32.self))
        }
    }

    /// Full restore of caches + position. Returns the restored tokens; the
    /// caller has already verified they prefix the new conversation.
    @discardableResult
    public func restoreKVCheckpoint(from url: URL) throws -> [Int32] {
        guard let tokens = peekKVCheckpoint(at: url) else {
            throw MetalError.unsupported(
                "GLM 5.2 disk-KV: checkpoint assente o estraneo")
        }
        let count = tokens.count
        guard let handle = FileHandle(forReadingAtPath: url.path) else {
            throw MetalError.unsupported(
                "GLM 5.2 disk-KV: checkpoint non leggibile")
        }
        defer { try? handle.close() }
        try handle.seek(toOffset: UInt64(24 + count * 4))
        for caches in allCaches {
            let compactBytes = count * caches.compactRowBytes
            guard let compactData = try handle.read(
                      upToCount: compactBytes),
                  compactData.count == compactBytes else {
                throw MetalError.unsupported(
                    "GLM 5.2 disk-KV: checkpoint troncato")
            }
            var indexerData: Data?
            if caches.indexerKeys != nil {
                let indexerBytes = count * caches.indexerRowBytes
                guard let data = try handle.read(upToCount: indexerBytes),
                      data.count == indexerBytes else {
                    throw MetalError.unsupported(
                        "GLM 5.2 disk-KV: checkpoint troncato (indexer)")
                }
                indexerData = data
            }
            try caches.restoreCheckpoint(
                compact: compactData, indexer: indexerData, rows: count)
        }
        position = count
        return tokens
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
        let hiddens = try runLayerMajor(tokens)
        let tHead = Date()
        let gpuHead = GLM52GraphTelemetry.gpuSeconds
        defer {
            profile.headS += Date().timeIntervalSince(tHead)
            gpuHeadS += GLM52GraphTelemetry.gpuSeconds - gpuHead
        }
        return try runtime.glm52ResidentLogits(
            outputHead: head, hidden: hiddens[hiddens.count - 1])
    }

    /// Speculative VERIFY: layer-major forward of a window returning the
    /// logits of EVERY position (weights read once for the whole window —
    /// the same amortization that makes speculation pay on streaming).
    public func forwardBatch(_ tokens: [Int32]) throws -> [[Float]] {
        guard !tokens.isEmpty else {
            throw MetalError.unsupported(
                "GLM 5.2 forwardBatch requires at least one token")
        }
        if tokens.count == 1 { return [try forwardNext(tokens[0])] }
        let hiddens = try runLayerMajor(tokens)
        let tHead = Date()
        let gpuHead = GLM52GraphTelemetry.gpuSeconds
        defer {
            profile.headS += Date().timeIntervalSince(tHead)
            gpuHeadS += GLM52GraphTelemetry.gpuSeconds - gpuHead
        }
        return try hiddens.map {
            try runtime.glm52ResidentLogits(outputHead: head, hidden: $0)
        }
    }

    /// Speculative REJECTION: drop the cache rows past `target` and rewind
    /// the position — the next forward overwrites the abandoned rows.
    public func rollback(to target: Int) throws {
        guard target >= 0, target <= position else {
            throw MetalError.unsupported(
                "GLM 5.2 rollback a \(target) fuori da 0...\(position)")
        }
        for caches in allCaches { caches.rollback(to: target) }
        position = target
    }

    private func runLayerMajor(_ tokens: [Int32]) throws -> [[Float]] {
        let tEmbed = Date()
        var hiddens: [[Float]] = try tokens.map { try embeddingRow($0) }
        profile.embedS += Date().timeIntervalSince(tEmbed)
        profile.forwards += tokens.count
        let basePosition = position
        counters.tokens += tokens.count
        var lastSelections: [(source: Int, rows: [UInt32])?] =
            Array(repeating: nil, count: tokens.count)

        // Per-token planes of the two-phase expert path, allocated on the
        // first sparse layer that needs them (100 MB at the 4096-token cap,
        // freed with the prefill).
        let planeStride = embeddingWidth * MemoryLayout<Float>.stride
        var planes: (hidden: MTLBuffer, ffnIn: MTLBuffer)?
        func expertPlanes() throws -> (hidden: MTLBuffer, ffnIn: MTLBuffer) {
            if let planes { return planes }
            guard let hidden = runtime.device.makeBuffer(
                      length: tokens.count * planeStride,
                      options: .storageModeShared),
                  let ffnIn = runtime.device.makeBuffer(
                      length: tokens.count * planeStride,
                      options: .storageModeShared) else {
                throw MetalError.bufferAlloc
            }
            planes = (hidden, ffnIn)
            return (hidden, ffnIn)
        }

        func reusedSelection(index: Int, isFull: Bool, token: Int) throws
            -> [UInt32]? {
            if isFull { return nil }
            guard let source = GLM52IndexSharePolicy
                      .selectionSourceLayer(for: index),
                  let last = lastSelections[token],
                  last.source == source else {
                throw MetalError.unsupported(
                    "GLM 5.2 IndexShare layer \(index) has no "
                    + "selection from its source layer")
            }
            return last.rows
        }

        func sweep(index: Int, isFull: Bool,
                   weights: GLM52ResidentDecodeWeights,
                   ffn: GLM52ResidentFFN,
                   caches: GLM52ResidentDecodeCaches) throws {
            // Prefill layer-major: niente fusione (ogni chiamata committa),
            // il carry resta inerte.
            var noCarry: MTLCommandBuffer?
            // TWO-PHASE expert path for sparse layers on the staged fetch:
            // phase A runs attention+router+shared per token (routed FFN
            // deferred) snapshotting hidden/ffnIn planes; phase B stages
            // each UNIQUE routed expert once for the whole prompt and
            // applies it to every token that selected it. Expert I/O drops
            // from selections×record to unique×record — the difference is
            // the whole game on long prompts.
            if case .sparse = ffn.kind, let stage = stagedFetch[index] {
                let planes = try expertPlanes()
                var users: [UInt32: [(token: Int, weight: Float)]] = [:]
                var order: [UInt32] = []
                for t in 0..<tokens.count {
                    // Stop/disconnessione durante un prefill lungo: la cella
                    // (layer, token) è la granularità giusta (~ms). Chi
                    // chiama risana con resetContext (che drena anche i
                    // prefetch dello streamer).
                    try Task.checkCancellation()
                    scratch.loadHidden(hiddens[t])
                    let result = try runtime.glm52ChainedDecodeLayer(
                        weights: weights, ffn: ffn, caches: caches,
                        scratch: scratch,
                        reusedSelection: try reusedSelection(
                            index: index, isFull: isFull, token: t),
                        position: basePosition + t,
                        activeExperts: activeExperts,
                        deferSparseFFN: true, carry: &noCarry)
                    accumulate(result.phases)
                    if isFull {
                        lastSelections[t] = (index, result.selection)
                    }
                    noteRouting(index, result.routing)
                    if let routing = result.routing {
                        let used = min(routing.selected.count,
                                       max(1, activeExperts
                                               ?? routing.selected.count))
                        for rank in 0..<used {
                            let id = UInt32(
                                bitPattern: routing.selected[rank])
                            if users[id] == nil {
                                users[id] = []
                                order.append(id)
                            }
                            users[id]?.append((t, routing.weights[rank]))
                        }
                    }
                    memcpy(planes.hidden.contents() + t * planeStride,
                           scratch.hidden.contents(), planeStride)
                    memcpy(planes.ffnIn.contents() + t * planeStride,
                           scratch.ffnIn.contents(), planeStride)
                }
                var start = 0
                while start < order.count {
                    try Task.checkCancellation()
                    let batch = Array(
                        order[start..<min(start + 8, order.count)])
                    let tStage = Date()
                    let staged = try stage(batch)
                    let tApply = Date()
                    let gpuApply = GLM52GraphTelemetry.gpuSeconds
                    profile.gatherS += tApply.timeIntervalSince(tStage)
                    var applications:
                        [(slot: Int, token: Int, weight: Float)] = []
                    for (slot, id) in batch.enumerated() {
                        for use in users[id] ?? [] {
                            applications.append(
                                (slot, use.token, use.weight))
                        }
                    }
                    try runtime.glm52ApplyRoutedExperts(
                        staged: staged, applications: applications,
                        hiddenAll: planes.hidden, ffnInAll: planes.ffnIn,
                        scratch: scratch, embeddingWidth: embeddingWidth,
                        expertHiddenWidth:
                            geometry.layer.expertHiddenWidth)
                    profile.expertsS += Date().timeIntervalSince(tApply)
                    gpuExpertsS += GLM52GraphTelemetry.gpuSeconds - gpuApply
                    start += 8
                }
                for t in 0..<tokens.count {
                    let pointer = (planes.hidden.contents()
                        + t * planeStride).bindMemory(
                            to: Float.self, capacity: embeddingWidth)
                    hiddens[t] = Array(UnsafeBufferPointer(
                        start: pointer, count: embeddingWidth))
                }
                return
            }
            for t in 0..<tokens.count {
                try Task.checkCancellation()
                scratch.loadHidden(hiddens[t])
                let result = try runtime.glm52ChainedDecodeLayer(
                    weights: weights, ffn: ffn, caches: caches,
                    scratch: scratch,
                    reusedSelection: try reusedSelection(
                        index: index, isFull: isFull, token: t),
                    position: basePosition + t,
                    activeExperts: activeExperts, carry: &noCarry)
                accumulate(result.phases)
                if isFull {
                    lastSelections[t] = (index, result.selection)
                }
                noteRouting(index, result.routing)
                hiddens[t] = scratch.readHidden(count: embeddingWidth)
            }
        }

        if let streamer {
            for ahead in 0..<min(prefetchDepth, streamedLayers.count) {
                streamer.prefetch(streamedLayers[ahead].tensors,
                                  parallel: true)
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
            try Task.checkCancellation()
            let index = streamedLayer.tensors.index
            let waitStart = Date()
            let buffers = try streamer!.wait(for: index)
            let stall = Date().timeIntervalSince(waitStart)
            counters.layerStallSeconds += stall
            counters.layerBytes += UInt64(streamedLayer.bigTensorBytes)
            profile.layerStreamS += stall
            profile.layerStreamBytes += streamedLayer.bigTensorBytes
            if rank + prefetchDepth < streamedLayers.count {
                streamer!.prefetch(
                    streamedLayers[rank + prefetchDepth].tensors,
                    parallel: true)
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
        return hiddens
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
            expertSpeculativeBytes: arenaStats.speculativeBytes,
            gpuSeconds: GLM52GraphTelemetry.gpuSeconds,
            gpuCommits: GLM52GraphTelemetry.commits)
    }

    /// Hit/miss dell'arena esperti in record dall'ultimo reset — per la riga
    /// di tuning della GUI. Snapshot lock-protetto: sicuro da ogni thread
    /// anche durante una generazione.
    public func expertArenaCounters() -> (hits: Int, misses: Int)? {
        guard let arena else { return nil }
        let stats = arena.statsSnapshot()
        return (stats.hitCount, stats.readCount)
    }

    public func resetStreamingStats() {
        counters.reset()
        arena?.resetStats()
        GLM52GraphTelemetry.reset()
        // Il profilo per-fase e le stats dell'arena vivono e muoiono insieme:
        // profileReport() innesta le seconde nel primo.
        profile = DecodeProfile()
        gpuRouteS = 0; gpuExpertsS = 0; gpuDenseS = 0; gpuHeadS = 0
    }

    /// Report per-fase nello STESSO formato del decoder DeepSeek
    /// (DecodeProfile.report): ms/token e percentuali per embed, route/attn,
    /// gather IO, experts, layer IO (attesa stream layer, solo GLM), output
    /// head, più le righe cache con i contatori dell'arena esperti — hit e
    /// miss in record e in byte, speculativi come "look-ahead" (I/O nascosto
    /// sotto il compute dell'attenzione).
    public func profileReport(title: String = "Profilo decode") -> String {
        var merged = profile
        if let arena {
            let stats = arena.statsSnapshot()
            merged.expertHits = stats.hitCount
            merged.expertMisses = stats.readCount
            merged.expertHitBytes = Int(stats.hitBytes)
            merged.expertMissBytes = Int(stats.readBytes)
            merged.gatherBytes = Int(stats.readBytes)
            merged.expertPrefilled = stats.speculativeCount
            merged.expertPrefilledBytes = Int(stats.speculativeBytes)
        }
        var report = merged.report(title: title)
        // Scomposizione GPU/host per fase (solo GLM: i commit sincroni la
        // rendono gratuita): quanto delle fasi qui sopra è ESECUZIONE GPU.
        // wall − gpu = host: encode, sync round-trip, router CPU, allocazioni.
        if profile.forwards > 0 {
            let f = Double(profile.forwards)
            func ms(_ s: Double) -> String {
                String(format: "%.0f", s / f * 1000)
            }
            report += "\n  gpu          route/attn \(ms(gpuRouteS))"
                + " · experts \(ms(gpuExpertsS))"
                + " · layer(alt) \(ms(gpuDenseS))"
                + " · head \(ms(gpuHeadS)) ms/token — il resto è host"
                + " (encode/sync/router)"
        }
        return report
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
        // Con stallo sotto la soglia i GiB/stallo esplodono in un numero
        // privo di senso fisico (byte serviti dal prefetch, non dal wait):
        // il dato utile è "overlap completo", non una banda.
        let layerStall = stats.layerStallSeconds >= 0.05
            ? String(format: "stallo %.1fs, %.1f GB/s",
                     stats.layerStallSeconds,
                     rate(stats.layerBytes, stats.layerStallSeconds))
            : "stallo ~0s, overlap completo"
        return String(
            format: "streaming: %d token · layer %.1f GiB "
                + "(%.2f GiB/token, %@) · esperti "
                + "%.2f GiB letti (stallo %.1fs, %.1f GB/s) · riuso arena "
                + "%.2f GiB · speculativi %.2f GiB · gpu %.1fs "
                + "(%d commit)",
            stats.tokens, gib(stats.layerBytes),
            gib(stats.layerBytes) / Double(stats.tokens),
            layerStall,
            gib(stats.expertBytes),
            stats.expertStallSeconds,
            rate(stats.expertBytes, stats.expertStallSeconds),
            gib(stats.expertHitBytes),
            gib(stats.expertSpeculativeBytes),
            stats.gpuSeconds, stats.gpuCommits)
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
    /// REAL GPU execution time of the graph's synchronous commits — the
    /// gap between this and (wall − stalls) is host overhead: encode,
    /// commit round-trips, router, readbacks.
    public let gpuSeconds: Double
    public let gpuCommits: Int
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

// MARK: - Auto-tune

/// Esito dell'auto-tune GLM: report leggibile più i knob vincenti
/// (nome env → valore; assente = default del motore).
public struct GLM52AutoTuneOutcome: Sendable {
    public let report: String
    public let winners: [String: String]
}

extension GLM52ResidentModel {
    /// I knob di CARICAMENTO che l'auto-tune esplora: tutti ESATTI (stessi
    /// logits) e tutti letti a init del motore, quindi applicabili con
    /// setenv + reload (~3 s: il load streaming GLM rende possibile ciò che
    /// per DeepSeek è proibitivo). Esclusi per costruzione: i knob di
    /// qualità (DS4_GLM_ACTIVE_EXPERTS, sidecar Q4 lossy) e quelli
    /// process-static (DS4_GLM_NSG/SG, DS4_GLM_MLOCK, DS4_GLM_READ_SPLIT —
    /// cache in static let, un setenv a metà processo non li muove).
    static let autoTuneKnobs: [(knob: String, alternatives: [String?])] = [
        ("DS4_GLM_MTLIO", ["0", "1"]),
        ("DS4_GLM_STREAM_SLOTS", [nil, "5"]),
        ("DS4_GLM_EXPERT_ARENA", [nil, "48"]),
        ("DS4_GLM_RESIDENT_LAYERS", [nil, "7"]),
        ("DS4_GLM_FUSE", [nil, "0"]),
    ]

    /// Auto-tune "un gradino alla volta" — l'analogo pragmatico del
    /// record-holder DeepSeek: misura la baseline, poi prova l'alternativa
    /// di OGNI knob tenendo il vincitore solo se batte il campione oltre la
    /// soglia di rumore. Ogni misura è un motore NUOVO (config al load) su
    /// prefill sintetico + decode greedy. A fine corsa l'ambiente del
    /// processo resta impostato sulla configurazione campione.
    public static func autoTune(runtime: MetalRuntime, path: String,
                                prompt: [Int32], genTokens: Int = 12,
                                progress: ((String) -> Void)? = nil) throws
        -> GLM52AutoTuneOutcome {
        /// Sopra il 3% il decode dei run brevi è segnale, sotto è rumore
        /// (misurato ±2-3% fra run identici in questa stessa sessione).
        let noiseThreshold = 1.03
        var report: [String] = []
        var winners: [String: String] = [:]

        func apply(_ knob: String, _ value: String?) {
            if let value { _ = setenv(knob, value, 1) }
            else { unsetenv(knob) }
        }

        func measure(_ label: String) throws -> Double {
            // Le opzioni si costruiscono dall'ambiente COME FA IL DEMO:
            // residenza e slot stream passano dalle options (non dall'env
            // del motore) e il default delle options è tutto-residente —
            // esattamente ciò che qui non si vuole mai.
            let env = ProcessInfo.processInfo.environment
            var options = GLM52ResidentModelOptions()
            options.residentLayerCount = env["DS4_GLM_RESIDENT_LAYERS"]
                .flatMap(Int.init)
                ?? GLM52ResidentModelOptions.adaptiveResidentLayerCount()
            options.activeExperts = env["DS4_GLM_ACTIVE_EXPERTS"]
                .flatMap(Int.init)
            if let slots = env["DS4_GLM_EXPERT_SLOTS"].flatMap(Int.init) {
                options.expertSlotCount = slots
            }
            if let slots = env["DS4_GLM_STREAM_SLOTS"].flatMap(Int.init) {
                options.streamSlotCount = slots
            }
            // Motore nuovo per ogni config; ARC libera il precedente (e i
            // suoi mlock) all'uscita dello scope, prima del load successivo.
            let engine = try GLM52ResidentModel(runtime: runtime, path: path,
                                                options: options)
            var logits = try engine.prefill(prompt)
            let start = Date()
            var produced = 0
            for _ in 0..<genTokens {
                guard let token = GLM52GreedyDecoding.argmax(logits) else {
                    break
                }
                logits = try engine.forwardNext(token)
                produced += 1
            }
            let tps = Double(produced)
                / max(Date().timeIntervalSince(start), 0.001)
            progress?(String(format: "%@: %.2f tok/s", label, tps))
            return tps
        }

        var best = try measure("baseline")
        report.append(String(format: "baseline: %.2f tok/s", best))
        for (knob, alternatives) in autoTuneKnobs {
            let original = ProcessInfo.processInfo.environment[knob]
            for candidate in alternatives where candidate != original {
                // "nil vs nil" e valori già attivi non si rimisurano.
                if candidate == nil && original == nil { continue }
                apply(knob, candidate)
                let label = "\(knob)=\(candidate ?? "default")"
                let tps = try measure(label)
                if tps > best * noiseThreshold {
                    best = tps
                    winners[knob] = candidate ?? ""
                    report.append(String(
                        format: "%@: %.2f tok/s — PROMOSSO", label, tps))
                    // Un gradino alla volta: promosso il primo che vince,
                    // si passa al knob successivo (stile DeepSeek).
                    break
                } else {
                    report.append(String(
                        format: "%@: %.2f tok/s — scartato", label, tps))
                    apply(knob, original)
                }
            }
        }
        report.append(String(
            format: "campione finale: %.2f tok/s — %@", best,
            winners.isEmpty ? "configurazione di partenza"
                : winners.map { "\($0.key)=\($0.value.isEmpty ? "default" : $0.value)" }
                    .sorted().joined(separator: "  ")))
        return GLM52AutoTuneOutcome(
            report: report.joined(separator: "\n"), winners: winners)
    }
}
