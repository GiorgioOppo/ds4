import Foundation
import DS4Core
import DS4Metal

/// Motore VERTICALE del worker (expert parallelism, Fase B — vedi
/// docs/EXPERT_PARALLELISM.md): possiede un sottoinsieme degli esperti dichiarati
/// dal GGUF (256 per Flash, 384 per Pro)
/// (mask, la STESSA per tutti i layer) e serve richieste `expertWork` →
/// `expertSum`: attivazione + id/pesi in ingresso, somma parziale pesata
/// dei SUOI esperti in uscita. Nessun KV, nessuna attention, nessuno stato
/// di sequenza: puro gather dall'SSD + FFN esperti. Il backbone denso
/// (route/attention/KV/head) vive sul coordinatore.
///
/// Fase B = correttezza prima: il gather va dritto su mmap/pread
/// (gatherLayerExperts); slot-cache, bundle sidecar e pre-warm dalla usage
/// imatrix arrivano in Fase D — sono ottimizzazioni, non semantica.
public final class ExpertShardEngine: @unchecked Sendable {
    private let rt: MetalRuntime
    private let model: GGUFModel
    private let geometry: DSV4RuntimeGeometry
    private let dims: DSV4Dims
    private let scratch: DecodeScratch
    private let idsPacked: GPUTensor      // 0..<k rimappati (gli slab del gather sono impacchettati)
    private let partialOut: GPUTensor     // nEmbd f32
    /// D2: la stessa terna del motore locale — gather (bundle sidecar) +
    /// slot-cache LRU + stride del pool interleaved. La cache tiene caldi gli
    /// esperti POSSEDUTI: lo shard non ha KV né backbone, quindi può
    /// permettersi più slot per layer di quanti ne abbia il locale.
    private let gatherFn: (Int, [Int32]) throws -> (GPUTensor, GPUTensor, GPUTensor)
    private let slotCache: ExpertSlotCache?
    private let slotStride: Int?
    private let usage: ExpertUsageStats
    private let slotsStage: GPUTensor     // id di slot per il matvec sul pool
    /// Quant dei tensori esperti PER LAYER (mixed-precision): il dispatch dei
    /// kernel deve seguire il layer, non il globale del primo layer routed.
    private let layerQuants: [(gate: MoEQuant, up: MoEQuant, down: MoEQuant)]
    public let nLayers: Int
    public let nExperts: Int
    /// mask[e] = questo shard possiede l'esperto e (per tutti i layer routed).
    public let mask: [Bool]
    public var ownedCount: Int { mask.lazy.filter { $0 }.count }
    /// Una richiesta alla volta: lo scratch è condiviso (il parallelismo è
    /// TRA i worker, non dentro uno shard).
    private let lock = NSLock()

    public init(modelPath: String, expertMask: Data,
                expertCacheSlots: Int = 0, usageJSON: Data = Data(),
                onLoadLog: (@Sendable (String) -> Void)? = nil) throws {
        onLoadLog?("init Metal runtime (compilazione kernel)…")
        self.rt = try MetalRuntime()
        onLoadLog?("mmap gguf…")
        self.model = try GGUFModel(path: modelPath, metalMapping: true, prefetchCPU: false)
        _ = try RuntimeBackendFactory.prepare(model: model)
        // Stessa validazione di carico del motore locale/pipeline: metadata
        // (config_validate_model), geometria per istanza e quant chiusi.
        let config = try ModelConfig(model: model)
        let runtimeGeometry = DSV4RuntimeGeometry(configuration: config)
        var d = runtimeGeometry.dims
        let mq = GGUFWeights.detectMoEQuant(model)
        d.gateQuant = mq.gate; d.upQuant = mq.up; d.downQuant = mq.down; d.routerF16 = mq.routerF16
        try GGUFWeights.validateRuntimeLayout(model, geometry: runtimeGeometry)
        self.geometry = runtimeGeometry
        self.dims = d
        self.nLayers = runtimeGeometry.nLayers
        self.nExperts = d.nExperts
        // Quant PER-LAYER (GGUF mixed-precision, "boosted layer"): il percorso
        // locale dispatcha su w.*Quant per layer — usare il globale del primo
        // layer routed decodificherebbe gli slab di un layer boosted col
        // kernel sbagliato: garbage finito e SILENZIOSO. Stessa mappatura di
        // GGUFWeights.setExpertQuant; fallback al globale sui layer densi.
        var lq: [(gate: MoEQuant, up: MoEQuant, down: MoEQuant)] = []
        lq.reserveCapacity(runtimeGeometry.nLayers)
        for il in 0..<runtimeGeometry.nLayers {
            let p = "blk.\(il)."
            let g = model.findTensor(p + "ffn_gate_exps.weight").flatMap { MoEQuant.from(ggufType: $0.type) } ?? d.gateQuant
            let u = model.findTensor(p + "ffn_up_exps.weight").flatMap { MoEQuant.from(ggufType: $0.type) } ?? d.upQuant
            let dn = model.findTensor(p + "ffn_down_exps.weight").flatMap { MoEQuant.from(ggufType: $0.type) } ?? d.downQuant
            lq.append((gate: g, up: u, down: dn))
        }
        self.layerQuants = lq
        let expectedMaskBytes = (d.nExperts + 7) / 8
        guard expertMask.count == expectedMaskBytes else {
            throw GGUFWeights.LoadError.message(
                "ExpertShard: mask di \(expertMask.count) byte, attesi \(expectedMaskBytes) per \(d.nExperts) esperti")
        }
        if d.nExperts % 8 != 0, let last = expertMask.last {
            let validMask = UInt8((1 << (d.nExperts % 8)) - 1)
            guard last & ~validMask == 0 else {
                throw GGUFWeights.LoadError.message("ExpertShard: bit di padding non zero nella mask")
            }
        }
        var m = [Bool](repeating: false, count: d.nExperts)
        for e in 0..<d.nExperts {
            let byte = expertMask[expertMask.startIndex + e / 8]
            m[e] = (byte >> UInt8(e % 8)) & 1 == 1
        }
        guard m.contains(true) else {
            throw GGUFWeights.LoadError.message("ExpertShard: mask di esperti vuota")
        }
        self.mask = m
        // Scratch minimo: niente KV/attention — maxKeys simbolico.
        self.scratch = try DecodeScratch(rt, d, maxKeys: 8)
        self.idsPacked = try GPUTensor.bytes(rt, Array(0..<Int32(d.k)).withUnsafeBytes { Array($0) },
                                             elementCount: d.k)
        self.partialOut = try GPUTensor.zeros(rt, floatCount: d.nEmbd)
        // D2: bundle + slot-cache + pre-warm dalla usage imatrix — la stessa
        // terna del motore locale, assemblata dal helper condiviso.
        let stats = ExpertUsageStats(nLayers: runtimeGeometry.nLayers,
                                     nExperts: runtimeGeometry.dims.nExperts)
        if !usageJSON.isEmpty { stats.load(usageJSON) }
        self.usage = stats
        let mlock = ProcessInfo.processInfo.environment["DS4_MLOCK"] == "1"
        let stack = StreamingDecoder.makeExpertGatherStack(rt: rt, model: model, dims: d,
                                                           nLayers: runtimeGeometry.nLayers,
                                                           slots: expertCacheSlots, usage: stats,
                                                           lockResident: mlock)
        self.gatherFn = stack.gather
        self.slotCache = stack.cache
        self.slotStride = stack.stride
        self.slotsStage = try GPUTensor.zerosBytes(rt, byteLength: d.k * 4)
        onLoadLog?("shard \(config.shape.name) pronto: \(ownedCount)/\(d.nExperts) esperti · \(runtimeGeometry.nLayers) layer"
                   + (stack.cache != nil ? " · cache \(max(8, expertCacheSlots)) slot/layer" : " · cache OFF"))
    }

    /// Somma parziale pesata per una richiesta `expertWork`. Sincrona (gather
    /// SSD + un command buffer): il chiamante la esegue fuori dal main actor.
    public func partial(_ req: DistExpertWork) throws -> DistExpertSum {
        lock.lock(); defer { lock.unlock() }
        let nE = dims.nEmbd
        guard req.layer >= 0, req.layer < nLayers else {
            throw GGUFWeights.LoadError.message("expertWork: layer \(req.layer) fuori range")
        }
        guard !req.ids.isEmpty, req.ids.count <= dims.k,
              req.weights.count == req.ids.count else {
            throw GGUFWeights.LoadError.message(
                "expertWork: k=\(req.ids.count) fuori range o pesi disallineati (\(req.weights.count))")
        }
        guard req.ids.allSatisfy({ $0 >= 0 && Int($0) < dims.nExperts && mask[Int($0)] }),
              Set(req.ids).count == req.ids.count else {
            throw GGUFWeights.LoadError.message("expertWork: id non posseduti da questo shard o duplicati")
        }
        let k = req.ids.count
        let quant = layerQuants[req.layer]
        // Attivazione → s.cur (f32; f16 di rete convertita CPU-side, sempre
        // via memcpy staging: una Data di rete può essere una slice non
        // allineata).
        var act = [Float](repeating: 0, count: nE)
        if req.bits == 32 {
            guard req.activation.count == nE * 4 else {
                throw GGUFWeights.LoadError.message("expertWork: attivazione f32 di taglia errata")
            }
            _ = act.withUnsafeMutableBytes { dst in
                req.activation.withUnsafeBytes { src in memcpy(dst.baseAddress!, src.baseAddress!, nE * 4) }
            }
        } else {
            guard req.activation.count == nE * 2 else {
                throw GGUFWeights.LoadError.message("expertWork: attivazione f16 di taglia errata")
            }
            var halves = [Float16](repeating: 0, count: nE)
            _ = halves.withUnsafeMutableBytes { dst in
                req.activation.withUnsafeBytes { src in memcpy(dst.baseAddress!, src.baseAddress!, nE * 2) }
            }
            for i in 0..<nE { act[i] = Float(halves[i]) }
        }
        scratch.writeCurActivation(act)
        // Pesi di route (gli slot oltre k restano 0: gli slot inerti del
        // percorso fuso non contribuiscono).
        scratch.writeRouteWeights(req.weights, padTo: dims.k)
        // Usage: il routing reale visto dallo shard alimenta pre-warm e
        // allocazione per-layer (stessa imatrix del locale).
        usage.record(layer: req.layer, ids: req.ids)
        // Esperti: slot-cache (hit = zero I/O) quando il layer è nella classe
        // di quant globale (la cache è mono-classe, come nel locale); i layer
        // mixed-precision e la cache spenta passano dal gather (bundle/mmap).
        let gExp: GPUTensor, uExp: GPUTensor, dExp: GPUTensor
        let idsTensor: GPUTensor
        let stride: Int?
        let onClass = quant.gate == dims.gateQuant && quant.up == dims.upQuant
            && quant.down == dims.downQuant
        if let cache = slotCache, onClass {
            let (pool, slots) = try cache.acquire(layer: req.layer, ids: req.ids)
            _ = slots.withUnsafeBytes {
                memcpy(slotsStage.buffer.contents() + slotsStage.byteOffset, $0.baseAddress!, $0.count)
            }
            gExp = pool.gate; uExp = pool.up; dExp = pool.down
            idsTensor = slotsStage
            stride = slotStride
        } else {
            let (g, u, dn) = try gatherFn(req.layer, req.ids)
            gExp = g; uExp = u; dExp = dn
            idsTensor = idsPacked
            stride = nil
        }
        // moe_sum6 somma SEMPRE 6 righe: con k<6 sul percorso non fuso le
        // righe k..<6 di down6 vanno azzerate (richieste diverse hanno k
        // diversi — una riga stantia di una richiesta a k più alto
        // avvelenerebbe la somma).
        let sumFused = dims.fusedMoE && k == 6
            && (quant.down == .q2_K || quant.down == .q4_K)
        if !sumFused { scratch.zeroDown6Rows(fromRow: k, nEmbd: nE) }
        let c = GraphContext(rt)
        try c.begin()
        try c.decodeExpertPartial(s: scratch, d: dims,
                                  gateQuant: quant.gate, upQuant: quant.up,
                                  downQuant: quant.down,
                                  gateExp: gExp, upExp: uExp, downExp: dExp,
                                  ids: idsTensor, k: k, out: partialOut,
                                  expertStride: stride)
        c.commit()
        // Un fault GPU (es. slab fuori misura) completa "con errore" lasciando
        // partialOut stantio: fallire FORTE invece di serializzare spazzatura.
        if let err = c.lastError {
            throw GGUFWeights.LoadError.message("expertWork: command buffer fallito (\(err))")
        }
        // Somma parziale → payload (f32 o f16, come la richiesta).
        let outPtr = partialOut.buffer.contents().advanced(by: partialOut.byteOffset)
            .bindMemory(to: Float.self, capacity: nE)
        let payload: Data
        if req.bits == 32 {
            payload = Data(bytes: outPtr, count: nE * 4)
        } else {
            var halves = [Float16](repeating: 0, count: nE)
            for i in 0..<nE { halves[i] = Float16(outPtr[i]) }
            payload = halves.withUnsafeBytes { Data($0) }
        }
        return DistExpertSum(seq: req.seq, layer: req.layer, partial: payload, bits: req.bits)
    }
}
