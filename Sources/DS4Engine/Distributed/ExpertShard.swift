import Foundation
import DS4Core
import DS4Metal

/// Motore VERTICALE del worker (expert parallelism, Fase B — vedi
/// docs/EXPERT_PARALLELISM.md): possiede un sottoinsieme dei 256 esperti
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
    private let dims: DSV4Dims
    private let scratch: DecodeScratch
    private let idsPacked: GPUTensor      // 0..<k rimappati (gli slab del gather sono impacchettati)
    private let partialOut: GPUTensor     // nEmbd f32
    private let preadFD: Int32?           // DS4_EXPERT_PREAD: gather via F_NOCACHE
    /// Quant dei tensori esperti PER LAYER (mixed-precision): il dispatch dei
    /// kernel deve seguire il layer, non il globale del primo layer routed.
    private let layerQuants: [(gate: MoEQuant, up: MoEQuant, down: MoEQuant)]
    public let nLayers: Int
    /// mask[e] = questo shard possiede l'esperto e (per tutti i layer routed).
    public let mask: [Bool]
    public var ownedCount: Int { mask.lazy.filter { $0 }.count }
    /// Una richiesta alla volta: lo scratch è condiviso (il parallelismo è
    /// TRA i worker, non dentro uno shard).
    private let lock = NSLock()

    public init(modelPath: String, expertMask: Data,
                onLoadLog: (@Sendable (String) -> Void)? = nil) throws {
        onLoadLog?("init Metal runtime (compilazione kernel)…")
        self.rt = try MetalRuntime()
        onLoadLog?("mmap gguf…")
        self.model = try GGUFModel(path: modelPath, metalMapping: true, prefetchCPU: false)
        // Stessa validazione di carico del motore locale/pipeline: metadata
        // (config_validate_model), solo profilo Flash, quant chiusi.
        let config = try ModelConfig(model: model)
        guard config.shape.variant == .flash else {
            throw ModelConfigError.unsupportedShape(
                "\(config.shape.name): il runtime supporta solo il profilo Flash")
        }
        var d = DSV4Shape.dims
        let mq = GGUFWeights.detectMoEQuant(model)
        d.gateQuant = mq.gate; d.upQuant = mq.up; d.downQuant = mq.down; d.routerF16 = mq.routerF16
        try GGUFWeights.validateRoutedExperts(model, dims: d, nLayers: DSV4Shape.nLayer)
        self.dims = d
        self.nLayers = DSV4Shape.nLayer
        // Quant PER-LAYER (GGUF mixed-precision, "boosted layer"): il percorso
        // locale dispatcha su w.*Quant per layer — usare il globale del primo
        // layer routed decodificherebbe gli slab di un layer boosted col
        // kernel sbagliato: garbage finito e SILENZIOSO. Stessa mappatura di
        // GGUFWeights.setExpertQuant; fallback al globale sui layer densi.
        var lq: [(gate: MoEQuant, up: MoEQuant, down: MoEQuant)] = []
        lq.reserveCapacity(DSV4Shape.nLayer)
        for il in 0..<DSV4Shape.nLayer {
            let p = "blk.\(il)."
            let g = model.findTensor(p + "ffn_gate_exps.weight").flatMap { MoEQuant.from(ggufType: $0.type) } ?? d.gateQuant
            let u = model.findTensor(p + "ffn_up_exps.weight").flatMap { MoEQuant.from(ggufType: $0.type) } ?? d.upQuant
            let dn = model.findTensor(p + "ffn_down_exps.weight").flatMap { MoEQuant.from(ggufType: $0.type) } ?? d.downQuant
            lq.append((gate: g, up: u, down: dn))
        }
        self.layerQuants = lq
        var m = [Bool](repeating: false, count: d.nExperts)
        for e in 0..<min(d.nExperts, expertMask.count * 8) {
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
        self.preadFD = ProcessInfo.processInfo.environment["DS4_EXPERT_PREAD"] == "1"
            ? model.uncachedFD() : nil
        onLoadLog?("shard pronto: \(ownedCount)/\(d.nExperts) esperti")
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
        // Attivazione → s.cur (f32; f16 di rete convertita CPU-side).
        let curPtr = scratch.cur.buffer.contents().advanced(by: scratch.cur.byteOffset)
        if req.bits == 32 {
            guard req.activation.count == nE * 4 else {
                throw GGUFWeights.LoadError.message("expertWork: attivazione f32 di taglia errata")
            }
            req.activation.withUnsafeBytes { _ = memcpy(curPtr, $0.baseAddress!, nE * 4) }
        } else {
            guard req.activation.count == nE * 2 else {
                throw GGUFWeights.LoadError.message("expertWork: attivazione f16 di taglia errata")
            }
            // Staging con memcpy: una Data di rete può essere una slice NON
            // allineata — bindMemory(to: Float16) su base disallineata è UB.
            var halves = [Float16](repeating: 0, count: nE)
            _ = halves.withUnsafeMutableBytes { dst in
                req.activation.withUnsafeBytes { src in memcpy(dst.baseAddress!, src.baseAddress!, nE * 2) }
            }
            let dst = curPtr.bindMemory(to: Float.self, capacity: nE)
            for i in 0..<nE { dst[i] = Float(halves[i]) }
        }
        // Pesi di route → s.rw (le righe oltre k restano 0: gli slot inerti
        // del percorso fuso non contribuiscono).
        let rwPtr = scratch.rw.buffer.contents().advanced(by: scratch.rw.byteOffset)
            .bindMemory(to: Float.self, capacity: dims.k)
        for i in 0..<dims.k { rwPtr[i] = i < k ? req.weights[i] : 0 }
        // Gather dei SUOI esperti: slab impacchettati, id rimappati 0..<k.
        let (g, u, dn) = try GGUFWeights.gatherLayerExperts(rt, model, req.layer, ids: req.ids,
                                                            dims: dims, willNeed: true,
                                                            uncachedFD: preadFD)
        // moe_sum6 somma SEMPRE 6 righe: con k<6 sul percorso non fuso le
        // righe k..<6 di down6 vanno azzerate (richieste diverse hanno k
        // diversi — una riga stantia di una richiesta a k più alto
        // avvelenerebbe la somma).
        let sumFused = dims.fusedMoE && k == 6
            && (quant.down == .q2_K || quant.down == .q4_K)
        if !sumFused && k < 6 {
            let d6 = scratch.down6.buffer.contents().advanced(by: scratch.down6.byteOffset)
            memset(d6 + k * nE * 4, 0, (6 - k) * nE * 4)
        }
        let c = GraphContext(rt)
        try c.begin()
        try c.decodeExpertPartial(s: scratch, d: dims,
                                  gateQuant: quant.gate, upQuant: quant.up,
                                  downQuant: quant.down,
                                  gateExp: g, upExp: u, downExp: dn,
                                  ids: idsPacked, k: k, out: partialOut)
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
