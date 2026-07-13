import Foundation
import DS4Core

// MARK: - Expert parallelism payloads (Fase A — docs/EXPERT_PARALLELISM.md)
//
// Scissione VERTICALE: i worker possiedono un sottoinsieme dei 256 esperti
// (lo stesso per tutti i layer), il coordinatore tiene l'intero backbone
// denso (route/attention/KV/head). Definiti e testati in Fase A; nessun
// percorso attivo li emette ancora (i loop v9 ignorano i tipi ignoti).

/// EXPERT ASSIGN: lo shard di ESPERTI di un worker verticale. La proprietà è
/// una MASK esplicita (bit e = esperto posseduto) e non un range: il
/// bilanciamento viene dalla usage imatrix (carico, non conteggio). Ogni id
/// deve appartenere a ESATTAMENTE un worker (il coordinatore valida la
/// partizione al connect).
public struct DistExpertAssign: Sendable {
    public var modelName: String
    public var expertCacheSlots: Int
    public var useExpertBundle: Bool
    /// 32 byte (256 bit): bit e (byte e/8, bit e%8) = 1 ⇔ questo worker
    /// possiede l'esperto e, per TUTTI i layer routed.
    public var expertMask: Data
    /// Knob di performance del coordinatore (whitelist Dist.perfKnobKeys).
    public var envKnobs: [(key: String, value: String)]
    /// Usage imatrix per pre-scaldare la slot-cache dello shard.
    public var usageJSON: Data

    public init(modelName: String, expertCacheSlots: Int, useExpertBundle: Bool,
                expertMask: Data, envKnobs: [(key: String, value: String)] = [],
                usageJSON: Data = Data()) {
        self.modelName = modelName; self.expertCacheSlots = expertCacheSlots
        self.useExpertBundle = useExpertBundle
        self.expertMask = expertMask
        self.envKnobs = envKnobs; self.usageJSON = usageJSON
    }

    public func encoded() -> Data {
        var d = Data()
        let name = Data(modelName.utf8)
        d.appendLE(UInt32(name.count)); d.append(name)
        d.appendLE(UInt32(expertCacheSlots))
        d.appendLE(UInt32(useExpertBundle ? 1 : 0))
        // ESATTAMENTE 32 byte sul filo: una mask corta scalerebbe i campi
        // successivi (usageLen letto dentro la mask) — pad a zero, mai meno.
        var mask32 = expertMask.prefix(32)
        if mask32.count < 32 { mask32.append(Data(repeating: 0, count: 32 - mask32.count)) }
        d.append(mask32)
        d.appendLE(UInt32(usageJSON.count)); d.append(usageJSON)
        d.appendLE(UInt32(envKnobs.count))
        for (k, v) in envKnobs {
            let kd = Data(k.utf8), vd = Data(v.utf8)
            d.appendLE(UInt32(kd.count)); d.append(kd)
            d.appendLE(UInt32(vd.count)); d.append(vd)
        }
        return d
    }

    public static func decode(_ d: Data) -> DistExpertAssign? {
        var o = d.startIndex
        guard d.count >= 4 else { return nil }
        let nameLen = Int(d.readLE(&o) as UInt32)
        guard nameLen > 0, nameLen < 1024, o + nameLen + 8 + 32 + 4 <= d.endIndex else { return nil }
        let name = String(decoding: d[o..<o+nameLen], as: UTF8.self); o += nameLen
        let slots = Int(d.readLE(&o) as UInt32)
        let bundle = (d.readLE(&o) as UInt32) != 0
        let mask = Data(d[o..<o+32]); o += 32
        let usageLen = Int(d.readLE(&o) as UInt32)
        guard usageLen >= 0, o + usageLen + 4 <= d.endIndex else { return nil }
        let usage = Data(d[o..<o+usageLen]); o += usageLen
        let knobCount = Int(d.readLE(&o) as UInt32)
        guard knobCount >= 0, knobCount <= 64 else { return nil }
        var knobs: [(key: String, value: String)] = []
        for _ in 0..<knobCount {
            guard o + 4 <= d.endIndex else { return nil }
            let kLen = Int(d.readLE(&o) as UInt32)
            guard kLen > 0, kLen <= 256, o + kLen + 4 <= d.endIndex else { return nil }
            let k = String(decoding: d[o..<o+kLen], as: UTF8.self); o += kLen
            let vLen = Int(d.readLE(&o) as UInt32)
            guard vLen >= 0, vLen <= 256, o + vLen <= d.endIndex else { return nil }
            let v = String(decoding: d[o..<o+vLen], as: UTF8.self); o += vLen
            knobs.append((key: k, value: v))
        }
        return DistExpertAssign(modelName: name, expertCacheSlots: slots, useExpertBundle: bundle,
                                expertMask: mask, envKnobs: knobs, usageJSON: usage)
    }
}

/// EXPERT WORK: un layer routed di un token — l'attivazione post-attention
/// (`s.cur`, nEmbd float) più GLI ID POSSEDUTI DAL DESTINATARIO tra i 6
/// selezionati dal router, con i rispettivi pesi di route. `seq` accoppia la
/// risposta (le richieste ai worker viaggiano in parallelo e possono tornare
/// fuori ordine).
public struct DistExpertWork: Sendable {
    public var seq: UInt32
    public var layer: Int
    public var ids: [Int32]
    public var weights: [Float]
    /// nEmbd valori: f32 o f16 secondo `bits` (stessa convenzione
    /// activationBits della pipeline).
    public var activation: Data
    public var bits: Int

    public init(seq: UInt32, layer: Int, ids: [Int32], weights: [Float],
                activation: Data, bits: Int) {
        self.seq = seq; self.layer = layer; self.ids = ids; self.weights = weights
        self.activation = activation; self.bits = bits
    }

    public func encoded() -> Data {
        var d = Data()
        d.appendLE(seq)
        d.appendLE(UInt32(layer))
        d.appendLE(UInt32(bits))
        d.appendLE(UInt32(ids.count))
        for i in ids { d.appendLE(UInt32(bitPattern: i)) }
        // Il decoder legge SEMPRE ids.count pesi: un chiamante disallineato
        // produrrebbe un frame che sfasa actLen — pad a zero, mai meno.
        for j in 0..<ids.count { d.appendLE(j < weights.count ? weights[j].bitPattern : Float(0).bitPattern) }
        d.appendLE(UInt32(activation.count)); d.append(activation)
        return d
    }

    public static func decode(_ d: Data) -> DistExpertWork? {
        var o = d.startIndex
        guard d.count >= 16 else { return nil }
        let seq = d.readLE(&o) as UInt32
        let layer = Int(d.readLE(&o) as UInt32)
        let bits = Int(d.readLE(&o) as UInt32)
        guard bits == 16 || bits == 32 else { return nil }
        let k = Int(d.readLE(&o) as UInt32)
        guard k >= 1, k <= 8, o + k * 8 + 4 <= d.endIndex else { return nil }
        var ids: [Int32] = []
        for _ in 0..<k { ids.append(Int32(bitPattern: d.readLE(&o) as UInt32)) }
        var weights: [Float] = []
        for _ in 0..<k { weights.append(Float(bitPattern: d.readLE(&o) as UInt32)) }
        let actLen = Int(d.readLE(&o) as UInt32)
        guard actLen >= 0, actLen <= 1 << 20, o + actLen <= d.endIndex else { return nil }
        let act = Data(d[o..<o+actLen])
        return DistExpertWork(seq: seq, layer: layer, ids: ids, weights: weights,
                              activation: act, bits: bits)
    }
}

/// EXPERT SUM: la somma parziale pesata (nEmbd valori, f32/f16 come la
/// richiesta) degli esperti computati dal worker per `seq`. Il coordinatore
/// somma le parziali dei worker coinvolti e prosegue col layer.
public struct DistExpertSum: Sendable {
    public var seq: UInt32
    public var layer: Int
    public var partial: Data
    public var bits: Int

    public init(seq: UInt32, layer: Int, partial: Data, bits: Int) {
        self.seq = seq; self.layer = layer; self.partial = partial; self.bits = bits
    }

    public func encoded() -> Data {
        var d = Data()
        d.appendLE(seq)
        d.appendLE(UInt32(layer))
        d.appendLE(UInt32(bits))
        d.appendLE(UInt32(partial.count)); d.append(partial)
        return d
    }

    public static func decode(_ d: Data) -> DistExpertSum? {
        var o = d.startIndex
        guard d.count >= 16 else { return nil }
        let seq = d.readLE(&o) as UInt32
        let layer = Int(d.readLE(&o) as UInt32)
        let bits = Int(d.readLE(&o) as UInt32)
        guard bits == 16 || bits == 32 else { return nil }
        let len = Int(d.readLE(&o) as UInt32)
        guard len >= 0, len <= 1 << 20, o + len <= d.endIndex else { return nil }
        return DistExpertSum(seq: seq, layer: layer, partial: Data(d[o..<o+len]), bits: bits)
    }
}

