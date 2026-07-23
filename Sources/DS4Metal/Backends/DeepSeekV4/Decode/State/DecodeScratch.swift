import Foundation
import Metal

/// Reusable scratch GPUTensors for one decode layer (allocate once, reuse across layers).
public final class DecodeScratch {
    /// Number of raw+compressed attention rows the context-sized tensors can
    /// hold.  `StreamingDecoder` grows the whole scratch geometrically as the
    /// live context crosses this high-water mark; the fixed-size tensors are
    /// recreated with it so no buffer from an older command buffer is reused.
    let attentionRows: Int

    // MARK: Accessi CPU per l'expert shard VERTICALE (modulo DS4Engine): i
    // campi restano interni al modulo — lo shard scrive i suoi input e azzera
    // le righe inerti senza esporre l'intero scratch.

    /// Scrive l'attivazione (nEmbd f32) in `cur` — l'input dei matvec esperti.
    public func writeCurActivation(_ values: [Float]) {
        let p = cur.buffer.contents().advanced(by: cur.byteOffset)
        _ = values.withUnsafeBytes { memcpy(p, $0.baseAddress!, $0.count) }
    }

    /// Scrive i pesi di route: `values` nei primi slot, zero fino a `padTo`
    /// (gli slot inerti del percorso fuso non devono contribuire).
    public func writeRouteWeights(_ values: [Float], padTo k: Int) {
        let p = (rw.buffer.contents() + rw.byteOffset).bindMemory(to: Float.self, capacity: k)
        for i in 0..<k { p[i] = i < values.count ? values[i] : 0 }
    }

    /// Azzera le righe [fromRow, 6) di down6: moe_sum6 somma SEMPRE 6 righe e
    /// con k variabile per richiesta una riga stantia avvelenerebbe la somma.
    public func zeroDown6Rows(fromRow k: Int, nEmbd: Int) {
        guard k < 6 else { return }
        let p = down6.buffer.contents().advanced(by: down6.byteOffset)
        memset(p + k * nEmbd * 4, 0, (6 - k) * nEmbd * 4)
    }

    let flat, mix, split, embd, cur: GPUTensor
    let qr, qrNorm, q, kvRaw, kv: GPUTensor
    let kvF16, mask, sinks, pad, tmp, heads, blockOut: GPUTensor
    let attnLow: GPUTensor   // low-rank attention output (nOutGroup * nLoraO)
    let afterAttn: GPUTensor
    let logits, sp, probs, selected, rw: GPUTensor
    let gate6, up6, mid6, down6, routed: GPUTensor
    let sgate, sup, smid, sdown, sharedOut, ffnOut: GPUTensor
    let idxQ, idxW, idxScores: GPUTensor   // NSA indexer: q [64×128], weights [64], scores [maxComp]
    // DS4_INDEXED_ATTN: top-K dell'indexer a INDICI (grezzi dallo heap +
    // ordinati per id crescente), int32 [indexerTopK] ciascuno.
    let idxTopK, idxTopKSorted: GPUTensor
    let id0: GPUTensor   // [Int32 0] — dense matvec via the MoE id-kernel (k=1, DS4_DENSE_Q4)
    let idsGroup: GPUTensor   // [0..nOutGroup-1] — grouped attn-out as an all-selected "MoE" (DS4_DENSE_Q4)

    public init(_ rt: MetalRuntime, _ d: DSV4Dims, maxKeys: Int) throws {
        let attentionRows = max(1, maxKeys)
        self.attentionRows = attentionRows
        let hcDim = d.nHC * d.nEmbd
        flat = try .zeros(rt, floatCount: hcDim); mix = try .zeros(rt, floatCount: 24)
        split = try .zeros(rt, floatCount: 24); embd = try .zeros(rt, floatCount: d.nEmbd)
        cur = try .zeros(rt, floatCount: d.nEmbd)
        qr = try .zeros(rt, floatCount: d.qRank); qrNorm = try .zeros(rt, floatCount: d.qRank)
        q = try .zeros(rt, floatCount: d.qDim)
        kvRaw = try .zeros(rt, floatCount: d.headDim); kv = try .zeros(rt, floatCount: d.headDim)
        let sb = GraphContext.flashScratchBytes(nHead: d.nHead, nKeys: attentionRows)
        kvF16 = try .zerosBytes(rt, byteLength: sb.kvF16); mask = try .zerosBytes(rt, byteLength: sb.mask)
        sinks = try .zerosBytes(rt, byteLength: sb.sinks); pad = try .zerosBytes(rt, byteLength: sb.pad)
        tmp = try .zerosBytes(rt, byteLength: sb.tmp); heads = try .zeros(rt, floatCount: d.nHead * d.headDim)
        attnLow = try .zeros(rt, floatCount: d.attnLowDim)
        blockOut = try .zeros(rt, floatCount: d.nEmbd); afterAttn = try .zeros(rt, floatCount: d.nHC * d.nEmbd)
        logits = try .zeros(rt, floatCount: d.nExperts); sp = try .zeros(rt, floatCount: d.nExperts)
        probs = try .zeros(rt, floatCount: d.nExperts); selected = try .zerosBytes(rt, byteLength: d.k * 4)
        rw = try .zeros(rt, floatCount: d.k)
        gate6 = try .zeros(rt, floatCount: d.k * d.expertFfn); up6 = try .zeros(rt, floatCount: d.k * d.expertFfn)
        mid6 = try .zeros(rt, floatCount: d.k * d.expertFfn); down6 = try .zeros(rt, floatCount: d.k * d.nEmbd)
        routed = try .zeros(rt, floatCount: d.nEmbd)
        sgate = try .zeros(rt, floatCount: d.sharedFfn); sup = try .zeros(rt, floatCount: d.sharedFfn)
        smid = try .zeros(rt, floatCount: d.sharedFfn); sdown = try .zeros(rt, floatCount: d.nEmbd)
        sharedOut = try .zeros(rt, floatCount: d.nEmbd); ffnOut = try .zeros(rt, floatCount: d.nEmbd)
        idxQ = try .zeros(rt, floatCount: d.nIndexerHead * d.nIndexerHeadDim)
        idxW = try .zeros(rt, floatCount: d.nIndexerHead)
        idxScores = try .zeros(rt, floatCount: attentionRows)   // ≥ maxComp rows, generous
        idxTopK = try .zerosBytes(rt, byteLength: max(1, d.indexerTopK) * 4)
        idxTopKSorted = try .zerosBytes(rt, byteLength: max(1, d.indexerTopK) * 4)
        id0 = try .zerosBytes(rt, byteLength: 4)          // Int32(0)
        idsGroup = try .bytes(rt, Array(0..<Int32(d.nOutGroup)).withUnsafeBytes { Array($0) },
                              elementCount: d.nOutGroup)
    }
}
