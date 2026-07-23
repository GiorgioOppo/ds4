import XCTest
@testable import DS4Metal

/// Attention DSA indicizzata (DS4_INDEXED_ATTN, GraphIndexedAttention.swift):
/// 1) il top-K a INDICI su GPU deve selezionare ESATTAMENTE lo stesso set del
///    riferimento CPU IndexerSelect.allowedTopK (stesso ordinamento del
///    percorso a maschera: score decrescente, id crescente sui pari);
/// 2) il kernel indicizzato rb16 (raw ring SWA + topK righe compresse + sink)
///    deve coincidere con il percorso storico maschera+flash sugli stessi
///    input, wrap del ring incluso. Tolleranza: entrambi i kernel lavorano in
///    half sulle righe K/V ma con tiling diverso.
final class GraphIndexedAttentionTests: XCTestCase {
    private let nHead = 8
    private let headDim = 512
    private let nRaw = 16       // finestra SWA visibile
    private let rawCap = 32     // capacità fisica del ring raw
    private let rawStart = 25   // 25+16 > 32: il wrap è esercitato
    private let nComp = 30
    private let topK = 8        // potenza di due (sort bitonic)
    private let pos = 139       // visible comp = 140/4 = 35 -> min(30) = 30

    private func makeRuntime() throws -> MetalRuntime {
        do { return try MetalRuntime() }
        catch MetalError.noDevice { throw XCTSkip("Metal device unavailable") }
    }

    private struct Generator {
        var state: UInt64
        init(seed: UInt64) { state = seed }
        mutating func float(_ scale: Float) -> Float {
            state = state &* 6_364_136_223_846_793_005 &+ 1_442_695_040_888_963_407
            return (Float(state >> 33) / Float(1 << 30) - 1.0) * scale
        }
    }

    private func fill(_ tensor: GPUTensor, count: Int, seed: UInt64,
                      scale: Float) {
        var g = Generator(seed: seed)
        let p = (tensor.buffer.contents() + tensor.byteOffset)
            .bindMemory(to: Float.self, capacity: count)
        for i in 0..<count { p[i] = g.float(scale) }
    }

    private func cpuSelection(_ scores: [Float]) -> [Int32] {
        var ids: [Int32] = []
        scores.withUnsafeBufferPointer {
            let allowed = IndexerSelect.allowedTopK(
                scores: $0.baseAddress!, count: nComp, k: topK)
            for (i, a) in allowed.enumerated() where a { ids.append(Int32(i)) }
        }
        return ids   // già in ordine di id crescente
    }

    func testTopKIndicesMatchCPUSelection() throws {
        let rt = try makeRuntime()
        var g = Generator(seed: 7)
        let scores = (0..<nComp).map { _ in g.float(1.0) }
        let scoresT = try GPUTensor.zeros(rt, floatCount: nComp)
        scores.withUnsafeBytes {
            (scoresT.buffer.contents() + scoresT.byteOffset)
                .copyMemory(from: $0.baseAddress!, byteCount: $0.count)
        }
        let raw = try GPUTensor.zerosBytes(rt, byteLength: topK * 4)
        let sorted = try GPUTensor.zerosBytes(rt, byteLength: topK * 4)

        let c = GraphContext(rt); try c.begin()
        try c.indexerTopKIndices(scores: scoresT, out: raw,
                                 nScores: nComp, topK: topK)
        try c.sortTopKAsc(indices: raw, sorted: sorted, topK: topK)
        c.commit()

        let p = (sorted.buffer.contents() + sorted.byteOffset)
            .bindMemory(to: Int32.self, capacity: topK)
        let got = Array(UnsafeBufferPointer(start: p, count: topK))
            .filter { $0 >= 0 }
        XCTAssertEqual(got, cpuSelection(scores),
                       "set/ordine del top-K GPU diverso dal riferimento CPU")
    }

    func testIndexedAttentionMatchesMaskedFlash() throws {
        let rt = try makeRuntime()
        let total = nRaw + nComp

        let q = try GPUTensor.zeros(rt, floatCount: nHead * headDim)
        let raw = try GPUTensor.zeros(rt, floatCount: rawCap * headDim)
        let comp = try GPUTensor.zeros(rt, floatCount: nComp * headDim)
        let sinks = try GPUTensor.zeros(rt, floatCount: nHead)
        fill(q, count: nHead * headDim, seed: 11, scale: 0.1)
        fill(raw, count: rawCap * headDim, seed: 12, scale: 0.1)
        fill(comp, count: nComp * headDim, seed: 13, scale: 0.1)
        fill(sinks, count: nHead, seed: 14, scale: 0.5)

        var g = Generator(seed: 15)
        let scores = (0..<nComp).map { _ in g.float(1.0) }
        let selection = cpuSelection(scores)

        // ── Percorso storico: maschera (0 raw / -inf comp non selezionate)
        // + flashAttnCore sull'intero span.
        let sb = GraphContext.flashScratchBytes(nHead: nHead, nKeys: 128)
        let kvF16 = try GPUTensor.zerosBytes(rt, byteLength: sb.kvF16)
        let mask = try GPUTensor.zerosBytes(rt, byteLength: sb.mask)
        let pad = try GPUTensor.zerosBytes(rt, byteLength: sb.pad)
        let tmp = try GPUTensor.zerosBytes(rt, byteLength: sb.tmp)
        let headsMasked = try GPUTensor.zeros(rt, floatCount: nHead * headDim)
        let maskP = (mask.buffer.contents() + mask.byteOffset)
            .bindMemory(to: Float16.self, capacity: total)
        for i in 0..<total { maskP[i] = i < nRaw ? 0 : -.infinity }
        for id in selection { maskP[nRaw + Int(id)] = 0 }

        let a = GraphContext(rt); try a.begin()
        try a.flashAttnCore(q: q, kvF32: raw, kvF16: kvF16, mask: mask,
                            sinks: sinks, pad: pad, tmp: tmp,
                            heads: headsMasked, nHead: nHead, nKeys: nRaw,
                            rawStartRow: rawStart, hasSinks: true,
                            comp: comp, nComp: nComp)
        a.commit()

        // ── Percorso indicizzato: stessi input, selezione via indici.
        let scoresT = try GPUTensor.zeros(rt, floatCount: nComp)
        scores.withUnsafeBytes {
            (scoresT.buffer.contents() + scoresT.byteOffset)
                .copyMemory(from: $0.baseAddress!, byteCount: $0.count)
        }
        let rawIdx = try GPUTensor.zerosBytes(rt, byteLength: topK * 4)
        let sortedIdx = try GPUTensor.zerosBytes(rt, byteLength: topK * 4)
        let headsIndexed = try GPUTensor.zeros(rt, floatCount: nHead * headDim)

        let b = GraphContext(rt); try b.begin()
        try b.indexerTopKIndices(scores: scoresT, out: rawIdx,
                                 nScores: nComp, topK: topK)
        try b.sortTopKAsc(indices: rawIdx, sorted: sortedIdx, topK: topK)
        try b.indexedMixedAttention(q: q, rawKv: raw, comp: comp,
                                    topk: sortedIdx, sinks: sinks,
                                    heads: headsIndexed, nHead: nHead,
                                    nRaw: nRaw, rawCap: rawCap,
                                    rawStart: rawStart, nComp: nComp,
                                    topK: topK, pos: pos,
                                    window: nRaw, ratio: 4)
        b.commit()

        let pm = (headsMasked.buffer.contents() + headsMasked.byteOffset)
            .bindMemory(to: Float.self, capacity: nHead * headDim)
        let pi = (headsIndexed.buffer.contents() + headsIndexed.byteOffset)
            .bindMemory(to: Float.self, capacity: nHead * headDim)
        var maxDiff: Float = 0
        var maxVal: Float = 0
        for i in 0..<(nHead * headDim) {
            maxDiff = max(maxDiff, abs(pm[i] - pi[i]))
            maxVal = max(maxVal, abs(pm[i]))
        }
        XCTAssertGreaterThan(maxVal, 1e-3, "output nullo: test mal costruito")
        XCTAssertLessThan(maxDiff, 2e-2,
                          "attention indicizzata diverge dal percorso a maschera")
    }
}
