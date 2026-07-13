import Foundation
import Metal
import DS4Core

extension StreamingDecoder {
    /// Embed a whole prefill chunk in ONE command buffer: the tokens' table rows
    /// are CPU-staged into a transient n-row table, then n embedTokenHC encodes
    /// run back-to-back (the encoder is serial, so the shared `embd` intermediate
    /// is safe). Numerically identical to n embedToken calls — it only removes
    /// the n-1 per-token commit+wait round-trips (512 GPU syncs per chunk).
    func embedTokensBatch(_ toks: [Int], into hcs: [GPUTensor]) throws {
        guard toks.count > 1 else {
            if let t = toks.first { try embedToken(t, into: hcs[0]) }
            return
        }
        let t = Date()
        let rowBytes = embedRowStage.byteLength
        let stage = try GPUTensor.zerosBytes(rt, byteLength: toks.count * rowBytes)
        for (j, token) in toks.enumerated() {
            precondition(token >= 0 && token < d.vocab, "embedTokensBatch: token \(token) out of vocab")
            memcpy(stage.buffer.contents() + j * rowBytes,
                   embedTable.buffer.contents() + embedTable.byteOffset + token * rowBytes,
                   rowBytes)
        }
        let ec = GraphContext(rt)
        try ec.begin()
        for j in 0..<toks.count {
            try ec.embedTokenHC(table: stage, token: j, embd: embd, hc: hcs[j],
                                nEmbd: d.nEmbd, nVocab: toks.count, nHC: d.nHC)
        }
        ec.commit()
        profile.embedS += Date().timeIntervalSince(t)
    }

    /// Embed one token into the HC state buffer `hc` (own command buffer).
    /// The token's table row is CPU-staged into `embedRowStage` and the kernel
    /// runs on that (token 0 of a 1-row table) — see embedRowStage for why.
    func embedToken(_ token: Int, into hc: GPUTensor) throws {
        let t = Date()
        let rowBytes = embedRowStage.byteLength
        precondition(token >= 0 && token < d.vocab, "embedToken: token \(token) out of vocab")
        memcpy(embedRowStage.buffer.contents(),
               embedTable.buffer.contents() + embedTable.byteOffset + token * rowBytes,
               rowBytes)
        let ec = GraphContext(rt)
        try ec.begin()
        try ec.embedTokenHC(table: embedRowStage, token: 0, embd: embd, hc: hc,
                            nEmbd: d.nEmbd, nVocab: 1, nHC: d.nHC)
        ec.commit()
        profile.embedS += Date().timeIntervalSince(t)
    }

    /// Read back the router's selection after a committed decodeRoute, applying
    /// the activeExperts top-K reduction (route weights renormalized to the
    /// original total). Returns the final (ids, weights), both of count K ≤ d.k.
    /// Also feeds the usage statistics ("usage imatrix") for `layer`.
    func readRouteSelection(layer: Int) -> (ids: [Int32], rw: [Float]) {
        selection(sel: scratch.selected, weights: scratch.rw, layer: layer)
    }

    /// Core of readRouteSelection, reading from ARBITRARY buffers — the batched
    /// route phase snapshots each token's selection into per-token buffers and
    /// reads them all back after one commit.
    func selection(sel: GPUTensor, weights: GPUTensor, layer: Int) -> (ids: [Int32], rw: [Float]) {
        let selPtr = (sel.buffer.contents() + sel.byteOffset).bindMemory(to: Int32.self, capacity: d.k)
        var ids = Array(UnsafeBufferPointer(start: selPtr, count: d.k))
        let wptr = (weights.buffer.contents() + weights.byteOffset).bindMemory(to: Float.self, capacity: d.k)
        var rw = Array(UnsafeBufferPointer(start: wptr, count: d.k))
        let K = max(1, min(d.activeExperts, d.k))
        if K < d.k {
            let keep = (0..<d.k).sorted { rw[$0] > rw[$1] }.prefix(K)
            let origSum = rw.reduce(0, +)
            let keptSum = keep.reduce(Float(0)) { $0 + rw[$1] }
            let scale = keptSum > 0 ? origSum / keptSum : 1
            ids = keep.map { ids[$0] }
            rw = keep.map { rw[$0] * scale }
        }
        usage?.record(layer: layer, ids: ids)
        return (ids, rw)
    }

    /// CPU-write `a` into the head of a shared GPU buffer (safe between commits).
    func writeFloats(_ a: [Float], into t: GPUTensor) {
        a.withUnsafeBytes {
            _ = memcpy(t.buffer.contents().advanced(by: t.byteOffset), $0.baseAddress!, $0.count)
        }
    }

    /// CPU-copy `count` floats between shared GPU buffers (after a commit).
    func copyFloats(from src: GPUTensor, to dst: GPUTensor, count: Int) {
        memcpy(dst.buffer.contents().advanced(by: dst.byteOffset),
               src.buffer.contents().advanced(by: src.byteOffset), count * 4)
    }

    /// Zero s.down6 rows K..d.k-1 so the fixed moeSum6 adds zeros for unused slots.
    func zeroDown6(from K: Int) {
        guard K < d.k else { return }
        let dptr = scratch.down6.buffer.contents().bindMemory(to: Float.self, capacity: d.k * d.nEmbd)
        for r in K..<d.k { for c in 0..<d.nEmbd { dptr[r * d.nEmbd + c] = 0 } }
    }
}
