import Foundation
import Metal
import DS4Core

extension StreamingDecoder {
    public func forward(token: Int, pos: Int, nKeys: Int) throws -> [Float] {
        // Fresh sequence: reset the recurrent compressor state (score=-inf, count=0).
        if pos == 0 { for c in compStates { try c?.reset(rt) }; for c in indexStates { try c?.reset(rt) } }
        // Layer 0's expert I/O can start NOW: the token id is known before any
        // GPU work (hash layer -> exact ids), so its fill overlaps embed+route(0).
        if remoteExperts == nil { kickLookahead(after: -1, token: token) }
        try embedToken(token, into: hcA)
        var cur = hcA, other = hcB
        do {
            for i in 0..<nLayers {
                // Per-layer pool drain, like the prefill loops: the command buffers/
                // encoders are autoreleased ObjC objects — without this a LONG
                // generation (hundreds of tokens x ~3 cb/layer) accumulates them
                // for the whole turn instead of freeing at each layer.
                try autoreleasepool {
                    let w = try layerProvider(i)        // LOAD layer i (dense; experts on demand if cached)
                    if i + 1 < nLayers { prefetch?(i + 1) }   // read-ahead next layer (overlaps its I/O)
                    try runLayer(i, w: w, layerRope: DSV4Shape.ropeParams(layer: i),
                                 cur: cur, other: other, pos: pos, nKeys: nKeys, token: token)
                    swap(&cur, &other)
                    // w (and any gathered experts) drop here -> Metal buffers freed (EVICT)
                }
            }
        } catch {
            drainFFN()   // never leave the routed FFN in flight on a torn-down token
            throw error
        }
        // Join the last layer's async FFN: the output head's own commit+wait
        // would cover the GPU ordering, but exportKV/readHC and error paths
        // must find NOTHING in flight — one explicit drain keeps the invariant.
        drainFFN()
        profile.forwards += 1
        return try outputHead(cur)
    }

    // MARK: - Distributed slice execution (pipeline parallelism)
    //
    // These let a node run only PART of the model: the coordinator owns the
    // embedding + output head, each worker owns a contiguous layer range and runs
    // it over an incoming HC state. The HC state (nHC*nEmbd floats) is what crosses
    // the wire between nodes. They reuse embedToken/runLayer/outputHead, so a slice
    // [start,end] is numerically identical to the same layers inside forward().

    /// HC state width that crosses the wire (nHC * nEmbd floats).
    public var hcStateCount: Int { d.nHC * d.nEmbd }

    /// Coordinator: embed `token` into the HC state (the start of the pipeline).
    public func embed(token: Int, pos: Int) throws -> [Float] {
        try embedToken(token, into: hcA)
        return readHC(hcA)
    }

    /// Worker: run layers `start...end` over an incoming HC state at absolute `pos`,
    /// returning the produced HC state to forward to the next slice. Resets only this
    /// slice's recurrent compressor state on a fresh sequence (pos == 0).
    public func forwardSlice(hc hcIn: [Float], pos: Int, nKeys: Int, start: Int, end: Int,
                             token: Int = -1) throws -> [Float] {
        precondition(start >= 0 && end < nLayers && start <= end, "invalid layer slice \(start)...\(end)")
        if pos == 0 { for i in start...end { try compStates[i]?.reset(rt); try indexStates[i]?.reset(rt) } }
        writeFloats(hcIn, into: hcA)
        if remoteExperts == nil { kickLookahead(after: start - 1, token: token) }
        var cur = hcA, other = hcB
        do {
            for i in start...end {
                try autoreleasepool {   // per-layer drain (see forward)
                    let w = try layerProvider(i)
                    if i + 1 <= end { prefetch?(i + 1) }
                    try runLayer(i, w: w, layerRope: DSV4Shape.ropeParams(layer: i),
                                 cur: cur, other: other, pos: pos, nKeys: nKeys, token: token)
                    swap(&cur, &other)
                }
            }
        } catch {
            drainFFN()
            throw error
        }
        drainFFN()   // readHC reads `cur` CPU-side: the async FFN must be complete
        profile.forwards += 1
        return readHC(cur)
    }

    /// Worker, chunked prefill: run layers `start...end` over `hcs.count` consecutive
    /// tokens' HC states starting at absolute `posBase`. Token-outer (numerically
    /// identical to consecutive forwardSlice calls); amortizes the NETWORK round
    /// trip over the chunk — one WORK/RESULT per chunk instead of per token.
    public func forwardSliceBatch(hcs: [[Float]], posBase: Int, start: Int, end: Int,
                                  tokens: [Int]? = nil) throws -> [[Float]] {
        var out: [[Float]] = []
        out.reserveCapacity(hcs.count)
        for (i, hc) in hcs.enumerated() {
            let pos = posBase + i
            out.append(try forwardSlice(hc: hc, pos: pos, nKeys: pos + 1, start: start, end: end,
                                        token: tokens.map { $0[i] } ?? -1))
        }
        return out
    }

    /// Coordinator/last node: run the output head over the final HC state → logits.
    public func head(hc hcIn: [Float]) throws -> [Float] {
        writeFloats(hcIn, into: hcA)
        return try outputHead(hcA)
    }

    /// Read the HC state (nHC*nEmbd floats) out of a GPU buffer.
    private func readHC(_ t: GPUTensor) -> [Float] {
        let n = d.nHC * d.nEmbd
        let p = t.buffer.contents().advanced(by: t.byteOffset).bindMemory(to: Float.self, capacity: n)
        return Array(UnsafeBufferPointer(start: p, count: n))
    }
}
