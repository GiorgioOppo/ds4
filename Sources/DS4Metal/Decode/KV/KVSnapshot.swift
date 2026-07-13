import Foundation

// CPU-side snapshot of the decoder's KV state at a token boundary — everything
// needed to resume generation at position `nKeys` without re-prefilling:
// per layer the raw SWA window rows (the only raw rows attention can see) and,
// on compressed layers, the full NSA compressor state (recurrent accumulators +
// emitted rows). Mirrors what ds4_kvstore.c checkpoints to disk.

public struct CompSnapshot: Sendable, Equatable {
    public var count: Int            // emitted compressed rows
    public var stateKv: [Float]      // recurrent accumulator [coff*ratio × width]
    public var stateScore: [Float]   // recurrent score state [coff*ratio × width]
    public var cacheRows: [Float]    // emitted rows [count × headDim]
    public init(count: Int, stateKv: [Float], stateScore: [Float], cacheRows: [Float]) {
        self.count = count; self.stateKv = stateKv
        self.stateScore = stateScore; self.cacheRows = cacheRows
    }
}

public struct KVLayerSnapshot: Sendable, Equatable {
    public var rawStart: Int         // absolute position of the first stored raw row
    public var raw: [Float]          // SWA window rows [(nKeys-rawStart) × headDim]
    public var comp: CompSnapshot?   // nil on non-compressed layers
    public var idx: CompSnapshot?    // NSA indexer compressor (ratio-4 layers only)
    public init(rawStart: Int, raw: [Float], comp: CompSnapshot?, idx: CompSnapshot? = nil) {
        self.rawStart = rawStart; self.raw = raw; self.comp = comp; self.idx = idx
    }
}

public struct KVSnapshot: Sendable, Equatable {
    public var nKeys: Int            // tokens in the KV when exported
    public var headDim: Int
    public var layers: [KVLayerSnapshot]
    public init(nKeys: Int, headDim: Int, layers: [KVLayerSnapshot]) {
        self.nKeys = nKeys; self.headDim = headDim; self.layers = layers
    }
}

extension StreamingDecoder {
    /// Export the live KV/compressor state for the first `nKeys` positions.
    /// Call only between generations (no in-flight GPU work). Layers outside the
    /// allocated `kvRange` (distributed slices) export empty.
    public func exportKV(nKeys: Int) -> KVSnapshot {
        var layers: [KVLayerSnapshot] = []
        layers.reserveCapacity(nLayers)
        for i in 0..<nLayers {
            guard kvRange.contains(i), nKeys > 0 else {
                layers.append(KVLayerSnapshot(rawStart: 0, raw: [], comp: nil)); continue
            }
            let rawStart = max(0, nKeys - d.nSWA)
            let rows = nKeys - rawStart
            // De-rotate the (possibly ring-buffer) raw cache into a chronological window.
            // Full cache: physStart == rawStart and it never wraps -> identical to before.
            let rawRows = rawCaches[i].count / d.headDim
            let physStart = rawStart % rawRows
            let raw: [Float] = (physStart + rows <= rawRows)
                ? readFloats(rawCaches[i], from: physStart * d.headDim, count: rows * d.headDim)
                : readFloats(rawCaches[i], from: physStart * d.headDim, count: (rawRows - physStart) * d.headDim)
                    + readFloats(rawCaches[i], from: 0, count: (rows - (rawRows - physStart)) * d.headDim)
            layers.append(KVLayerSnapshot(rawStart: rawStart, raw: raw,
                                          comp: snapshotComp(compStates[i]),
                                          idx: snapshotComp(indexStates[i])))
        }
        return KVSnapshot(nKeys: nKeys, headDim: d.headDim, layers: layers)
    }

    /// Restore a previously exported state: positions 0..<snapshot.nKeys become
    /// valid (raw window at absolute offsets + full compressor state), so the
    /// caller can continue prefilling from `snapshot.nKeys`. Call only between
    /// generations.
    public func importKV(_ s: KVSnapshot) throws {
        try beginImportKV(nKeys: s.nKeys, headDim: s.headDim, layerCount: s.layers.count)
        for i in 0..<nLayers { try importKVLayer(s.layers[i], at: i) }
    }

    /// Shape gate for a STREAMED restore (disk KV): callers that parse a
    /// checkpoint one layer at a time validate here BEFORE reading any tensor
    /// body, then feed each layer to `importKVLayer` and release it — peak RAM
    /// is one layer's slabs, not the whole checkpoint.
    public func beginImportKV(nKeys: Int, headDim: Int, layerCount: Int) throws {
        guard headDim == d.headDim, layerCount == nLayers, nKeys <= maxKeys else {
            throw KVSnapshotError.shapeMismatch
        }
    }

    /// Restore ONE layer (raw window + compressors). Layers outside the
    /// allocated `kvRange` (distributed slices) are skipped, mirroring importKV.
    public func importKVLayer(_ layer: KVLayerSnapshot, at i: Int) throws {
        guard i >= 0, i < nLayers else { throw KVSnapshotError.shapeMismatch }
        guard kvRange.contains(i) else { return }
        // Re-rotate the chronological window back into the (possibly ring) raw cache.
        // Full cache: physStart == rawStart and it never wraps -> identical to before.
        let rawRows = rawCaches[i].count / d.headDim
        let rows = layer.raw.count / d.headDim
        // The lengths come from the FILE: a corrupt/truncated checkpoint must be
        // rejected, never memcpy'd past the ring (C: "session tensor is smaller
        // than the payload", ds4.c:23477).
        guard layer.rawStart >= 0, rows <= rawRows,
              layer.raw.count == rows * d.headDim else {
            throw KVSnapshotError.shapeMismatch
        }
        let physStart = layer.rawStart % rawRows
        if physStart + rows <= rawRows {
            writeFloatsArray(layer.raw, into: rawCaches[i], at: physStart * d.headDim)
        } else {
            let seg1 = (rawRows - physStart) * d.headDim
            writeFloatsArray(Array(layer.raw[0..<seg1]), into: rawCaches[i], at: physStart * d.headDim)
            writeFloatsArray(Array(layer.raw[seg1...]), into: rawCaches[i], at: 0)
        }
        try restoreComp(compStates[i], from: layer.comp, rowDim: d.headDim)
        try restoreComp(indexStates[i], from: layer.idx, rowDim: d.nIndexerHeadDim)
    }

    /// Read one compressor's full state (recurrent accumulators + emitted rows).
    private func snapshotComp(_ c: CompressorState?) -> CompSnapshot? {
        guard let c else { return nil }
        let coff = c.ratio == 4 ? 2 : 1
        let stateLen = coff * c.ratio * c.width
        return CompSnapshot(count: c.count,
                            stateKv: readFloats(c.stateKv, from: 0, count: stateLen),
                            stateScore: readFloats(c.stateScore, from: 0, count: stateLen),
                            cacheRows: readFloats(c.cache, from: 0, count: c.count * c.headDim))
    }

    private func restoreComp(_ c: CompressorState?, from snap: CompSnapshot?, rowDim: Int) throws {
        guard let c else { return }
        try c.reset(rt)
        guard let snap else { return }
        // Every length below comes from the FILE — validate against the live
        // tensors' capacities before any memcpy (C: raw_cap / "invalid compressed
        // row count", ds4.c:24669/24702).
        let coff = c.ratio == 4 ? 2 : 1
        let stateLen = coff * c.ratio * c.width
        // Bound on the EMISSION SCHEDULE (1 row per `ratio` positions over at
        // most maxKeys positions), not on the allocation (maxComp carries +8
        // rows of slack): a corrupt checkpoint must never restore a row count
        // no legitimate run can reach — the lazy indexer-scoring skip PROVES
        // "top-K never activates" from exactly this bound.
        guard rowDim == c.headDim,
              snap.count >= 0, snap.count <= maxKeys / c.ratio,
              snap.count * rowDim == snap.cacheRows.count,
              snap.stateKv.count == stateLen,
              snap.stateScore.count == stateLen else {
            throw KVSnapshotError.shapeMismatch
        }
        writeFloatsArray(snap.stateKv, into: c.stateKv, at: 0)
        writeFloatsArray(snap.stateScore, into: c.stateScore, at: 0)
        writeFloatsArray(snap.cacheRows, into: c.cache, at: 0)
        c.count = snap.count
    }

    private func readFloats(_ t: GPUTensor, from offset: Int, count: Int) -> [Float] {
        guard count > 0 else { return [] }
        let p = t.buffer.contents().advanced(by: t.byteOffset + offset * 4)
            .bindMemory(to: Float.self, capacity: count)
        return Array(UnsafeBufferPointer(start: p, count: count))
    }

    private func writeFloatsArray(_ a: [Float], into t: GPUTensor, at offset: Int) {
        guard !a.isEmpty else { return }
        a.withUnsafeBytes {
            _ = memcpy(t.buffer.contents().advanced(by: t.byteOffset + offset * 4),
                       $0.baseAddress!, $0.count)
        }
    }
}

public enum KVSnapshotError: Error { case shapeMismatch }
