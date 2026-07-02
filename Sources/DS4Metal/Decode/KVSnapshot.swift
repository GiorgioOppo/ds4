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
        guard s.headDim == d.headDim, s.layers.count == nLayers, s.nKeys <= maxKeys else {
            throw KVSnapshotError.shapeMismatch
        }
        for i in 0..<nLayers {
            guard kvRange.contains(i) else { continue }
            let layer = s.layers[i]
            // Re-rotate the chronological window back into the (possibly ring) raw cache.
            // Full cache: physStart == rawStart and it never wraps -> identical to before.
            let rawRows = rawCaches[i].count / d.headDim
            let rows = layer.raw.count / d.headDim
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
        guard let snap, snap.count * rowDim == snap.cacheRows.count else {
            if snap != nil { throw KVSnapshotError.shapeMismatch }
            return
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
