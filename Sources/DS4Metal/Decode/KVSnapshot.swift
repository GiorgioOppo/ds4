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
    public init(rawStart: Int, raw: [Float], comp: CompSnapshot?) {
        self.rawStart = rawStart; self.raw = raw; self.comp = comp
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
            let raw = readFloats(rawCaches[i], from: rawStart * d.headDim, count: rows * d.headDim)
            var comp: CompSnapshot? = nil
            if let c = compStates[i] {
                let coff = c.ratio == 4 ? 2 : 1
                let stateLen = coff * c.ratio * c.width
                comp = CompSnapshot(count: c.count,
                                    stateKv: readFloats(c.stateKv, from: 0, count: stateLen),
                                    stateScore: readFloats(c.stateScore, from: 0, count: stateLen),
                                    cacheRows: readFloats(c.cache, from: 0, count: c.count * d.headDim))
            }
            layers.append(KVLayerSnapshot(rawStart: rawStart, raw: raw, comp: comp))
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
            writeFloatsArray(layer.raw, into: rawCaches[i], at: layer.rawStart * d.headDim)
            if let c = compStates[i] {
                try c.reset(rt)
                guard let comp = layer.comp, comp.count * d.headDim == comp.cacheRows.count else {
                    if layer.comp != nil { throw KVSnapshotError.shapeMismatch }
                    continue
                }
                writeFloatsArray(comp.stateKv, into: c.stateKv, at: 0)
                writeFloatsArray(comp.stateScore, into: c.stateScore, at: 0)
                writeFloatsArray(comp.cacheRows, into: c.cache, at: 0)
                c.count = comp.count
            }
        }
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
            memcpy(t.buffer.contents().advanced(by: t.byteOffset + offset * 4),
                   $0.baseAddress!, $0.count)
        }
    }
}

public enum KVSnapshotError: Error { case shapeMismatch }
