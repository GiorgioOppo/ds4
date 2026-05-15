import Foundation

/// Captures the runtime KV cache state of every layer (main +
/// MTP blocks) so it can be later restored — either in the same
/// process (B2: live RAM round-trip) or persisted to a mmap-backed
/// `.vec` file (B3: project KV reuse across app launches).
///
/// The snapshot stores raw bytes side-by-side with shape + dtype so
/// the restore path can detect a mismatched checkpoint and refuse
/// to load garbage into the buffers. This is intentionally a value
/// type that carries `Data` blobs — `Transformer.snapshotKVCache`
/// performs an explicit `memcpy` from each `MTLBuffer.contents()`
/// so the snapshot is decoupled from the live cache (callers can
/// keep mutating the model afterwards).
public struct KVCacheSnapshot {
    /// Which logical buffer inside one transformer block a slot
    /// describes. The blob alone is meaningless — restore needs
    /// the role to route it back to the right layer field.
    public enum SlotRole: String, Codable {
        case mlaKV
        case mlaCompKVState
        case mlaCompScoreState
        case indexerKV
        case indexerCompKVState
        case indexerCompScoreState
    }

    public struct Slot {
        public let layerIndex: Int
        public let isMTP: Bool
        public let role: SlotRole
        public let shape: [Int]
        public let dtype: DType
        /// Raw little-endian bytes, length == `shape.product *
        /// stride(dtype)`. Owns the storage; not aliased back into
        /// the live cache.
        public let bytes: Data

        public init(layerIndex: Int, isMTP: Bool, role: SlotRole,
                    shape: [Int], dtype: DType, bytes: Data) {
            self.layerIndex = layerIndex; self.isMTP = isMTP
            self.role = role; self.shape = shape
            self.dtype = dtype; self.bytes = bytes
        }
    }

    public var slots: [Slot]

    public init(slots: [Slot]) { self.slots = slots }

    /// Total bytes the snapshot would occupy on disk. Useful as a
    /// progress / quota signal for the persistence path.
    public var totalBytes: Int {
        slots.reduce(0) { $0 + $1.bytes.count }
    }
}

extension Transformer {

    /// Capture the live KV cache state of every layer into a value
    /// snapshot. Cheap relative to the GPU work that produced the
    /// cache, but still O(cache bytes) — for V4 with windowSize
    /// 4096 the snapshot is on the order of a few hundred MB, so
    /// the caller should only call this when about to persist or
    /// hand off (not "every N forwards").
    ///
    /// Must be called between forward passes — concurrent reads on
    /// a buffer whose GPU writes haven't drained will see garbage.
    /// `Transformer.forward` commits and waits per layer, so
    /// "between forwards" simply means "not from inside a layer
    /// callback".
    public func snapshotKVCache() -> KVCacheSnapshot {
        var slots: [KVCacheSnapshot.Slot] = []
        for (i, block) in layers.enumerated() {
            Self.collectSlots(from: block,
                               layerIndex: i, isMTP: false,
                               into: &slots)
        }
        for (i, m) in mtp.enumerated() {
            Self.collectSlots(from: m.block,
                               layerIndex: i, isMTP: true,
                               into: &slots)
        }
        return KVCacheSnapshot(slots: slots)
    }

    /// Re-populate every layer's KV cache from a snapshot.
    /// Two-pass design:
    ///   1) restore single-slot fields (MLA kvCache, Indexer kvCache)
    ///      directly as each is encountered;
    ///   2) collect kvState/scoreState pairs per compressor and
    ///      restore them atomically — `Compressor.restoreState`
    ///      takes both together so the rolling decode state stays
    ///      consistent. A compressor present in the snapshot only
    ///      partially (one of the two states missing) is skipped
    ///      entirely so the next forward lazily re-allocates a
    ///      fresh, zeroed pair.
    ///
    /// Slots whose `(layerIndex, isMTP)` is out of range are
    /// silently dropped — the snapshot might have been produced
    /// against a different revision of the model and we'd rather
    /// load what we can than refuse the whole blob.
    public func restoreKVCache(_ snap: KVCacheSnapshot) {
        struct CompKey: Hashable {
            let layerIndex: Int
            let isMTP: Bool
            let isIndexerSide: Bool
        }
        var compStates: [CompKey: (kv: KVCacheSnapshot.Slot?,
                                     score: KVCacheSnapshot.Slot?)] = [:]

        for slot in snap.slots {
            let block: Block?
            if slot.isMTP {
                block = (slot.layerIndex < mtp.count)
                    ? mtp[slot.layerIndex].block : nil
            } else {
                block = (slot.layerIndex < layers.count)
                    ? layers[slot.layerIndex] : nil
            }
            guard let b = block else { continue }
            switch slot.role {
            case .mlaKV:
                b.attn.restoreKVCacheBytes(
                    shape: slot.shape, dtype: slot.dtype, bytes: slot.bytes)
            case .indexerKV:
                b.attn.indexer?.restoreKVCacheBytes(
                    shape: slot.shape, dtype: slot.dtype, bytes: slot.bytes)
            case .mlaCompKVState:
                let k = CompKey(layerIndex: slot.layerIndex,
                                  isMTP: slot.isMTP, isIndexerSide: false)
                compStates[k, default: (nil, nil)].kv = slot
            case .mlaCompScoreState:
                let k = CompKey(layerIndex: slot.layerIndex,
                                  isMTP: slot.isMTP, isIndexerSide: false)
                compStates[k, default: (nil, nil)].score = slot
            case .indexerCompKVState:
                let k = CompKey(layerIndex: slot.layerIndex,
                                  isMTP: slot.isMTP, isIndexerSide: true)
                compStates[k, default: (nil, nil)].kv = slot
            case .indexerCompScoreState:
                let k = CompKey(layerIndex: slot.layerIndex,
                                  isMTP: slot.isMTP, isIndexerSide: true)
                compStates[k, default: (nil, nil)].score = slot
            }
        }

        for (key, pair) in compStates {
            guard let kvSlot = pair.kv, let scoreSlot = pair.score else {
                // Partial pair: let the compressor re-alloc lazily
                // on next forward. Better than restoring half of a
                // rolling state with garbage on the other side.
                continue
            }
            let block: Block?
            if key.isMTP {
                block = (key.layerIndex < mtp.count)
                    ? mtp[key.layerIndex].block : nil
            } else {
                block = (key.layerIndex < layers.count)
                    ? layers[key.layerIndex] : nil
            }
            guard let b = block else { continue }
            let compressor: Compressor?
            if key.isIndexerSide {
                compressor = b.attn.indexer?.compressor
            } else {
                compressor = b.attn.compressor
            }
            guard let comp = compressor else { continue }
            let kvT = Tensor.empty(shape: kvSlot.shape, dtype: kvSlot.dtype)
            kvT.writeBytes(kvSlot.bytes)
            let scT = Tensor.empty(shape: scoreSlot.shape, dtype: scoreSlot.dtype)
            scT.writeBytes(scoreSlot.bytes)
            comp.restoreState(kvState: kvT, scoreState: scT)
        }
    }

    // MARK: - internals

    private static func collectSlots(from block: Block,
                                      layerIndex: Int, isMTP: Bool,
                                      into slots: inout [KVCacheSnapshot.Slot]) {
        let mla = block.attn
        if let kv = mla.kvCache {
            slots.append(.init(layerIndex: layerIndex, isMTP: isMTP,
                                role: .mlaKV,
                                shape: kv.shape, dtype: kv.dtype,
                                bytes: kv.readBytes()))
        }
        if let comp = mla.compressor {
            if let s = comp.kvState {
                slots.append(.init(layerIndex: layerIndex, isMTP: isMTP,
                                    role: .mlaCompKVState,
                                    shape: s.shape, dtype: s.dtype,
                                    bytes: s.readBytes()))
            }
            if let s = comp.scoreState {
                slots.append(.init(layerIndex: layerIndex, isMTP: isMTP,
                                    role: .mlaCompScoreState,
                                    shape: s.shape, dtype: s.dtype,
                                    bytes: s.readBytes()))
            }
        }
        if let idx = mla.indexer {
            if let kv = idx.kvCache {
                slots.append(.init(layerIndex: layerIndex, isMTP: isMTP,
                                    role: .indexerKV,
                                    shape: kv.shape, dtype: kv.dtype,
                                    bytes: kv.readBytes()))
            }
            if let s = idx.compressor.kvState {
                slots.append(.init(layerIndex: layerIndex, isMTP: isMTP,
                                    role: .indexerCompKVState,
                                    shape: s.shape, dtype: s.dtype,
                                    bytes: s.readBytes()))
            }
            if let s = idx.compressor.scoreState {
                slots.append(.init(layerIndex: layerIndex, isMTP: isMTP,
                                    role: .indexerCompScoreState,
                                    shape: s.shape, dtype: s.dtype,
                                    bytes: s.readBytes()))
            }
        }
    }

}
