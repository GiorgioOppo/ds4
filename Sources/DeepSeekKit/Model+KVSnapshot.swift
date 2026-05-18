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

    /// Disk format magic: 'KVS1' (little-endian uint32).
    public static let diskMagic: UInt32 = 0x4B565331

    /// File version (bump on incompatible format change).
    public static let diskVersion: UInt32 = 1

    /// Serializza l'intero snapshot in formato binario, scrivibile
    /// atomicamente a disco. Layout:
    ///
    ///   [ u32 magic 'KVS1' ][ u32 version ][ u32 slotCount ][ u32 reserved ]
    ///   per slot:
    ///     [ u32 layerIndex ][ u8 isMTP ][ u8 role ][ u8 dtype ][ u8 rank ]
    ///     [ rank × i32 shape ][ u64 bytesLen ][ bytesLen × u8 raw ]
    ///   [ u32 endMagic 'KVS1' ]  // sanity check
    ///
    /// `role` è codificato come ordinal di `SlotRole.allCases` (FUTURE:
    /// più robusto come stringa, ma per ora ordinal compact).
    /// `dtype` codificato similarmente.
    public func encodeForDisk() -> Data {
        var data = Data(capacity: 16 + totalBytes + slots.count * 32)
        appendUInt32LE(&data, Self.diskMagic)
        appendUInt32LE(&data, Self.diskVersion)
        appendUInt32LE(&data, UInt32(slots.count))
        appendUInt32LE(&data, 0)  // reserved
        for slot in slots {
            appendUInt32LE(&data, UInt32(slot.layerIndex))
            data.append(slot.isMTP ? 1 : 0)
            data.append(UInt8(Self.encodeRole(slot.role)))
            data.append(UInt8(Self.encodeDType(slot.dtype)))
            data.append(UInt8(slot.shape.count))
            for d in slot.shape {
                appendUInt32LE(&data, UInt32(d))
            }
            appendUInt64LE(&data, UInt64(slot.bytes.count))
            data.append(slot.bytes)
        }
        appendUInt32LE(&data, Self.diskMagic)
        return data
    }

    /// Salva atomicamente a `url`. Tmp + rename per consistency.
    public func save(to url: URL) throws {
        let tmp = url.appendingPathExtension("tmp")
        try encodeForDisk().write(to: tmp, options: .atomic)
        let fm = FileManager.default
        if fm.fileExists(atPath: url.path) {
            _ = try fm.replaceItemAt(url, withItemAt: tmp)
        } else {
            try fm.moveItem(at: tmp, to: url)
        }
    }

    /// Carica da disco. Ritorna nil se il file non esiste, magic
    /// mismatch, version incompatibile o parse fallisce (preferiamo
    /// "ripartire pulito" piuttosto che caricare bytes ambigui in
    /// MTLBuffer).
    public static func load(from url: URL) -> KVCacheSnapshot? {
        guard let data = try? Data(contentsOf: url),
              data.count >= 20 else { return nil }
        var cursor = 0
        guard let magic = readUInt32LE(data, &cursor), magic == diskMagic,
              let version = readUInt32LE(data, &cursor), version == diskVersion,
              let slotCount = readUInt32LE(data, &cursor),
              let _reserved = readUInt32LE(data, &cursor)
        else { return nil }
        _ = _reserved
        var slots: [Slot] = []
        slots.reserveCapacity(Int(slotCount))
        for _ in 0..<slotCount {
            guard let layerIdx = readUInt32LE(data, &cursor),
                  let mtpByte = readUInt8(data, &cursor),
                  let roleByte = readUInt8(data, &cursor),
                  let dtypeByte = readUInt8(data, &cursor),
                  let rank = readUInt8(data, &cursor),
                  let role = decodeRole(Int(roleByte)),
                  let dtype = decodeDType(Int(dtypeByte))
            else { return nil }
            var shape: [Int] = []
            shape.reserveCapacity(Int(rank))
            for _ in 0..<rank {
                guard let d = readUInt32LE(data, &cursor)
                else { return nil }
                shape.append(Int(d))
            }
            guard let bytesLen = readUInt64LE(data, &cursor) else { return nil }
            guard cursor + Int(bytesLen) <= data.count else { return nil }
            let bytes = data.subdata(in: cursor..<(cursor + Int(bytesLen)))
            cursor += Int(bytesLen)
            slots.append(Slot(layerIndex: Int(layerIdx),
                               isMTP: mtpByte != 0,
                               role: role, shape: shape,
                               dtype: dtype, bytes: bytes))
        }
        // Verifica end magic.
        guard let endMagic = readUInt32LE(data, &cursor),
              endMagic == diskMagic
        else { return nil }
        return KVCacheSnapshot(slots: slots)
    }

    // ---- Codifica role/dtype ----

    private static func encodeRole(_ r: SlotRole) -> Int {
        switch r {
        case .mlaKV: return 0
        case .mlaCompKVState: return 1
        case .mlaCompScoreState: return 2
        case .indexerKV: return 3
        case .indexerCompKVState: return 4
        case .indexerCompScoreState: return 5
        }
    }

    private static func decodeRole(_ i: Int) -> SlotRole? {
        switch i {
        case 0: return .mlaKV
        case 1: return .mlaCompKVState
        case 2: return .mlaCompScoreState
        case 3: return .indexerKV
        case 4: return .indexerCompKVState
        case 5: return .indexerCompScoreState
        default: return nil
        }
    }

    /// DType ordinal stabile per la persistenza. Aggiungere nuovi
    /// dtype IN FONDO (mai riordinare) per non rompere la lettura
    /// di snapshot vecchi.
    private static func encodeDType(_ d: DType) -> Int {
        switch d {
        case .f32: return 0
        case .f16: return 1
        case .bf16: return 2
        case .i32: return 3
        case .i8: return 4
        default: return 255  // unknown / unsupported per snapshot
        }
    }

    private static func decodeDType(_ i: Int) -> DType? {
        switch i {
        case 0: return .f32
        case 1: return .f16
        case 2: return .bf16
        case 3: return .i32
        case 4: return .i8
        default: return nil
        }
    }
}

// ---- Byte helpers (host endian assumed little = Apple Silicon) ----

@inline(__always)
private func appendUInt32LE(_ data: inout Data, _ v: UInt32) {
    var le = v.littleEndian
    withUnsafeBytes(of: &le) { data.append(contentsOf: $0) }
}

@inline(__always)
private func appendUInt64LE(_ data: inout Data, _ v: UInt64) {
    var le = v.littleEndian
    withUnsafeBytes(of: &le) { data.append(contentsOf: $0) }
}

@inline(__always)
private func readUInt32LE(_ data: Data, _ cursor: inout Int) -> UInt32? {
    guard cursor + 4 <= data.count else { return nil }
    let v = data.withUnsafeBytes { (raw: UnsafeRawBufferPointer) -> UInt32 in
        raw.load(fromByteOffset: cursor, as: UInt32.self)
    }
    cursor += 4
    return UInt32(littleEndian: v)
}

@inline(__always)
private func readUInt64LE(_ data: Data, _ cursor: inout Int) -> UInt64? {
    guard cursor + 8 <= data.count else { return nil }
    let v = data.withUnsafeBytes { (raw: UnsafeRawBufferPointer) -> UInt64 in
        raw.load(fromByteOffset: cursor, as: UInt64.self)
    }
    cursor += 8
    return UInt64(littleEndian: v)
}

@inline(__always)
private func readUInt8(_ data: Data, _ cursor: inout Int) -> UInt8? {
    guard cursor < data.count else { return nil }
    let v = data[data.startIndex + cursor]
    cursor += 1
    return v
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
