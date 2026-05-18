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

    /// File version. v1 = no compression. v2 = aggiunge `diskDtype`
    /// per slot per supportare salvataggi quantizzati (F16, BF16,
    /// ecc.). Reader auto-detects la versione.
    public static let diskVersionV1: UInt32 = 1
    public static let diskVersionV2: UInt32 = 2

    /// Compressione opt-in per il salvataggio su disco. La memoria
    /// in-process resta nei dtype originali (es. F32 per kvState);
    /// la quantizzazione avviene solo al `save` e la dequantize al
    /// `load`. Niente kernel Metal richiesto — tutto CPU-side
    /// (Float16 nativo Swift su Apple Silicon).
    ///
    /// Trade-off:
    ///   - `.f32`: lossless, file size = original.
    ///   - `.f16`: half precision (11-bit mantissa). ~2× compression.
    ///     Range overflow possibile su valori > 65504; il save
    ///     satura silenziosamente. Per KV cache la perdita è
    ///     percettivamente nulla.
    ///   - `.bf16`: brain-float16 (7-bit mantissa, stesso esponente
    ///     di F32). 2× compression, range completo F32 ma
    ///     precisione ridotta. Implementato via shift bit (no
    ///     hardware Apple Silicon, ma trivial in Swift).
    ///
    /// FP8 / INT8 / INT4 richiedono scale per riga e sono più
    /// invasive; lasciate come future enhancement.
    public enum DiskCompression: UInt8, Codable, Sendable {
        case f32  = 0
        case f16  = 1
        case bf16 = 2
    }

    /// Serializza l'intero snapshot in formato binario, scrivibile
    /// atomicamente a disco. Layout v2:
    ///
    ///   [u32 magic 'KVS1'][u32 version=2][u32 slotCount][u32 reserved]
    ///   per slot:
    ///     [u32 layerIndex][u8 isMTP][u8 role][u8 origDtype][u8 diskDtype]
    ///     [u8 rank][3 byte pad][rank × u32 shape dims]
    ///     [u64 bytesLen][bytesLen × u8 raw]
    ///   [u32 endMagic 'KVS1']
    ///
    /// `origDtype` è il dtype dell'in-memory tensor (es. F32).
    /// `diskDtype` è quello effettivamente sul disco (può essere
    /// uguale o un formato quantizzato). Al load, se diversi, viene
    /// applicata la dequantizzazione per ripristinare `origDtype`.
    public func encodeForDisk(compression: DiskCompression = .f32) -> Data {
        let useV2 = compression != .f32
        let version = useV2 ? Self.diskVersionV2 : Self.diskVersionV1
        var data = Data(capacity: 16 + totalBytes + slots.count * 32)
        appendUInt32LE(&data, Self.diskMagic)
        appendUInt32LE(&data, version)
        appendUInt32LE(&data, UInt32(slots.count))
        appendUInt32LE(&data, 0)  // reserved
        for slot in slots {
            appendUInt32LE(&data, UInt32(slot.layerIndex))
            data.append(slot.isMTP ? 1 : 0)
            data.append(UInt8(Self.encodeRole(slot.role)))
            data.append(UInt8(Self.encodeDType(slot.dtype)))
            if useV2 {
                // diskDtype byte (1 = F16, 2 = BF16, 0 = same as
                // origDtype). Applichiamo solo agli slot F32 in
                // memoria; gli altri restano verbatim.
                let useDiskCompression = (slot.dtype == .f32) && (compression != .f32)
                let diskByte: UInt8 = useDiskCompression
                    ? UInt8(Self.encodeDiskCompression(compression))
                    : 0
                data.append(diskByte)
            }
            data.append(UInt8(slot.shape.count))
            if useV2 {
                // Pad 3 bytes per allineamento u32 della shape
                data.append(0); data.append(0); data.append(0)
            }
            for d in slot.shape {
                appendUInt32LE(&data, UInt32(d))
            }
            let payload: Data
            if useV2 && slot.dtype == .f32 && compression != .f32 {
                payload = Self.quantizeF32Bytes(slot.bytes,
                                                  to: compression)
            } else {
                payload = slot.bytes
            }
            appendUInt64LE(&data, UInt64(payload.count))
            data.append(payload)
        }
        appendUInt32LE(&data, Self.diskMagic)
        return data
    }

    /// Salva atomicamente a `url`. Tmp + rename per consistency.
    public func save(to url: URL,
                      compression: DiskCompression = .f32) throws {
        let tmp = url.appendingPathExtension("tmp")
        try encodeForDisk(compression: compression)
            .write(to: tmp, options: .atomic)
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
              let version = readUInt32LE(data, &cursor),
              version == diskVersionV1 || version == diskVersionV2,
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
                  let role = decodeRole(Int(roleByte)),
                  let dtype = decodeDType(Int(dtypeByte))
            else { return nil }
            var diskCompression: DiskCompression = .f32
            if version == diskVersionV2 {
                guard let diskByte = readUInt8(data, &cursor),
                      let comp = decodeDiskCompression(Int(diskByte))
                else { return nil }
                diskCompression = comp
            }
            guard let rank = readUInt8(data, &cursor) else { return nil }
            if version == diskVersionV2 {
                // Skip 3 pad bytes.
                guard cursor + 3 <= data.count else { return nil }
                cursor += 3
            }
            var shape: [Int] = []
            shape.reserveCapacity(Int(rank))
            for _ in 0..<rank {
                guard let d = readUInt32LE(data, &cursor)
                else { return nil }
                shape.append(Int(d))
            }
            guard let bytesLen = readUInt64LE(data, &cursor) else { return nil }
            guard cursor + Int(bytesLen) <= data.count else { return nil }
            let rawBytes = data.subdata(in: cursor..<(cursor + Int(bytesLen)))
            cursor += Int(bytesLen)
            // Dequantize se il disk dtype differisce dall'orig dtype.
            let bytes: Data
            if diskCompression != .f32 && dtype == .f32 {
                bytes = dequantizeToF32Bytes(rawBytes, from: diskCompression)
            } else {
                bytes = rawBytes
            }
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

    // ---- Codifica DiskCompression ----

    private static func encodeDiskCompression(_ c: DiskCompression) -> Int {
        return Int(c.rawValue)
    }

    private static func decodeDiskCompression(_ i: Int) -> DiskCompression? {
        return DiskCompression(rawValue: UInt8(i))
    }

    // ---- Quantize / dequantize CPU-side ----

    /// Converte un blob F32 in F16 o BF16. Per F16 usa il tipo Swift
    /// nativo (hardware-accelerated su Apple Silicon). Per BF16 usa
    /// bit-shift manuale (F32 → BF16 = truncate dei 16 bit bassi
    /// della mantissa).
    static func quantizeF32Bytes(_ src: Data,
                                   to compression: DiskCompression) -> Data {
        let count = src.count / MemoryLayout<Float>.stride
        guard count > 0 else { return Data() }
        return src.withUnsafeBytes { (raw: UnsafeRawBufferPointer) -> Data in
            let srcPtr = raw.baseAddress!.assumingMemoryBound(to: Float.self)
            switch compression {
            case .f32:
                return src   // no-op
            case .f16:
                var out = Data(count: count * 2)
                out.withUnsafeMutableBytes { (raw: UnsafeMutableRawBufferPointer) in
                    let dst = raw.baseAddress!.assumingMemoryBound(to: Float16.self)
                    for i in 0..<count {
                        dst[i] = Float16(srcPtr[i])
                    }
                }
                return out
            case .bf16:
                var out = Data(count: count * 2)
                out.withUnsafeMutableBytes { (raw: UnsafeMutableRawBufferPointer) in
                    let dst = raw.baseAddress!.assumingMemoryBound(to: UInt16.self)
                    for i in 0..<count {
                        // F32 → BF16: prendi i 16 bit alti (sign + exp + 7 mantissa).
                        // Rounding nearest-even sarebbe più accurato; per ora
                        // truncate (RTZ) — sufficienti per KV cache.
                        let bits = srcPtr[i].bitPattern
                        dst[i] = UInt16(truncatingIfNeeded: bits >> 16)
                    }
                }
                return out
            }
        }
    }

    /// Converte un blob F16 o BF16 in F32 (espansione 2×). Usato al
    /// load dello snapshot per restore in memoria al dtype originale.
    static func dequantizeToF32Bytes(_ src: Data,
                                       from compression: DiskCompression) -> Data {
        let count = src.count / 2
        guard count > 0 else { return Data() }
        return src.withUnsafeBytes { (raw: UnsafeRawBufferPointer) -> Data in
            var out = Data(count: count * MemoryLayout<Float>.stride)
            out.withUnsafeMutableBytes { (rawOut: UnsafeMutableRawBufferPointer) in
                let dst = rawOut.baseAddress!.assumingMemoryBound(to: Float.self)
                switch compression {
                case .f32:
                    // Identity (shouldn't happen but defensive)
                    let srcF = raw.baseAddress!.assumingMemoryBound(to: Float.self)
                    for i in 0..<count { dst[i] = srcF[i] }
                case .f16:
                    let srcH = raw.baseAddress!.assumingMemoryBound(to: Float16.self)
                    for i in 0..<count { dst[i] = Float(srcH[i]) }
                case .bf16:
                    let srcU = raw.baseAddress!.assumingMemoryBound(to: UInt16.self)
                    for i in 0..<count {
                        // BF16 → F32: shift back into the high half,
                        // low 16 bits stay zero.
                        let bits = UInt32(srcU[i]) << 16
                        dst[i] = Float(bitPattern: bits)
                    }
                }
            }
            return out
        }
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
