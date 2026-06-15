import Foundation
#if canImport(Darwin)
import Darwin
#endif

// Faithful Swift port of the GGUF loader in ds4.c (model_open, parse_metadata,
// parse_tensors, the cursor reader, the tensor-type table, and the metadata
// accessors). The model file is mmap'd once and tensor bytes stay in place;
// callers reach weights by absolute offset into the mapping.
//
// Differences from C, none behavioral: where ds4.c calls ds4_die() (aborting the
// process) this throws GGUFError, and KV keys / tensor names are decoded to Swift
// String at parse time (lookups are identical). Phase 2 of the C->Swift port.

public enum GGUFValueType: UInt32, Sendable {
    case uint8 = 0, int8 = 1, uint16 = 2, int16 = 3, uint32 = 4, int32 = 5
    case float32 = 6, bool = 7, string = 8, array = 9
    case uint64 = 10, int64 = 11, float64 = 12
}

public enum GGUFError: Error, CustomStringConvertible {
    case cannotOpen(String)
    case mmapFailed(String)
    case tooSmall
    case notGGUF
    case unsupportedVersion(UInt32)
    case truncated(UInt64)
    case message(String)

    public var description: String {
        switch self {
        case .cannotOpen(let p): return "cannot open model: \(p)"
        case .mmapFailed(let p): return "cannot mmap model: \(p)"
        case .tooSmall: return "model file is too small to be GGUF"
        case .notGGUF: return "model is not a GGUF file"
        case .unsupportedVersion(let v): return "only GGUF v3 is supported (got \(v))"
        case .truncated(let at): return "truncated GGUF file at byte \(at)"
        case .message(let m): return m
        }
    }
}

/// GGUF tensor-type descriptor (matches gguf_type_info / gguf_types[] in ds4.c).
public struct GGUFTypeInfo: Sendable {
    public let name: String
    public let blockElems: UInt32
    public let blockBytes: UInt32
}

public enum GGUF {
    public static let magic: UInt32 = 0x4655_4747 // "GGUF" little-endian
    public static let maxDims = 8

    /// The sparse type table from ds4.c (indices without an entry are unknown).
    public static let typeTable: [UInt32: GGUFTypeInfo] = [
        0:  .init(name: "f32",     blockElems: 1,   blockBytes: 4),
        1:  .init(name: "f16",     blockElems: 1,   blockBytes: 2),
        2:  .init(name: "q4_0",    blockElems: 32,  blockBytes: 18),
        3:  .init(name: "q4_1",    blockElems: 32,  blockBytes: 20),
        6:  .init(name: "q5_0",    blockElems: 32,  blockBytes: 22),
        7:  .init(name: "q5_1",    blockElems: 32,  blockBytes: 24),
        8:  .init(name: "q8_0",    blockElems: 32,  blockBytes: 34),
        9:  .init(name: "q8_1",    blockElems: 32,  blockBytes: 40),
        10: .init(name: "q2_k",    blockElems: 256, blockBytes: 84),
        11: .init(name: "q3_k",    blockElems: 256, blockBytes: 110),
        12: .init(name: "q4_k",    blockElems: 256, blockBytes: 144),
        13: .init(name: "q5_k",    blockElems: 256, blockBytes: 176),
        14: .init(name: "q6_k",    blockElems: 256, blockBytes: 210),
        15: .init(name: "q8_k",    blockElems: 256, blockBytes: 292),
        16: .init(name: "iq2_xxs", blockElems: 256, blockBytes: 66),
        17: .init(name: "iq2_xs",  blockElems: 256, blockBytes: 74),
        18: .init(name: "iq3_xxs", blockElems: 256, blockBytes: 98),
        19: .init(name: "iq1_s",   blockElems: 256, blockBytes: 110),
        20: .init(name: "iq4_nl",  blockElems: 256, blockBytes: 50),
        21: .init(name: "iq3_s",   blockElems: 256, blockBytes: 110),
        22: .init(name: "iq2_s",   blockElems: 256, blockBytes: 82),
        23: .init(name: "iq4_xs",  blockElems: 256, blockBytes: 136),
        24: .init(name: "i8",      blockElems: 1,   blockBytes: 1),
        25: .init(name: "i16",     blockElems: 1,   blockBytes: 2),
        26: .init(name: "i32",     blockElems: 1,   blockBytes: 4),
        27: .init(name: "i64",     blockElems: 1,   blockBytes: 8),
        28: .init(name: "f64",     blockElems: 1,   blockBytes: 8),
        29: .init(name: "iq1_m",   blockElems: 256, blockBytes: 56),
        30: .init(name: "bf16",    blockElems: 1,   blockBytes: 2),
    ]

    public static func typeInfo(_ type: UInt32) -> GGUFTypeInfo? { typeTable[type] }
    public static func typeName(_ type: UInt32) -> String { typeTable[type]?.name ?? "unknown" }

    /// Port of tensor_nbytes: block-rounded byte size, or nil for unknown types.
    public static func tensorNBytes(type: UInt32, elements: UInt64) -> UInt64? {
        guard let info = typeTable[type], info.blockElems != 0 else { return nil }
        let be = UInt64(info.blockElems)
        let bb = UInt64(info.blockBytes)
        let blocks = (elements + be - 1) / be
        guard blocks <= UInt64.max / bb else { return nil }
        return blocks * bb
    }

    static func alignUp(_ value: UInt64, _ alignment: UInt64) -> UInt64 {
        let rem = value % alignment
        return rem == 0 ? value : value + alignment - rem
    }
}

/// Bounds-checked little-endian reader over the mmap (port of ds4_cursor).
struct GGUFCursor {
    let base: UnsafePointer<UInt8>
    let size: UInt64
    var pos: UInt64

    init(base: UnsafePointer<UInt8>, size: UInt64, pos: UInt64 = 0) {
        self.base = base; self.size = size; self.pos = pos
    }

    private func has(_ n: UInt64) throws {
        if n > size || pos > size - n { throw GGUFError.truncated(pos) }
    }

    mutating func skip(_ n: UInt64) throws { try has(n); pos += n }

    mutating func u32() throws -> UInt32 {
        try has(4)
        let v = UnsafeRawPointer(base + Int(pos)).loadUnaligned(as: UInt32.self)
        pos += 4
        return v
    }

    mutating func u64() throws -> UInt64 {
        try has(8)
        let v = UnsafeRawPointer(base + Int(pos)).loadUnaligned(as: UInt64.self)
        pos += 8
        return v
    }

    mutating func read<T>(as type: T.Type) throws -> T {
        let n = UInt64(MemoryLayout<T>.size)
        try has(n)
        let v = UnsafeRawPointer(base + Int(pos)).loadUnaligned(as: T.self)
        pos += n
        return v
    }

    /// A GGUF string: u64 length then raw UTF-8 bytes (not NUL terminated).
    mutating func string() throws -> String {
        let len = try u64()
        try has(len)
        let bytes = UnsafeBufferPointer(start: base + Int(pos), count: Int(len))
        pos += len
        return String(decoding: bytes, as: UTF8.self)
    }

    /// Raw bytes of a GGUF string (no UTF-8 decode). Used by the tokenizer,
    /// which keys its tables by exact bytes like the C implementation.
    mutating func stringBytes() throws -> [UInt8] {
        let len = try u64()
        try has(len)
        let buf = UnsafeBufferPointer(start: base + Int(pos), count: Int(len))
        pos += len
        return Array(buf)
    }

    static func scalarSize(_ type: UInt32) -> UInt64 {
        switch type {
        case GGUFValueType.uint8.rawValue, GGUFValueType.int8.rawValue, GGUFValueType.bool.rawValue:
            return 1
        case GGUFValueType.uint16.rawValue, GGUFValueType.int16.rawValue:
            return 2
        case GGUFValueType.uint32.rawValue, GGUFValueType.int32.rawValue, GGUFValueType.float32.rawValue:
            return 4
        case GGUFValueType.uint64.rawValue, GGUFValueType.int64.rawValue, GGUFValueType.float64.rawValue:
            return 8
        default:
            return 0
        }
    }

    /// Port of skip_value: advance past a metadata value of `type`.
    mutating func skipValue(_ type: UInt32, depth: Int = 0) throws {
        if depth > 8 { throw GGUFError.message("metadata array nesting is too deep at byte \(pos)") }

        let scalar = GGUFCursor.scalarSize(type)
        if scalar != 0 { try skip(scalar); return }

        if type == GGUFValueType.string.rawValue {
            _ = try string()
            return
        }
        if type == GGUFValueType.array.rawValue {
            let itemType = try u32()
            let len = try u64()
            let itemSize = GGUFCursor.scalarSize(itemType)
            if itemSize != 0 {
                if len > UInt64.max / itemSize { throw GGUFError.message("metadata array is too large at byte \(pos)") }
                try skip(len * itemSize)
                return
            }
            var i: UInt64 = 0
            while i < len { try skipValue(itemType, depth: depth + 1); i += 1 }
            return
        }
        throw GGUFError.message("unknown GGUF metadata type at byte \(pos)")
    }
}

/// A loaded GGUF model: header, metadata table, tensor directory, and the live
/// mmap. Mirrors ds4_model + model_open/parse_metadata/parse_tensors.
public final class GGUFModel {

    public struct Tensor: Sendable {
        public let name: String
        public let dims: [UInt64]
        public let type: UInt32
        public let elements: UInt64
        public let relOffset: UInt64
        public let absOffset: UInt64
        public let bytes: UInt64
        public var typeName: String { GGUF.typeName(type) }
    }

    struct KV { let key: String; let type: UInt32; let valuePos: UInt64 }

    public let version: UInt32
    public let alignment: UInt64
    public let size: UInt64
    public let tensorDataPos: UInt64
    public let maxTensorBytes: UInt64
    public let tensors: [Tensor]

    private let map: UnsafeMutableRawPointer
    private let base: UnsafePointer<UInt8>
    private let fd: Int32
    private let kvs: [KV]
    private let kvIndex: [String: Int]
    private let tensorIndex: [String: Int]

    public var n_kv: UInt64 { UInt64(kvs.count) }
    public var n_tensors: UInt64 { UInt64(tensors.count) }
    /// Base of the mmap, for reading tensor bytes by absolute offset.
    public var mapBase: UnsafeRawPointer { UnsafeRawPointer(base) }

    /// Hint the OS to read these mmap byte ranges ahead (POSIX_MADV_WILLNEED), so
    /// the SSD I/O for the NEXT layer overlaps the current layer's compute. A pure
    /// advisory on the read-only mapping — it cannot affect correctness. Static so
    /// a caller can run it on a background queue capturing only Sendable values
    /// (the base pointer + the precomputed ranges), never the model itself.
    public static func prefetch(base: UnsafeRawPointer, ranges: [(offset: UInt64, bytes: UInt64)]) {
        let page = Int(getpagesize())
        let m = UnsafeMutableRawPointer(mutating: base)
        for r in ranges where r.bytes > 0 {
            let start = Int(r.offset)
            let lo = start - (start % page)                       // page-align down
            _ = posix_madvise(m.advanced(by: lo), (start - lo) + Int(r.bytes), POSIX_MADV_WILLNEED)
        }
    }

    /// Open and map the GGUF. `metalMapping` uses MAP_SHARED (for no-copy Metal
    /// buffers); otherwise MAP_PRIVATE.
    public init(path: String, metalMapping: Bool = true, prefetchCPU: Bool = false) throws {
        let fd = open(path, O_RDONLY)
        if fd == -1 { throw GGUFError.cannotOpen(path) }

        var st = stat()
        if fstat(fd, &st) == -1 { close(fd); throw GGUFError.cannotOpen(path) }
        if st.st_size < 32 { close(fd); throw GGUFError.tooSmall }

        let flags = metalMapping ? MAP_SHARED : MAP_PRIVATE
        guard let region = mmap(nil, Int(st.st_size), PROT_READ, flags, fd, 0),
              region != MAP_FAILED else {
            close(fd); throw GGUFError.mmapFailed(path)
        }

        self.fd = fd
        self.map = region
        self.size = UInt64(st.st_size)
        let base = UnsafePointer(region.assumingMemoryBound(to: UInt8.self))
        self.base = base

        var c = GGUFCursor(base: base, size: self.size)
        let magic = try Self.guard(close: fd, region: region, size: self.size) { try c.u32() }
        if magic != GGUF.magic { munmap(region, Int(st.st_size)); close(fd); throw GGUFError.notGGUF }
        let version = try Self.guard(close: fd, region: region, size: self.size) { try c.u32() }
        let nTensors = try Self.guard(close: fd, region: region, size: self.size) { try c.u64() }
        let nKV = try Self.guard(close: fd, region: region, size: self.size) { try c.u64() }
        if version != 3 { munmap(region, Int(st.st_size)); close(fd); throw GGUFError.unsupportedVersion(version) }
        self.version = version

        // --- parse_metadata ---
        var kvs: [KV] = []
        kvs.reserveCapacity(Int(nKV))
        var alignment: UInt64 = 32
        do {
            for _ in 0..<nKV {
                let key = try c.string()
                let type = try c.u32()
                let valuePos = c.pos
                if key == "general.alignment" && type == GGUFValueType.uint32.rawValue {
                    var tmp = GGUFCursor(base: base, size: self.size, pos: valuePos)
                    if let a = try? tmp.u32(), a != 0 { alignment = UInt64(a) }
                }
                try c.skipValue(type)
                kvs.append(KV(key: key, type: type, valuePos: valuePos))
            }
        } catch {
            munmap(region, Int(st.st_size)); close(fd); throw error
        }
        self.alignment = alignment
        self.kvs = kvs
        var kvIndex: [String: Int] = [:]
        for (i, kv) in kvs.enumerated() where kvIndex[kv.key] == nil { kvIndex[kv.key] = i }
        self.kvIndex = kvIndex

        // --- parse_tensors ---
        var tensors: [Tensor] = []
        tensors.reserveCapacity(Int(nTensors))
        var rawTensors: [(name: String, dims: [UInt64], type: UInt32, elements: UInt64, relOffset: UInt64, bytes: UInt64)] = []
        do {
            for _ in 0..<nTensors {
                let name = try c.string()
                let ndim = try c.u32()
                if ndim == 0 || ndim > UInt32(GGUF.maxDims) {
                    throw GGUFError.message("tensor has an unsupported number of dimensions")
                }
                var dims: [UInt64] = []
                var elements: UInt64 = 1
                for _ in 0..<ndim {
                    let d = try c.u64()
                    if d != 0 && elements > UInt64.max / d {
                        throw GGUFError.message("tensor element count overflow")
                    }
                    elements *= d
                    dims.append(d)
                }
                let type = try c.u32()
                let relOffset = try c.u64()
                let bytes = GGUF.tensorNBytes(type: type, elements: elements) ?? 0
                rawTensors.append((name, dims, type, elements, relOffset, bytes))
            }
        } catch {
            munmap(region, Int(st.st_size)); close(fd); throw error
        }

        let dataPos = GGUF.alignUp(c.pos, alignment)
        self.tensorDataPos = dataPos

        var maxBytes: UInt64 = 0
        do {
            for r in rawTensors {
                if r.relOffset > UInt64.max - dataPos {
                    throw GGUFError.message("tensor offset overflow")
                }
                let absOffset = dataPos + r.relOffset
                if r.bytes != 0 && (absOffset > self.size || r.bytes > self.size - absOffset) {
                    throw GGUFError.message("tensor points outside GGUF file")
                }
                if r.bytes > maxBytes { maxBytes = r.bytes }
                tensors.append(Tensor(name: r.name, dims: r.dims, type: r.type,
                                      elements: r.elements, relOffset: r.relOffset,
                                      absOffset: absOffset, bytes: r.bytes))
            }
        } catch {
            munmap(region, Int(st.st_size)); close(fd); throw error
        }
        self.maxTensorBytes = maxBytes
        self.tensors = tensors
        var tensorIndex: [String: Int] = [:]
        for (i, t) in tensors.enumerated() where tensorIndex[t.name] == nil { tensorIndex[t.name] = i }
        self.tensorIndex = tensorIndex

        if !metalMapping && prefetchCPU {
            #if canImport(Darwin)
            _ = posix_madvise(region, Int(st.st_size), POSIX_MADV_WILLNEED)
            #endif
        }
    }

    deinit {
        munmap(map, Int(size))
        close(fd)
    }

    /// Helper that tears the mapping down if a header read throws mid-init.
    private static func `guard`<T>(close fd: Int32, region: UnsafeMutableRawPointer,
                                   size: UInt64, _ body: () throws -> T) throws -> T {
        do { return try body() }
        catch { munmap(region, Int(size)); close(fd); throw error }
    }

    // MARK: - Metadata accessors (port of model_get_*)

    private func kv(_ key: String) -> KV? {
        guard let i = kvIndex[key] else { return nil }
        return kvs[i]
    }

    private func cursor(at pos: UInt64) -> GGUFCursor {
        GGUFCursor(base: base, size: size, pos: pos)
    }

    public func string(_ key: String) -> String? {
        guard let kv = kv(key), kv.type == GGUFValueType.string.rawValue else { return nil }
        var c = cursor(at: kv.valuePos)
        return try? c.string()
    }

    public func u32(_ key: String) -> UInt32? {
        guard let kv = kv(key), kv.type == GGUFValueType.uint32.rawValue else { return nil }
        var c = cursor(at: kv.valuePos)
        return try? c.u32()
    }

    public func u64(_ key: String) -> UInt64? {
        guard let kv = kv(key), kv.type == GGUFValueType.uint64.rawValue else { return nil }
        var c = cursor(at: kv.valuePos)
        return try? c.u64()
    }

    /// Port of model_get_u64_compat: accept u64 or u32.
    public func u64Compat(_ key: String) -> UInt64? {
        guard let kv = kv(key) else { return nil }
        var c = cursor(at: kv.valuePos)
        switch kv.type {
        case GGUFValueType.uint64.rawValue: return try? c.u64()
        case GGUFValueType.uint32.rawValue: return (try? c.u32()).map(UInt64.init)
        default: return nil
        }
    }

    /// Port of model_get_f32_compat: accept f32/f64/u32/i32.
    public func f32Compat(_ key: String) -> Float? {
        guard let kv = kv(key) else { return nil }
        var c = cursor(at: kv.valuePos)
        switch kv.type {
        case GGUFValueType.float32.rawValue: return try? c.read(as: Float.self)
        case GGUFValueType.float64.rawValue: return (try? c.read(as: Double.self)).map { Float($0) }
        case GGUFValueType.uint32.rawValue: return (try? c.u32()).map { Float($0) }
        case GGUFValueType.int32.rawValue: return (try? c.read(as: Int32.self)).map { Float($0) }
        default: return nil
        }
    }

    public func bool(_ key: String) -> Bool? {
        guard let kv = kv(key), kv.type == GGUFValueType.bool.rawValue else { return nil }
        var c = cursor(at: kv.valuePos)
        return (try? c.read(as: UInt8.self)).map { $0 != 0 }
    }

    /// Port of model_get_array: returns the element type, count, and data offset.
    public func array(_ key: String) -> (type: UInt32, len: UInt64, dataPos: UInt64)? {
        guard let kv = kv(key), kv.type == GGUFValueType.array.rawValue else { return nil }
        var c = cursor(at: kv.valuePos)
        guard let t = try? c.u32(), let n = try? c.u64() else { return nil }
        return (t, n, c.pos)
    }

    public func findTensor(_ name: String) -> Tensor? {
        guard let i = tensorIndex[name] else { return nil }
        return tensors[i]
    }

    /// Read a GGUF array whose elements are u32 or i32, sign-preserved as Int64.
    public func intArray(_ key: String) -> [Int64]? {
        guard let (t, n, dataPos) = array(key) else { return nil }
        var c = cursor(at: dataPos)
        var out: [Int64] = []
        out.reserveCapacity(Int(n))
        for _ in 0..<n {
            switch t {
            case GGUFValueType.uint32.rawValue:
                guard let v = try? c.u32() else { return nil }; out.append(Int64(v))
            case GGUFValueType.int32.rawValue:
                guard let v = try? c.read(as: Int32.self) else { return nil }; out.append(Int64(v))
            default: return nil
            }
        }
        return out
    }

    /// Read a GGUF array whose elements are f32 or f64, as Double.
    public func floatArray(_ key: String) -> [Double]? {
        guard let (t, n, dataPos) = array(key) else { return nil }
        var c = cursor(at: dataPos)
        var out: [Double] = []
        out.reserveCapacity(Int(n))
        for _ in 0..<n {
            switch t {
            case GGUFValueType.float32.rawValue:
                guard let v = try? c.read(as: Float.self) else { return nil }; out.append(Double(v))
            case GGUFValueType.float64.rawValue:
                guard let v = try? c.read(as: Double.self) else { return nil }; out.append(v)
            default: return nil
            }
        }
        return out
    }

    /// Read a GGUF array of strings as raw byte arrays (token / merge tables).
    public func stringArrayBytes(_ key: String) -> [[UInt8]]? {
        guard let (t, n, dataPos) = array(key), t == GGUFValueType.string.rawValue else { return nil }
        var c = cursor(at: dataPos)
        var out: [[UInt8]] = []
        out.reserveCapacity(Int(n))
        for _ in 0..<n {
            guard let bytes = try? c.stringBytes() else { return nil }
            out.append(bytes)
        }
        return out
    }
}
