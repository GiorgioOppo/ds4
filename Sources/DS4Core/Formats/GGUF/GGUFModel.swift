import Foundation
#if canImport(Darwin)
import Darwin
#endif

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

    /// Path the model was opened from (for auxiliary descriptors / diagnostics).
    public let path: String

    private let map: UnsafeMutableRawPointer
    private let base: UnsafePointer<UInt8>
    private let fd: Int32
    /// Lazily-opened second descriptor with F_NOCACHE (see uncachedFD()).
    private var noCacheFD: Int32 = -1
    private let kvs: [KV]
    private let kvIndex: [String: Int]
    private let tensorIndex: [String: Int]

    public var n_kv: UInt64 { UInt64(kvs.count) }
    public var n_tensors: UInt64 { UInt64(tensors.count) }
    /// Base of the mmap, for reading tensor bytes by absolute offset.
    public var mapBase: UnsafeRawPointer { UnsafeRawPointer(base) }

    /// A second descriptor on the same file with F_NOCACHE set: pread()s on it
    /// go (mostly) straight from disk to the destination buffer WITHOUT filling
    /// the unified page cache. Used for expert streaming (DS4_EXPERT_PREAD):
    /// the slabs get copied into wired GPU pools anyway, so caching them a
    /// second time only evicts the dense weights on tight-RAM machines.
    /// Lazily opened, closed in deinit. Call from ONE thread (model load);
    /// the returned fd itself is safe for concurrent pread().
    public func uncachedFD() -> Int32? {
        if noCacheFD >= 0 { return noCacheFD }
        let f = open(path, O_RDONLY)
        guard f >= 0 else { return nil }
        _ = fcntl(f, F_NOCACHE, 1)
        noCacheFD = f
        return f
    }

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
        if fd == -1 {
            throw GGUFError.cannotOpen("\(path) — \(String(cString: strerror(errno)))")
        }

        var st = stat()
        if fstat(fd, &st) == -1 {
            let detail = String(cString: strerror(errno))
            close(fd)
            throw GGUFError.cannotOpen("\(path) — \(detail)")
        }
        if st.st_size < 32 { close(fd); throw GGUFError.tooSmall }

        let flags = metalMapping ? MAP_SHARED : MAP_PRIVATE
        guard let region = mmap(nil, Int(st.st_size), PROT_READ, flags, fd, 0),
              region != MAP_FAILED else {
            close(fd); throw GGUFError.mmapFailed(path)
        }

        self.path = path
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
        if noCacheFD >= 0 { close(noCacheFD) }
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

