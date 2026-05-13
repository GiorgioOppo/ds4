import Foundation
import Metal
#if canImport(Darwin)
import Darwin
#endif

/// Safetensors reader. Supports two backings, both exposed as a single
/// shared `MTLBuffer` so `Tensor.load(_:)` returns zero-copy slices:
///
///   1. **mmap** (default; `init(url:)`): `mmap(PROT_READ, MAP_PRIVATE)` +
///      `MTLBuffer(bytesNoCopy:)`. OS demand-pages weight bytes; lower
///      idle RSS but every fresh page costs a fault. Required for files
///      that don't fit in RAM, the only viable mode for a 900 GB V4-Pro
///      checkpoint on a 192-512 GB Mac.
///   2. **preload** (`init(preloadedURL:byteCount:)`): allocate a fresh
///      `storageModeShared` MTLBuffer the size of the file, then
///      `read(2)` the whole file into it. Used when `LoadPlan` decides
///      the entire checkpoint fits in available RAM with margin —
///      eliminates per-tensor page faults at the cost of a one-shot
///      RSS spike at startup.
///
/// File layout (both modes):
///   `[u64 little-endian header length][JSON header][tensor data...]`
public final class SafeTensorsFile {
    public struct Entry: Decodable {
        public let dtype: String
        public let shape: [Int]
        public let dataOffsets: [Int]
        enum CodingKeys: String, CodingKey {
            case dtype, shape
            case dataOffsets = "data_offsets"
        }
    }

    public let url: URL
    public let entries: [String: Entry]
    private let dataStart: Int

    /// Shared MTLBuffer covering the entire file (mmap window or
    /// preloaded copy). All Tensors returned by `load(_:)` reference
    /// this buffer with their `offset` field set to the absolute byte
    /// position in the file. `internal` so the streaming path
    /// (`WeightLoader.adviseShard`) can call `madvise` against the
    /// underlying mmap pages.
    internal let sharedBuffer: MTLBuffer

    // ---- mmap path ----
    /// - Parameter mapNoCache: pass `true` for streaming-mode loads.
    ///   Adds the Darwin `MAP_NOCACHE` flag to mmap so the file's
    ///   pages bypass the unified buffer cache — they're read from
    ///   disk into our mapping but NEVER promoted to the system
    ///   file cache. Saves multi-GB of competing cache when many
    ///   apps + the model share unified memory; the cost is that
    ///   re-faulting a page after `MADV_FREE_REUSABLE` hits disk
    ///   directly instead of a warm cache.
    public init(url: URL, mapNoCache: Bool = false) throws {
        self.url = url

        let fd = open(url.path, O_RDONLY)
        guard fd >= 0 else {
            throw NSError(domain: "SafeTensors", code: 10, userInfo: [
                NSLocalizedDescriptionKey: "open failed: \(url.path)"
            ])
        }
        defer { close(fd) }

        var st = stat()
        guard fstat(fd, &st) == 0 else {
            throw NSError(domain: "SafeTensors", code: 11)
        }
        let fileSize = Int(st.st_size)

        // mmap rounded up to a page boundary so `bytesNoCopy:` accepts it.
        let pageSize = Int(sysconf(_SC_PAGESIZE))
        let alignedSize = ((fileSize + pageSize - 1) / pageSize) * pageSize
        // mmap reserves virtual address space immediately. Physical
        // pages fault in on first read, but the kernel maintains a
        // page table for the whole region from the moment mmap
        // returns — non-trivial overhead for multi-GB ranges.
        MemoryLogger.willAllocate(bytes: alignedSize,
                                   label: "mmap \(url.lastPathComponent)\(mapNoCache ? " NOCACHE" : "")")
        var mapFlags: Int32 = MAP_PRIVATE
        if mapNoCache {
            mapFlags |= MAP_NOCACHE
        }
        guard let raw = mmap(nil, alignedSize, PROT_READ, mapFlags, fd, 0),
              raw != MAP_FAILED else {
            throw NSError(domain: "SafeTensors", code: 12, userInfo: [
                NSLocalizedDescriptionKey: "mmap failed for \(url.path)"
            ])
        }

        let dealloc: (UnsafeMutableRawPointer, Int) -> Void = { p, n in
            munmap(p, n)
        }
        let device = Device.shared.mtl
        guard let buf = device.makeBuffer(
                bytesNoCopy: raw, length: alignedSize,
                options: .storageModeShared,
                deallocator: dealloc) else {
            munmap(raw, alignedSize)
            throw Self.bufferCreationError(url: url, fileSize: fileSize,
                                            alignedSize: alignedSize,
                                            pageSize: pageSize)
        }
        self.sharedBuffer = buf

        let parsed = try Self.parseHeader(at: raw)
        self.entries = parsed.entries
        self.dataStart = parsed.dataStart
    }

    // ---- preload path ----
    /// Allocates a fresh `storageModeShared` MTLBuffer of `byteCount`
    /// bytes, then `read(2)`s the whole file into it. Use only when
    /// the caller has already verified there's enough free RAM
    /// (`LoadPlan.decide`).
    public init(preloadedURL url: URL, byteCount: UInt64) throws {
        self.url = url

        let device = Device.shared.mtl
        MemoryLogger.willAllocate(bytes: Int(byteCount),
                                   label: "preload \(url.lastPathComponent)")
        guard let buf = device.makeBuffer(
                length: Int(byteCount), options: .storageModeShared) else {
            throw Self.bufferCreationError(url: url,
                                            fileSize: Int(byteCount),
                                            alignedSize: Int(byteCount),
                                            pageSize: 0)
        }
        self.sharedBuffer = buf

        let fd = open(url.path, O_RDONLY)
        guard fd >= 0 else {
            throw NSError(domain: "SafeTensors", code: 14, userInfo: [
                NSLocalizedDescriptionKey: "open failed: \(url.path)"
            ])
        }
        defer { close(fd) }

        var off = 0
        let total = Int(byteCount)
        let base = buf.contents()
        while off < total {
            let n = read(fd, base.advanced(by: off), total - off)
            if n > 0 {
                off += n
            } else if n == 0 {
                throw NSError(domain: "SafeTensors", code: 15, userInfo: [
                    NSLocalizedDescriptionKey:
                        "short read: got \(off)/\(total) bytes from \(url.path)"
                ])
            } else if errno != EINTR {
                throw NSError(domain: "SafeTensors", code: 16, userInfo: [
                    NSLocalizedDescriptionKey:
                        "read failed at offset \(off): \(String(cString: strerror(errno)))"
                ])
            }
        }

        let parsed = try Self.parseHeader(at: base)
        self.entries = parsed.entries
        self.dataStart = parsed.dataStart
    }

    /// Returns a Tensor referencing this shard's shared buffer. No
    /// bytes are copied — the GPU reads directly.
    public func load(_ name: String, on device: Device = .shared) throws -> Tensor {
        guard let e = entries[name] else {
            throw NSError(domain: "SafeTensors", code: 3,
                          userInfo: [NSLocalizedDescriptionKey: "missing tensor \(name)"])
        }
        let absOffset = dataStart + e.dataOffsets[0]
        let dt = Self.parseDType(e.dtype)
        return Tensor(shape: e.shape, dtype: dt,
                      buffer: sharedBuffer, offset: absOffset)
    }

    // ---- internals ----
    private struct ParsedHeader {
        let entries: [String: Entry]
        let dataStart: Int
    }

    /// Parses the `[u64 header_len][JSON]` prefix starting at `base`.
    /// Both backings produce a pointer to byte 0 of the file, so this
    /// is shared between `init(url:)` and `init(preloadedURL:)`.
    private static func parseHeader(at base: UnsafeMutableRawPointer) throws -> ParsedHeader {
        let lenPtr = base.bindMemory(to: UInt64.self, capacity: 1)
        let headerLen = Int(lenPtr.pointee)
        let headerData = Data(bytesNoCopy: base.advanced(by: 8),
                              count: headerLen,
                              deallocator: .none)
        let rawJSON = try JSONSerialization.jsonObject(with: headerData) as? [String: Any] ?? [:]
        var parsed: [String: Entry] = [:]
        for (k, v) in rawJSON {
            if k == "__metadata__" { continue }
            let entryData = try JSONSerialization.data(withJSONObject: v)
            parsed[k] = try JSONDecoder().decode(Entry.self, from: entryData)
        }
        return ParsedHeader(entries: parsed, dataStart: 8 + headerLen)
    }

    private static func bufferCreationError(url: URL, fileSize: Int,
                                             alignedSize: Int, pageSize: Int) -> NSError {
        let gib = 1024.0 * 1024.0 * 1024.0
        let maxBuf = Device.shared.mtl.maxBufferLength
        let detail = """
        MTLBuffer creation failed for \(url.path)
          file size     : \(String(format: "%.2f", Double(fileSize) / gib)) GiB
          aligned size  : \(String(format: "%.2f", Double(alignedSize) / gib)) GiB \
        (page=\(pageSize))
          maxBufferLen  : \(String(format: "%.2f", Double(maxBuf) / gib)) GiB
        Likely causes:
          - shard exceeds device.maxBufferLength (lower --shard-size-gb in converter)
          - filesystem is not APFS (exFAT/NTFS mmap may be rejected by Metal)
        """
        return NSError(domain: "SafeTensors", code: 13, userInfo: [
            NSLocalizedDescriptionKey: detail
        ])
    }

    private static func parseDType(_ s: String) -> DType {
        switch s.uppercased() {
        case "F32": return .f32
        case "F16": return .f16
        case "BF16": return .bf16
        case "I32", "U32": return .i32
        case "I64", "U64": return .i64
        case "I8", "U8": return .i8
        case "I4", "U4": return .i4
        case "I2", "U2": return .i2
        case "F8_E4M3", "F8E4M3", "FLOAT8_E4M3FN": return .fp8E4M3
        case "F4_E2M1", "F4E2M1", "FLOAT4_E2M1FN_X2": return .fp4E2M1
        case "F8_E8M0", "F8E8M0", "FLOAT8_E8M0FNU": return .e8m0
        default:
            fatalError("unsupported safetensors dtype: \(s)")
        }
    }
}
