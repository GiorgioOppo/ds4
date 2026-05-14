import Foundation
import Metal
import Darwin

/// Persistent, mmap-backed KV cache storage for a single conversation.
///
/// A `KVCacheFile` owns a file on disk laid out as:
///
///   [ HEADER (4 KiB, fixed) ][ PAYLOAD (page-aligned, variable) ]
///
/// The payload region is exposed as a single `MTLBuffer` over the mmap'd
/// pages (`storageModeShared` + `bytesNoCopy`), and the layers ask for
/// sub-views at known offsets. Apple Silicon's unified memory plus
/// `MADV_RANDOM` lets the kernel keep hot pages resident and page out
/// the rest on memory pressure — no copies on the critical path.
///
/// The header carries the metadata needed for cross-turn reuse (Step 3):
///   - `prefilledTokens`: how far the cache is filled
///   - `historyHash`: 128-bit hash of the messages that produced it
///   - `modelPathHash`: 64-bit hash of `modelDirPath`, so a model swap
///     invalidates the file
///   - `payloadBytes`: declared size of the payload region; mismatch
///     means the model config changed (different layer count, head
///     dim, window, etc.) and the file should be wiped.
///
/// Step 1 only builds and exposes this storage. Steps 2 and 3 plug it
/// into the layers and the inference lifecycle.
public final class KVCacheFile {
    public static let headerBytes: Int = 4096
    public static let magic: UInt32 = 0x4B564331   // 'KVC1' big-endian

    public struct Header {
        public var version: UInt32
        public var payloadBytes: UInt64
        public var prefilledTokens: UInt64
        public var historyHashLow: UInt64
        public var historyHashHigh: UInt64
        public var modelPathHash: UInt64
    }

    public let url: URL
    public let payloadBuffer: MTLBuffer
    public let payloadOffset: Int
    public let payloadBytes: Int

    // Retained so the deallocator can hand them back to the kernel
    // once Metal is done with the buffer.
    private let basePointer: UnsafeMutableRawPointer
    private let totalBytes: Int

    /// Opens (or creates) the file at `url`, sizes it to fit a payload
    /// of `payloadBytes`, mmaps it, and wraps the payload region as a
    /// shared-storage `MTLBuffer`. Each `KVCacheFile` instance owns
    /// the mapping for its lifetime; the `MTLBuffer`'s deallocator
    /// releases the mapping when Metal drops its last reference.
    public init(url: URL,
                 payloadBytes: Int,
                 device: MTLDevice = Device.shared.mtl) throws {
        precondition(payloadBytes > 0, "payloadBytes must be positive")

        let pageSize = Int(getpagesize())
        let alignedPayload = roundUp(payloadBytes, to: pageSize)
        let total = Self.headerBytes + alignedPayload

        // 1) Open + size the file.
        let fd = open(url.path,
                       O_RDWR | O_CREAT,
                       0o644)
        guard fd >= 0 else {
            throw KVCacheFileError.openFailed(errno: errno, path: url.path)
        }
        if ftruncate(fd, off_t(total)) != 0 {
            let e = errno
            close(fd)
            throw KVCacheFileError.truncateFailed(errno: e, path: url.path)
        }

        // 2) Map the whole file. PROT_READ|WRITE + MAP_SHARED so writes
        //    persist; MADV_RANDOM tells the kernel not to prefetch — the
        //    attention kernel reads scattered rows.
        //
        //    Bridge through `Int` so no `UnsafeMutableRawPointer?` ever
        //    appears in this block — every prior attempt to unwrap the
        //    mmap result inside a `guard let` had the type checker
        //    re-widen the binding back to Optional at the use sites.
        //
        //      Int(bitPattern: pointer?)  →  Int   (nil maps to 0)
        //      0  → mmap returned NULL
        //      -1 → mmap returned MAP_FAILED ((void *)-1)
        //      _  → valid mapping; rebuild a non-optional pointer.
        let addr = Int(bitPattern: mmap(nil, total,
                                         PROT_READ | PROT_WRITE,
                                         MAP_SHARED, fd, 0))
        close(fd)  // the mapping holds its own reference
        if addr == 0 || addr == -1 {
            throw KVCacheFileError.mmapFailed(errno: errno, path: url.path)
        }
        // `UnsafeMutableRawPointer(bitPattern:)` is failable only for 0,
        // which we just rejected — force-unwrap is safe and produces a
        // strictly non-optional binding the rest of the function uses.
        let raw: UnsafeMutableRawPointer =
            UnsafeMutableRawPointer(bitPattern: addr)!
        madvise(raw, total, MADV_RANDOM)

        // 3) Wrap as a Metal buffer. Use the full mapping so payload
        //    offsets are simple integer adds; layers will subscript
        //    via `offset:` on `Tensor`.
        let deallocator: (UnsafeMutableRawPointer, Int) -> Void = { ptr, len in
            munmap(ptr, len)
        }
        guard let buf = device.makeBuffer(
            bytesNoCopy: raw,
            length: total,
            options: .storageModeShared,
            deallocator: deallocator)
        else {
            munmap(raw, total)
            throw KVCacheFileError.metalWrapFailed(path: url.path)
        }

        self.url = url
        self.payloadBuffer = buf
        self.payloadOffset = Self.headerBytes
        self.payloadBytes = alignedPayload
        self.basePointer = raw
        self.totalBytes = total

        // 4) Initialise (or validate) the header.
        let h = currentHeader()
        if !isInitialised(h) {
            writeFreshHeader(payloadBytes: UInt64(alignedPayload))
        }
    }

    // ---- header access ----

    /// Reads the current on-disk header. Returns a header with
    /// `version == 0` and zeroed payload size when the file was just
    /// created (uninitialised).
    public func readHeader() -> Header {
        currentHeader()
    }

    /// Stamps a brand-new header into the file. Use this when the
    /// cache contents become invalid (model swap, config change, hash
    /// mismatch) — the payload bytes stay on disk but readers will
    /// treat the cache as empty.
    public func resetHeader(modelPathHash: UInt64) {
        let h = Header(
            version: 1,
            payloadBytes: UInt64(payloadBytes),
            prefilledTokens: 0,
            historyHashLow: 0,
            historyHashHigh: 0,
            modelPathHash: modelPathHash)
        writeHeader(h)
    }

    /// Bumps the prefill checkpoint after a generation step has
    /// extended the cache. Caller must also update the history hash
    /// when the prefilled prefix actually grew.
    public func updateCheckpoint(prefilledTokens: UInt64,
                                  historyHashLow: UInt64,
                                  historyHashHigh: UInt64,
                                  modelPathHash: UInt64) {
        var h = currentHeader()
        h.version = 1
        h.payloadBytes = UInt64(payloadBytes)
        h.prefilledTokens = prefilledTokens
        h.historyHashLow = historyHashLow
        h.historyHashHigh = historyHashHigh
        h.modelPathHash = modelPathHash
        writeHeader(h)
    }

    // ---- payload access ----

    /// Returns a Metal buffer + byte offset describing a sub-region
    /// of the payload. Layers compose a `Tensor` over this by passing
    /// `buffer:` and `offset:` to `Tensor.init`. Step 2 will add a
    /// `Tensor.mapped(file:offset:shape:dtype:)` convenience.
    public func region(offset: Int, length: Int) -> (MTLBuffer, Int) {
        precondition(offset >= 0)
        precondition(length > 0)
        precondition(offset + length <= payloadBytes,
                      "region [\(offset), \(offset+length)) exceeds payload size \(payloadBytes)")
        return (payloadBuffer, payloadOffset + offset)
    }

    // ---- internals ----

    private func currentHeader() -> Header {
        var h = Header(version: 0, payloadBytes: 0,
                        prefilledTokens: 0,
                        historyHashLow: 0, historyHashHigh: 0,
                        modelPathHash: 0)
        // Layout (little-endian on Apple Silicon):
        //   u32 magic | u32 version | u64 payloadBytes
        //   u64 prefilledTokens | u64 hashLow | u64 hashHigh
        //   u64 modelPathHash
        let p = basePointer.assumingMemoryBound(to: UInt8.self)
        let magic = readUInt32(p, 0)
        guard magic == Self.magic else { return h }
        h.version          = readUInt32(p, 4)
        h.payloadBytes     = readUInt64(p, 8)
        h.prefilledTokens  = readUInt64(p, 16)
        h.historyHashLow   = readUInt64(p, 24)
        h.historyHashHigh  = readUInt64(p, 32)
        h.modelPathHash    = readUInt64(p, 40)
        return h
    }

    private func isInitialised(_ h: Header) -> Bool {
        h.version != 0 && h.payloadBytes == UInt64(payloadBytes)
    }

    private func writeFreshHeader(payloadBytes: UInt64) {
        let h = Header(version: 1,
                        payloadBytes: payloadBytes,
                        prefilledTokens: 0,
                        historyHashLow: 0,
                        historyHashHigh: 0,
                        modelPathHash: 0)
        writeHeader(h)
    }

    private func writeHeader(_ h: Header) {
        let p = basePointer.assumingMemoryBound(to: UInt8.self)
        writeUInt32(p, 0,  Self.magic)
        writeUInt32(p, 4,  h.version)
        writeUInt64(p, 8,  h.payloadBytes)
        writeUInt64(p, 16, h.prefilledTokens)
        writeUInt64(p, 24, h.historyHashLow)
        writeUInt64(p, 32, h.historyHashHigh)
        writeUInt64(p, 40, h.modelPathHash)
        // Force the dirty header pages to disk so a crash mid-turn
        // doesn't leave a stale prefill checkpoint pointing at
        // partially-written payload data.
        msync(basePointer, Self.headerBytes, MS_ASYNC)
    }
}

public enum KVCacheFileError: Error, LocalizedError {
    case openFailed(errno: Int32, path: String)
    case truncateFailed(errno: Int32, path: String)
    case mmapFailed(errno: Int32, path: String)
    case metalWrapFailed(path: String)

    public var errorDescription: String? {
        switch self {
        case .openFailed(let e, let p):
            return "open(\(p)) failed: \(String(cString: strerror(e)))"
        case .truncateFailed(let e, let p):
            return "ftruncate(\(p)) failed: \(String(cString: strerror(e)))"
        case .mmapFailed(let e, let p):
            return "mmap(\(p)) failed: \(String(cString: strerror(e)))"
        case .metalWrapFailed(let p):
            return "MTLDevice.makeBuffer(bytesNoCopy:) failed for \(p)"
        }
    }
}

// ---- byte helpers (host endian = little on Apple Silicon) ----

@inline(__always)
private func roundUp(_ x: Int, to multiple: Int) -> Int {
    let r = x % multiple
    return r == 0 ? x : x + (multiple - r)
}

@inline(__always)
private func readUInt32(_ p: UnsafePointer<UInt8>, _ off: Int) -> UInt32 {
    var v: UInt32 = 0
    memcpy(&v, p + off, 4)
    return v
}

@inline(__always)
private func readUInt64(_ p: UnsafePointer<UInt8>, _ off: Int) -> UInt64 {
    var v: UInt64 = 0
    memcpy(&v, p + off, 8)
    return v
}

@inline(__always)
private func writeUInt32(_ p: UnsafeMutablePointer<UInt8>, _ off: Int, _ v: UInt32) {
    var x = v
    memcpy(p + off, &x, 4)
}

@inline(__always)
private func writeUInt64(_ p: UnsafeMutablePointer<UInt8>, _ off: Int, _ v: UInt64) {
    var x = v
    memcpy(p + off, &x, 8)
}
