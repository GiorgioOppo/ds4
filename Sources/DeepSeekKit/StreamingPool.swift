import Foundation
import Metal
#if canImport(Darwin)
import Darwin
#endif

/// Pool architecture for true single-shard-at-a-time streaming on
/// Apple Silicon. Replaces the `MTLBuffer(bytesNoCopy:)` over mmap
/// path used by `.mmap`/`.preload` strategies for the `.streaming`
/// strategy.
///
/// **Why this exists.** On Apple Silicon, when you wrap an mmap'd
/// region as a Metal buffer via `bytesNoCopy:`, the driver pins
/// those pages whenever the buffer is referenced. `madvise(...)`
/// returns success but `mincore()` shows zero pages dropped — the
/// kernel won't evict pages the driver claims. With 147 GB of
/// V4-Flash mmap'd through 45 such buffers, the system runs at
/// 100% memory pressure permanently and either crashes or makes
/// inference unusably slow.
///
/// **The pool.** Two MTLBuffer slots, both `.storageModeShared`,
/// allocated ONCE at load time:
///
///   - `sharedSlot`: holds concatenated data of "shared" shards
///     (top-level tensors: embed, head, RMSNorm gains, hc_head_*).
///     `mlock`'d so it never gets evicted. Sized at typical ~2 GB
///     for V4-Flash.
///   - `rotatingSlot`: sized to the largest per-layer shard
///     (~3.4 GB for V4-Flash). At runtime, `ensureLayer(K)`
///     `pread`s layer K's shard data into this slot. Layer K's
///     `Tensor`s all point into this single buffer with offsets
///     within the shard; when we `pread` shard K+1 over the same
///     bytes, layer K's tensors are invalidated (their data is
///     overwritten) but the buffer object is the same, so the
///     refs remain "valid" — we just have to ensure we never use
///     a previous layer's tensors after `ensureLayer(K+1)`.
///
/// Total memory: shared + rotating + KV cache + activations ≈ 6 GB
/// for V4-Flash, fits comfortably in 16 GB with the OS and GUI.
public final class StreamingPool {
    public let sharedSlot: MTLBuffer
    public let rotatingSlot: MTLBuffer

    public enum Slot: Sendable { case shared, rotating }

    /// Resolved location of every tensor in the model. Built once
    /// at init from parsed shard headers.
    public struct TensorLocation: Sendable {
        public let slot: Slot
        public let offsetInSlot: Int
        public let shape: [Int]
        public let dtype: DType
    }
    public let tensorLocation: [String: TensorLocation]

    /// One entry per non-shared shard: which file to `pread`
    /// from and what byte range.
    private struct LayerShardSource {
        let url: URL
        let dataStart: Int
        let dataByteCount: Int
        /// Pre-allocated open fd or -1 to open lazily. Today we
        /// open on demand because we may have 43 shards and only
        /// 1 active at a time.
    }
    private let layerToShard: [Int: LayerShardSource]
    private var currentRotatingLayer: Int = -1
    private let rotatingCapacity: Int

    /// Build the pool.
    ///
    /// - Parameter shards: parsed headers, in directory order
    ///   (same as `WeightLoader.discoverShards`).
    /// - Parameter shardLayers: from `WeightLoader.buildShardLayers`
    ///   — same shard ownership classification we already use for
    ///   `.streaming`'s madvise path.
    public init(shards: [SafeTensorsHeader],
                 shardLayers: [Int]) throws {
        precondition(shards.count == shardLayers.count)

        // Partition shards: shared vs per-layer.
        var sharedIndices: [Int] = []
        var layerToShard: [Int: LayerShardSource] = [:]
        for (i, owner) in shardLayers.enumerated() {
            if owner == -1 {
                sharedIndices.append(i)
            } else {
                layerToShard[owner] = LayerShardSource(
                    url: shards[i].url,
                    dataStart: shards[i].dataStart,
                    dataByteCount: shards[i].dataByteCount)
            }
        }
        self.layerToShard = layerToShard

        // Allocate sharedSlot at size = sum of shared shard data
        // sizes. Aligned to 4096 bytes so we can `mlock` cleanly.
        let sharedTotal = sharedIndices.reduce(0) { $0 + shards[$1].dataByteCount }
        let sharedAligned = ((sharedTotal + 4095) / 4096) * 4096
        MemoryLogger.willAllocate(bytes: sharedAligned, label: "sharedSlot")
        guard let sharedBuf = Device.shared.mtl.makeBuffer(
                length: max(sharedAligned, 16),
                options: .storageModeShared) else {
            throw NSError(domain: "StreamingPool", code: 30, userInfo: [
                NSLocalizedDescriptionKey: "sharedSlot allocation failed"
            ])
        }
        self.sharedSlot = sharedBuf

        // Allocate rotatingSlot at size = max per-layer shard
        // data size (so any layer's shard fits).
        let maxLayerBytes = layerToShard.values.map(\.dataByteCount).max() ?? 0
        let rotatingAligned = ((maxLayerBytes + 4095) / 4096) * 4096
        MemoryLogger.willAllocate(bytes: rotatingAligned, label: "rotatingSlot")
        guard let rotBuf = Device.shared.mtl.makeBuffer(
                length: max(rotatingAligned, 16),
                options: .storageModeShared) else {
            throw NSError(domain: "StreamingPool", code: 31, userInfo: [
                NSLocalizedDescriptionKey: "rotatingSlot allocation failed"
            ])
        }
        self.rotatingSlot = rotBuf
        self.rotatingCapacity = rotatingAligned

        // Fill sharedSlot via pread of each shared shard's data
        // section, recording the offset for each tensor name.
        var locations: [String: TensorLocation] = [:]
        var sharedOffsetCursor = 0
        for i in sharedIndices {
            let header = shards[i]
            try Self.preadInto(buffer: sharedSlot,
                                bufferOffset: sharedOffsetCursor,
                                url: header.url,
                                fileOffset: header.dataStart,
                                byteCount: header.dataByteCount)
            // Record tensor locations within sharedSlot.
            for (name, entry) in header.entries {
                let inShard = entry.dataOffsets[0]
                locations[name] = TensorLocation(
                    slot: .shared,
                    offsetInSlot: sharedOffsetCursor + inShard,
                    shape: entry.shape,
                    dtype: Self.parseDType(entry.dtype))
            }
            sharedOffsetCursor += header.dataByteCount
        }

        // Record per-layer tensor locations (slot=rotating).
        for (layerK, _) in layerToShard {
            // Find the shard this layer owns.
            guard let shardIdx = shardLayers.firstIndex(of: layerK) else { continue }
            let header = shards[shardIdx]
            for (name, entry) in header.entries {
                let inShard = entry.dataOffsets[0]
                locations[name] = TensorLocation(
                    slot: .rotating,
                    offsetInSlot: inShard,
                    shape: entry.shape,
                    dtype: Self.parseDType(entry.dtype))
            }
        }
        self.tensorLocation = locations

        // mlock the shared slot so it's truly resident-and-pinned.
        if let addr = sharedSlot.contents() as UnsafeMutableRawPointer?,
           sharedAligned > 0 {
            let rc = mlock(addr, sharedAligned)
            if rc == 0 {
                Self.log(String(format:
                    "[pool] sharedSlot %.2f GB mlocked (%d shard(s))\n",
                    Double(sharedAligned) / 1_073_741_824,
                    sharedIndices.count))
            } else {
                let errnoStr = String(cString: strerror(errno))
                Self.log(String(format:
                    "[pool] sharedSlot mlock FAILED: %s — proceeding unpinned\n",
                    errnoStr))
            }
        }
        Self.log(String(format:
            "[pool] rotatingSlot capacity %.2f GB (largest layer shard)\n",
            Double(rotatingAligned) / 1_073_741_824))
    }

    /// Gate diagnostic output behind `MemoryLogger.enabled` so the
    /// pool stays quiet in normal runs and verbose under the
    /// `DEEPSEEK_MEM_LOG=1` env var.
    @inline(__always)
    private static func log(_ s: String) {
        guard MemoryLogger.enabled else { return }
        FileHandle.standardError.write(Data(s.utf8))
    }

    /// Ensure layer K's shard is loaded into `rotatingSlot`.
    /// Idempotent: no-op if already loaded.
    public func ensureLayer(_ K: Int) throws {
        if currentRotatingLayer == K { return }
        guard let src = layerToShard[K] else {
            // Layer has no per-layer shard (probably entirely in
            // shared shard, e.g. a one-block toy model). Nothing
            // to load.
            return
        }
        precondition(src.dataByteCount <= rotatingCapacity,
                     "shard for layer \(K) exceeds rotatingSlot capacity")
        try Self.preadInto(buffer: rotatingSlot,
                            bufferOffset: 0,
                            url: src.url,
                            fileOffset: src.dataStart,
                            byteCount: src.dataByteCount)
        currentRotatingLayer = K
        Self.log(String(format:
            "[pool] layer=%d shard preaded %.2f GB into rotatingSlot\n",
            K, Double(src.dataByteCount) / 1_073_741_824))
    }

    // ---- internals ----

    private static func preadInto(buffer: MTLBuffer,
                                   bufferOffset: Int,
                                   url: URL,
                                   fileOffset: Int,
                                   byteCount: Int) throws {
        let fd = open(url.path, O_RDONLY)
        guard fd >= 0 else {
            throw NSError(domain: "StreamingPool", code: 32, userInfo: [
                NSLocalizedDescriptionKey: "open failed: \(url.path)"
            ])
        }
        defer { close(fd) }

        let base = buffer.contents().advanced(by: bufferOffset)
        var off = 0
        while off < byteCount {
            let n = pread(fd, base.advanced(by: off),
                          byteCount - off,
                          off_t(fileOffset + off))
            if n > 0 {
                off += n
            } else if n == 0 {
                throw NSError(domain: "StreamingPool", code: 33, userInfo: [
                    NSLocalizedDescriptionKey:
                        "short pread (got \(off)/\(byteCount)) from \(url.lastPathComponent)"
                ])
            } else if errno != EINTR {
                let errnoStr = String(cString: strerror(errno))
                throw NSError(domain: "StreamingPool", code: 34, userInfo: [
                    NSLocalizedDescriptionKey:
                        "pread failed at \(off): \(errnoStr) — \(url.lastPathComponent)"
                ])
            }
        }
    }

    /// Same switch as SafeTensorsFile.parseDType — copied here so
    /// StreamingPool doesn't reach into SafeTensorsFile's private
    /// surface.
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
            fatalError("StreamingPool: unsupported safetensors dtype: \(s)")
        }
    }
}
