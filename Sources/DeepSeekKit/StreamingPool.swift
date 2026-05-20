import Foundation
import Metal
#if canImport(Darwin)
import Darwin
#endif

/// Pool architecture for sliding-window streaming on Apple Silicon.
/// Replaces the `MTLBuffer(bytesNoCopy:)` over mmap path used by
/// `.mmap`/`.preload` strategies for the `.streaming` strategy.
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
/// **The pool.** Two MTLBuffer regions, both `.storageModeShared`,
/// allocated ONCE at load time:
///
///   - `sharedSlot`: holds concatenated data of "shared" shards
///     (top-level tensors: embed, head, RMSNorm gains, hc_head_*).
///     `mlock`'d so it never gets evicted.
///   - `rotatingSlot`: a single MTLBuffer carved into **N sub-slots**
///     of `slotSize` bytes each (slotSize = aligned max per-layer
///     shard size). Each per-layer shard K is permanently assigned
///     to sub-slot `K mod N` — the slot index is a property of K,
///     not of access order. Layer K's `Tensor` objects all point
///     into the rotating buffer at offset
///     `(K mod N) * slotSize + tensorOffsetInShard`, fixed at init.
///
/// **Why modular assignment.** `Tensor` captures `MTLBuffer + offset`
/// at construction time (Assembly.swift builds every block's
/// weights up-front, before the first forward). To avoid an
/// indirection layer on every tensor access, layer K's bytes must
/// always live at the same address. With sub-slot = `K mod N`,
/// the address is `rotatingSlot.contents() + (K mod N) * slotSize +
/// inShardOffset` — stable for the lifetime of the pool.
///
/// **Sliding window.** With N sub-slots and strictly sequential
/// forward (layer 0, 1, ..., L-1), the working set per layer is 1
/// and the prefetched window is N-1. After computing layer K and
/// `releaseLayer(K)`, the pool schedules a background pread of
/// layer K+N into sub-slot `(K+N) mod N = K mod N` — i.e. the slot
/// holding K, which is no longer needed because the GPU finished
/// with it before `releaseLayer` ran (`cmdL.waitUntilCompleted`).
/// By the time `ensureLayer(K+N)` is called, the prefetch is
/// already complete and the fast path returns without I/O.
///
/// **Memory.** Total rotating bytes = `N * slotSize`. With N=1 the
/// behaviour is identical to the previous single-slot design (a
/// blocking pread per layer transition). With N >= numLayerShards,
/// every per-layer shard is pre-loaded once at init and never
/// pread again — equivalent to a partial preload but only for the
/// per-layer subset of shards.
public final class StreamingPool {
    public let sharedSlot: MTLBuffer
    /// Backing buffer for the rotating region. Internally divided
    /// into `slotCount` sub-slots of `slotSize` bytes; layer K
    /// always lives in sub-slot `K mod slotCount`. Exposed as a
    /// single buffer so existing callers don't need to know about
    /// the partitioning — the per-tensor `offsetInSlot` already
    /// encodes the sub-slot offset.
    public let rotatingSlot: MTLBuffer
    /// Number of rotating sub-slots (N). Always >= 1.
    public let slotCount: Int
    /// Bytes per sub-slot. Aligned to 4 KiB. The maximum per-layer
    /// shard data size, rounded up.
    public let slotSize: Int

    public enum Slot: Sendable { case shared, rotating }

    /// Resolved location of every tensor in the model. Built once
    /// at init from parsed shard headers.
    public struct TensorLocation: Sendable {
        public let slot: Slot
        /// For `.shared`, offset within `sharedSlot`.
        /// For `.rotating`, ABSOLUTE offset within `rotatingSlot`
        /// (i.e. already includes `(K mod N) * slotSize`).
        public let offsetInSlot: Int
        public let shape: [Int]
        public let dtype: DType
    }
    public let tensorLocation: [String: TensorLocation]

    /// Per per-layer-shard source needed to `pread` it on demand.
    private struct LayerShardSource {
        let url: URL
        let dataStart: Int
        let dataByteCount: Int
    }
    private let layerToShard: [Int: LayerShardSource]

    /// Resident state of each rotating sub-slot. Lock-protected.
    /// `loadedLayer == nil` means the sub-slot is empty or its
    /// content is stale (e.g. a load failed mid-way and poisoned
    /// the slot — the next ensure/prefetch for that K will retry).
    private struct SlotState {
        var loadedLayer: Int?
    }
    private var slotStates: [SlotState]
    private let stateLock = NSLock()

    /// Serial queue for all `pread` I/O. `ensureLayer` uses
    /// `sync` (blocks the caller); `prefetchLayer` uses `async`
    /// (returns immediately). Serial because APFS-on-NVMe peaks
    /// around 3-4 concurrent reads and we'd rather have one
    /// in-flight at a time than fight for disk bandwidth.
    private let ioQueue: DispatchQueue

    /// Long-lived file descriptors keyed by shard URL. Opened lazily
    /// on the first pread for each shard, kept open for the pool's
    /// lifetime, and closed in `deinit`. Two reasons to cache:
    ///   - One `open()` + `close()` per layer transition was a measurable
    ///     fraction of the per-token I/O latency on cold APFS reads.
    ///   - Each cached fd is opened with `F_RDAHEAD`, telling the
    ///     Darwin kernel to issue aggressive sequential readahead.
    ///     That hint is per-fd, so it survives across preads on the
    ///     same shard — exactly what the rotating-slot access pattern
    ///     needs (each layer is read sequentially in 1-GiB chunks).
    /// Accessed only from `ioQueue` (serial), so no locking needed.
    private var fdCache: [URL: Int32] = [:]

    /// Per-layer index into its shard for fast byte-range lookup.
    /// `tensorIO[name]` returns `(url, fileOffset, byteCount, slotOffset)`
    /// for every per-layer tensor we know about (rotating slot only —
    /// shared tensors live in mlocked sharedSlot and aren't paged).
    /// Built once at init from the shard headers; read-only from there
    /// on, so safe to access without locking from any thread.
    private struct TensorIO {
        let url: URL
        let fileOffset: Int
        let byteCount: Int
        let slotOffset: Int
    }
    private let tensorIO: [String: TensorIO]

    /// Names of non-expert tensors per layer, precomputed at init so
    /// lazy-expert mode can call `ensureTensors(layer:names:)` without
    /// re-filtering the header on every token. Empty for layers that
    /// have no expert tensors (e.g. dense layers before MoE kicks in).
    private let nonExpertNamesByLayer: [Int: [String]]

    /// Build the pool.
    ///
    /// - Parameter shards: parsed headers, in directory order
    ///   (same as `WeightLoader.discoverShards`).
    /// - Parameter shardLayers: from `WeightLoader.buildShardLayers`
    ///   — same shard ownership classification we already use for
    ///   `.streaming`'s madvise path.
    /// - Parameter targetSlotCount: number of rotating sub-slots
    ///   to allocate. Clamped to `[1, numLayerShards]`. Caller is
    ///   responsible for sizing this against the available
    ///   unified-memory budget; the pool itself does not probe
    ///   `SystemProbe`. Pass 1 for the legacy single-slot
    ///   behaviour, or higher to keep multiple per-layer shards
    ///   resident simultaneously.
    public init(shards: [SafeTensorsHeader],
                 shardLayers: [Int],
                 targetSlotCount: Int) throws {
        precondition(shards.count == shardLayers.count)
        precondition(targetSlotCount >= 1,
                      "targetSlotCount must be >= 1; got \(targetSlotCount)")

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

        // -------- sharedSlot: same as before --------
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

        // -------- rotatingSlot: N sub-slots in one buffer --------
        let maxLayerBytes = layerToShard.values.map(\.dataByteCount).max() ?? 0
        let slotSize = ((maxLayerBytes + 4095) / 4096) * 4096
        // Clamp slot count to the number of distinct per-layer
        // shards: anything beyond is wasted address space (we'd
        // have empty sub-slots forever).
        let layerShardCount = layerToShard.count
        let slotCount = max(1, min(targetSlotCount, max(1, layerShardCount)))
        let rotatingTotal = slotSize * slotCount
        self.slotSize = slotSize
        self.slotCount = slotCount
        self.slotStates = Array(repeating: SlotState(loadedLayer: nil),
                                  count: slotCount)
        self.ioQueue = DispatchQueue(label: "deepseek.streaming-pool.io",
                                       qos: .userInitiated)

        MemoryLogger.willAllocate(bytes: rotatingTotal, label: "rotatingSlot")
        guard let rotBuf = Device.shared.mtl.makeBuffer(
                length: max(rotatingTotal, 16),
                options: .storageModeShared) else {
            throw NSError(domain: "StreamingPool", code: 31, userInfo: [
                NSLocalizedDescriptionKey:
                    "rotatingSlot allocation failed (slotCount=\(slotCount), slotSize=\(slotSize))"
            ])
        }
        self.rotatingSlot = rotBuf

        // -------- Fill sharedSlot from shared shards --------
        var locations: [String: TensorLocation] = [:]
        var sharedOffsetCursor = 0
        for i in sharedIndices {
            let header = shards[i]
            // Init path: shared shards are read once at startup, not
            // touched again. Open and close a one-shot fd; the cache
            // is reserved for the rotating-slot hot path.
            try Self.preadOneShot(buffer: sharedSlot,
                                   bufferOffset: sharedOffsetCursor,
                                   url: header.url,
                                   fileOffset: header.dataStart,
                                   byteCount: header.dataByteCount)
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

        // -------- Resolve per-layer tensor locations (offsets
        //          already include the sub-slot base address) --------
        var tensorIO: [String: TensorIO] = [:]
        var nonExpertByLayer: [Int: [String]] = [:]
        for (layerK, src) in layerToShard {
            guard let shardIdx = shardLayers.firstIndex(of: layerK) else { continue }
            let header = shards[shardIdx]
            let slotBase = (layerK % slotCount) * slotSize
            var nonExpert: [String] = []
            for (name, entry) in header.entries {
                let inShard = entry.dataOffsets[0]
                let byteCount = entry.dataOffsets[1] - entry.dataOffsets[0]
                locations[name] = TensorLocation(
                    slot: .rotating,
                    offsetInSlot: slotBase + inShard,
                    shape: entry.shape,
                    dtype: Self.parseDType(entry.dtype))
                tensorIO[name] = TensorIO(
                    url: src.url,
                    fileOffset: src.dataStart + inShard,
                    byteCount: byteCount,
                    slotOffset: slotBase + inShard)
                // ".ffn.experts." separator is stable across DeepSeek
                // V2 / V3 / V4 checkpoints; the shared expert is
                // named ".ffn.shared_experts." and stays in the
                // non-expert bucket so it's preloaded with the rest
                // of the layer "core" (attention proj, norms, gate).
                if !name.contains(".ffn.experts.") {
                    nonExpert.append(name)
                }
            }
            nonExpertByLayer[layerK] = nonExpert
        }
        self.tensorLocation = locations
        self.tensorIO = tensorIO
        self.nonExpertNamesByLayer = nonExpertByLayer

        // -------- mlock the shared slot --------
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
            "[pool] rotatingSlot %d sub-slot(s) × %.2f GB = %.2f GB total\n",
            slotCount,
            Double(slotSize) / 1_073_741_824,
            Double(rotatingTotal) / 1_073_741_824))

        // -------- Pre-fill: load the first N per-layer shards --------
        // We bias the pre-fill toward the lowest layer indices since
        // forward starts at layer 0 and proceeds upward. With N slots
        // covering layers 0..N-1, the first N-1 ensureLayer() calls
        // hit the cache; the first miss (ensureLayer(N)) triggers a
        // pread for layer N that overwrites slot 0 (which held layer
        // 0 and is no longer needed). MTP layers (1000+) and any
        // sparse layer indices fill in afterwards as ensureLayer hits.
        let sortedLayers = layerToShard.keys.sorted()
        for layerK in sortedLayers.prefix(slotCount) {
            try loadLayerSync(layerK)
        }
    }

    /// Gate diagnostic output behind `MemoryLogger.enabled`.
    @inline(__always)
    private static func log(_ s: String) {
        guard MemoryLogger.enabled else { return }
        FileHandle.standardError.write(Data(s.utf8))
    }

    /// Prints a single banner on the FIRST pread of the run, so the
    /// log makes it obvious which path is active. Always fires (not
    /// gated on MemoryLogger) — one line, useful when troubleshooting
    /// "did my env var actually apply?".
    private static var modeBannerPrinted = false
    private static let modeBannerLock = NSLock()
    private static func logModeOnce(_ lazy: Bool) {
        modeBannerLock.lock()
        let already = modeBannerPrinted
        modeBannerPrinted = true
        modeBannerLock.unlock()
        guard !already else { return }
        let msg = lazy
            ? "[pool] streaming mode = LAZY-EXPERT (DEEPSEEK_LAZY_EXPERT=1)\n"
            : "[pool] streaming mode = FULL-SHARD (set DEEPSEEK_LAZY_EXPERT=1 to enable lazy)\n"
        FileHandle.standardError.write(Data(msg.utf8))
    }


    /// Lazy-expert mode toggle. Picked up at the first `ensureLayer`
    /// call so a single flag at process launch decides the strategy:
    ///   DEEPSEEK_LAZY_EXPERT=1 → only pread non-expert tensors at
    ///     `ensureLayer`; the per-MoE expert tensors are loaded on
    ///     demand by `ensureTensors(layer:names:)` once the gate has
    ///     picked the active topK.
    ///   anything else (default) → legacy full-shard pread.
    /// Reduces per-token I/O on huge MoE checkpoints from ~full-shard
    /// (~3 GB / layer at V4-Flash sizes) to ~core + 8/256 experts
    /// (~600 MB / layer), and as a side effect keeps each GPU command
    /// buffer's working set small enough that macOS no longer aborts
    /// it with `kIOGPUCommandBufferCallbackErrorImpactingInteractivity`.
    public static let lazyExpertEnabled: Bool = {
        ProcessInfo.processInfo
            .environment["DEEPSEEK_LAZY_EXPERT"] == "1"
    }()

    /// Ensure layer K's shard is loaded into its sub-slot. Blocks
    /// until the data is resident. Idempotent: no-op if already
    /// loaded (fast path skips the I/O queue entirely).
    public func ensureLayer(_ K: Int) throws {
        // Fast path: already resident, no queue wait. Safe because
        // `loadedLayer` is set to K *only after* the pread completes.
        // In lazy-expert mode the same "resident" cache still applies
        // — `loadLayerSync` writes the marker after the (smaller)
        // non-expert pread finishes; expert tensors are tracked
        // separately by `ensureTensors`.
        let slotIdx = K % slotCount
        stateLock.lock()
        let alreadyLoaded = slotStates[slotIdx].loadedLayer == K
        stateLock.unlock()
        if alreadyLoaded { return }

        // Slow path: serialize through the I/O queue. If a prefetch
        // for the same slot is in flight (likely for the same K),
        // we land behind it and find the slot already loaded by the
        // time our job runs.
        var caught: Error?
        ioQueue.sync {
            do { try self.loadLayerSync(K) }
            catch { caught = error }
        }
        if let e = caught { throw e }
    }

    /// Pread a specific set of named tensors from layer K's shard into
    /// their existing slot offsets. Used by lazy-expert mode after the
    /// MoE gate has decided which experts are active for the current
    /// token — the caller passes only the active expert weight names
    /// instead of triggering a full-layer pread.
    ///
    /// Names that aren't part of layer K's shard (or aren't in the
    /// rotating slot at all) are silently skipped. Always serializes
    /// through `ioQueue` so it interleaves correctly with prefetches.
    public func ensureTensors(layer K: Int, names: [String]) throws {
        guard !names.isEmpty else { return }
        var caught: Error?
        ioQueue.sync {
            do { try self.loadTensorsSync(layer: K, names: names) }
            catch { caught = error }
        }
        if let e = caught { throw e }
    }

    /// Schedule a background pread of layer K's shard. Returns
    /// immediately. Errors are logged but not propagated — the
    /// next `ensureLayer(K)` will retry synchronously and surface
    /// any persistent failure.
    public func prefetchLayer(_ K: Int) {
        // Skip layers that have no per-layer shard (top-level only)
        // without queueing — cheap predicate, avoids cluttering
        // ioQueue with no-ops on every release.
        guard layerToShard[K] != nil else { return }
        ioQueue.async { [weak self] in
            guard let self = self else { return }
            do {
                try self.loadLayerSync(K)
            } catch {
                Self.log("[pool] prefetchLayer(\(K)) failed: " +
                          "\(error.localizedDescription)\n")
            }
        }
    }

    /// Run on `ioQueue` (sync caller) or as the queue's async job.
    /// Performs the pread for layer K into its sub-slot and
    /// updates `slotStates` atomically. Branches between full-shard
    /// pread (default) and lazy-expert pread (only non-expert tensors).
    private func loadLayerSync(_ K: Int) throws {
        guard let src = layerToShard[K] else {
            // Layer has no per-layer shard (probably entirely in
            // shared shard, e.g. a one-block toy model). Nothing
            // to load.
            return
        }
        precondition(src.dataByteCount <= slotSize,
                     "shard for layer \(K) (\(src.dataByteCount) B) exceeds slotSize \(slotSize) B")

        let slotIdx = K % slotCount
        stateLock.lock()
        if slotStates[slotIdx].loadedLayer == K {
            // Another job loaded it while we waited on the queue.
            stateLock.unlock()
            return
        }
        // Mark the slot as "in flight": clear loadedLayer so any
        // concurrent fast-path reader takes the slow path and ends
        // up serialized behind this job.
        slotStates[slotIdx].loadedLayer = nil
        stateLock.unlock()

        do {
            if Self.lazyExpertEnabled,
               let nonExpert = nonExpertNamesByLayer[K],
               !nonExpert.isEmpty {
                // Lazy mode: only fill the non-expert tensor ranges.
                // Expert ranges in the slot are left dormant; MoE's
                // routing observer triggers `ensureTensors` for the
                // active topK before the dispatch loop runs.
                Self.logModeOnce(true)
                try preadTensors(layer: K, names: nonExpert,
                                  shardURL: src.url)
            } else {
                // Full-shard pread (legacy path / non-MoE layers).
                Self.logModeOnce(false)
                let slotBase = rotatingSlot.contents()
                    .advanced(by: slotIdx * slotSize)
                _ = posix_madvise(slotBase, src.dataByteCount,
                                  POSIX_MADV_WILLNEED)
                let fd = try cachedFD(for: src.url)
                try Self.preadInto(buffer: rotatingSlot,
                                    bufferOffset: slotIdx * slotSize,
                                    fd: fd,
                                    fileOffset: src.dataStart,
                                    byteCount: src.dataByteCount,
                                    shardName: src.url.lastPathComponent)
            }
        } catch {
            // Pread failed; leave the slot poisoned (loadedLayer
            // = nil) so the next caller retries instead of
            // trusting stale bytes.
            throw error
        }

        stateLock.lock()
        slotStates[slotIdx].loadedLayer = K
        stateLock.unlock()
        Self.log(String(format:
            "[pool] layer=%d preaded %.2f GB into sub-slot %d\n",
            K, Double(src.dataByteCount) / 1_073_741_824, slotIdx))
    }

    /// Run on `ioQueue` (sync caller). Performs targeted preads for
    /// `names` into their pre-computed slot offsets, without touching
    /// or rewriting any other tensor's bytes in the slot. Used by
    /// `ensureTensors(layer:names:)`.
    private func loadTensorsSync(layer K: Int, names: [String]) throws {
        guard let src = layerToShard[K] else { return }
        try preadTensors(layer: K, names: names, shardURL: src.url)
        // The slot's "loadedLayer" marker is not flipped here: lazy
        // mode tracks per-tensor freshness implicitly through the
        // caller (MoE asks for the experts it needs, attention/norms
        // are loaded by the prior ensureLayer call). If we *were* to
        // overwrite the marker, the next prefetch for the rotating
        // slot would skip on the cache-hit fast path even though the
        // expert ranges from a different token are now stale.
    }

    /// Bulk pread the supplied tensor names from a single shard file.
    /// Caller is responsible for queue serialization. Uses the cached
    /// fd (with F_RDAHEAD) and madvises each slot range before write.
    private func preadTensors(layer K: Int,
                               names: [String],
                               shardURL: URL) throws {
        let fd = try cachedFD(for: shardURL)
        let baseAddr = rotatingSlot.contents()
        var totalBytes = 0
        var hitCount = 0
        for name in names {
            guard let io = tensorIO[name] else { continue }
            // Hint VM: about to write this region, keep it resident.
            _ = posix_madvise(baseAddr.advanced(by: io.slotOffset),
                              io.byteCount,
                              POSIX_MADV_WILLNEED)
            try Self.preadInto(buffer: rotatingSlot,
                                bufferOffset: io.slotOffset,
                                fd: fd,
                                fileOffset: io.fileOffset,
                                byteCount: io.byteCount,
                                shardName: shardURL.lastPathComponent)
            totalBytes += io.byteCount
            hitCount += 1
        }
        Self.log(String(format:
            "[pool] layer=%d preadTensors n=%d/%d bytes=%.2f MB shard=%@\n",
            K, hitCount, names.count,
            Double(totalBytes) / 1_048_576,
            shardURL.lastPathComponent as CVarArg))
    }

    // ---- internals ----

    /// Init-path helper: open a fresh fd, pread, close. Used for
    /// shared shards that aren't part of the rotating hot path.
    private static func preadOneShot(buffer: MTLBuffer,
                                      bufferOffset: Int,
                                      url: URL,
                                      fileOffset: Int,
                                      byteCount: Int) throws {
        let fd = open(url.path, O_RDONLY)
        guard fd >= 0 else {
            throw NSError(domain: "StreamingPool", code: 32, userInfo: [
                NSLocalizedDescriptionKey:
                    "open failed: \(url.path) (\(String(cString: strerror(errno))))"
            ])
        }
        defer { close(fd) }
        try preadInto(buffer: buffer, bufferOffset: bufferOffset,
                       fd: fd, fileOffset: fileOffset,
                       byteCount: byteCount,
                       shardName: url.lastPathComponent)
    }

    private static func preadInto(buffer: MTLBuffer,
                                   bufferOffset: Int,
                                   fd: Int32,
                                   fileOffset: Int,
                                   byteCount: Int,
                                   shardName: String) throws {
        // macOS / Darwin `pread` returns EINVAL when `nbyte` exceeds
        // a per-syscall cap (observed at ~2 GiB on M-series Macs,
        // even though POSIX permits up to SSIZE_MAX). For per-layer
        // shards larger than that (V4-Flash's biggest is 3.37 GB)
        // the very first iteration fails before the short-read loop
        // can chunk it. We cap each call at 1 GiB to stay safely
        // below INT_MAX with headroom; the loop iterates as before.
        //
        // This is silently catastrophic if not capped: ensureLayer
        // swallows the error (`DEEPSEEK_MEM_LOG=0` is the default),
        // leaving the rotating slot full of stale or uninitialised
        // bytes, so every block above the failed layer computes its
        // attention / MLP against zero (or previous layer's) weights
        // and produces tokens that look uniform-random across scripts
        // — the exact symptom we've been chasing.
        let kPreadCap = 1 << 30   // 1 GiB
        let base = buffer.contents().advanced(by: bufferOffset)
        var off = 0
        while off < byteCount {
            let toRead = min(byteCount - off, kPreadCap)
            let n = pread(fd, base.advanced(by: off),
                          toRead,
                          off_t(fileOffset + off))
            if n > 0 {
                off += n
            } else if n == 0 {
                throw NSError(domain: "StreamingPool", code: 33, userInfo: [
                    NSLocalizedDescriptionKey:
                        "short pread (got \(off)/\(byteCount)) from \(shardName)"
                ])
            } else if errno != EINTR {
                let errnoStr = String(cString: strerror(errno))
                throw NSError(domain: "StreamingPool", code: 34, userInfo: [
                    NSLocalizedDescriptionKey:
                        "pread failed at \(off) (req \(toRead) bytes): \(errnoStr) — \(shardName)"
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

    /// Open (or fetch from cache) a long-lived fd for `url`. Sets
    /// `F_RDAHEAD` on first open so the Darwin kernel issues
    /// aggressive sequential prefetch on subsequent preads — a single
    /// layer shard is read in one or more 1-GiB chunks in increasing
    /// file offset order, exactly the access pattern readahead is
    /// optimised for.
    ///
    /// Called only from `ioQueue` (serial), so the cache mutation is
    /// race-free without an explicit lock.
    private func cachedFD(for url: URL) throws -> Int32 {
        if let fd = fdCache[url] { return fd }
        let fd = open(url.path, O_RDONLY)
        guard fd >= 0 else {
            throw NSError(domain: "StreamingPool", code: 32, userInfo: [
                NSLocalizedDescriptionKey:
                    "open failed: \(url.path) (\(String(cString: strerror(errno))))"
            ])
        }
        // F_RDAHEAD takes 0/1 on Darwin; the return value is unused.
        // Best-effort hint: if the syscall fails (sandbox / unusual
        // filesystem) we keep the fd anyway — pread still works, just
        // without prefetch acceleration.
        _ = fcntl(fd, F_RDAHEAD, 1)
        fdCache[url] = fd
        return fd
    }

    deinit {
        for fd in fdCache.values where fd >= 0 {
            close(fd)
        }
    }
}
