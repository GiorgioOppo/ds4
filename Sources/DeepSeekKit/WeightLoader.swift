import Foundation
import Dispatch
import Metal
#if canImport(Darwin)
import Darwin
#endif

/// Indexes every `.safetensors` shard in a directory and exposes a
/// `load(name:)` API that returns the tensor regardless of which shard it
/// lives in. Returns `nil` for names that aren't present (the caller can
/// then fall back to random init).
///
/// Expected input directory layout: the post-`convert.py` form, i.e. one or
/// more files named `model0-mp1.safetensors`, `model1-mp1.safetensors`, …
/// or any other set of `.safetensors` files. Names follow the convention in
/// `Reference/inference/convert.py` (renames `self_attn → attn`,
/// `mlp → ffn`, `weight_scale_inv → scale`, etc.).
public final class WeightLoader {
    public let directory: URL
    private var shards: [SafeTensorsFile] = []
    private var index: [String: Int] = [:]   // name → shards[index]
    public private(set) var missing: Set<String> = []

    /// Per-shard "dominant layer index" used by the streaming hints.
    /// Built once at init by inspecting tensor names: a shard whose
    /// entries are all `layers.K.*` is owned by layer K and can be
    /// page-evicted (`madvise MADV_DONTNEED`) when layer K is done
    /// with the current token's forward. -1 means "shared / always
    /// resident" — embedding, output head, RMSNorm gains, and any
    /// shard mixing tensors from more than one layer.
    private var shardLayers: [Int] = []
    public var streamingEnabled: Bool = false

    /// Non-nil when `plan.strategy == .streaming` and the pool
    /// path activated successfully. Replaces the per-shard mmap
    /// MTLBuffers with two pre-allocated `.storageModeShared`
    /// slots (sharedSlot + rotatingSlot). `load(_:)` returns
    /// Tensors pointing into the pool; `Transformer.forward`
    /// calls `ensureLayer(_:)` before each block to swap layer
    /// K's data into the rotating slot.
    private var pool: StreamingPool? = nil

    /// Construct from an already-decided `LoadPlan`. The plan owns the
    /// list of shards and chooses mmap vs preload; this initializer
    /// just opens them. Preload is parallelized via
    /// `DispatchQueue.concurrentPerform`; mmap stays sequential (the
    /// VM-mapping syscall is microseconds, parallelism is noise).
    public init(plan: LoadPlan) throws {
        guard let first = plan.shards.first else {
            throw NSError(domain: "WeightLoader", code: 1, userInfo: [
                NSLocalizedDescriptionKey: "LoadPlan has no shards"
            ])
        }
        self.directory = first.url.deletingLastPathComponent()

        switch plan.strategy {
        case .mmap:
            for (url, _) in plan.shards {
                let f = try SafeTensorsFile(url: url)
                let shardIdx = shards.count
                shards.append(f)
                for name in f.entries.keys { index[name] = shardIdx }
            }

        case .streaming:
            // Pool path: parse every shard's HEADER only (no
            // mmap of the data section), then build a
            // StreamingPool that owns two MTLBuffer slots. The
            // legacy `shards: [SafeTensorsFile]` array stays
            // empty — `load(_:)` short-circuits to the pool
            // instead.
            var headers: [SafeTensorsHeader] = []
            for (url, _) in plan.shards {
                let h = try SafeTensorsHeader.parse(url: url)
                headers.append(h)
                // Populate name → shardIdx index so the existing
                // shape(of:) / missing reporting still works.
                let shardIdx = headers.count - 1
                for name in h.entries.keys { index[name] = shardIdx }
            }
            // Classify shards (same logic as buildShardLayers,
            // but on SafeTensorsHeader.entries instead of
            // SafeTensorsFile.entries).
            var classifyLayers = [Int](repeating: -1, count: headers.count)
            for (i, h) in headers.enumerated() {
                var seenLayers: Set<Int> = []
                var hasTopLevel = false
                for name in h.entries.keys {
                    if let layer = Self.parseLayerIndex(from: name) {
                        seenLayers.insert(layer)
                    } else {
                        hasTopLevel = true
                    }
                }
                classifyLayers[i] = (hasTopLevel || seenLayers.count != 1)
                    ? -1 : seenLayers.first!
            }
            self.shardLayers = classifyLayers
            self.streamingEnabled = true
            self.pool = try StreamingPool(shards: headers,
                                            shardLayers: classifyLayers)
            // The legacy `shards: [SafeTensorsFile]` array stays
            // empty in this strategy — `load(_:)` checks `pool`
            // first.

        case .preload:
            // Pre-size and fill in parallel; flatten + index after.
            // `concurrentPerform` saturates the GCD default-pool width
            // (≈ active cores). On a single APFS-on-NVMe volume that
            // exceeds the ~4-stream sweet spot, but the extra threads
            // mostly block on read syscalls — measured penalty is small
            // and not worth gating with a semaphore.
            let n = plan.shards.count
            var slots: [SafeTensorsFile?] = Array(repeating: nil, count: n)
            var firstError: Error?
            let lock = NSLock()
            DispatchQueue.concurrentPerform(iterations: n) { i in
                lock.lock()
                let abort = firstError != nil
                lock.unlock()
                if abort { return }
                do {
                    let (url, bytes) = plan.shards[i]
                    let f = try SafeTensorsFile(preloadedURL: url, byteCount: bytes)
                    slots[i] = f
                } catch {
                    lock.lock()
                    if firstError == nil { firstError = error }
                    lock.unlock()
                }
            }
            if let e = firstError { throw e }
            for f in slots {
                guard let f else {
                    throw NSError(domain: "WeightLoader", code: 4, userInfo: [
                        NSLocalizedDescriptionKey: "preload returned a nil slot"
                    ])
                }
                let shardIdx = shards.count
                shards.append(f)
                for name in f.entries.keys { index[name] = shardIdx }
            }
        }

        // For `.mmap` / `.preload` we build the per-shard layer
        // ownership table over the legacy `shards: [SafeTensorsFile]`
        // array. `.streaming` already populated `shardLayers` and
        // `streamingEnabled` inside its switch arm via the pool's
        // header-classified data; nothing more to do.
        if pool == nil {
            self.shardLayers = Self.buildShardLayers(shards: shards)
            self.streamingEnabled = false
        }
    }

    /// Backwards-compat convenience: build a default `mmap` plan
    /// covering the whole directory, then delegate. Kept so existing
    /// tests and callers that don't care about strategy keep
    /// compiling.
    public convenience init(directory: URL) throws {
        let shards = try Self.discoverShards(in: directory)
        let total = shards.reduce(0 as UInt64) { $0 + $1.byteCount }
        let maxShard = shards.map(\.byteCount).max() ?? 0
        let plan = LoadPlan(
            strategy: .mmap, shards: shards.map { ($0.url, $0.byteCount) },
            totalBytes: total, maxShardBytes: maxShard,
            availableRAM: 0, physicalRAM: 0, mtlWorkingSet: 0,
            cores: 0, reason: "legacy WeightLoader(directory:) — mmap default")
        try self.init(plan: plan)
    }

    /// Enumerate `.safetensors` shards in `dir`, skipping LFS pointer
    /// stubs (3-line text files < 1 KiB), and return them sorted by
    /// filename together with their byte size. Used by both
    /// `LoadPlan.decide` (to total / cap-check) and `WeightLoader.init`
    /// (to actually open them).
    public static func discoverShards(in dir: URL) throws -> [(url: URL, byteCount: UInt64)] {
        let fm = FileManager.default
        let contents = (try? fm.contentsOfDirectory(at: dir,
                                                     includingPropertiesForKeys: nil)) ?? []
        let candidates = contents
            .filter { $0.pathExtension == "safetensors" }
            .sorted { $0.lastPathComponent < $1.lastPathComponent }

        if candidates.isEmpty {
            throw NSError(domain: "WeightLoader", code: 1, userInfo: [
                NSLocalizedDescriptionKey:
                    "no .safetensors files in \(dir.path) — did you run convert.py?"
            ])
        }

        var out: [(URL, UInt64)] = []
        for url in candidates {
            let attrs = try fm.attributesOfItem(atPath: url.path)
            let size = (attrs[.size] as? Int) ?? 0
            if size < 1024 { continue }   // LFS pointer stub
            out.append((url, UInt64(size)))
        }
        if out.isEmpty {
            throw NSError(domain: "WeightLoader", code: 2, userInfo: [
                NSLocalizedDescriptionKey:
                    "all safetensors files in \(dir.path) were LFS pointers — run `git lfs pull` or download the actual blobs"
            ])
        }
        return out
    }

    /// Returns the tensor for `name`, or nil if not present.
    /// In streaming mode, returns a Tensor pointing into the
    /// StreamingPool's sharedSlot or rotatingSlot — the rotating
    /// slot's BACKING DATA isn't valid until `ensureLayer(K)` has
    /// been called for the right K (the caller, `Transformer.forward`,
    /// does this between blocks).
    public func load(_ name: String) throws -> Tensor? {
        if let pool = pool {
            guard let loc = pool.tensorLocation[name] else {
                missing.insert(name)
                return nil
            }
            let buf: MTLBuffer
            switch loc.slot {
            case .shared:   buf = pool.sharedSlot
            case .rotating: buf = pool.rotatingSlot
            }
            return Tensor(shape: loc.shape, dtype: loc.dtype,
                          buffer: buf, offset: loc.offsetInSlot)
        }
        guard let s = index[name] else {
            missing.insert(name)
            return nil
        }
        return try shards[s].load(name)
    }

    /// In streaming-pool mode, swap layer K's shard data into the
    /// rotating slot. Idempotent. Called by `Transformer.forward`
    /// before each block. No-op when not streaming.
    public func ensureLayer(_ K: Int) {
        guard let pool = pool else { return }
        do {
            try pool.ensureLayer(K)
        } catch {
            FileHandle.standardError.write(Data(
                "[pool] ensureLayer(\(K)) failed: \(error.localizedDescription)\n"
                    .utf8))
        }
    }

    /// Convenience: load with a fallback name list. Tries each in order.
    public func tryLoad(_ candidates: [String]) throws -> Tensor? {
        for n in candidates {
            if let t = try load(n) { return t }
        }
        for n in candidates { missing.insert(n) }
        return nil
    }

    public var totalKnownNames: Int { index.count }
    public var shardCount: Int { shards.count }

    /// Queries the on-disk shape of a tensor without loading its data.
    /// Useful for auto-inferring missing fields in ModelConfig when
    /// config.json is incomplete.
    public func shape(of name: String) -> [Int]? {
        // Pool mode: look up in the StreamingPool's pre-resolved
        // index — `shards` array is empty under this strategy.
        if let pool = pool {
            return pool.tensorLocation[name]?.shape
        }
        guard let s = index[name] else { return nil }
        return shards[s].entries[name]?.shape
    }

    /// Convenience: try a list of candidate names and return the first
    /// shape found.
    public func shape(ofAny candidates: [String]) -> [Int]? {
        for n in candidates {
            if let s = shape(of: n) { return s }
        }
        return nil
    }

    // ----------- Streaming hints (madvise-based) -----------
    //
    // For very-oversubscribed checkpoints (e.g. 147 GB INT4 V4-Flash
    // on a 16 GB Mac) the all-resident mmap path lets macOS evict
    // pages from the GUI compositor and the kernel itself when
    // weights compete for the same unified-memory pool — freezing
    // the box. With streaming hints enabled, `Transformer.forward`
    // calls `prefetchLayer(K+1)` while computing layer K and
    // `releaseLayer(K-1)` right after, telling the kernel which
    // ranges are hot and which are cold so eviction stays inside
    // OUR mapping rather than ranging across the system.
    //
    // The hints use Darwin's `madvise(MADV_WILLNEED / MADV_DONTNEED)`
    // — they don't actually load or unload anything synchronously,
    // they steer the kernel's LRU. The MTLBuffer references stay
    // valid throughout; only the physical-memory residency of the
    // backing mmap pages changes.

    /// Build a map from each shard index to the "dominant layer"
    /// whose tensors it contains. The converter writes layer-aligned
    /// shards (one bucket per layer), so most shards have a single
    /// owning layer. Shards containing top-level tensors (embed,
    /// head, hc_head_*, RMSNorm gains) or mixing multiple layers
    /// get `-1` → never page-evicted.
    private static func buildShardLayers(shards: [SafeTensorsFile]) -> [Int] {
        var out = [Int](repeating: -1, count: shards.count)
        for (idx, shard) in shards.enumerated() {
            var seenLayers: Set<Int> = []
            var hasTopLevel = false
            for name in shard.entries.keys {
                if let layer = Self.parseLayerIndex(from: name) {
                    seenLayers.insert(layer)
                } else {
                    hasTopLevel = true
                }
            }
            if hasTopLevel || seenLayers.count != 1 {
                out[idx] = -1
            } else {
                out[idx] = seenLayers.first!
            }
        }
        return out
    }

    /// Parse `N` from `layers.N.<anything>` or `mtp.N.<anything>`.
    /// MTP layers get encoded as `1000 + N` so they don't collide
    /// with the main-layer numbering — the model's MTP modules call
    /// `prefetchLayer(1000 + mtpIdx)` if they want streaming.
    private static func parseLayerIndex(from name: String) -> Int? {
        let parts = name.split(separator: ".").map(String.init)
        guard parts.count >= 2 else { return nil }
        if parts[0] == "layers", let n = Int(parts[1]) { return n }
        if parts[0] == "mtp",    let n = Int(parts[1]) { return 1000 + n }
        return nil
    }

    /// Hint to the kernel that layer K's pages will be touched
    /// soon. Currently a no-op even when streamingEnabled — the
    /// MADV_WILLNEED variant of the previous revision caused
    /// triple-residency (K-1 pending eviction + K active + K+1
    /// pre-faulted) that OOM'd 16 GB Macs during forward. Leaving
    /// the API in place for future re-introduction once we have a
    /// better way to bound residency.
    public func prefetchLayer(_ layerIndex: Int) {
        // Intentional no-op. See doc comment.
        _ = layerIndex
    }

    /// Aggressively reclaim layer K's pages. Three-step Darwin
    /// sequence after which `mincore()` is used to verify how
    /// many pages actually got dropped:
    ///
    ///   1. `MADV_DONTNEED`        — soft hint "don't need soon"
    ///   2. `MADV_ZERO_WIRED_PAGES` — drop wired-down content
    ///   3. `MADV_FREE_REUSABLE`    — mark pages immediately
    ///                                 reclaimable for reuse
    ///
    /// `MADV_FREE_REUSABLE` alone is often ignored under low
    /// pressure; the cascade gives the kernel three increasingly
    /// strong nudges. The `mincore` verification logs the
    /// resident-page count BEFORE and AFTER so we can see in the
    /// trace whether the OS is actually honouring our hints.
    ///
    /// Called from `Transformer.forward` right after each layer's
    /// `commit+wait`. No-op unless `streamingEnabled`. Top-level
    /// shards (owner == -1) skipped.
    public func releaseLayer(_ layerIndex: Int) {
        // Pool mode: nothing to release — the next `ensureLayer`
        // overwrites the rotating slot's bytes via pread. Layer K's
        // tensors become silently invalid the moment ensureLayer(K')
        // is called for K' != K, but that's the design contract:
        // Transformer.forward only accesses layer K's tensors
        // BETWEEN ensureLayer(K) and ensureLayer(K+1).
        if pool != nil { return }
        // Legacy mmap path (unused in current streaming, kept in
        // case we re-enable madvise-only experimentation).
        guard streamingEnabled else { return }
        for (shardIdx, owner) in shardLayers.enumerated()
            where owner == layerIndex {
            let before = residentPageCount(shardIdx)
            adviseShard(shardIdx, advice: MADV_DONTNEED)
            adviseShard(shardIdx, advice: Self.MADV_ZERO_WIRED_PAGES)
            adviseShard(shardIdx, advice: Self.MADV_FREE_REUSABLE)
            let after = residentPageCount(shardIdx)
            let line = String(format:
                "[release shard=%d layer=%d resident-pages: %d → %d (-%d, %.0f%%)]\n",
                shardIdx, layerIndex,
                before, after,
                before > after ? before - after : 0,
                before > 0 ? Double(before - after) * 100.0 / Double(before) : 0)
            FileHandle.standardError.write(Data(line.utf8))
        }
    }

    /// Darwin's MADV_* constants (sys/mman.h). Defined statically
    /// because Swift's Darwin shim doesn't always re-export all of
    /// them.
    private static let MADV_FREE_REUSABLE: Int32 = 7
    private static let MADV_ZERO_WIRED_PAGES: Int32 = 6

    /// Pin the shared shards into physical RAM via `mlock(2)`. The
    /// top-level shard(s) carry embed/head/norms — touched on every
    /// forward, never long enough offline to amortise a refault.
    /// Pinning them spends ~500 MB of wired-memory budget for a
    /// large per-token speedup AND removes them from the pool of
    /// pages the OS could decide to swap during a memory crunch.
    private func pinSharedShards() {
        var totalPinned: UInt64 = 0
        for (shardIdx, owner) in shardLayers.enumerated() where owner == -1 {
            let buf = shards[shardIdx].sharedBuffer
            let len = buf.length
            guard let addr = buf.contents() as UnsafeMutableRawPointer? else { continue }
            let rc = mlock(addr, len)
            if rc == 0 {
                totalPinned &+= UInt64(len)
                let line = String(format:
                    "[pin shard=%d  shared  %.2f MB mlocked]\n",
                    shardIdx, Double(len) / (1024 * 1024))
                FileHandle.standardError.write(Data(line.utf8))
            } else {
                let errnoStr = String(cString: strerror(errno))
                let line = String(format:
                    "[pin shard=%d  shared  %.2f MB mlock FAILED: %s]\n",
                    shardIdx, Double(len) / (1024 * 1024),
                    errnoStr)
                FileHandle.standardError.write(Data(line.utf8))
            }
        }
        let summary = String(format:
            "[pin] total wired %.2f GB across shared shards\n",
            Double(totalPinned) / (1024 * 1024 * 1024))
        FileHandle.standardError.write(Data(summary.utf8))
    }

    private func adviseShard(_ idx: Int, advice: Int32) {
        let buf = shards[idx].sharedBuffer
        let len = buf.length
        guard let addr = buf.contents() as UnsafeMutableRawPointer? else { return }
        _ = madvise(addr, len, advice)
    }

    /// Counts pages still physically resident in the given shard's
    /// mapping using `mincore(2)`. Returns 0 on error. Cheap: one
    /// byte-per-page in a stack buffer, O(pages-in-shard).
    private func residentPageCount(_ idx: Int) -> Int {
        let buf = shards[idx].sharedBuffer
        let len = buf.length
        guard let addr = buf.contents() as UnsafeMutableRawPointer? else { return 0 }
        let pageSize = Int(sysconf(_SC_PAGESIZE))
        let nPages = (len + pageSize - 1) / pageSize
        var vec = [Int8](repeating: 0, count: nPages)
        let rc = vec.withUnsafeMutableBufferPointer { mincore(addr, len, $0.baseAddress) }
        guard rc == 0 else { return 0 }
        // mincore's vec[i] low-order bit set when page is resident.
        return vec.reduce(0) { $0 + (Int($1) & 1) }
    }
}
