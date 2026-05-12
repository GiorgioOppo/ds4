import Foundation
import Dispatch
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
        case .mmap, .streaming:
            // Identical load-time path. `.streaming` only differs at
            // runtime: `Transformer.forward` calls `prefetchLayer` /
            // `releaseLayer` between blocks so the kernel can drop
            // cold pages without thrashing the GUI. The shards
            // themselves are mmap'd the same way.
            for (url, _) in plan.shards {
                let f = try SafeTensorsFile(url: url)
                let shardIdx = shards.count
                shards.append(f)
                for name in f.entries.keys { index[name] = shardIdx }
            }

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

        // Populate the per-shard layer ownership table (for the
        // streaming hints API). Streaming is gated on `plan.strategy
        // == .streaming` below; for `.mmap` / `.preload` the table
        // is still built but never consulted.
        self.shardLayers = Self.buildShardLayers(shards: shards)
        self.streamingEnabled = (plan.strategy == .streaming)
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
    public func load(_ name: String) throws -> Tensor? {
        guard let s = index[name] else {
            missing.insert(name)
            return nil
        }
        return try shards[s].load(name)
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
    /// soon. No-op unless `streamingEnabled` is true. Idempotent.
    public func prefetchLayer(_ layerIndex: Int) {
        guard streamingEnabled else { return }
        for (shardIdx, owner) in shardLayers.enumerated()
            where owner == layerIndex {
            adviseShard(shardIdx, advice: MADV_WILLNEED)
        }
    }

    /// Hint to the kernel that layer K's pages are cold and can be
    /// reclaimed. Top-level / shared shards (`owner == -1`) are
    /// always skipped. No-op unless `streamingEnabled`.
    public func releaseLayer(_ layerIndex: Int) {
        guard streamingEnabled else { return }
        for (shardIdx, owner) in shardLayers.enumerated()
            where owner == layerIndex {
            adviseShard(shardIdx, advice: MADV_DONTNEED)
        }
    }

    private func adviseShard(_ idx: Int, advice: Int32) {
        let buf = shards[idx].sharedBuffer
        let len = buf.length
        guard let addr = buf.contents() as UnsafeMutableRawPointer? else { return }
        _ = madvise(addr, len, advice)
    }
}
