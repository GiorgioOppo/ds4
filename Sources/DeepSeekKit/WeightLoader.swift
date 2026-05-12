import Foundation
import Dispatch

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
}
