import Foundation

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

    public init(directory: URL) throws {
        self.directory = directory
        let fm = FileManager.default
        let contents = (try? fm.contentsOfDirectory(at: directory,
                                                     includingPropertiesForKeys: nil)) ?? []
        let sortedShards = contents
            .filter { $0.pathExtension == "safetensors" }
            .sorted { $0.lastPathComponent < $1.lastPathComponent }

        if sortedShards.isEmpty {
            throw NSError(domain: "WeightLoader", code: 1, userInfo: [
                NSLocalizedDescriptionKey:
                    "no .safetensors files in \(directory.path) — did you run convert.py?"
            ])
        }

        for url in sortedShards {
            // Skip the LFS pointer files (3-line text files masquerading as
            // safetensors when LFS payload wasn't fetched).
            let attrs = try fm.attributesOfItem(atPath: url.path)
            if let size = attrs[.size] as? Int, size < 1024 { continue }

            let f = try SafeTensorsFile(url: url)
            let shardIdx = shards.count
            shards.append(f)
            for name in f.entries.keys { index[name] = shardIdx }
        }

        if shards.isEmpty {
            throw NSError(domain: "WeightLoader", code: 2, userInfo: [
                NSLocalizedDescriptionKey:
                    "all safetensors files in \(directory.path) were LFS pointers — run `git lfs pull` or download the actual blobs"
            ])
        }
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
