import Foundation

// Multi-shard GGUF: presents several on-disk GGUF files as ONE logical model by
// unioning their tensor directories by name. The upstream DeepSeek V4 Pro Q4
// package ships as two layer-range shards, each a self-contained GGUF with a
// DISJOINT subset of tensors:
//   DeepSeek-V4-Pro-Q4K-Layers00-30.gguf        (blk.0..30.*)
//   DeepSeek-V4-Pro-Q4K-Layers-31-output.gguf   (blk.31..N.* + output/embedding)
// Upstream maps one shard per machine for the distributed run; this type also
// lets a high-RAM machine open both as a single local model (the piece the port
// was missing — the shards were download-only, "not presented as a local model").
//
// Contract:
//  - Tensor names must be DISJOINT across shards (they describe different layers);
//    a duplicate name is an error, not a silent last-wins.
//  - Global metadata (general.*, architecture, geometry) is read first-shard-wins;
//    shards are expected to agree on it. Order the shards as given.
//
// Pure Swift, no Metal: this is loader infrastructure. Each returned Tensor's
// `absOffset` is relative to ITS shard's mapping — always fetch bytes through
// this type (`tensorData`/`tensor(named:)`) so the owning shard is used.

public final class GGUFShardSet {

    /// A tensor together with the shard that stores its bytes.
    public struct Located {
        public let shardIndex: Int
        public let model: GGUFModel
        public let tensor: GGUFModel.Tensor
    }

    /// The underlying shards, in the order provided.
    public let shards: [GGUFModel]
    /// Unified tensor directory (all shards), in shard-then-file order.
    public let tensors: [GGUFModel.Tensor]

    private let located: [String: (shard: Int, tensorIndex: Int)]
    /// Which shard owns each `blk.<il>.*` layer (all of a layer's tensors live
    /// in one shard — the split is by layer range).
    private let layerToShard: [Int: Int]

    public var n_tensors: UInt64 { UInt64(tensors.count) }
    /// Alignment of the first shard (shards are expected to agree).
    public var alignment: UInt64 { shards.first?.alignment ?? 32 }
    /// The first shard — the authority for global (non-layer) metadata.
    public var primary: GGUFModel { shards[0] }

    /// Open every path as a GGUF and merge. Throws if a tensor name appears in
    /// more than one shard, or if `paths` is empty.
    public convenience init(paths: [String], metalMapping: Bool = true) throws {
        guard !paths.isEmpty else { throw GGUFError.message("GGUFShardSet needs at least one shard") }
        let models = try paths.map { try GGUFModel(path: $0, metalMapping: metalMapping) }
        try self.init(shards: models)
    }

    /// Merge already-open shards (used by tests and callers that map explicitly).
    public init(shards: [GGUFModel]) throws {
        guard !shards.isEmpty else { throw GGUFError.message("GGUFShardSet needs at least one shard") }
        self.shards = shards

        var all: [GGUFModel.Tensor] = []
        var index: [String: (shard: Int, tensorIndex: Int)] = [:]
        var layerShard: [Int: Int] = [:]
        for (si, m) in shards.enumerated() {
            for t in m.tensors {
                if let prev = index[t.name] {
                    throw GGUFError.message(
                        "GGUFShardSet: tensor '\(t.name)' is in shard \(prev.shard) and \(si)")
                }
                index[t.name] = (si, all.count)
                all.append(t)
                if let il = Self.layerIndex(of: t.name) {
                    if let prev = layerShard[il], prev != si {
                        throw GGUFError.message(
                            "GGUFShardSet: layer \(il) is split across shards \(prev) and \(si)")
                    }
                    layerShard[il] = si
                }
            }
        }
        self.tensors = all
        self.located = index
        self.layerToShard = layerShard
    }

    /// Parse the layer index from a `blk.<il>.…` tensor name, or nil.
    static func layerIndex(of name: String) -> Int? {
        guard name.hasPrefix("blk.") else { return nil }
        let rest = name.dropFirst(4)
        guard let dot = rest.firstIndex(of: ".") else { return nil }
        return Int(rest[..<dot])
    }

    // MARK: - Tensor access (routed to the owning shard)

    public func find(_ name: String) -> Located? {
        guard let loc = located[name] else { return nil }
        return Located(shardIndex: loc.shard, model: shards[loc.shard],
                       tensor: tensors[loc.tensorIndex])
    }

    /// The merged tensor descriptor (bytes must still be read via `tensorData`).
    public func findTensor(_ name: String) -> GGUFModel.Tensor? { find(name)?.tensor }

    /// The shard (a plain `GGUFModel`) that owns a given tensor, or nil.
    public func shard(owning name: String) -> GGUFModel? { find(name)?.model }

    /// The shard that owns layer `il`'s tensors (`blk.<il>.*`), or nil if no
    /// shard declares that layer. Because the split is by layer range, all of a
    /// layer's tensors live in one shard — so weight loaders can pass THIS shard
    /// to the existing single-file `GGUFWeights` code unchanged.
    public func shard(forLayer il: Int) -> GGUFModel? {
        layerToShard[il].map { shards[$0] }
    }

    /// Raw bytes of a tensor, read from the shard that owns it.
    public func tensorData(_ name: String) -> Data? {
        guard let l = find(name) else { return nil }
        return l.model.tensorData(l.tensor)
    }

    // MARK: - Metadata (first shard that declares the key wins)

    public func string(_ key: String) -> String? { firstNonNil { $0.string(key) } }
    public func u32(_ key: String) -> UInt32? { firstNonNil { $0.u32(key) } }
    public func u64(_ key: String) -> UInt64? { firstNonNil { $0.u64(key) } }
    public func u64Compat(_ key: String) -> UInt64? { firstNonNil { $0.u64Compat(key) } }
    public func f32Compat(_ key: String) -> Float? { firstNonNil { $0.f32Compat(key) } }
    public func bool(_ key: String) -> Bool? { firstNonNil { $0.bool(key) } }
    public func intArray(_ key: String) -> [Int64]? { firstNonNil { $0.intArray(key) } }
    public func floatArray(_ key: String) -> [Double]? { firstNonNil { $0.floatArray(key) } }
    public func stringArrayBytes(_ key: String) -> [[UInt8]]? { firstNonNil { $0.stringArrayBytes(key) } }

    private func firstNonNil<T>(_ pick: (GGUFModel) -> T?) -> T? {
        for m in shards { if let v = pick(m) { return v } }
        return nil
    }
}
