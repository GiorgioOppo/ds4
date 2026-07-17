import Foundation

/// Loads each layer's weights once and reuses them across tokens (weights are
/// read-only during decode). Keeps non-routed weights resident instead of
/// re-streaming them from the mmap every token.
final class CachedLayerProvider {
    private let make: (Int) throws -> LayerWeights
    private var cache: [Int: LayerWeights] = [:]
    init(_ make: @escaping (Int) throws -> LayerWeights) { self.make = make }
    func get(_ il: Int) throws -> LayerWeights {
        if let w = cache[il] { return w }
        let w = try make(il); cache[il] = w; return w
    }
}
