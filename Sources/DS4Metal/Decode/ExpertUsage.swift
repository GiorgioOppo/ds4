import Foundation

/// Per-(layer, expert) routing-frequency statistics — the runtime equivalent of
/// an "importance matrix" for experts. Measured from the router's actual
/// selections during real use, persisted across sessions, and used to pre-warm
/// the expert slot-cache with the historically hottest experts ("persistent"
/// experts always in RAM; the LRU handles the "changing" ones).
public final class ExpertUsageStats {
    private(set) var counts: [[Int32: Int]]   // per layer: expert id -> times routed
    public private(set) var totalRoutes = 0

    public init(nLayers: Int) {
        counts = Array(repeating: [:], count: nLayers)
    }

    public func record(layer: Int, ids: [Int32]) {
        guard layer >= 0, layer < counts.count else { return }
        for id in ids { counts[layer][id, default: 0] += 1 }
        totalRoutes += ids.count
    }

    /// The historically hottest experts of a layer (descending by count).
    public func top(layer: Int, n: Int) -> [Int32] {
        guard layer >= 0, layer < counts.count else { return [] }
        return counts[layer].sorted { $0.value > $1.value }.prefix(n).map(\.key)
    }

    /// Share of all routes in `layer` captured by its hottest `n` experts
    /// (1.0 = perfectly concentrated, n/256 ≈ uniform). The honest signal for
    /// whether expert caching can pay on this workload.
    public func concentration(layer: Int, n: Int) -> Double {
        guard layer >= 0, layer < counts.count else { return 0 }
        let total = counts[layer].values.reduce(0, +)
        guard total > 0 else { return 0 }
        let top = counts[layer].values.sorted(by: >).prefix(n).reduce(0, +)
        return Double(top) / Double(total)
    }

    // MARK: Persistence — compact JSON [[ [id, count], … ] × nLayers].

    public func serialize() -> Data? {
        let arr = counts.map { layer in
            layer.map { [Int($0.key), $0.value] }.sorted { $0[1] > $1[1] }
        }
        return try? JSONSerialization.data(withJSONObject: arr)
    }

    /// Merge persisted counts into the live ones (additive).
    public func load(_ data: Data) {
        guard let arr = try? JSONSerialization.jsonObject(with: data) as? [[[Int]]] else { return }
        for (il, layer) in arr.enumerated() where il < counts.count {
            for pair in layer where pair.count == 2 {
                counts[il][Int32(pair[0]), default: 0] += pair[1]
                totalRoutes += pair[1]
            }
        }
    }

    public func reset() {
        for i in counts.indices { counts[i] = [:] }
        totalRoutes = 0
    }
}
