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

    /// Usage-driven per-layer slot allocation for the expert cache: keep the
    /// SAME total wired budget (`base` slots × layers with data) but move slots
    /// to the layers where the routing mass concentrates. Greedy on the marginal
    /// hit gain: slot #(r+1) of a layer is worth the routing share of its
    /// (r+1)-th hottest expert, so after giving every layer the LRU floor, the
    /// remaining budget goes to the highest marginal shares anywhere. Layers
    /// with flat routing (uniform ≈ 1/256 shares) naturally shrink toward the
    /// floor; concentrated layers grow toward `cap`.
    ///
    /// Returns nil when there is not enough history to trust (< ~43 tokens per
    /// active layer on average) — the caller falls back to the uniform `base`.
    /// Layers with no data are absent from the result (caller default applies).
    public func slotAllocation(base: Int, floor: Int = 8, cap: Int = 64) -> [Int: Int]? {
        let active = counts.indices.filter { !counts[$0].isEmpty }
        guard !active.isEmpty, base > floor else { return nil }
        guard totalRoutes >= 256 * active.count else { return nil }   // ≥ ~43 tokens of history
        let capped = max(floor, cap)
        var alloc: [Int: Int] = [:]
        // Marginal gains: (layer, share of the layer's routes captured by its
        // rank-r expert) for ranks floor..<cap — the value of ONE MORE slot.
        var gains: [(layer: Int, share: Double)] = []
        for il in active {
            alloc[il] = floor
            let total = Double(counts[il].values.reduce(0, +))
            guard total > 0 else { continue }
            let sorted = counts[il].values.sorted(by: >)
            for r in floor..<min(capped, sorted.count) {
                gains.append((layer: il, share: Double(sorted[r]) / total))
            }
        }
        var remaining = (base - floor) * active.count
        for g in gains.sorted(by: { $0.share > $1.share }) {
            if remaining == 0 { break }
            alloc[g.layer]! += 1
            remaining -= 1
        }
        return alloc
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

    /// Swap in another profile (agent switch): clear and load `data` if present.
    public func replace(with data: Data?) {
        reset()
        if let data { load(data) }
    }
}
