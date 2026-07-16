import Foundation

/// Per-(layer, expert) routing-frequency statistics — the runtime equivalent of
/// an "importance matrix" for experts. Measured from the router's actual
/// selections during real use, persisted across sessions, and used to pre-warm
/// the expert slot-cache with the historically hottest experts ("persistent"
/// experts always in RAM; the LRU handles the "changing" ones).
///
/// THREAD-SAFE: `record` runs on the decode thread while the slot-cache's
/// speculative look-ahead reads `top`/`slotAllocation` from a background
/// queue (pool creation included) — every access goes through one lock.
/// The lock is uncontended in practice (short critical sections).
public final class ExpertUsageStats: @unchecked Sendable {
    private var counts: [[Int32: Int]]        // per layer: expert id -> times routed
    private let nExperts: Int
    private var _totalRoutes = 0
    private let lock = NSLock()

    public var totalRoutes: Int { lock.lock(); defer { lock.unlock() }; return _totalRoutes }

    public init(nLayers: Int, nExperts: Int = 256) {
        precondition(nLayers >= 0 && nExperts > 0)
        counts = Array(repeating: [:], count: nLayers)
        self.nExperts = nExperts
    }

    public func record(layer: Int, ids: [Int32]) {
        lock.lock(); defer { lock.unlock() }
        guard layer >= 0, layer < counts.count else { return }
        // Ids are validated (≥ 0) so a corrupt readback can never poison the
        // profile — a bad id persisted here would break the cache pre-warm
        // (GGUFWeights.copyExpert throws on out-of-range) on every later run.
        var valid = 0
        for id in ids where id >= 0 && Int(id) < nExperts {
            counts[layer][id, default: 0] += 1
            valid += 1
        }
        _totalRoutes += valid
    }

    /// Total routed picks recorded for a layer (0 = no data / dense layer).
    public func routes(layer: Int) -> Int {
        lock.lock(); defer { lock.unlock() }
        guard layer >= 0, layer < counts.count else { return 0 }
        return counts[layer].values.reduce(0, +)
    }

    /// The historically hottest experts of a layer (descending by count).
    public func top(layer: Int, n: Int) -> [Int32] {
        lock.lock(); defer { lock.unlock() }
        guard layer >= 0, layer < counts.count else { return [] }
        return counts[layer].sorted { $0.value > $1.value }.prefix(n).map(\.key)
    }

    /// Share of all routes in `layer` captured by its hottest `n` experts
    /// (1.0 = perfectly concentrated, n/nExperts ≈ uniform). The honest signal for
    /// whether expert caching can pay on this workload.
    public func concentration(layer: Int, n: Int) -> Double {
        lock.lock(); defer { lock.unlock() }
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
    /// with flat routing (uniform ≈ 1/nExperts shares) naturally shrink toward the
    /// floor; concentrated layers grow toward `cap`.
    ///
    /// Returns nil until roughly one route per expert has been observed in each
    /// active layer — the caller then falls back to the uniform `base`.
    /// Layers with no data are absent from the result (caller default applies).
    public func slotAllocation(base: Int, floor: Int = 8, cap: Int = 64) -> [Int: Int]? {
        lock.lock(); defer { lock.unlock() }
        let active = counts.indices.filter { !counts[$0].isEmpty }
        guard !active.isEmpty, base > floor else { return nil }
        guard _totalRoutes >= nExperts * active.count else { return nil }
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
        lock.lock(); defer { lock.unlock() }
        let arr = counts.map { layer in
            layer.map { [Int($0.key), $0.value] }.sorted { $0[1] > $1[1] }
        }
        return try? JSONSerialization.data(withJSONObject: arr)
    }

    /// Merge persisted counts into the live ones (additive).
    public func load(_ data: Data) {
        lock.lock(); defer { lock.unlock() }
        loadLocked(data)
    }

    private func loadLocked(_ data: Data) {
        guard let arr = try? JSONSerialization.jsonObject(with: data) as? [[[Int]]] else { return }
        for (il, layer) in arr.enumerated() where il < counts.count {
            // A corrupt/foreign profile must degrade to "entry ignored" (like the
            // C hotlist loader), never trap (Int32(truncating)) or poison the
            // warm set with out-of-range ids that break pool creation.
            for pair in layer where pair.count == 2 {
                guard let id = Int32(exactly: pair[0]), id >= 0,
                      Int(id) < nExperts, pair[1] > 0 else { continue }
                counts[il][id, default: 0] += pair[1]
                _totalRoutes += pair[1]
            }
        }
    }

    public func reset() {
        lock.lock(); defer { lock.unlock() }
        resetLocked()
    }

    private func resetLocked() {
        for i in counts.indices { counts[i] = [:] }
        _totalRoutes = 0
    }

    /// Swap in another profile (agent switch): clear and load `data` if present.
    public func replace(with data: Data?) {
        lock.lock(); defer { lock.unlock() }
        resetLocked()
        if let data { loadLocked(data) }
    }
}
