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
        let observedRoutes = active.reduce(0) { partial, il in
            partial + counts[il].values.reduce(0, +)
        }
        guard observedRoutes >= nExperts * active.count else { return nil }
        let unitCosts = Dictionary(uniqueKeysWithValues: active.map { ($0, 1) })
        return slotAllocationLocked(active: active,
                                    budgetBytes: base * active.count,
                                    bytesPerSlot: unitCosts,
                                    fixedFloorLayers: [],
                                    floor: floor, cap: cap)
    }

    /// Byte-budgeted usage allocation for mixed-quant expert caches.
    ///
    /// Unlike ``slotAllocation(base:floor:cap:)``, one slot can have a
    /// different cost in every layer (for example an IQ2/Q2 expert record is
    /// much smaller than a Q4/Q4/Q4 one). One slot's expected I/O saving is its
    /// marginal routing share multiplied by the record size, while its RAM
    /// cost is the same record size; the benefit/cost score therefore remains
    /// the marginal share. Dividing by the record size here would penalize Q4
    /// twice and strand the layers with the most bytes to save. The returned
    /// allocation never exceeds `budgetBytes`.
    ///
    /// - Parameters:
    ///   - budgetBytes: Total byte budget available to all returned pools.
    ///   - bytesPerSlot: Exact gate+up+down record size for each layer.
    ///   - cacheableLayers: Optional allow-list. Layers absent from it are not
    ///     allocated and, importantly, do not count toward history readiness.
    ///   - fixedFloorLayers: Layers that must stay at `floor`. Hash-routed
    ///     layers with exact look-ahead are the main use: extra historical-hot
    ///     slots bring less value there than on unpredictable router layers.
    ///
    /// Returns nil when the budget cannot provide the mandatory floor, or
    /// until **every** eligible layer has accumulated roughly one route per
    /// expert. Layers with no positive byte cost are absent.
    public func slotAllocation(budgetBytes: Int,
                               bytesPerSlot: [Int: Int],
                               cacheableLayers: Set<Int>? = nil,
                               fixedFloorLayers: Set<Int> = [],
                               floor: Int = 8,
                               cap: Int = 64) -> [Int: Int]? {
        lock.lock(); defer { lock.unlock() }
        guard budgetBytes > 0, floor > 0 else { return nil }
        let expected = counts.indices.filter { il in
            (cacheableLayers?.contains(il) ?? true) && (bytesPerSlot[il] ?? 0) > 0
        }
        guard !expected.isEmpty else { return nil }
        // Every pool in the all-layer plan needs a trustworthy allocation. If
        // even one expected layer is missing history, returning a partial map
        // would make the cache fall back to `base` for that layer and could
        // silently exceed the byte budget. Bypassed layers are excluded by the
        // allow-list and cannot satisfy readiness for a cacheable one.
        guard expected.allSatisfy({ il in
            counts[il].values.reduce(0, +) >= nExperts
        }) else { return nil }
        return slotAllocationLocked(active: expected,
                                    budgetBytes: budgetBytes,
                                    bytesPerSlot: bytesPerSlot,
                                    fixedFloorLayers: fixedFloorLayers,
                                    floor: floor, cap: cap)
    }

    /// History-independent fallback for a mixed-quant cache. It gives every
    /// eligible layer the mandatory LRU floor, then repeatedly grows the pool
    /// with the smallest resident byte footprint. The result is byte-balanced:
    /// a layer whose records cost about 2x naturally receives about half as
    /// many slots, instead of a uniform-slot fallback silently exceeding RAM.
    public static func byteBalancedSlotAllocation(budgetBytes: Int,
                                                  bytesPerSlot: [Int: Int],
                                                  cacheableLayers: Set<Int>? = nil,
                                                  fixedFloorLayers: Set<Int> = [],
                                                  floor: Int = 8,
                                                  cap: Int = 64) -> [Int: Int]? {
        guard budgetBytes > 0, floor > 0 else { return nil }
        let active = bytesPerSlot.keys.filter { il in
            (cacheableLayers?.contains(il) ?? true) && (bytesPerSlot[il] ?? 0) > 0
        }.sorted()
        guard !active.isEmpty else { return nil }
        var alloc = Dictionary(uniqueKeysWithValues: active.map { ($0, floor) })
        let floorBytes = active.reduce(0) { partial, il in
            partial + floor * (bytesPerSlot[il] ?? 0)
        }
        guard floorBytes <= budgetBytes else { return nil }
        let capped = max(floor, cap)
        var remaining = budgetBytes - floorBytes
        while let next = active
            .filter({ il in
                !fixedFloorLayers.contains(il)
                    && (alloc[il] ?? floor) < capped
                    && (bytesPerSlot[il] ?? 0) <= remaining
            })
            .min(by: { lhs, rhs in
                let lhsBytes = ((alloc[lhs] ?? floor) + 1) * (bytesPerSlot[lhs] ?? 0)
                let rhsBytes = ((alloc[rhs] ?? floor) + 1) * (bytesPerSlot[rhs] ?? 0)
                return lhsBytes == rhsBytes ? lhs < rhs : lhsBytes < rhsBytes
            }) {
            let cost = bytesPerSlot[next]!
            alloc[next, default: floor] += 1
            remaining -= cost
        }
        return alloc
    }

    /// Lock must be held by the caller.
    private func slotAllocationLocked(active: [Int],
                                      budgetBytes: Int,
                                      bytesPerSlot: [Int: Int],
                                      fixedFloorLayers: Set<Int>,
                                      floor: Int,
                                      cap: Int) -> [Int: Int]? {
        let capped = max(floor, cap)
        var alloc = Dictionary(uniqueKeysWithValues: active.map { ($0, floor) })
        let floorBytes = active.reduce(0) { partial, il in
            partial + floor * (bytesPerSlot[il] ?? 0)
        }
        guard floorBytes <= budgetBytes else { return nil }

        // Expected I/O bytes saved divided by RAM bytes spent equals `share`:
        // both numerator and denominator contain the record size. Rank r is
        // naturally considered after ranks < r because per-layer counts are
        // non-increasing and every rank of a layer has the same byte cost.
        var gains: [(layer: Int, rank: Int, share: Double, cost: Int)] = []
        for il in active where !fixedFloorLayers.contains(il) {
            guard let cost = bytesPerSlot[il], cost > 0 else { continue }
            let total = Double(counts[il].values.reduce(0, +))
            guard total > 0 else { continue }
            let sorted = counts[il].values.sorted(by: >)
            for rank in floor..<min(capped, sorted.count) {
                let share = Double(sorted[rank]) / total
                gains.append((layer: il, rank: rank, share: share, cost: cost))
            }
        }
        gains.sort {
            if $0.share != $1.share { return $0.share > $1.share }
            if $0.layer != $1.layer { return $0.layer < $1.layer }
            return $0.rank < $1.rank
        }
        var remaining = budgetBytes - floorBytes
        for gain in gains where gain.cost <= remaining {
            alloc[gain.layer, default: floor] += 1
            remaining -= gain.cost
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
