import Foundation

/// Per-(layer, expert) usage count produced by `ExpertAnalyzer`. The
/// analyzer aggregates one of these for every routed expert it saw
/// during the calibration forward, even when the count is zero —
/// having the full grid makes the decision math and the dry-run
/// preview trivial.
public struct ExpertUsageRow: Sendable, Codable, Equatable {
    public let layerId: Int
    public let expertId: Int
    public let count: Int
    public init(layerId: Int, expertId: Int, count: Int) {
        self.layerId = layerId
        self.expertId = expertId
        self.count = count
    }
}

/// Decision produced by `ExpertAnalyzer`. Mirrors the shape of
/// `KeepDecision` from the vocab side: an explicit list of kept
/// ids per layer, plus the stats that drove the choice.
public struct ExpertKeepDecision: Sendable, Codable {

    /// Per-layer keep sets. `keepIds[layerId]` is the sorted list of
    /// routed-expert ids that survived; the complement `[0..<nRouted)
    /// \ keepIds[layerId]` is what the rewriter physically removes.
    public let keepIds: [[Int]]

    /// Per-layer dropped sets, precomputed for convenience. The
    /// rewriter uses this directly. Always equal to
    /// `[0..<nRouted) \ keepIds[i]` per layer.
    public let droppedIds: [[Int]]

    /// Number of routed experts in the source model (mirrors
    /// `config.n_routed_experts`). Constant across layers by design.
    public let nRoutedExperts: Int

    /// Number of activated experts per token (mirrors
    /// `config.n_activated_experts`). We refuse to drop below this
    /// per layer — there must always be at least topK alive experts
    /// or the gate top-K kernel would dereference nil slots.
    public let nActivatedExperts: Int

    /// Coverage threshold used to drive the decision (e.g. 0.99 =
    /// keep top-K experts that together account for 99% of routing
    /// assignments in the calibration corpus).
    public let coverage: Double

    /// Total routing assignments observed during analyzer
    /// (= `tokensSeen * topK` summed across layers, ÷ nLayers).
    public let totalAssignments: Int

    /// Per-layer fraction of assignments covered by the kept set.
    /// `actualCoveragePerLayer[i] ∈ [coverage, 1.0]`.
    public let actualCoveragePerLayer: [Double]

    /// Full grid of `(layerId, expertId) → count`. Useful for the
    /// dry-run preview and for reproducibility (caller can rebuild
    /// any decision threshold by re-running the math on this grid).
    public let usage: [ExpertUsageRow]

    public init(keepIds: [[Int]],
                droppedIds: [[Int]],
                nRoutedExperts: Int,
                nActivatedExperts: Int,
                coverage: Double,
                totalAssignments: Int,
                actualCoveragePerLayer: [Double],
                usage: [ExpertUsageRow])
    {
        self.keepIds = keepIds
        self.droppedIds = droppedIds
        self.nRoutedExperts = nRoutedExperts
        self.nActivatedExperts = nActivatedExperts
        self.coverage = coverage
        self.totalAssignments = totalAssignments
        self.actualCoveragePerLayer = actualCoveragePerLayer
        self.usage = usage
    }

    /// Convenience: count of experts dropped, summed across layers.
    public var totalDropped: Int {
        droppedIds.reduce(0) { $0 + $1.count }
    }

    /// Convenience: count of experts kept, summed across layers.
    /// `totalKept + totalDropped == nLayers * nRoutedExperts`.
    public var totalKept: Int {
        keepIds.reduce(0) { $0 + $1.count }
    }

    /// Number of main transformer layers covered by this decision.
    public var nLayers: Int { keepIds.count }

    // MARK: - Decision math

    /// Build a decision from raw usage counts via coverage threshold.
    ///
    /// - Parameter usage: full grid of `(layer, expert, count)`. Rows
    ///   with `count == 0` are tolerated (and tend to be the first to
    ///   drop).
    /// - Parameter nLayers, nRoutedExperts, nActivatedExperts: shape
    ///   parameters straight from `ModelConfig`.
    /// - Parameter coverage: target fraction of routing assignments
    ///   to cover with the kept set, per layer. `0.99` means: keep
    ///   the minimum number of top-frequency experts that together
    ///   account for ≥ 99% of the layer's routing decisions.
    /// - Parameter minKept: floor on the number of kept experts per
    ///   layer. The caller should pass `max(nActivatedExperts, 4)`
    ///   so the top-K kernel always has live targets even on
    ///   out-of-distribution prompts.
    ///
    /// Returns a decision with `actualCoveragePerLayer[i]` ≥ coverage
    /// whenever the layer received any routing at all. Layers that
    /// saw zero tokens (degenerate corpus) keep ALL experts — we
    /// can't justify dropping anything without evidence.
    public static func build(usage: [ExpertUsageRow],
                              nLayers: Int,
                              nRoutedExperts: Int,
                              nActivatedExperts: Int,
                              coverage: Double,
                              minKept: Int) -> ExpertKeepDecision
    {
        precondition(coverage > 0 && coverage <= 1)
        precondition(minKept >= nActivatedExperts,
                     "minKept (\(minKept)) must be ≥ nActivatedExperts (\(nActivatedExperts))")
        precondition(minKept <= nRoutedExperts,
                     "minKept (\(minKept)) must be ≤ nRoutedExperts (\(nRoutedExperts))")

        // Bucket usage by layer for quick lookup.
        var grid: [[Int]] = Array(repeating: Array(repeating: 0,
                                                     count: nRoutedExperts),
                                    count: nLayers)
        for r in usage {
            guard r.layerId >= 0 && r.layerId < nLayers,
                  r.expertId >= 0 && r.expertId < nRoutedExperts else { continue }
            grid[r.layerId][r.expertId] = r.count
        }

        var keepIds: [[Int]] = []
        var droppedIds: [[Int]] = []
        var actualCov: [Double] = []
        var totalAssignments = 0

        for L in 0..<nLayers {
            let row = grid[L]
            let layerTotal = row.reduce(0, +)
            totalAssignments += layerTotal

            // Layer saw no tokens → keep everything (nothing to drop
            // safely; the analyzer's coverage corpus was too narrow).
            guard layerTotal > 0 else {
                keepIds.append(Array(0..<nRoutedExperts))
                droppedIds.append([])
                actualCov.append(1.0)
                continue
            }

            // Sort expert ids by count desc; ties broken by ascending
            // id for determinism.
            let sortedIds = (0..<nRoutedExperts).sorted { a, b in
                if row[a] != row[b] { return row[a] > row[b] }
                return a < b
            }

            // Accumulate counts until we hit the coverage target.
            var keep: Set<Int> = []
            var running = 0
            let target = Int((coverage * Double(layerTotal)).rounded(.up))
            for eid in sortedIds {
                keep.insert(eid)
                running += row[eid]
                if running >= target { break }
            }

            // Floor: ensure at least `minKept` experts survive (even
            // if coverage hit earlier). Pad with the next highest-
            // usage experts until reaching the floor.
            if keep.count < minKept {
                for eid in sortedIds {
                    if keep.contains(eid) { continue }
                    keep.insert(eid)
                    running += row[eid]
                    if keep.count >= minKept { break }
                }
            }

            let keptSorted = keep.sorted()
            let droppedSorted = (0..<nRoutedExperts).filter { !keep.contains($0) }
            keepIds.append(keptSorted)
            droppedIds.append(droppedSorted)
            actualCov.append(Double(running) / Double(layerTotal))
        }

        return ExpertKeepDecision(
            keepIds: keepIds,
            droppedIds: droppedIds,
            nRoutedExperts: nRoutedExperts,
            nActivatedExperts: nActivatedExperts,
            coverage: coverage,
            totalAssignments: totalAssignments,
            actualCoveragePerLayer: actualCov,
            usage: usage)
    }
}
