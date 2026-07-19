import Foundation
import DS4Core
import DS4Metal

extension InferenceService {
public struct BenchPoint: Sendable {
        public let contextTokens: Int
        public let prefillTps: Double
        /// Media semplice (token generati / tempo totale) — sporca dei costi
        /// una-tantum (primo token freddo, stalli).
        public let genTps: Double
        /// 99° percentile della VELOCITÀ per-token (1/durata di ogni token,
        /// ordinati): la velocità di regime raggiunta, robusta agli outlier
        /// lenti. È la metrica riportata dal Bench.
        public let genTpsP99: Double
        public let kvBytes: UInt64
        /// Velocità per-token IN ORDINE DI GENERAZIONE (1/durata di ciascuno):
        /// distingue la partenza fredda (primi token lenti, poi regime — normale
        /// dopo un reload) dal degrado progressivo da pressione di memoria
        /// (coda più lenta della testa) — media e p99 da soli non li separano.
        public let genSpeeds: [Double]
        /// Optional deterministic quality trace used by the in-app machine
        /// autotuner. It is deliberately bounded to the generated decisions:
        /// token ids, full-vocabulary Float32 hashes and the ordered top-3.
        public let qualitySignature: MachineAutoTuneQualitySignature?
        public init(contextTokens: Int, prefillTps: Double, genTps: Double, kvBytes: UInt64,
                    genTpsP99: Double = 0, genSpeeds: [Double] = [],
                    qualitySignature: MachineAutoTuneQualitySignature? = nil) {
            self.contextTokens = contextTokens; self.prefillTps = prefillTps
            self.genTps = genTps; self.kvBytes = kvBytes
            self.genTpsP99 = genTpsP99 > 0 ? genTpsP99 : genTps
            self.genSpeeds = genSpeeds
            self.qualitySignature = qualitySignature
        }
    }

    /// One exact next-token comparison emitted while teacher forcing advances.
    /// `index` is one-based within this benchmark; `sourceTokenIndex` is the
    /// zero-based target position in the tokenized source text.
    public struct AccuracyObservation: Sendable, Equatable, Identifiable {
        public let index: Int
        /// Zero-based sampled-piece index. Legacy single-piece runs use 0.
        public let pieceIndex: Int
        public let sourceTokenIndex: Int
        public let expectedTokenId: Int
        /// Up to three unique candidates, ordered by descending model logit.
        public let predictedTokenIds: [Int]
        public let top1Correct: Bool
        public let top2Correct: Bool
        public let top3Correct: Bool
        /// Fractions in 0...1 after including this observation.
        public let cumulativeTop1Accuracy: Double
        public let cumulativeTop2Accuracy: Double
        public let cumulativeTop3Accuracy: Double

        /// Backward-compatible top-1 aliases.
        public var predictedTokenId: Int { predictedTokenIds.first ?? -1 }
        public var correct: Bool { top1Correct }
        public var cumulativeAccuracy: Double { cumulativeTop1Accuracy }
        public var id: Int { index }

        public init(index: Int, pieceIndex: Int = 0,
                    sourceTokenIndex: Int, expectedTokenId: Int,
                    predictedTokenIds: [Int], top1Correct: Bool,
                    top2Correct: Bool, top3Correct: Bool,
                    cumulativeTop1Accuracy: Double,
                    cumulativeTop2Accuracy: Double,
                    cumulativeTop3Accuracy: Double) {
            self.index = index
            self.pieceIndex = pieceIndex
            self.sourceTokenIndex = sourceTokenIndex
            self.expectedTokenId = expectedTokenId
            self.predictedTokenIds = predictedTokenIds
            self.top1Correct = top1Correct
            self.top2Correct = top2Correct
            self.top3Correct = top3Correct
            self.cumulativeTop1Accuracy = cumulativeTop1Accuracy
            self.cumulativeTop2Accuracy = cumulativeTop2Accuracy
            self.cumulativeTop3Accuracy = cumulativeTop3Accuracy
        }

        /// Source-compatible initializer for existing top-1 clients.
        public init(index: Int, pieceIndex: Int = 0,
                    sourceTokenIndex: Int, expectedTokenId: Int,
                    predictedTokenId: Int, correct: Bool,
                    cumulativeAccuracy: Double) {
            self.init(
                index: index,
                pieceIndex: pieceIndex,
                sourceTokenIndex: sourceTokenIndex,
                expectedTokenId: expectedTokenId,
                predictedTokenIds: [predictedTokenId],
                top1Correct: correct,
                top2Correct: correct,
                top3Correct: correct,
                cumulativeTop1Accuracy: cumulativeAccuracy,
                cumulativeTop2Accuracy: cumulativeAccuracy,
                cumulativeTop3Accuracy: cumulativeAccuracy
            )
        }
    }

    /// One graph-ready block of consecutive observations. Start/end are
    /// one-based, inclusive observation indices (not source-token offsets).
    /// `index` itself is zero-based for direct chart/collection addressing.
    public struct AccuracyBucket: Sendable, Equatable, Identifiable {
        public let index: Int
        public let startIndex: Int
        public let endIndex: Int
        public let top1CorrectTokens: Int
        public let top2CorrectTokens: Int
        public let top3CorrectTokens: Int
        public let evaluatedTokens: Int
        /// Fractions in 0...1 within this bucket.
        public let localTop1Accuracy: Double
        public let localTop2Accuracy: Double
        public let localTop3Accuracy: Double
        /// Fractions in 0...1 over all observations through this bucket.
        public let cumulativeTop1Accuracy: Double
        public let cumulativeTop2Accuracy: Double
        public let cumulativeTop3Accuracy: Double

        /// Backward-compatible top-1 aliases.
        public var correctTokens: Int { top1CorrectTokens }
        public var localAccuracy: Double { localTop1Accuracy }
        public var cumulativeAccuracy: Double { cumulativeTop1Accuracy }
        public var id: Int { index }

        public init(index: Int, startIndex: Int, endIndex: Int,
                    top1CorrectTokens: Int, top2CorrectTokens: Int,
                    top3CorrectTokens: Int, evaluatedTokens: Int,
                    localTop1Accuracy: Double, localTop2Accuracy: Double,
                    localTop3Accuracy: Double,
                    cumulativeTop1Accuracy: Double,
                    cumulativeTop2Accuracy: Double,
                    cumulativeTop3Accuracy: Double) {
            self.index = index
            self.startIndex = startIndex
            self.endIndex = endIndex
            self.top1CorrectTokens = top1CorrectTokens
            self.top2CorrectTokens = top2CorrectTokens
            self.top3CorrectTokens = top3CorrectTokens
            self.evaluatedTokens = evaluatedTokens
            self.localTop1Accuracy = localTop1Accuracy
            self.localTop2Accuracy = localTop2Accuracy
            self.localTop3Accuracy = localTop3Accuracy
            self.cumulativeTop1Accuracy = cumulativeTop1Accuracy
            self.cumulativeTop2Accuracy = cumulativeTop2Accuracy
            self.cumulativeTop3Accuracy = cumulativeTop3Accuracy
        }

        /// Source-compatible initializer for existing top-1 clients.
        public init(index: Int, startIndex: Int, endIndex: Int,
                    correctTokens: Int, evaluatedTokens: Int,
                    localAccuracy: Double, cumulativeAccuracy: Double) {
            self.init(
                index: index,
                startIndex: startIndex,
                endIndex: endIndex,
                top1CorrectTokens: correctTokens,
                top2CorrectTokens: correctTokens,
                top3CorrectTokens: correctTokens,
                evaluatedTokens: evaluatedTokens,
                localTop1Accuracy: localAccuracy,
                localTop2Accuracy: localAccuracy,
                localTop3Accuracy: localAccuracy,
                cumulativeTop1Accuracy: cumulativeAccuracy,
                cumulativeTop2Accuracy: cumulativeAccuracy,
                cumulativeTop3Accuracy: cumulativeAccuracy
            )
        }
    }

    /// One deterministic corpus segment selected for teacher-forced scoring.
    public struct AccuracyPiecePlan: Sendable, Equatable, Identifiable {
        /// Zero-based execution order within the sampling plan.
        public let index: Int
        public let sourceStartTokenIndex: Int
        public let targetStartTokenIndex: Int
        public let contextTokens: Int
        public let evaluatedTokens: Int
        /// True when fewer than the requested tokens can be evaluated.
        public let truncated: Bool
        public var id: Int { index }

        public init(index: Int, sourceStartTokenIndex: Int,
                    targetStartTokenIndex: Int, contextTokens: Int,
                    evaluatedTokens: Int, truncated: Bool) {
            self.index = index
            self.sourceStartTokenIndex = sourceStartTokenIndex
            self.targetStartTokenIndex = targetStartTokenIndex
            self.contextTokens = contextTokens
            self.evaluatedTokens = evaluatedTokens
            self.truncated = truncated
        }
    }

    /// Pure, reproducible plan produced before any Metal execution.
    public struct AccuracySamplingPlan: Sendable, Equatable {
        public let pieces: [AccuracyPiecePlan]
        public let requestedPieceCount: Int
        public let seed: UInt64
        public let effectiveMinContextTokens: Int
        public let effectiveMaxContextTokens: Int
        /// True when any requested bound/count/length had to be reduced.
        public let truncated: Bool
        public var actualPieceCount: Int { pieces.count }

        public init(pieces: [AccuracyPiecePlan], requestedPieceCount: Int,
                    seed: UInt64, effectiveMinContextTokens: Int,
                    effectiveMaxContextTokens: Int, truncated: Bool) {
            self.pieces = pieces
            self.requestedPieceCount = requestedPieceCount
            self.seed = seed
            self.effectiveMinContextTokens = effectiveMinContextTokens
            self.effectiveMaxContextTokens = effectiveMaxContextTokens
            self.truncated = truncated
        }
    }

    /// Accuracy counters for one sampled corpus segment.
    public struct AccuracyPieceResult: Sendable, Equatable, Identifiable {
        public let index: Int
        public let sourceStartTokenIndex: Int
        public let targetStartTokenIndex: Int
        public let contextTokens: Int
        public let evaluatedTokens: Int
        public let top1CorrectTokens: Int
        public let top2CorrectTokens: Int
        public let top3CorrectTokens: Int
        public let top1Accuracy: Double
        public let top2Accuracy: Double
        public let top3Accuracy: Double
        public let truncated: Bool
        public var id: Int { index }

        /// Backward-style top-1 aliases for generic result rendering.
        public var correctTokens: Int { top1CorrectTokens }
        public var accuracy: Double { top1Accuracy }

        public init(index: Int, sourceStartTokenIndex: Int,
                    targetStartTokenIndex: Int, contextTokens: Int,
                    evaluatedTokens: Int, top1CorrectTokens: Int,
                    top2CorrectTokens: Int, top3CorrectTokens: Int,
                    top1Accuracy: Double, top2Accuracy: Double,
                    top3Accuracy: Double, truncated: Bool) {
            self.index = index
            self.sourceStartTokenIndex = sourceStartTokenIndex
            self.targetStartTokenIndex = targetStartTokenIndex
            self.contextTokens = contextTokens
            self.evaluatedTokens = evaluatedTokens
            self.top1CorrectTokens = top1CorrectTokens
            self.top2CorrectTokens = top2CorrectTokens
            self.top3CorrectTokens = top3CorrectTokens
            self.top1Accuracy = top1Accuracy
            self.top2Accuracy = top2Accuracy
            self.top3Accuracy = top3Accuracy
            self.truncated = truncated
        }
    }

    public struct AccuracyResult: Sendable, Equatable {
        public let originalTokens: Int
        /// Unscored source tokens used as initial context.
        public let contextTokens: Int
        public let evaluatedTokens: Int
        public let top1CorrectTokens: Int
        public let top2CorrectTokens: Int
        public let top3CorrectTokens: Int
        /// Exact top-k next-token accuracies as fractions in 0...1.
        public let top1Accuracy: Double
        public let top2Accuracy: Double
        public let top3Accuracy: Double
        public let duration: Double
        public let evaluatedTps: Double
        public let truncated: Bool
        public let bucketSize: Int
        public let buckets: [AccuracyBucket]
        public let pieces: [AccuracyPieceResult]
        public let requestedPieceCount: Int
        public let seed: UInt64
        public let effectiveMinContextTokens: Int
        public let effectiveMaxContextTokens: Int

        /// Backward-compatible top-1 aliases.
        public var correctTokens: Int { top1CorrectTokens }
        public var accuracy: Double { top1Accuracy }

        public init(originalTokens: Int, contextTokens: Int,
                    evaluatedTokens: Int, top1CorrectTokens: Int,
                    top2CorrectTokens: Int, top3CorrectTokens: Int,
                    top1Accuracy: Double, top2Accuracy: Double,
                    top3Accuracy: Double, duration: Double, evaluatedTps: Double,
                    truncated: Bool, bucketSize: Int,
                    buckets: [AccuracyBucket],
                    pieces: [AccuracyPieceResult], requestedPieceCount: Int,
                    seed: UInt64, effectiveMinContextTokens: Int,
                    effectiveMaxContextTokens: Int) {
            self.originalTokens = originalTokens
            self.contextTokens = contextTokens
            self.evaluatedTokens = evaluatedTokens
            self.top1CorrectTokens = top1CorrectTokens
            self.top2CorrectTokens = top2CorrectTokens
            self.top3CorrectTokens = top3CorrectTokens
            self.top1Accuracy = top1Accuracy
            self.top2Accuracy = top2Accuracy
            self.top3Accuracy = top3Accuracy
            self.duration = duration
            self.evaluatedTps = evaluatedTps
            self.truncated = truncated
            self.bucketSize = bucketSize
            self.buckets = buckets
            self.pieces = pieces
            self.requestedPieceCount = requestedPieceCount
            self.seed = seed
            self.effectiveMinContextTokens = effectiveMinContextTokens
            self.effectiveMaxContextTokens = effectiveMaxContextTokens
        }

        /// Source-compatible top-k initializer used before sampled pieces.
        public init(originalTokens: Int, contextTokens: Int,
                    evaluatedTokens: Int, top1CorrectTokens: Int,
                    top2CorrectTokens: Int, top3CorrectTokens: Int,
                    top1Accuracy: Double, top2Accuracy: Double,
                    top3Accuracy: Double, duration: Double, evaluatedTps: Double,
                    truncated: Bool, bucketSize: Int,
                    buckets: [AccuracyBucket]) {
            self.init(
                originalTokens: originalTokens,
                contextTokens: contextTokens,
                evaluatedTokens: evaluatedTokens,
                top1CorrectTokens: top1CorrectTokens,
                top2CorrectTokens: top2CorrectTokens,
                top3CorrectTokens: top3CorrectTokens,
                top1Accuracy: top1Accuracy,
                top2Accuracy: top2Accuracy,
                top3Accuracy: top3Accuracy,
                duration: duration,
                evaluatedTps: evaluatedTps,
                truncated: truncated,
                bucketSize: bucketSize,
                buckets: buckets,
                pieces: [],
                requestedPieceCount: 1,
                seed: 0,
                effectiveMinContextTokens: contextTokens,
                effectiveMaxContextTokens: contextTokens
            )
        }

        /// Source-compatible initializer for existing top-1 clients.
        public init(originalTokens: Int, contextTokens: Int,
                    evaluatedTokens: Int, correctTokens: Int,
                    accuracy: Double, duration: Double, evaluatedTps: Double,
                    truncated: Bool, bucketSize: Int,
                    buckets: [AccuracyBucket]) {
            self.init(
                originalTokens: originalTokens,
                contextTokens: contextTokens,
                evaluatedTokens: evaluatedTokens,
                top1CorrectTokens: correctTokens,
                top2CorrectTokens: correctTokens,
                top3CorrectTokens: correctTokens,
                top1Accuracy: accuracy,
                top2Accuracy: accuracy,
                top3Accuracy: accuracy,
                duration: duration,
                evaluatedTps: evaluatedTps,
                truncated: truncated,
                bucketSize: bucketSize,
                buckets: buckets
            )
        }
    }

    public enum AccuracyBenchmarkError: Error, Sendable, Equatable,
                                        CustomStringConvertible, LocalizedError {
        case insufficientTokens(actual: Int)
        case contextTooSmall(actual: Int)

        public var description: String {
            switch self {
            case .insufficientTokens(let actual):
                return "Correctness benchmark requires at least 2 text tokens (found \(actual))."
            case .contextTooSmall(let actual):
                return "The loaded context must hold at least BOS plus one text token (found \(actual))."
            }
        }

        public var errorDescription: String? { description }
    }

    /// Build a deterministic multi-piece plan without tokenization, model state
    /// or Metal access. Distinct target starts are sampled without replacement.
    /// Starts that can provide the full requested continuation are consumed
    /// first; short tail starts are used only when the requested piece count
    /// exceeds that pool. A start before `effectiveMaxContextTokens` locally
    /// clamps its context ceiling to the number of preceding source tokens.
    public static func makeAccuracySamplingPlan(
        sourceTokenCount: Int,
        contextSize: Int,
        minContextTokens: Int,
        maxContextTokens: Int,
        maxTokensPerPiece: Int,
        pieceCount: Int,
        seed: UInt64
    ) -> AccuracySamplingPlan {
        let sourceCount = max(0, sourceTokenCount)
        let kvCapacity = max(0, contextSize)
        // A correctness run always evaluates at least one independently sampled
        // piece. This mirrors the other scalar inputs (`maxTokensPerPiece`,
        // context bounds) and prevents a successful-looking zero-token result
        // when API clients pass zero or a negative count.
        let requestedPieces = max(1, pieceCount)
        let requestedTokens = max(1, maxTokensPerPiece)
        let contextCeiling = max(0, min(sourceCount - 1, kvCapacity - 1))

        guard contextCeiling > 0 else {
            return AccuracySamplingPlan(
                pieces: [],
                requestedPieceCount: requestedPieces,
                seed: seed,
                effectiveMinContextTokens: 0,
                effectiveMaxContextTokens: 0,
                truncated: true
            )
        }

        let effectiveMin = min(max(1, minContextTokens), contextCeiling)
        let effectiveMax = min(max(effectiveMin, maxContextTokens), contextCeiling)
        // Domain-separated streams keep selection of each pool and context
        // lengths reproducible independently. In particular, increasing the
        // requested piece count preserves the already selected prefix and its
        // context lengths.
        var completeRNG = AccuracySamplingRNG(
            seed: seed ^ 0xC6BC_2796_92B5_C323
        )
        var tailRNG = AccuracySamplingRNG(
            seed: seed ^ 0xD1B5_4A32_D192_ED03
        )
        var contextRNG = AccuracySamplingRNG(
            seed: seed ^ 0x94D0_49BB_1331_11EB
        )

        // A target has a complete corpus continuation when target + R <= N.
        // Sample the complete and short-tail pools independently so selecting
        // additional pieces never perturbs the already selected complete pool.
        // The sparse partial Fisher-Yates sampler stores only the requested
        // swaps: O(selected pieces) memory/time instead of O(corpus tokens).
        let firstTail = max(effectiveMin,
                            min(sourceCount, sourceCount - requestedTokens + 1))
        var selected = sampleAccuracyStarts(
            in: effectiveMin..<firstTail,
            count: requestedPieces,
            using: &completeRNG
        )
        let remaining = requestedPieces - selected.count
        if remaining > 0 {
            selected.append(contentsOf: sampleAccuracyStarts(
                in: firstTail..<sourceCount,
                count: remaining,
                using: &tailRNG
            ))
        }

        var pieces: [AccuracyPiecePlan] = []
        pieces.reserveCapacity(selected.count)
        for (index, targetStart) in selected.enumerated() {
            let pieceMaxContext = min(effectiveMax, targetStart)
            let contextSpan = pieceMaxContext - effectiveMin + 1
            let context = effectiveMin + contextRNG.nextInt(upperBound: contextSpan)
            let evaluated = min(requestedTokens,
                                sourceCount - targetStart,
                                kvCapacity - context)
            pieces.append(AccuracyPiecePlan(
                index: index,
                sourceStartTokenIndex: targetStart - context,
                targetStartTokenIndex: targetStart,
                contextTokens: context,
                evaluatedTokens: evaluated,
                truncated: evaluated < requestedTokens
            ))
        }

        let wasClamped = sourceCount != sourceTokenCount
            || kvCapacity != contextSize
            || requestedPieces != pieceCount
            || requestedTokens != maxTokensPerPiece
            || effectiveMin != minContextTokens
            || effectiveMax != maxContextTokens
            || pieces.count != requestedPieces
            || pieces.contains(where: { $0.truncated })
        return AccuracySamplingPlan(
            pieces: pieces,
            requestedPieceCount: requestedPieces,
            seed: seed,
            effectiveMinContextTokens: effectiveMin,
            effectiveMaxContextTokens: effectiveMax,
            truncated: wasClamped
        )
    }

    private struct AccuracySamplingRNG {
        private var state: UInt64

        init(seed: UInt64) {
            state = seed
        }

        mutating func next() -> UInt64 {
            state &+= 0x9E37_79B9_7F4A_7C15
            var value = state
            value = (value ^ (value >> 30)) &* 0xBF58_476D_1CE4_E5B9
            value = (value ^ (value >> 27)) &* 0x94D0_49BB_1331_11EB
            return value ^ (value >> 31)
        }

        mutating func nextInt(upperBound: Int) -> Int {
            precondition(upperBound > 0)
            guard upperBound > 1 else { return 0 }
            let bound = UInt64(upperBound)
            let threshold = (0 &- bound) % bound
            var value = next()
            while value < threshold { value = next() }
            return Int(value % bound)
        }
    }

    /// Return the first `count` values of a uniformly shuffled integer range
    /// without materializing the whole range. `swaps` is the sparse state of a
    /// forward Fisher-Yates shuffle; consumed positions need not be retained.
    private static func sampleAccuracyStarts(
        in range: Range<Int>,
        count: Int,
        using rng: inout AccuracySamplingRNG
    ) -> [Int] {
        let poolCount = range.count
        let sampleCount = min(max(0, count), poolCount)
        guard sampleCount > 0 else { return [] }

        var swaps: [Int: Int] = [:]
        swaps.reserveCapacity(sampleCount)
        var sampled: [Int] = []
        sampled.reserveCapacity(sampleCount)

        for index in 0..<sampleCount {
            let selectedIndex = index + rng.nextInt(upperBound: poolCount - index)
            let selectedValue = swaps[selectedIndex] ?? selectedIndex
            let displacedValue = swaps[index] ?? index
            if selectedIndex != index {
                swaps[selectedIndex] = displacedValue
            }
            swaps.removeValue(forKey: index)
            sampled.append(range.lowerBound + selectedValue)
        }
        return sampled
    }

    /// Pure aggregation state kept separate from Metal so bucket boundaries,
    /// percentages and empty results can be unit-tested without a model.
    public struct AccuracyAccumulator: Sendable {
        public let bucketSize: Int
        public private(set) var evaluatedTokens = 0
        public private(set) var top1CorrectTokens = 0
        public private(set) var top2CorrectTokens = 0
        public private(set) var top3CorrectTokens = 0

        /// Backward-compatible top-1 alias.
        public var correctTokens: Int { top1CorrectTokens }

        private var completedBuckets: [AccuracyBucket] = []
        private var currentBucketTop1Correct = 0
        private var currentBucketTop2Correct = 0
        private var currentBucketTop3Correct = 0
        private var currentBucketEvaluated = 0
        private var currentBucketStart = 1

        public init(bucketSize: Int) {
            self.bucketSize = max(1, bucketSize)
        }

        @discardableResult
        public mutating func append(sourceTokenIndex: Int, expectedTokenId: Int,
                                    predictedTokenIds: [Int],
                                    pieceIndex: Int = 0) -> AccuracyObservation {
            var seen = Set<Int>()
            var normalized: [Int] = []
            normalized.reserveCapacity(min(3, predictedTokenIds.count))
            for tokenId in predictedTokenIds where seen.insert(tokenId).inserted {
                normalized.append(tokenId)
                if normalized.count == 3 { break }
            }
            let top1Correct = normalized.first == expectedTokenId
            let top2Correct = normalized.prefix(2).contains(expectedTokenId)
            let top3Correct = normalized.prefix(3).contains(expectedTokenId)

            evaluatedTokens += 1
            if top1Correct { top1CorrectTokens += 1 }
            if top2Correct { top2CorrectTokens += 1 }
            if top3Correct { top3CorrectTokens += 1 }
            currentBucketEvaluated += 1
            if top1Correct { currentBucketTop1Correct += 1 }
            if top2Correct { currentBucketTop2Correct += 1 }
            if top3Correct { currentBucketTop3Correct += 1 }

            let observation = AccuracyObservation(
                index: evaluatedTokens,
                pieceIndex: pieceIndex,
                sourceTokenIndex: sourceTokenIndex,
                expectedTokenId: expectedTokenId,
                predictedTokenIds: normalized,
                top1Correct: top1Correct,
                top2Correct: top2Correct,
                top3Correct: top3Correct,
                cumulativeTop1Accuracy: Double(top1CorrectTokens) / Double(evaluatedTokens),
                cumulativeTop2Accuracy: Double(top2CorrectTokens) / Double(evaluatedTokens),
                cumulativeTop3Accuracy: Double(top3CorrectTokens) / Double(evaluatedTokens)
            )
            if currentBucketEvaluated == bucketSize {
                closeCurrentBucket(endIndex: evaluatedTokens)
            }
            return observation
        }

        /// Source-compatible top-1 append overload.
        @discardableResult
        public mutating func append(sourceTokenIndex: Int, expectedTokenId: Int,
                                    predictedTokenId: Int,
                                    pieceIndex: Int = 0) -> AccuracyObservation {
            append(sourceTokenIndex: sourceTokenIndex,
                   expectedTokenId: expectedTokenId,
                   predictedTokenIds: [predictedTokenId],
                   pieceIndex: pieceIndex)
        }

        /// Build an immutable snapshot. This is intentionally nonmutating so an
        /// empty `let AccuracyAccumulator` can produce a finite zero result and
        /// repeated UI reads cannot duplicate the final partial bucket.
        public func result(originalTokens: Int, contextTokens: Int,
                           duration: Double, truncated: Bool,
                           pieces: [AccuracyPieceResult] = [],
                           requestedPieceCount: Int = 1,
                           seed: UInt64 = 0,
                           effectiveMinContextTokens: Int? = nil,
                           effectiveMaxContextTokens: Int? = nil) -> AccuracyResult {
            let safeDuration = max(0, duration)
            let top1Accuracy = evaluatedTokens > 0
                ? Double(top1CorrectTokens) / Double(evaluatedTokens) : 0
            let top2Accuracy = evaluatedTokens > 0
                ? Double(top2CorrectTokens) / Double(evaluatedTokens) : 0
            let top3Accuracy = evaluatedTokens > 0
                ? Double(top3CorrectTokens) / Double(evaluatedTokens) : 0
            let tps = safeDuration > 0 && evaluatedTokens > 0
                ? Double(evaluatedTokens) / safeDuration : 0
            var buckets = completedBuckets
            if currentBucketEvaluated > 0 {
                buckets.append(AccuracyBucket(
                    index: buckets.count,
                    startIndex: currentBucketStart,
                    endIndex: evaluatedTokens,
                    top1CorrectTokens: currentBucketTop1Correct,
                    top2CorrectTokens: currentBucketTop2Correct,
                    top3CorrectTokens: currentBucketTop3Correct,
                    evaluatedTokens: currentBucketEvaluated,
                    localTop1Accuracy: Double(currentBucketTop1Correct) / Double(currentBucketEvaluated),
                    localTop2Accuracy: Double(currentBucketTop2Correct) / Double(currentBucketEvaluated),
                    localTop3Accuracy: Double(currentBucketTop3Correct) / Double(currentBucketEvaluated),
                    cumulativeTop1Accuracy: top1Accuracy,
                    cumulativeTop2Accuracy: top2Accuracy,
                    cumulativeTop3Accuracy: top3Accuracy
                ))
            }
            return AccuracyResult(
                originalTokens: originalTokens,
                contextTokens: contextTokens,
                evaluatedTokens: evaluatedTokens,
                top1CorrectTokens: top1CorrectTokens,
                top2CorrectTokens: top2CorrectTokens,
                top3CorrectTokens: top3CorrectTokens,
                top1Accuracy: top1Accuracy,
                top2Accuracy: top2Accuracy,
                top3Accuracy: top3Accuracy,
                duration: safeDuration,
                evaluatedTps: tps,
                truncated: truncated,
                bucketSize: bucketSize,
                buckets: buckets,
                pieces: pieces,
                requestedPieceCount: requestedPieceCount,
                seed: seed,
                effectiveMinContextTokens: effectiveMinContextTokens ?? contextTokens,
                effectiveMaxContextTokens: effectiveMaxContextTokens ?? contextTokens
            )
        }

        private mutating func closeCurrentBucket(endIndex: Int) {
            completedBuckets.append(AccuracyBucket(
                index: completedBuckets.count,
                startIndex: currentBucketStart,
                endIndex: endIndex,
                top1CorrectTokens: currentBucketTop1Correct,
                top2CorrectTokens: currentBucketTop2Correct,
                top3CorrectTokens: currentBucketTop3Correct,
                evaluatedTokens: currentBucketEvaluated,
                localTop1Accuracy: Double(currentBucketTop1Correct) / Double(currentBucketEvaluated),
                localTop2Accuracy: Double(currentBucketTop2Correct) / Double(currentBucketEvaluated),
                localTop3Accuracy: Double(currentBucketTop3Correct) / Double(currentBucketEvaluated),
                cumulativeTop1Accuracy: Double(top1CorrectTokens) / Double(evaluatedTokens),
                cumulativeTop2Accuracy: Double(top2CorrectTokens) / Double(evaluatedTokens),
                cumulativeTop3Accuracy: Double(top3CorrectTokens) / Double(evaluatedTokens)
            ))
            currentBucketStart = endIndex + 1
            currentBucketTop1Correct = 0
            currentBucketTop2Correct = 0
            currentBucketTop3Correct = 0
            currentBucketEvaluated = 0
        }
    }

    /// Exact top-1/top-2/top-3 next-token accuracy over plain text, evaluated
    /// with teacher forcing. `contextTokens` source tokens are consumed but not
    /// scored; up to `maxTokens` following tokens are evaluated. The model
    /// always receives the true source continuation, so one miss never shifts
    /// later comparisons.
    public func accuracyBenchmark(
        text: String,
        contextTokens requestedContextTokens: Int,
        maxTokens requestedMaxTokens: Int,
        bucketSize: Int,
        onObservation: @Sendable (AccuracyObservation) -> Void = { _ in }
    ) throws -> AccuracyResult {
        // The decoder KV is overwritten from position zero. Preserve all logical
        // conversation metadata and force the normal generation path to rebuild
        // that exact committed transcript after every return, error or cancel.
        let usage = decoder.usage
        let usageSnapshot = usage?.serialize()
        defer {
            // `replace(nil)` deliberately clears any routes recorded by this
            // run even if serializing the prior profile failed.
            usage?.replace(with: usageSnapshot)
            kvDirty = true
        }

        let source = tok.tokenize(text).map(Int.init)
        guard source.count >= 2 else {
            throw AccuracyBenchmarkError.insufficientTokens(actual: source.count)
        }
        guard contextSize >= 2 else {
            throw AccuracyBenchmarkError.contextTooSmall(actual: contextSize)
        }

        // One KV position is BOS. Leave room for at least one scored target;
        // the target itself need not be fed, hence end is also the input count.
        let maxContext = min(source.count - 1, contextSize - 1)
        let prefix = min(max(1, requestedContextTokens), maxContext)
        let requested = max(1, requestedMaxTokens)
        let end = min(source.count, contextSize, prefix + requested)
        let inputsEnd = end - 1
        var inputs = [Int(tok.bosId)]
        inputs.append(contentsOf: source.prefix(inputsEnd))
        let scoreRange = prefix..<end

        var accumulator = AccuracyAccumulator(bucketSize: bucketSize)
        decoder.resetProfile()
        let started = Date()
        _ = try decoder.prefillTopK(tokens: inputs, startPos: 0, k: 3,
                                    scoreRange: scoreRange) { range, predictions in
            for (offset, candidates) in predictions.enumerated() {
                let sourceIndex = range.lowerBound + offset
                let observation = accumulator.append(
                    sourceTokenIndex: sourceIndex,
                    expectedTokenId: source[sourceIndex],
                    predictedTokenIds: candidates
                )
                onObservation(observation)
            }
        }
        let duration = Date().timeIntervalSince(started)
        return accumulator.result(
            originalTokens: source.count,
            contextTokens: prefix,
            duration: duration,
            truncated: end < source.count
        )
    }

    /// Deterministic multi-segment next-token evaluation. The corpus is
    /// tokenized once, then every planned segment runs an independent prefill
    /// from position zero. Global accuracy is token-weighted naturally by the
    /// shared accumulator rather than averaging per-piece percentages.
    public func accuracyBenchmark(
        text: String,
        minContextTokens: Int,
        maxContextTokens: Int,
        maxTokensPerPiece: Int,
        pieceCount: Int,
        seed: UInt64,
        bucketSize: Int,
        onObservation: @Sendable (AccuracyObservation) -> Void = { _ in }
    ) throws -> AccuracyResult {
        let usage = decoder.usage
        let usageSnapshot = usage?.serialize()
        defer {
            usage?.replace(with: usageSnapshot)
            kvDirty = true
        }

        try Task.checkCancellation()
        let source = tok.tokenize(text).map(Int.init)
        try Task.checkCancellation()
        guard source.count >= 2 else {
            throw AccuracyBenchmarkError.insufficientTokens(actual: source.count)
        }
        guard contextSize >= 2 else {
            throw AccuracyBenchmarkError.contextTooSmall(actual: contextSize)
        }

        // The pure planner intentionally remains synchronous/nonthrowing for
        // deterministic unit tests. Its sparse sampler is O(pieceCount); the
        // checks after tokenization and planning keep cancellation visible
        // before Metal execution.
        let plan = Self.makeAccuracySamplingPlan(
            sourceTokenCount: source.count,
            contextSize: contextSize,
            minContextTokens: minContextTokens,
            maxContextTokens: maxContextTokens,
            maxTokensPerPiece: maxTokensPerPiece,
            pieceCount: pieceCount,
            seed: seed
        )
        try Task.checkCancellation()
        var accumulator = AccuracyAccumulator(bucketSize: bucketSize)
        var pieceResults: [AccuracyPieceResult] = []
        pieceResults.reserveCapacity(plan.pieces.count)
        decoder.resetProfile()
        let started = Date()

        for piece in plan.pieces {
            try Task.checkCancellation()
            let beforeEvaluated = accumulator.evaluatedTokens
            let beforeTop1 = accumulator.top1CorrectTokens
            let beforeTop2 = accumulator.top2CorrectTokens
            let beforeTop3 = accumulator.top3CorrectTokens

            let sourceInputEnd = piece.targetStartTokenIndex
                + piece.evaluatedTokens - 1
            var inputs = [Int(tok.bosId)]
            inputs.append(contentsOf: source[
                piece.sourceStartTokenIndex..<sourceInputEnd
            ])
            let scoreRange = piece.contextTokens..<(piece.contextTokens + piece.evaluatedTokens)

            _ = try decoder.prefillTopK(
                tokens: inputs,
                startPos: 0,
                k: 3,
                scoreRange: scoreRange
            ) { range, predictions in
                for (offset, candidates) in predictions.enumerated() {
                    let sourceIndex = piece.targetStartTokenIndex
                        + range.lowerBound - piece.contextTokens + offset
                    let observation = accumulator.append(
                        sourceTokenIndex: sourceIndex,
                        expectedTokenId: source[sourceIndex],
                        predictedTokenIds: candidates,
                        pieceIndex: piece.index
                    )
                    onObservation(observation)
                }
            }

            let evaluated = accumulator.evaluatedTokens - beforeEvaluated
            let top1Correct = accumulator.top1CorrectTokens - beforeTop1
            let top2Correct = accumulator.top2CorrectTokens - beforeTop2
            let top3Correct = accumulator.top3CorrectTokens - beforeTop3
            let denominator = Double(max(1, evaluated))
            pieceResults.append(AccuracyPieceResult(
                index: piece.index,
                sourceStartTokenIndex: piece.sourceStartTokenIndex,
                targetStartTokenIndex: piece.targetStartTokenIndex,
                contextTokens: piece.contextTokens,
                evaluatedTokens: evaluated,
                top1CorrectTokens: top1Correct,
                top2CorrectTokens: top2Correct,
                top3CorrectTokens: top3Correct,
                top1Accuracy: evaluated > 0 ? Double(top1Correct) / denominator : 0,
                top2Accuracy: evaluated > 0 ? Double(top2Correct) / denominator : 0,
                top3Accuracy: evaluated > 0 ? Double(top3Correct) / denominator : 0,
                truncated: piece.truncated || evaluated < piece.evaluatedTokens
            ))
        }

        let duration = Date().timeIntervalSince(started)
        return accumulator.result(
            originalTokens: source.count,
            contextTokens: plan.effectiveMaxContextTokens,
            duration: duration,
            truncated: plan.truncated,
            pieces: pieceResults,
            requestedPieceCount: plan.requestedPieceCount,
            seed: plan.seed,
            effectiveMinContextTokens: plan.effectiveMinContextTokens,
            effectiveMaxContextTokens: plan.effectiveMaxContextTokens
        )
    }

    /// Native benchmark (replaces the removed `ds4-bench` binary): prefill a
    /// synthetic prompt of `contextTokens` tokens and decode `genTokens` from it,
    /// returning prefill/generation throughput at that context frontier. Resets
    /// the conversation; `contextTokens` is clamped to fit the loaded context.
    public func benchmark(contextTokens: Int, genTokens: Int,
                          greedy: Bool = false,
                          captureQuality: Bool = false) throws -> BenchPoint {
        resetConversation(systemPrompt: nil)
        let ctx = max(8, min(contextTokens, contextSize - genTokens - 4))
        // Synthetic prompt: BOS + a tiled filler tokenization (output quality is
        // irrelevant for timing; the work — attention, MoE gather — is the same).
        var ids: [Int] = [Int(tok.bosId)]
        let filler = tok.tokenizeRenderedChat("The quick brown fox jumps over the lazy dog. ").map { Int($0) }
        let pad = filler.isEmpty ? [Int(tok.eosId)] : filler
        var i = 0
        while ids.count < ctx { ids.append(pad[i % pad.count]); i += 1 }
        ids = Array(ids.prefix(ctx))

        let t0 = Date()
        var lastLogits = try decoder.prefill(tokens: ids, startPos: 0)
        let prefillDt = Date().timeIntervalSince(t0)

        var pos = ids.count
        var rng: UInt64 = 0xD54
        var produced = 0
        var tokenSpeeds: [Double] = []          // 1/durata di OGNI token generato
        tokenSpeeds.reserveCapacity(genTokens)
        var generatedTokens: [Int] = []
        var capturedLogits: [[Float]] = []
        if captureQuality {
            generatedTokens.reserveCapacity(genTokens)
            capturedLogits.reserveCapacity(genTokens)
        }
        let g0 = Date()
        while produced < genTokens {
            try Task.checkCancellation()
            // Keep trace construction outside the timed forward interval. Array
            // uses copy-on-write, so retaining each immutable logits frame is a
            // bounded snapshot rather than work charged to the kernel timing.
            if captureQuality { capturedLogits.append(lastLogits) }
            let next = Sampler.sample(
                lastLogits,
                temperature: greedy ? 0 : 0.6,
                topK: 0,
                topP: greedy ? 1 : 0.95,
                minP: greedy ? 0 : 0.05,
                rng: &rng
            )
            if captureQuality { generatedTokens.append(next) }
            let t0 = Date()
            lastLogits = try decoder.forward(token: next, pos: pos, nKeys: pos + 1)
            let dt = Date().timeIntervalSince(t0)
            if dt > 0 { tokenSpeeds.append(1.0 / dt) }
            pos += 1; produced += 1
        }
        let genDt = Date().timeIntervalSince(g0)
        kvDirty = true   // synthetic KV state — force a rebuild on the next real turn
        let kv = UInt64(runtimeGeometry.nLayers) * UInt64(ctx) * UInt64(dims.headDim) * 4
        // p99 della velocità per-token: ordina le velocità e prendi il valore
        // al 99° percentile — il regime raggiunto, insensibile al primo token
        // freddo e agli stalli che schiacciano la media.
        var p99 = 0.0
        if !tokenSpeeds.isEmpty {
            let sorted = tokenSpeeds.sorted()
            p99 = sorted[min(sorted.count - 1, Int(Double(sorted.count - 1) * 0.99))]
        }
        let signature: MachineAutoTuneQualitySignature? = captureQuality
            ? MachineAutoTuneQualitySignature(
                generatedTokens: generatedTokens,
                frames: capturedLogits.map(Self.machineAutoTuneQualityFrame)
            )
            : nil
        return BenchPoint(contextTokens: ctx,
                          prefillTps: prefillDt > 0 ? Double(ctx) / prefillDt : 0,
                          genTps: genDt > 0 && produced > 0 ? Double(produced) / genDt : 0,
                          kvBytes: kv,
                          genTpsP99: p99,
                          genSpeeds: tokenSpeeds,
                          qualitySignature: signature)
    }

    /// Small prompt-restoring inference probe used by the machine auto-tuner
    /// after the selected agent and DiskKV are installed. `benchmark` resets
    /// the conversation to a synthetic prompt and deliberately leaves its KV
    /// dirty; the probe restores the active system prompt even when execution
    /// throws. Callers must still mark any non-empty conversation for history
    /// re-prefill because synthetic inference intentionally replaces its KV.
    @discardableResult
    public func machineAutoTuneProbe(
        contextTokens: Int = 96,
        genTokens: Int = 32,
        captureQuality: Bool = false
    ) throws -> BenchPoint {
        let savedSystemPrompt = systemPrompt
        defer { resetConversation(systemPrompt: savedSystemPrompt) }
        return try benchmark(
            contextTokens: contextTokens,
            genTokens: genTokens,
            greedy: true,
            captureQuality: captureQuality
        )
    }

    /// Compact but fail-closed summary of one complete logits frame. The hash
    /// includes every Float32 bit pattern (NaNs included); finiteCount and top-3
    /// make malformed/non-finite output explicit in reports.
    private nonisolated static func machineAutoTuneQualityFrame(
        _ logits: [Float]
    ) -> MachineAutoTuneQualityFrame {
        var hash: UInt64 = 0xcbf29ce484222325
        var finiteCount = 0
        var top: [(token: Int, value: Float)] = []
        top.reserveCapacity(3)

        for (token, value) in logits.enumerated() {
            var bits = value.bitPattern
            for _ in 0..<4 {
                hash ^= UInt64(bits & 0xff)
                hash &*= 0x100000001b3
                bits >>= 8
            }
            guard value.isFinite else { continue }
            finiteCount += 1
            top.append((token, value))
            top.sort {
                $0.value == $1.value ? $0.token < $1.token : $0.value > $1.value
            }
            if top.count > 3 { top.removeLast() }
        }
        return MachineAutoTuneQualityFrame(
            count: logits.count,
            finiteCount: finiteCount,
            bitHash: String(format: "%016llx", hash),
            top3TokenIds: top.map(\.token)
        )
    }

    /// Riscalda il motore subito dopo il load. La slot-cache degli esperti crea
    /// i pool SOLO alla prima richiesta (allocazione Metal + fill dei top-usage:
    /// ~slot × ~7 MB × layer instradati ≈ GB letti da SSD) e i percorsi
    /// Metal/lookahead partono freddi: senza warmup è il PRIMO messaggio
    /// dell'utente a pagare tutto — primi chunk di prefill lenti e 4-7 s sul
    /// primo token. Un mini giro sintetico (12 token di prefill + 3 di decode)
    /// sposta quel costo al load. Idempotente; preserva il system prompt attivo
    /// e lascia lo stato pulito (il benchmark lo sporca di proposito).
    @discardableResult
    public func warmup() -> Bool {
        guard !warmedUp else { return true }
        let saved = systemPrompt
        let t0 = Date()
        do {
            _ = try benchmark(contextTokens: 12, genTokens: 3)
            // Mark the engine warm only after every synthetic token completed.
            // A cancelled auto-tune restoration must be allowed to retry this
            // work instead of publishing a cold engine as already warmed.
            warmedUp = true
            log(String(
                format: "warmup completato in %.1fs (pool esperti + kernel caldi)",
                Date().timeIntervalSince(t0)))
            resetConversation(systemPrompt: saved)
            return true
        } catch {
            // Best-effort: un warmup fallito non deve bloccare nulla — il primo
            // messaggio reale pagherà la partenza fredda come prima.
            warmedUp = false
            log("warmup fallito: \(error)")
            resetConversation(systemPrompt: saved)
            return false
        }
    }
}
