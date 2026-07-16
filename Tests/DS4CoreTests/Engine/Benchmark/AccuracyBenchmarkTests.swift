import XCTest
@testable import DS4Engine

final class AccuracyBenchmarkTests: XCTestCase {
    func testEmptyResultUsesFiniteZeroMetrics() {
        let accumulator = InferenceService.AccuracyAccumulator(bucketSize: 8)

        let result = accumulator.result(
            originalTokens: 1,
            contextTokens: 1,
            duration: 0,
            truncated: false
        )

        XCTAssertEqual(result.originalTokens, 1)
        XCTAssertEqual(result.contextTokens, 1)
        XCTAssertEqual(result.evaluatedTokens, 0)
        XCTAssertEqual(result.correctTokens, 0)
        XCTAssertEqual(result.accuracy, 0)
        XCTAssertEqual(result.evaluatedTps, 0)
        XCTAssertTrue(result.accuracy.isFinite)
        XCTAssertTrue(result.evaluatedTps.isFinite)
        XCTAssertTrue(result.buckets.isEmpty)
    }

    func testAllCorrectProducesOneHundredPercentAccuracy() {
        var accumulator = InferenceService.AccuracyAccumulator(bucketSize: 4)

        for targetIndex in 1...4 {
            let observation = accumulator.append(
                sourceTokenIndex: targetIndex,
                expectedTokenId: 100 + targetIndex,
                predictedTokenId: 100 + targetIndex
            )
            XCTAssertTrue(observation.correct)
            XCTAssertEqual(observation.cumulativeAccuracy, 1, accuracy: 1e-12)
        }

        let result = accumulator.result(
            originalTokens: 5,
            contextTokens: 1,
            duration: 2,
            truncated: false
        )

        XCTAssertEqual(result.evaluatedTokens, 4)
        XCTAssertEqual(result.correctTokens, 4)
        XCTAssertEqual(result.accuracy, 1, accuracy: 1e-12)
        XCTAssertEqual(result.evaluatedTps, 2, accuracy: 1e-12)
        XCTAssertEqual(result.buckets.count, 1)
        XCTAssertEqual(result.buckets[0].localAccuracy, 1, accuracy: 1e-12)
        XCTAssertEqual(result.buckets[0].cumulativeAccuracy, 1, accuracy: 1e-12)
    }

    func testAllWrongProducesZeroPercentAccuracy() {
        var accumulator = InferenceService.AccuracyAccumulator(bucketSize: 3)

        for targetIndex in 1...3 {
            let observation = accumulator.append(
                sourceTokenIndex: targetIndex,
                expectedTokenId: targetIndex,
                predictedTokenId: targetIndex + 100
            )
            XCTAssertFalse(observation.correct)
            XCTAssertEqual(observation.cumulativeAccuracy, 0)
        }

        let result = accumulator.result(
            originalTokens: 4,
            contextTokens: 1,
            duration: 1,
            truncated: false
        )

        XCTAssertEqual(result.evaluatedTokens, 3)
        XCTAssertEqual(result.correctTokens, 0)
        XCTAssertEqual(result.accuracy, 0)
        XCTAssertEqual(result.buckets.count, 1)
        XCTAssertEqual(result.buckets[0].localAccuracy, 0)
        XCTAssertEqual(result.buckets[0].cumulativeAccuracy, 0)
    }

    func testKnownMixTracksLocalAndCumulativeAccuracy() {
        var accumulator = InferenceService.AccuracyAccumulator(bucketSize: 2)
        let correctness = [true, false, true, false, true]
        let expectedCumulative = [1.0, 0.5, 2.0 / 3.0, 0.5, 0.6]

        for (offset, correct) in correctness.enumerated() {
            let targetIndex = offset + 1
            let expectedToken = 10 + targetIndex
            let observation = accumulator.append(
                sourceTokenIndex: targetIndex,
                expectedTokenId: expectedToken,
                predictedTokenId: correct ? expectedToken : expectedToken + 1_000
            )

            XCTAssertEqual(observation.index, targetIndex)
            XCTAssertEqual(observation.sourceTokenIndex, targetIndex)
            XCTAssertEqual(observation.expectedTokenId, expectedToken)
            XCTAssertEqual(observation.correct, correct)
            XCTAssertEqual(
                observation.cumulativeAccuracy,
                expectedCumulative[offset],
                accuracy: 1e-12
            )
        }

        let result = accumulator.result(
            originalTokens: 6,
            contextTokens: 1,
            duration: 2.5,
            truncated: false
        )

        XCTAssertEqual(result.evaluatedTokens, 5)
        XCTAssertEqual(result.correctTokens, 3)
        XCTAssertEqual(result.accuracy, 0.6, accuracy: 1e-12)
        XCTAssertEqual(result.evaluatedTps, 2, accuracy: 1e-12)
        XCTAssertEqual(result.bucketSize, 2)
        XCTAssertEqual(result.buckets.count, 3)

        assertBucket(
            result.buckets[0],
            evaluated: 2,
            correct: 1,
            local: 0.5,
            cumulative: 0.5
        )
        assertBucket(
            result.buckets[1],
            evaluated: 2,
            correct: 1,
            local: 0.5,
            cumulative: 0.5
        )
        assertBucket(
            result.buckets[2],
            evaluated: 1,
            correct: 1,
            local: 1,
            cumulative: 0.6
        )
    }

    func testFinalPartialBucketUsesItsActualDenominator() {
        var accumulator = InferenceService.AccuracyAccumulator(bucketSize: 4)

        for targetIndex in 1...6 {
            accumulator.append(
                sourceTokenIndex: targetIndex,
                expectedTokenId: targetIndex,
                predictedTokenId: targetIndex == 5 ? targetIndex : -1
            )
        }

        let result = accumulator.result(
            originalTokens: 7,
            contextTokens: 1,
            duration: 6,
            truncated: false
        )

        XCTAssertEqual(result.buckets.count, 2)
        XCTAssertEqual(result.buckets[0].evaluatedTokens, 4)
        XCTAssertEqual(result.buckets[1].evaluatedTokens, 2)
        XCTAssertEqual(result.buckets[1].correctTokens, 1)
        XCTAssertEqual(result.buckets[1].localAccuracy, 0.5, accuracy: 1e-12)
        XCTAssertEqual(result.buckets[1].cumulativeAccuracy, 1.0 / 6.0, accuracy: 1e-12)
    }

    func testNCorpusTokensProduceNMinusOneObservations() {
        let corpusTokenCount = 4
        var accumulator = InferenceService.AccuracyAccumulator(bucketSize: 16)

        // Token zero is context. Targets occupy the remaining zero-based source
        // positions 1...(N - 1), so neither the first nor last target is lost.
        for sourceTokenIndex in 1..<corpusTokenCount {
            accumulator.append(
                sourceTokenIndex: sourceTokenIndex,
                expectedTokenId: sourceTokenIndex,
                predictedTokenId: sourceTokenIndex
            )
        }

        let result = accumulator.result(
            originalTokens: corpusTokenCount,
            contextTokens: 1,
            duration: 1,
            truncated: false
        )

        XCTAssertEqual(result.evaluatedTokens, corpusTokenCount - 1)
        XCTAssertEqual(result.correctTokens, corpusTokenCount - 1)
        XCTAssertEqual(result.buckets.first?.startIndex, 1)
        XCTAssertEqual(result.buckets.first?.endIndex, corpusTokenCount - 1)
    }

    func testObservationProgressIsIndependentFromSourceTokenOffset() {
        var accumulator = InferenceService.AccuracyAccumulator(bucketSize: 8)

        let first = accumulator.append(
            sourceTokenIndex: 10,
            expectedTokenId: 42,
            predictedTokenId: 42
        )
        let second = accumulator.append(
            sourceTokenIndex: 11,
            expectedTokenId: 43,
            predictedTokenId: 99
        )
        let result = accumulator.result(
            originalTokens: 20,
            contextTokens: 10,
            duration: 1,
            truncated: true
        )

        XCTAssertEqual(first.index, 1)
        XCTAssertEqual(first.sourceTokenIndex, 10)
        XCTAssertEqual(second.index, 2)
        XCTAssertEqual(second.sourceTokenIndex, 11)
        XCTAssertEqual(result.buckets.first?.startIndex, 1)
        XCTAssertEqual(result.buckets.first?.endIndex, 2)
        XCTAssertTrue(result.truncated)
    }

    func testKnownTopThreeMixIsNestedAtEveryLevel() {
        var accumulator = InferenceService.AccuracyAccumulator(bucketSize: 3)
        let rows: [(expected: Int, candidates: [Int])] = [
            (10, [10, 20, 30]), // top-1 hit
            (20, [10, 20, 30]), // top-2 hit
            (30, [10, 20, 30]), // top-3 hit
            (40, [10, 20, 30]), // miss
            (50, [99, 50, 77]), // top-2 hit
        ]
        let expectedCumulativeTop1 = [1.0, 0.5, 1.0 / 3.0, 0.25, 0.2]
        let expectedCumulativeTop2 = [1.0, 1.0, 2.0 / 3.0, 0.5, 0.6]
        let expectedCumulativeTop3 = [1.0, 1.0, 1.0, 0.75, 0.8]

        for (offset, row) in rows.enumerated() {
            let observation = accumulator.append(
                sourceTokenIndex: offset + 1,
                expectedTokenId: row.expected,
                predictedTokenIds: row.candidates
            )

            if observation.top1Correct {
                XCTAssertTrue(observation.top2Correct)
                XCTAssertTrue(observation.top3Correct)
            }
            if observation.top2Correct {
                XCTAssertTrue(observation.top3Correct)
            }
            XCTAssertLessThanOrEqual(
                observation.cumulativeTop1Accuracy,
                observation.cumulativeTop2Accuracy
            )
            XCTAssertLessThanOrEqual(
                observation.cumulativeTop2Accuracy,
                observation.cumulativeTop3Accuracy
            )
            XCTAssertEqual(
                observation.cumulativeTop1Accuracy,
                expectedCumulativeTop1[offset],
                accuracy: 1e-12
            )
            XCTAssertEqual(
                observation.cumulativeTop2Accuracy,
                expectedCumulativeTop2[offset],
                accuracy: 1e-12
            )
            XCTAssertEqual(
                observation.cumulativeTop3Accuracy,
                expectedCumulativeTop3[offset],
                accuracy: 1e-12
            )
        }

        let result = accumulator.result(
            originalTokens: 6,
            contextTokens: 1,
            duration: 1,
            truncated: false
        )

        XCTAssertEqual(result.top1CorrectTokens, 1)
        XCTAssertEqual(result.top2CorrectTokens, 3)
        XCTAssertEqual(result.top3CorrectTokens, 4)
        XCTAssertEqual(result.top1Accuracy, 0.2, accuracy: 1e-12)
        XCTAssertEqual(result.top2Accuracy, 0.6, accuracy: 1e-12)
        XCTAssertEqual(result.top3Accuracy, 0.8, accuracy: 1e-12)
        XCTAssertLessThanOrEqual(result.top1Accuracy, result.top2Accuracy)
        XCTAssertLessThanOrEqual(result.top2Accuracy, result.top3Accuracy)

        for bucket in result.buckets {
            XCTAssertLessThanOrEqual(bucket.top1CorrectTokens, bucket.top2CorrectTokens)
            XCTAssertLessThanOrEqual(bucket.top2CorrectTokens, bucket.top3CorrectTokens)
            XCTAssertLessThanOrEqual(bucket.localTop1Accuracy, bucket.localTop2Accuracy)
            XCTAssertLessThanOrEqual(bucket.localTop2Accuracy, bucket.localTop3Accuracy)
            XCTAssertLessThanOrEqual(
                bucket.cumulativeTop1Accuracy,
                bucket.cumulativeTop2Accuracy
            )
            XCTAssertLessThanOrEqual(
                bucket.cumulativeTop2Accuracy,
                bucket.cumulativeTop3Accuracy
            )
        }
    }

    func testTopThreeFinalPartialBucketUsesActualDenominator() {
        var accumulator = InferenceService.AccuracyAccumulator(bucketSize: 3)
        let rows: [(expected: Int, candidates: [Int])] = [
            (1, [1, 8, 9]),
            (2, [8, 2, 9]),
            (3, [8, 9, 3]),
            (4, [4, 8, 9]),
            (5, [8, 9, 5]),
        ]

        for (offset, row) in rows.enumerated() {
            accumulator.append(
                sourceTokenIndex: offset + 1,
                expectedTokenId: row.expected,
                predictedTokenIds: row.candidates
            )
        }

        let result = accumulator.result(
            originalTokens: 6,
            contextTokens: 1,
            duration: 1,
            truncated: false
        )

        XCTAssertEqual(result.buckets.count, 2)
        let partial = result.buckets[1]
        XCTAssertEqual(partial.evaluatedTokens, 2)
        XCTAssertEqual(partial.top1CorrectTokens, 1)
        XCTAssertEqual(partial.top2CorrectTokens, 1)
        XCTAssertEqual(partial.top3CorrectTokens, 2)
        XCTAssertEqual(partial.localTop1Accuracy, 0.5, accuracy: 1e-12)
        XCTAssertEqual(partial.localTop2Accuracy, 0.5, accuracy: 1e-12)
        XCTAssertEqual(partial.localTop3Accuracy, 1.0, accuracy: 1e-12)
        XCTAssertEqual(partial.cumulativeTop1Accuracy, 0.4, accuracy: 1e-12)
        XCTAssertEqual(partial.cumulativeTop2Accuracy, 0.6, accuracy: 1e-12)
        XCTAssertEqual(partial.cumulativeTop3Accuracy, 1.0, accuracy: 1e-12)
    }

    func testTopThreeCandidatesAreUniqueOrderedAndLimitedToThree() {
        var accumulator = InferenceService.AccuracyAccumulator(bucketSize: 8)

        let duplicate = accumulator.append(
            sourceTokenIndex: 1,
            expectedTokenId: 7,
            predictedTokenIds: [1, 1, 7, 7, 9, 10]
        )
        XCTAssertEqual(duplicate.predictedTokenIds, [1, 7, 9])
        XCTAssertEqual(duplicate.predictedTokenId, 1)
        XCTAssertFalse(duplicate.top1Correct)
        XCTAssertTrue(duplicate.top2Correct)
        XCTAssertTrue(duplicate.top3Correct)

        let beyondTopThree = accumulator.append(
            sourceTokenIndex: 2,
            expectedTokenId: 4,
            predictedTokenIds: [1, 2, 3, 4]
        )
        XCTAssertEqual(beyondTopThree.predictedTokenIds, [1, 2, 3])
        XCTAssertFalse(beyondTopThree.top3Correct)

        let short = accumulator.append(
            sourceTokenIndex: 3,
            expectedTokenId: 8,
            predictedTokenIds: [8]
        )
        XCTAssertEqual(short.predictedTokenIds, [8])
        XCTAssertTrue(short.top1Correct)
        XCTAssertTrue(short.top2Correct)
        XCTAssertTrue(short.top3Correct)

        let empty = accumulator.append(
            sourceTokenIndex: 4,
            expectedTokenId: 8,
            predictedTokenIds: []
        )
        XCTAssertTrue(empty.predictedTokenIds.isEmpty)
        XCTAssertEqual(empty.predictedTokenId, -1)
        XCTAssertFalse(empty.top1Correct)
        XCTAssertFalse(empty.top2Correct)
        XCTAssertFalse(empty.top3Correct)
    }

    func testLegacyTopOneAPIIsAnExactAlias() throws {
        var accumulator = InferenceService.AccuracyAccumulator(bucketSize: 1)
        let observation = accumulator.append(
            sourceTokenIndex: 1,
            expectedTokenId: 42,
            predictedTokenId: 42
        )
        let result = accumulator.result(
            originalTokens: 2,
            contextTokens: 1,
            duration: 1,
            truncated: false
        )
        let bucket = try XCTUnwrap(result.buckets.first)

        XCTAssertEqual(observation.predictedTokenIds, [42])
        XCTAssertEqual(observation.predictedTokenId, 42)
        XCTAssertEqual(observation.correct, observation.top1Correct)
        XCTAssertEqual(observation.cumulativeAccuracy, observation.cumulativeTop1Accuracy)
        XCTAssertEqual(result.correctTokens, result.top1CorrectTokens)
        XCTAssertEqual(result.accuracy, result.top1Accuracy)
        XCTAssertEqual(bucket.correctTokens, bucket.top1CorrectTokens)
        XCTAssertEqual(bucket.localAccuracy, bucket.localTop1Accuracy)
        XCTAssertEqual(bucket.cumulativeAccuracy, bucket.cumulativeTop1Accuracy)
    }

    func testSamplingPlanIsReproducibleForTheSameSeed() {
        let first = InferenceService.makeAccuracySamplingPlan(
            sourceTokenCount: 1_000,
            contextSize: 256,
            minContextTokens: 16,
            maxContextTokens: 80,
            maxTokensPerPiece: 32,
            pieceCount: 20,
            seed: 0xD54
        )
        let second = InferenceService.makeAccuracySamplingPlan(
            sourceTokenCount: 1_000,
            contextSize: 256,
            minContextTokens: 16,
            maxContextTokens: 80,
            maxTokensPerPiece: 32,
            pieceCount: 20,
            seed: 0xD54
        )

        assertEquivalentPlans(first, second)
    }

    func testSamplingPlanChangesWithSeed() {
        let first = InferenceService.makeAccuracySamplingPlan(
            sourceTokenCount: 1_000,
            contextSize: 256,
            minContextTokens: 16,
            maxContextTokens: 80,
            maxTokensPerPiece: 32,
            pieceCount: 20,
            seed: 1
        )
        let second = InferenceService.makeAccuracySamplingPlan(
            sourceTokenCount: 1_000,
            contextSize: 256,
            minContextTokens: 16,
            maxContextTokens: 80,
            maxTokensPerPiece: 32,
            pieceCount: 20,
            seed: 2
        )

        XCTAssertNotEqual(
            first.pieces.map(\.targetStartTokenIndex),
            second.pieces.map(\.targetStartTokenIndex)
        )
    }

    func testSamplingPiecesRespectDistinctStartsAndAllBounds() {
        let sourceTokenCount = 500
        let contextSize = 128
        let maxTokensPerPiece = 24
        let plan = InferenceService.makeAccuracySamplingPlan(
            sourceTokenCount: sourceTokenCount,
            contextSize: contextSize,
            minContextTokens: 8,
            maxContextTokens: 64,
            maxTokensPerPiece: maxTokensPerPiece,
            pieceCount: 40,
            seed: 0xC0FFEE
        )

        XCTAssertEqual(plan.pieces.count, 40)
        XCTAssertEqual(Set(plan.pieces.map(\.targetStartTokenIndex)).count,
                       plan.pieces.count)
        for (offset, piece) in plan.pieces.enumerated() {
            XCTAssertEqual(piece.index, offset)
            XCTAssertGreaterThanOrEqual(piece.targetStartTokenIndex,
                                        plan.effectiveMinContextTokens)
            XCTAssertGreaterThanOrEqual(piece.contextTokens,
                                        plan.effectiveMinContextTokens)
            XCTAssertLessThanOrEqual(piece.contextTokens,
                                     plan.effectiveMaxContextTokens)
            XCTAssertLessThanOrEqual(piece.contextTokens,
                                     piece.targetStartTokenIndex)
            XCTAssertEqual(piece.sourceStartTokenIndex,
                           piece.targetStartTokenIndex - piece.contextTokens)
            XCTAssertGreaterThanOrEqual(piece.sourceStartTokenIndex, 0)
            XCTAssertGreaterThanOrEqual(piece.evaluatedTokens, 1)
            XCTAssertLessThanOrEqual(piece.evaluatedTokens, maxTokensPerPiece)
            XCTAssertLessThanOrEqual(piece.targetStartTokenIndex + piece.evaluatedTokens,
                                     sourceTokenCount)
            XCTAssertLessThanOrEqual(piece.contextTokens + piece.evaluatedTokens,
                                     contextSize)
        }
    }

    func testSamplingPrefersFullPiecesBeforeUsingShortTail() {
        let fullOnly = InferenceService.makeAccuracySamplingPlan(
            sourceTokenCount: 100,
            contextSize: 64,
            minContextTokens: 4,
            maxContextTokens: 16,
            maxTokensPerPiece: 10,
            pieceCount: 20,
            seed: 77
        )

        XCTAssertEqual(fullOnly.pieces.count, 20)
        XCTAssertTrue(fullOnly.pieces.allSatisfy { $0.evaluatedTokens == 10 })
        XCTAssertTrue(fullOnly.pieces.allSatisfy { !$0.truncated })

        // Targets 3...8 can provide all four requested evaluations. Asking for
        // eight distinct starts requires two additional targets from 9...11.
        let withTail = InferenceService.makeAccuracySamplingPlan(
            sourceTokenCount: 12,
            contextSize: 16,
            minContextTokens: 3,
            maxContextTokens: 3,
            maxTokensPerPiece: 4,
            pieceCount: 8,
            seed: 77
        )

        XCTAssertEqual(withTail.pieces.count, 8)
        XCTAssertTrue(withTail.pieces.prefix(6).allSatisfy { $0.evaluatedTokens == 4 })
        XCTAssertTrue(withTail.pieces.suffix(2).allSatisfy {
            $0.evaluatedTokens >= 1 && $0.evaluatedTokens < 4 && $0.truncated
        })
        XCTAssertTrue(withTail.truncated)
    }

    func testSamplingPlanClampsCorpusKVAndExcessivePieceCount() {
        let plan = InferenceService.makeAccuracySamplingPlan(
            sourceTokenCount: 6,
            contextSize: 4,
            minContextTokens: 100,
            maxContextTokens: 200,
            maxTokensPerPiece: 100,
            pieceCount: 99,
            seed: 9
        )

        XCTAssertEqual(plan.requestedPieceCount, 99)
        XCTAssertEqual(plan.effectiveMinContextTokens, 3)
        XCTAssertEqual(plan.effectiveMaxContextTokens, 3)
        XCTAssertEqual(plan.pieces.count, 3)
        XCTAssertEqual(Set(plan.pieces.map(\.targetStartTokenIndex)), Set([3, 4, 5]))
        XCTAssertTrue(plan.pieces.allSatisfy { $0.contextTokens == 3 })
        XCTAssertTrue(plan.pieces.allSatisfy { $0.evaluatedTokens == 1 })
        XCTAssertTrue(plan.pieces.allSatisfy { $0.truncated })
        XCTAssertTrue(plan.truncated)
    }

    func testSamplingPlanNormalizesInvertedContextRange() {
        let plan = InferenceService.makeAccuracySamplingPlan(
            sourceTokenCount: 100,
            contextSize: 64,
            minContextTokens: 40,
            maxContextTokens: 10,
            maxTokensPerPiece: 5,
            pieceCount: 10,
            seed: 42
        )

        // The lower bound is authoritative: an inverted upper bound is raised
        // to it after both are clamped to corpus/KV capacity.
        XCTAssertEqual(plan.effectiveMinContextTokens, 40)
        XCTAssertEqual(plan.effectiveMaxContextTokens, 40)
        XCTAssertTrue(plan.pieces.allSatisfy { $0.contextTokens == 40 })
        XCTAssertTrue(plan.pieces.allSatisfy { $0.targetStartTokenIndex >= 40 })
        XCTAssertTrue(plan.truncated)
    }

    func testSamplingPlanNormalizesNonPositivePieceCountToOne() {
        for pieceCount in [0, -7] {
            let plan = InferenceService.makeAccuracySamplingPlan(
                sourceTokenCount: 100,
                contextSize: 64,
                minContextTokens: 4,
                maxContextTokens: 16,
                maxTokensPerPiece: 8,
                pieceCount: pieceCount,
                seed: 123
            )

            XCTAssertEqual(plan.requestedPieceCount, 1)
            XCTAssertEqual(plan.pieces.count, 1)
            XCTAssertGreaterThanOrEqual(plan.pieces[0].evaluatedTokens, 1)
            XCTAssertTrue(plan.truncated)
        }
    }

    func testSparseSamplingHandlesHugeVirtualCorpusAndKeepsPrefixStable() {
        let smaller = InferenceService.makeAccuracySamplingPlan(
            sourceTokenCount: Int.max,
            contextSize: 512,
            minContextTokens: 16,
            maxContextTokens: 128,
            maxTokensPerPiece: 64,
            pieceCount: 8,
            seed: 0xD54
        )
        let larger = InferenceService.makeAccuracySamplingPlan(
            sourceTokenCount: Int.max,
            contextSize: 512,
            minContextTokens: 16,
            maxContextTokens: 128,
            maxTokensPerPiece: 64,
            pieceCount: 32,
            seed: 0xD54
        )

        XCTAssertEqual(Array(larger.pieces.prefix(smaller.pieces.count)),
                       smaller.pieces)
        XCTAssertEqual(Set(larger.pieces.map(\.targetStartTokenIndex)).count,
                       larger.pieces.count)
    }

    private func assertEquivalentPlans(
        _ lhs: InferenceService.AccuracySamplingPlan,
        _ rhs: InferenceService.AccuracySamplingPlan,
        file: StaticString = #filePath,
        line: UInt = #line
    ) {
        XCTAssertEqual(lhs.requestedPieceCount, rhs.requestedPieceCount,
                       file: file, line: line)
        XCTAssertEqual(lhs.seed, rhs.seed, file: file, line: line)
        XCTAssertEqual(lhs.effectiveMinContextTokens, rhs.effectiveMinContextTokens,
                       file: file, line: line)
        XCTAssertEqual(lhs.effectiveMaxContextTokens, rhs.effectiveMaxContextTokens,
                       file: file, line: line)
        XCTAssertEqual(lhs.truncated, rhs.truncated, file: file, line: line)
        XCTAssertEqual(lhs.pieces.count, rhs.pieces.count, file: file, line: line)
        for (left, right) in zip(lhs.pieces, rhs.pieces) {
            XCTAssertEqual(left.index, right.index, file: file, line: line)
            XCTAssertEqual(left.sourceStartTokenIndex, right.sourceStartTokenIndex,
                           file: file, line: line)
            XCTAssertEqual(left.targetStartTokenIndex, right.targetStartTokenIndex,
                           file: file, line: line)
            XCTAssertEqual(left.contextTokens, right.contextTokens,
                           file: file, line: line)
            XCTAssertEqual(left.evaluatedTokens, right.evaluatedTokens,
                           file: file, line: line)
            XCTAssertEqual(left.truncated, right.truncated, file: file, line: line)
        }
    }

    private func assertBucket(
        _ bucket: InferenceService.AccuracyBucket,
        evaluated: Int,
        correct: Int,
        local: Double,
        cumulative: Double,
        file: StaticString = #filePath,
        line: UInt = #line
    ) {
        XCTAssertEqual(bucket.evaluatedTokens, evaluated, file: file, line: line)
        XCTAssertEqual(bucket.correctTokens, correct, file: file, line: line)
        XCTAssertEqual(bucket.localAccuracy, local, accuracy: 1e-12, file: file, line: line)
        XCTAssertEqual(
            bucket.cumulativeAccuracy,
            cumulative,
            accuracy: 1e-12,
            file: file,
            line: line
        )
    }
}
