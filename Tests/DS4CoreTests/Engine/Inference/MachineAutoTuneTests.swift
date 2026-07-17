import XCTest
@testable import DS4Engine

final class MachineAutoTuneTests: XCTestCase {
    private func signature(
        hash: String = "0123456789abcdef",
        finiteCount: Int = 128
    ) -> MachineAutoTuneQualitySignature {
        MachineAutoTuneQualitySignature(
            generatedTokens: [7, 8],
            frames: [
                MachineAutoTuneQualityFrame(
                    count: 128,
                    finiteCount: finiteCount,
                    bitHash: hash,
                    top3TokenIds: [7, 8, 9]
                ),
                MachineAutoTuneQualityFrame(
                    count: 128,
                    finiteCount: finiteCount,
                    bitHash: hash,
                    top3TokenIds: [8, 7, 9]
                ),
            ]
        )
    }

    private func observation(
        primary: Double = 100,
        secondary: Double = 50,
        stability: Double = 0.9,
        freeMemory: Double = 20,
        swapout: Double = 0,
        quality: MachineAutoTuneQualitySignature? = nil
    ) -> MachineAutoTuneObservation {
        MachineAutoTuneObservation(
            primaryTps: primary,
            secondaryTps: secondary,
            stability: stability,
            memoryFreePercent: freeMemory,
            swapoutMiB: swapout,
            quality: quality ?? signature()
        )
    }

    func testConfigurationValueSettingHashAndCodableRoundTrip() throws {
        let original = MachineAutoTuneConfiguration()
            .setting(22, for: .expertCacheSlots)
            .setting(4, for: .preadSplit)

        XCTAssertEqual(original.value(for: .expertCacheSlots), 22)
        XCTAssertEqual(original.value(for: .preadSplit), 4)
        XCTAssertNil(original.value(for: .asyncFFN))
        XCTAssertEqual(original.setting(4, for: .q8NSG).value(for: .q8NSG), 4)

        let changed = original.setting(24, for: .expertCacheSlots)
        XCTAssertEqual(original.value(for: .expertCacheSlots), 22)
        XCTAssertEqual(changed.value(for: .expertCacheSlots), 24)
        XCTAssertNotEqual(original, changed)

        let decoded = try JSONDecoder().decode(
            MachineAutoTuneConfiguration.self,
            from: JSONEncoder().encode(original)
        )
        XCTAssertEqual(decoded, original)
        XCTAssertEqual(Set([decoded, original]).count, 1)
    }

    func testQualityFrameAndSignaturePublicInitializersAreCodable() throws {
        let original = signature()
        let decoded = try JSONDecoder().decode(
            MachineAutoTuneQualitySignature.self,
            from: JSONEncoder().encode(original)
        )
        XCTAssertEqual(decoded, original)
        XCTAssertEqual(decoded.frames.count, decoded.generatedTokens.count)
        XCTAssertEqual(decoded.frames[0].count, 128)
        XCTAssertEqual(decoded.frames[0].finiteCount, 128)
        XCTAssertEqual(decoded.frames[0].bitHash, "0123456789abcdef")
        XCTAssertEqual(decoded.frames[0].top3TokenIds, [7, 8, 9])
        XCTAssertEqual(decoded.frames[1].top3TokenIds, [8, 7, 9])
    }

    func testScreeningAdmitsCandidateInsideRegressionBudget() {
        let root = signature()
        let result = MachineAutoTuneEvaluator.screening(
            incumbent: observation(primary: 100, quality: root),
            candidate: observation(primary: 98, quality: root),
            rootQuality: root
        )

        XCTAssertTrue(result.qualified)
        XCTAssertTrue(result.reasons.isEmpty)
        XCTAssertEqual(result.primaryRatios, [0.98])
        XCTAssertEqual(result.balancedPrimaryRatio, 0.98)
        XCTAssertTrue(result.transitionQualityExact)
        XCTAssertTrue(result.cumulativeQualityExact)
    }

    func testScreeningRejectsClearlySlowerCandidate() {
        let root = signature()
        let result = MachineAutoTuneEvaluator.screening(
            incumbent: observation(primary: 100, quality: root),
            candidate: observation(primary: 96, quality: root),
            rootQuality: root
        )

        XCTAssertFalse(result.qualified)
        XCTAssertTrue(result.reasons.contains { $0.contains("primary screening ratio") })
    }

    func testBaselineRepeatabilityIsSymmetric() {
        let root = signature()
        let stable = MachineAutoTuneEvaluator.repeatability(
            first: observation(primary: 100, secondary: 50, quality: root),
            second: observation(primary: 102, secondary: 52, quality: root),
            rootQuality: root
        )
        XCTAssertTrue(stable.qualified, stable.reasons.joined(separator: "; "))
        XCTAssertTrue(stable.forward.qualified)
        XCTAssertTrue(stable.reverse.qualified)

        let stillWarming = MachineAutoTuneEvaluator.repeatability(
            first: observation(primary: 100, secondary: 50, quality: root),
            second: observation(primary: 110, secondary: 60, quality: root),
            rootQuality: root
        )
        XCTAssertTrue(stillWarming.forward.qualified)
        XCTAssertFalse(stillWarming.reverse.qualified)
        XCTAssertFalse(stillWarming.qualified)
        XCTAssertTrue(stillWarming.reasons.contains {
            $0.contains("reverse: primary screening ratio")
        })
    }

    func testBaselineRepeatabilityKeepsPerSampleSwapAndImmutableQualityGates() {
        let root = signature()
        let drift = signature(hash: "fedcba9876543210")
        let result = MachineAutoTuneEvaluator.repeatability(
            first: observation(swapout: 129, quality: root),
            second: observation(primary: 101, quality: drift),
            rootQuality: root
        )

        XCTAssertFalse(result.qualified)
        XCTAssertTrue(result.reasons.contains { $0.contains("swapout") })
        XCTAssertTrue(result.reasons.contains { $0.contains("immutable root") })
    }

    func testPublicQualityValidationUsesFailClosedStructuralChecks() {
        XCTAssertTrue(
            MachineAutoTuneEvaluator.qualityValidationReasons(signature()).isEmpty
        )
        let malformed = signature(finiteCount: 127)
        XCTAssertTrue(
            MachineAutoTuneEvaluator.qualityValidationReasons(
                malformed,
                label: "root"
            ).contains { $0.contains("non-finite logits") }
        )
    }

    func testBestValidObservationSelectsHighestDecode() {
        let root = signature()
        let observations = [
            observation(primary: 3.339, secondary: 4.383, quality: root),
            observation(primary: 3.424, secondary: 6.004, quality: root),
            observation(primary: 3.462, secondary: 6.626, quality: root),
        ]

        let best = MachineAutoTuneEvaluator.bestValidObservation(
            in: observations,
            rootQuality: root
        )

        XCTAssertEqual(best, observations[2])
    }

    func testBestValidObservationRejectsFasterInvalidSamples() {
        let root = signature()
        let drift = signature(hash: "fedcba9876543210")
        let valid = observation(primary: 3.462, secondary: 6.626, quality: root)
        let observations = [
            valid,
            observation(primary: 9, swapout: 129, quality: root),
            observation(primary: 8, freeMemory: 9, quality: root),
            observation(primary: 7, stability: 0.74, quality: root),
            observation(primary: 6, quality: drift),
        ]

        let best = MachineAutoTuneEvaluator.bestValidObservation(
            in: observations,
            rootQuality: root
        )

        XCTAssertEqual(best, valid)
    }

    func testBestValidObservationIsOrderIndependent() {
        let root = signature()
        let observations = [
            observation(primary: 100, secondary: 70, quality: root),
            observation(primary: 105, secondary: 50, quality: root),
            observation(primary: 103, secondary: 80, quality: root),
        ]

        let forward = MachineAutoTuneEvaluator.bestValidObservation(
            in: observations,
            rootQuality: root
        )
        let reverse = MachineAutoTuneEvaluator.bestValidObservation(
            in: Array(observations.reversed()),
            rootQuality: root
        )

        XCTAssertEqual(forward, observations[1])
        XCTAssertEqual(reverse, forward)
    }

    func testBestValidObservationUsesDeterministicTieBreaks() {
        let root = signature()

        let secondaryWinner = observation(
            primary: 100, secondary: 51, stability: 0.80,
            freeMemory: 11, swapout: 10, quality: root
        )
        XCTAssertEqual(
            MachineAutoTuneEvaluator.bestValidObservation(
                in: [
                    observation(primary: 100, secondary: 50, quality: root),
                    secondaryWinner,
                ],
                rootQuality: root
            ),
            secondaryWinner
        )

        let stabilityWinner = observation(
            primary: 100, secondary: 50, stability: 0.91,
            freeMemory: 11, swapout: 10, quality: root
        )
        XCTAssertEqual(
            MachineAutoTuneEvaluator.bestValidObservation(
                in: [
                    observation(
                        primary: 100, secondary: 50, stability: 0.90,
                        freeMemory: 20, swapout: 0, quality: root
                    ),
                    stabilityWinner,
                ],
                rootQuality: root
            ),
            stabilityWinner
        )

        let memoryWinner = observation(
            primary: 100, secondary: 50, stability: 0.90,
            freeMemory: 21, swapout: 20, quality: root
        )
        XCTAssertEqual(
            MachineAutoTuneEvaluator.bestValidObservation(
                in: [
                    observation(
                        primary: 100, secondary: 50, stability: 0.90,
                        freeMemory: 20, swapout: 0, quality: root
                    ),
                    memoryWinner,
                ],
                rootQuality: root
            ),
            memoryWinner
        )

        let lowerSwapWinner = observation(
            primary: 100, secondary: 50, stability: 0.90,
            freeMemory: 20, swapout: 1, quality: root
        )
        XCTAssertEqual(
            MachineAutoTuneEvaluator.bestValidObservation(
                in: [
                    observation(
                        primary: 100, secondary: 50, stability: 0.90,
                        freeMemory: 20, swapout: 2, quality: root
                    ),
                    lowerSwapWinner,
                ],
                rootQuality: root
            ),
            lowerSwapWinner
        )
    }

    func testBestValidObservationReturnsNilWithoutEligibleSamples() {
        let root = signature()

        XCTAssertNil(
            MachineAutoTuneEvaluator.bestValidObservation(
                in: [],
                rootQuality: root
            )
        )
        XCTAssertNil(
            MachineAutoTuneEvaluator.bestValidObservation(
                in: [
                    observation(primary: 200, swapout: 129, quality: root),
                    observation(primary: 300, freeMemory: 9, quality: root),
                ],
                rootQuality: root
            )
        )
    }

    func testHighWaterComparisonRequiresStrictlyHigherDecode() {
        let root = signature()
        let incumbent = observation(primary: 100, secondary: 50, quality: root)
        let lower = MachineAutoTuneEvaluator.highWaterComparison(
            incumbent: incumbent,
            candidate: observation(primary: 99.9999, secondary: 50, quality: root),
            rootQuality: root
        )
        let equal = MachineAutoTuneEvaluator.highWaterComparison(
            incumbent: incumbent,
            candidate: observation(primary: 100, secondary: 50, quality: root),
            rootQuality: root
        )
        let higher = MachineAutoTuneEvaluator.highWaterComparison(
            incumbent: incumbent,
            candidate: observation(primary: 100.0001, secondary: 50, quality: root),
            rootQuality: root
        )

        XCTAssertFalse(lower.qualified)
        XCTAssertFalse(equal.qualified)
        XCTAssertTrue(higher.qualified, higher.reasons.joined(separator: "; "))
        XCTAssertGreaterThan(higher.balancedPrimaryRatio!, 1)
    }

    func testHighWaterComparisonPrefillRegressionBoundary() {
        let root = signature()
        let incumbent = observation(primary: 100, secondary: 100, quality: root)
        let inside = MachineAutoTuneEvaluator.highWaterComparison(
            incumbent: incumbent,
            candidate: observation(primary: 101, secondary: 92.1, quality: root),
            rootQuality: root
        )
        let boundary = MachineAutoTuneEvaluator.highWaterComparison(
            incumbent: incumbent,
            candidate: observation(primary: 101, secondary: 92, quality: root),
            rootQuality: root
        )
        let outside = MachineAutoTuneEvaluator.highWaterComparison(
            incumbent: incumbent,
            candidate: observation(primary: 101, secondary: 91.9, quality: root),
            rootQuality: root
        )

        XCTAssertTrue(inside.qualified, inside.reasons.joined(separator: "; "))
        XCTAssertTrue(boundary.qualified, boundary.reasons.joined(separator: "; "))
        XCTAssertFalse(outside.qualified)
    }

    func testHighWaterComparisonEnforcesQualityAndResourceGates() {
        let root = signature()
        let drift = signature(hash: "fedcba9876543210")
        let incumbent = observation(primary: 100, secondary: 50, quality: root)
        let invalidCandidates: [(String, MachineAutoTuneObservation)] = [
            ("immutable quality", observation(primary: 103, quality: drift)),
            ("prefill", observation(primary: 103, secondary: 45, quality: root)),
            ("RAM", observation(primary: 103, freeMemory: 9, quality: root)),
            ("stability", observation(primary: 103, stability: 0.74, quality: root)),
            ("swap", observation(primary: 103, swapout: 129, quality: root)),
        ]

        for (label, candidate) in invalidCandidates {
            let result = MachineAutoTuneEvaluator.highWaterComparison(
                incumbent: incumbent,
                candidate: candidate,
                rootQuality: root
            )
            XCTAssertFalse(
                result.qualified,
                "Expected the high-water comparison to reject \(label)"
            )
        }
    }

    func testABBAUsesGeometricMeanAndPromotesQualifiedCandidate() {
        let root = signature()
        let result = MachineAutoTuneEvaluator.abba(
            a1: observation(primary: 100, secondary: 50, quality: root),
            b1: observation(primary: 104, secondary: 51, quality: root),
            b2: observation(primary: 103, secondary: 50.5, quality: root),
            a2: observation(primary: 100, secondary: 50, quality: root),
            rootQuality: root
        )

        XCTAssertTrue(result.qualified, result.reasons.joined(separator: "; "))
        XCTAssertEqual(result.primaryRatios[0], 1.04, accuracy: 1e-12)
        XCTAssertEqual(result.primaryRatios[1], 1.03, accuracy: 1e-12)
        XCTAssertEqual(result.balancedPrimaryRatio!, (1.04 * 1.03).squareRoot(), accuracy: 1e-12)
        XCTAssertEqual(result.balancedSecondaryRatio!, (1.02 * 1.01).squareRoot(), accuracy: 1e-12)
    }

    func testABBARequiresMinimumBalancedGain() {
        let root = signature()
        let result = MachineAutoTuneEvaluator.abba(
            a1: observation(primary: 100, quality: root),
            b1: observation(primary: 101, quality: root),
            b2: observation(primary: 101, quality: root),
            a2: observation(primary: 100, quality: root),
            rootQuality: root
        )

        XCTAssertFalse(result.qualified)
        XCTAssertTrue(result.reasons.contains { $0.contains("balanced primary ratio") })
    }

    func testABBARejectsOneBadPairEvenWhenGeometricMeanIsFast() {
        let root = signature()
        let result = MachineAutoTuneEvaluator.abba(
            a1: observation(primary: 100, quality: root),
            b1: observation(primary: 96, quality: root),
            b2: observation(primary: 120, quality: root),
            a2: observation(primary: 100, quality: root),
            rootQuality: root
        )

        XCTAssertGreaterThan(result.balancedPrimaryRatio!, 1.02)
        XCTAssertFalse(result.qualified)
        XCTAssertTrue(result.reasons.contains { $0.contains("primary pair 1") })
    }

    func testABBARejectsSecondaryRegression() {
        let root = signature()
        let result = MachineAutoTuneEvaluator.abba(
            a1: observation(primary: 100, secondary: 50, quality: root),
            b1: observation(primary: 104, secondary: 45, quality: root),
            b2: observation(primary: 104, secondary: 45, quality: root),
            a2: observation(primary: 100, secondary: 50, quality: root),
            rootQuality: root
        )

        XCTAssertFalse(result.qualified)
        XCTAssertTrue(result.reasons.contains { $0.contains("balanced secondary ratio") })
    }

    func testExactQualityRequiresTransitionAndImmutableRoot() {
        let root = signature()
        let drift = signature(hash: "fedcba9876543210")

        let transitionFailure = MachineAutoTuneEvaluator.abba(
            a1: observation(primary: 100, quality: root),
            b1: observation(primary: 104, quality: drift),
            b2: observation(primary: 104, quality: root),
            a2: observation(primary: 100, quality: root),
            rootQuality: root
        )
        XCTAssertFalse(transitionFailure.transitionQualityExact)
        XCTAssertFalse(transitionFailure.cumulativeQualityExact)
        XCTAssertFalse(transitionFailure.qualified)

        let cumulativeFailure = MachineAutoTuneEvaluator.screening(
            incumbent: observation(primary: 100, quality: drift),
            candidate: observation(primary: 104, quality: drift),
            rootQuality: root
        )
        XCTAssertTrue(cumulativeFailure.transitionQualityExact)
        XCTAssertFalse(cumulativeFailure.cumulativeQualityExact)
        XCTAssertFalse(cumulativeFailure.qualified)
    }

    func testMalformedOrNonFiniteQualityIsRejected() {
        let malformed = signature(finiteCount: 127)
        let result = MachineAutoTuneEvaluator.screening(
            incumbent: observation(quality: malformed),
            candidate: observation(primary: 104, quality: malformed),
            rootQuality: malformed
        )

        XCTAssertFalse(result.qualified)
        XCTAssertTrue(result.reasons.contains { $0.contains("malformed") })
    }

    func testQualityRejectsFrameTokenCountMismatch() {
        let valid = signature()
        let mismatched = MachineAutoTuneQualitySignature(
            generatedTokens: valid.generatedTokens,
            frames: [valid.frames[0]]
        )
        let result = MachineAutoTuneEvaluator.screening(
            incumbent: observation(quality: mismatched),
            candidate: observation(primary: 104, quality: mismatched),
            rootQuality: mismatched
        )

        XCTAssertFalse(result.qualified)
        XCTAssertTrue(result.reasons.contains { $0.contains("frame/token count mismatch") })
    }

    func testQualityRejectsGeneratedTokenThatIsNotGreedyTopOne() {
        let valid = signature()
        let nonGreedy = MachineAutoTuneQualitySignature(
            generatedTokens: [9, 8],
            frames: valid.frames
        )
        let result = MachineAutoTuneEvaluator.screening(
            incumbent: observation(quality: nonGreedy),
            candidate: observation(primary: 104, quality: nonGreedy),
            rootQuality: nonGreedy
        )

        XCTAssertFalse(result.qualified)
        XCTAssertTrue(result.reasons.contains { $0.contains("greedy top-1") })
    }

    func testQualityRejectsInconsistentOrZeroVocabularyCounts() {
        let valid = signature()
        let inconsistent = MachineAutoTuneQualitySignature(
            generatedTokens: valid.generatedTokens,
            frames: [
                valid.frames[0],
                MachineAutoTuneQualityFrame(
                    count: 127,
                    finiteCount: 127,
                    bitHash: "0123456789abcdef",
                    top3TokenIds: [8, 7, 9]
                ),
            ]
        )
        let inconsistentResult = MachineAutoTuneEvaluator.screening(
            incumbent: observation(quality: inconsistent),
            candidate: observation(primary: 104, quality: inconsistent),
            rootQuality: inconsistent
        )
        XCTAssertFalse(inconsistentResult.qualified)
        XCTAssertTrue(inconsistentResult.reasons.contains { $0.contains("vocabulary count") })

        let zeroVocabulary = MachineAutoTuneQualitySignature(
            generatedTokens: [0],
            frames: [
                MachineAutoTuneQualityFrame(
                    count: 0,
                    finiteCount: 0,
                    bitHash: "0123456789abcdef",
                    top3TokenIds: [0, 1, 2]
                ),
            ]
        )
        let zeroResult = MachineAutoTuneEvaluator.screening(
            incumbent: observation(quality: zeroVocabulary),
            candidate: observation(primary: 104, quality: zeroVocabulary),
            rootQuality: zeroVocabulary
        )
        XCTAssertFalse(zeroResult.qualified)
        XCTAssertTrue(zeroResult.reasons.contains { $0.contains("non-positive vocabulary") })
    }

    func testNaNAndInfinityMetricsAreRejected() {
        let root = signature()
        let nanResult = MachineAutoTuneEvaluator.screening(
            incumbent: observation(quality: root),
            candidate: observation(primary: .nan, quality: root),
            rootQuality: root
        )
        XCTAssertFalse(nanResult.qualified)
        XCTAssertNil(nanResult.balancedPrimaryRatio)

        let infiniteResult = MachineAutoTuneEvaluator.screening(
            incumbent: observation(quality: root),
            candidate: observation(secondary: .infinity, quality: root),
            rootQuality: root
        )
        XCTAssertFalse(infiniteResult.qualified)
        XCTAssertNil(infiniteResult.balancedSecondaryRatio)
    }

    func testStabilityMemoryAndSwapGates() {
        let root = signature()
        let unstable = MachineAutoTuneEvaluator.screening(
            incumbent: observation(quality: root),
            candidate: observation(primary: 104, stability: 0.74, quality: root),
            rootQuality: root
        )
        XCTAssertFalse(unstable.qualified)
        XCTAssertTrue(unstable.reasons.contains { $0.contains("stability") })

        let pressured = MachineAutoTuneEvaluator.screening(
            incumbent: observation(quality: root),
            candidate: observation(primary: 104, freeMemory: 7, quality: root),
            rootQuality: root
        )
        XCTAssertFalse(pressured.qualified)
        XCTAssertTrue(pressured.reasons.contains { $0.contains("free memory") })

        let swapping = MachineAutoTuneEvaluator.screening(
            incumbent: observation(quality: root),
            candidate: observation(primary: 104, swapout: 129, quality: root),
            rootQuality: root
        )
        XCTAssertFalse(swapping.qualified)
        XCTAssertTrue(swapping.reasons.contains { $0.contains("swapout") })
    }

    func testOrderedNeighborsPreferUpwardAndRespectBoundaries() {
        let middle = MachineAutoTuneEvaluator.orderedNeighbors(
            values: [1, 2, 3], current: 2
        )
        XCTAssertEqual(middle, [
            MachineAutoTuneNeighbor(value: 3, direction: .upward),
            MachineAutoTuneNeighbor(value: 1, direction: .downward),
        ])
        XCTAssertEqual(
            MachineAutoTuneEvaluator.orderedNeighbors(values: [1, 2, 3], current: 1),
            [MachineAutoTuneNeighbor(value: 2, direction: .upward)]
        )
        XCTAssertEqual(
            MachineAutoTuneEvaluator.orderedNeighbors(values: [1, 2, 3], current: 3),
            [MachineAutoTuneNeighbor(value: 2, direction: .downward)]
        )
        XCTAssertTrue(
            MachineAutoTuneEvaluator.orderedNeighbors(values: [1, 2, 3], current: 4).isEmpty
        )
        XCTAssertTrue(
            MachineAutoTuneEvaluator.orderedNeighbors(values: [1, 2, 2], current: 2).isEmpty
        )
    }

    func testDirectionalWalkContinuesUpwardAndStopsAtFirstRegression() {
        var walk = MachineAutoTuneDirectionalWalk(
            values: [18, 20, 22, 24, 26], current: 20
        )

        XCTAssertTrue(walk.isValid)
        XCTAssertFalse(walk.isStopped)
        XCTAssertNil(walk.direction)
        XCTAssertEqual(
            walk.nextCandidate(),
            MachineAutoTuneNeighbor(value: 22, direction: .upward)
        )
        // A candidate cannot be emitted twice while its result is pending.
        XCTAssertNil(walk.nextCandidate())

        XCTAssertTrue(walk.recordResult(qualified: true))
        XCTAssertEqual(walk.current, 22)
        XCTAssertEqual(walk.direction, .upward)
        XCTAssertEqual(
            walk.nextCandidate(),
            MachineAutoTuneNeighbor(value: 24, direction: .upward)
        )

        XCTAssertFalse(walk.recordResult(qualified: false))
        XCTAssertEqual(walk.current, 22)
        XCTAssertTrue(walk.isStopped)
        XCTAssertNil(walk.nextCandidate(), "26 must not be tried after 24 loses to 22")
    }

    func testDirectionalWalkTriesDownwardOnlyAfterInitialUpwardRejection() {
        var walk = MachineAutoTuneDirectionalWalk(
            values: [16, 18, 20, 22, 24], current: 20
        )

        XCTAssertEqual(
            walk.nextCandidate(),
            MachineAutoTuneNeighbor(value: 22, direction: .upward)
        )
        XCTAssertFalse(walk.recordResult(qualified: false))
        XCTAssertFalse(walk.isStopped)
        XCTAssertNil(walk.direction)
        XCTAssertEqual(walk.current, 20)

        XCTAssertEqual(
            walk.nextCandidate(),
            MachineAutoTuneNeighbor(value: 18, direction: .downward)
        )
        XCTAssertTrue(walk.recordResult(qualified: true))
        XCTAssertEqual(walk.current, 18)
        XCTAssertEqual(walk.direction, .downward)
        XCTAssertEqual(
            walk.nextCandidate(),
            MachineAutoTuneNeighbor(value: 16, direction: .downward)
        )
    }

    func testDirectionalWalkDoesNotRepeatACharacterizedFallback() {
        var walk = MachineAutoTuneDirectionalWalk(
            values: [18, 20, 22], current: 20
        )

        XCTAssertEqual(walk.nextCandidate()?.value, 22)
        XCTAssertFalse(walk.recordResult(qualified: false))
        XCTAssertEqual(walk.nextCandidate()?.value, 18)
        XCTAssertFalse(walk.recordResult(qualified: false))

        XCTAssertTrue(walk.isStopped)
        XCTAssertNil(walk.nextCandidate())
    }

    func testDirectionalWalkRejectsMalformedLaddersAndMissingCurrent() {
        for walk in [
            MachineAutoTuneDirectionalWalk(values: [], current: 20),
            MachineAutoTuneDirectionalWalk(values: [18, 20, 20, 22], current: 20),
            MachineAutoTuneDirectionalWalk(values: [18, 22, 20], current: 20),
            MachineAutoTuneDirectionalWalk(values: [18, 20, 22], current: 24),
        ] {
            var invalid = walk
            XCTAssertFalse(invalid.isValid)
            XCTAssertTrue(invalid.isStopped)
            XCTAssertNil(invalid.nextCandidate())
        }
    }

    func testMedianAveragesEvenMiddlePairAndRejectsInvalidSamples() {
        XCTAssertEqual(MachineAutoTuneEvaluator.median([9, 1, 5]), 5)
        XCTAssertEqual(MachineAutoTuneEvaluator.median([9, 1, 5, 3]), 4)
        XCTAssertNil(MachineAutoTuneEvaluator.median([]))
        XCTAssertNil(MachineAutoTuneEvaluator.median([1, .nan]))
        XCTAssertNil(MachineAutoTuneEvaluator.median([1, .infinity]))
    }

    func testStandardManifestCoversEverySafeKnobAndScalesSlotsWithRAM() {
        let small = MachineAutoTuneManifest.standard(ramGB: 16)
        let large = MachineAutoTuneParameter.standard(ramGB: 96)

        let expectedSafeKnobs = Set(MachineAutoTuneKnob.allCases.filter { $0 != .q8NSG })
        XCTAssertEqual(Set(small.map(\.knob)), expectedSafeKnobs)
        XCTAssertNil(small.first { $0.knob == .q8NSG })
        XCTAssertTrue(small.allSatisfy { !$0.values.isEmpty })
        XCTAssertTrue(small.allSatisfy { Set($0.values).count == $0.values.count })
        XCTAssertTrue(small.allSatisfy { $0.quality == .exact })

        let smallSlots = small.first { $0.knob == .expertCacheSlots }!
        let largeSlots = large.first { $0.knob == .expertCacheSlots }!
        XCTAssertEqual(smallSlots.values, [0, 8, 12, 16, 18, 20, 22, 24])
        XCTAssertTrue(smallSlots.values.contains(0))
        XCTAssertTrue(smallSlots.values.contains(8))
        XCTAssertFalse(smallSlots.values.contains(32))
        XCTAssertTrue(largeSlots.values.contains(32))
        XCTAssertTrue(largeSlots.values.contains(48))
        XCTAssertTrue(smallSlots.memoryRisk)
        XCTAssertEqual(smallSlots.search, .walk)
        XCTAssertEqual(small.first { $0.knob == .expertCacheUniform }?.values, [0, 1])
        XCTAssertEqual(small.first { $0.knob == .expertCacheUniform }?.search, .sweep)
        XCTAssertEqual(small.first { $0.knob == .denseAhead }?.search, .walk)
        XCTAssertEqual(small.first { $0.knob == .preadSplit }?.search, .sweep)
    }

    func testNormalMemoryEnvelopeKeepsTenPercentPolicy() {
        let envelope = MachineAutoTuneMemoryEnvelope(
            baselineFreePercent: 18,
            physicalMemoryBytes: 64 * 1_073_741_824
        )

        XCTAssertTrue(envelope.isValid)
        XCTAssertTrue(envelope.canStart)
        XCTAssertFalse(envelope.isConstrained)
        XCTAssertEqual(envelope.effectiveMinimumFreePercent, 10)
        XCTAssertTrue(envelope.allowsResidentDeltaBytes(512 * 1_048_576))
        XCTAssertEqual(envelope.policy().minimumMemoryFreePercent, 10)
    }

    func testConstrainedEnvelopeLetsKnownRootRunBelowTenPercent() {
        let envelope = MachineAutoTuneMemoryEnvelope(
            baselineFreePercent: 7,
            physicalMemoryBytes: 16 * 1_073_741_824
        )
        let policy = envelope.policy()
        let root = signature()

        XCTAssertTrue(envelope.canStart)
        XCTAssertTrue(envelope.isConstrained)
        XCTAssertEqual(envelope.hardReservePercent, 3.125, accuracy: 1e-12)
        XCTAssertEqual(envelope.effectiveMinimumFreePercent, 6, accuracy: 1e-12)
        XCTAssertEqual(policy.minimumMemoryFreePercent, 6, accuracy: 1e-12)

        let repeatResult = MachineAutoTuneEvaluator.screening(
            incumbent: observation(freeMemory: 7, quality: root),
            candidate: observation(primary: 100, freeMemory: 6.2, quality: root),
            rootQuality: root,
            policy: policy
        )
        XCTAssertTrue(repeatResult.qualified, repeatResult.reasons.joined(separator: "; "))
    }

    func testConstrainedEnvelopeRejectsMemoryGrowthAndBelowRootFloor() {
        let envelope = MachineAutoTuneMemoryEnvelope(
            baselineFreePercent: 7,
            physicalMemoryBytes: 64 * 1_073_741_824
        )
        let root = signature()

        XCTAssertFalse(envelope.allowsResidentDeltaBytes(1))
        XCTAssertTrue(envelope.allowsResidentDeltaBytes(0))
        XCTAssertTrue(envelope.allowsResidentDeltaBytes(-1))

        let result = MachineAutoTuneEvaluator.screening(
            incumbent: observation(freeMemory: 7, quality: root),
            candidate: observation(primary: 110, freeMemory: 5.99, quality: root),
            rootQuality: root,
            policy: envelope.policy()
        )
        XCTAssertFalse(result.qualified)
        XCTAssertTrue(result.reasons.contains { $0.contains("free memory") })
    }

    func testConstrainedEnvelopeDoesNotRelaxSpeedQualityOrSwapGates() {
        let envelope = MachineAutoTuneMemoryEnvelope(
            baselineFreePercent: 7,
            physicalMemoryBytes: 64 * 1_073_741_824
        )
        let root = signature()
        let drift = signature(hash: "fedcba9876543210")

        let slower = MachineAutoTuneEvaluator.abba(
            a1: observation(primary: 100, freeMemory: 7, quality: root),
            b1: observation(primary: 99, freeMemory: 7, quality: root),
            b2: observation(primary: 99, freeMemory: 7, quality: root),
            a2: observation(primary: 100, freeMemory: 7, quality: root),
            rootQuality: root,
            policy: envelope.policy()
        )
        XCTAssertFalse(slower.qualified)
        XCTAssertTrue(slower.reasons.contains { $0.contains("balanced primary ratio") })

        let wrongQuality = MachineAutoTuneEvaluator.screening(
            incumbent: observation(freeMemory: 7, quality: root),
            candidate: observation(primary: 110, freeMemory: 7, quality: drift),
            rootQuality: root,
            policy: envelope.policy()
        )
        XCTAssertFalse(wrongQuality.qualified)
        XCTAssertFalse(wrongQuality.cumulativeQualityExact)

        let swapping = MachineAutoTuneEvaluator.screening(
            incumbent: observation(freeMemory: 7, quality: root),
            candidate: observation(
                primary: 110, freeMemory: 7, swapout: 129, quality: root
            ),
            rootQuality: root,
            policy: envelope.policy()
        )
        XCTAssertFalse(swapping.qualified)
        XCTAssertTrue(swapping.reasons.contains { $0.contains("swapout") })
    }

    func testConstrainedEnvelopeFailsClosedBelowHardByteReserve() throws {
        let envelope = MachineAutoTuneMemoryEnvelope(
            baselineFreePercent: 3,
            physicalMemoryBytes: 16 * 1_073_741_824
        )

        XCTAssertTrue(envelope.isConstrained)
        XCTAssertFalse(envelope.canStart)
        XCTAssertEqual(
            envelope.effectiveMinimumFreePercent,
            envelope.hardReservePercent,
            accuracy: 1e-12
        )

        let invalid = MachineAutoTuneMemoryEnvelope(
            baselineFreePercent: .nan,
            physicalMemoryBytes: 0
        )
        XCTAssertFalse(invalid.isValid)
        XCTAssertFalse(invalid.canStart)
        XCTAssertFalse(invalid.allowsResidentDeltaBytes(-1))
        XCTAssertTrue(invalid.effectiveMinimumFreePercent.isInfinite)

        let zeroReserve = MachineAutoTuneMemoryEnvelope(
            baselineFreePercent: 7,
            physicalMemoryBytes: 16 * 1_073_741_824,
            hardReserveBytes: 0
        )
        XCTAssertFalse(zeroReserve.isValid)
        XCTAssertFalse(zeroReserve.canStart)

        let decoded = try JSONDecoder().decode(
            MachineAutoTuneMemoryEnvelope.self,
            from: JSONEncoder().encode(envelope)
        )
        XCTAssertEqual(decoded, envelope)
    }
}
