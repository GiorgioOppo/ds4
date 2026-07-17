import XCTest
import DS4Core
@testable import DS4Metal

final class GLM52CompactDSATests: XCTestCase {
    func testIndexShareScheduleMatchesGLM52() {
        let expected = [0, 1, 2] + Array(stride(from: 6, through: 74, by: 4))
        XCTAssertEqual(GLM52IndexSharePolicy.fullIndexerLayers(), expected)
        XCTAssertEqual(expected.count, 21)

        XCTAssertEqual(GLM52IndexSharePolicy.selectionSourceLayer(for: 0), 0)
        XCTAssertEqual(GLM52IndexSharePolicy.selectionSourceLayer(for: 2), 2)
        XCTAssertEqual(GLM52IndexSharePolicy.selectionSourceLayer(for: 3), 2)
        XCTAssertEqual(GLM52IndexSharePolicy.selectionSourceLayer(for: 5), 2)
        XCTAssertEqual(GLM52IndexSharePolicy.selectionSourceLayer(for: 6), 6)
        XCTAssertEqual(GLM52IndexSharePolicy.selectionSourceLayer(for: 9), 6)
        XCTAssertEqual(GLM52IndexSharePolicy.selectionSourceLayer(for: 10), 10)
        XCTAssertEqual(GLM52IndexSharePolicy.selectionSourceLayer(for: 77), 74)
        XCTAssertNil(GLM52IndexSharePolicy.selectionSourceLayer(for: 78),
                     "block 78 is nextn, not a normal inference layer")
    }

    func testF16CompactLayoutHasExactReferenceByteCount() throws {
        let layout = GLM52CompactDSALayout(precision: .float16)
        XCTAssertEqual(layout.normalLayerCount, 78)
        XCTAssertEqual(layout.fullIndexerLayers.count, 21)
        XCTAssertEqual(layout.kvLoRABytesPerToken, 79_872)
        XCTAssertEqual(layout.ropeTailBytesPerToken, 9_984)
        XCTAssertEqual(layout.indexerKeyBytesPerToken, 5_376)
        XCTAssertEqual(layout.bytesPerToken, 95_232)

        let allocation = try layout.allocation(rows: 4_096)
        XCTAssertEqual(allocation.totalBytes, 390_070_272)
        XCTAssertEqual(allocation.totalBytes,
                       allocation.kvLoRABytes + allocation.ropeTailBytes
                           + allocation.indexerKeyBytes)
    }

    func testF32OracleLayoutIsExactlyTwiceF16() throws {
        let f16 = GLM52CompactDSALayout(precision: .float16)
        let f32 = GLM52CompactDSALayout(precision: .float32)
        XCTAssertEqual(f32.bytesPerToken, 2 * f16.bytesPerToken)
        let f16Bytes = try f16.allocation(rows: 123).totalBytes
        let f32Bytes = try f32.allocation(rows: 123).totalBytes
        XCTAssertEqual(f32Bytes, 2 * f16Bytes)
    }

    func testLazySlabPlanDoesNotAllocateTheLogicalContextUpFront() throws {
        let policy = GLM52CompactDSACapacityPolicy(slabRows: 1_024)
        let layout = GLM52CompactDSALayout()

        let first = try policy.plan(
            currentCapacity: 0,
            requiredRows: 1,
            logicalContext: 100_000,
            layout: layout
        )
        XCTAssertEqual(first.targetCapacity, 1_024)
        XCTAssertEqual(first.newSlabRanges, [0..<1_024])
        XCTAssertEqual(first.additionalBytes, 97_517_568)
        XCTAssertLessThan(first.targetCapacity, 100_000)

        let unchanged = try policy.plan(
            currentCapacity: 1_024,
            requiredRows: 1_024,
            logicalContext: 100_000,
            layout: layout
        )
        XCTAssertFalse(unchanged.grows)
        XCTAssertTrue(unchanged.newSlabRanges.isEmpty)

        let second = try policy.plan(
            currentCapacity: 1_024,
            requiredRows: 1_025,
            logicalContext: 100_000,
            layout: layout
        )
        XCTAssertEqual(second.targetCapacity, 2_048)
        XCTAssertEqual(second.newSlabRanges, [1_024..<2_048])
        XCTAssertEqual(second.additionalBytes, first.additionalBytes)
    }

    func testLazyPlanUsesAPartialTailSlabAndEnforcesBudget() throws {
        let policy = GLM52CompactDSACapacityPolicy(slabRows: 4_096)
        let layout = GLM52CompactDSALayout()
        let plan = try policy.plan(
            currentCapacity: 98_304,
            requiredRows: 99_999,
            logicalContext: 100_000,
            layout: layout
        )
        XCTAssertEqual(plan.targetCapacity, 100_000)
        XCTAssertEqual(plan.newSlabRanges, [98_304..<100_000])
        XCTAssertEqual(plan.residentBytesAfterGrowth, 9_523_200_000)

        XCTAssertThrowsError(try policy.plan(
            currentCapacity: 0,
            requiredRows: 1,
            logicalContext: 100_000,
            layout: layout,
            maxResidentBytes: 100_000_000
        )) { error in
            guard case GLM52CompactDSAError.budgetExceeded(let required, let budget) = error else {
                return XCTFail("unexpected error: \(error)")
            }
            XCTAssertEqual(required, 390_070_272)
            XCTAssertEqual(budget, 100_000_000)
        }
    }

    func testSlabPlannerDoesNotOverflowForMaximumSlabSize() throws {
        let plan = try GLM52CompactDSACapacityPolicy(slabRows: .max).plan(
            currentCapacity: 1,
            requiredRows: 2,
            logicalContext: 4
        )
        XCTAssertEqual(plan.targetCapacity, 4)
        XCTAssertEqual(plan.newSlabRanges, [1..<4])
    }

    func testContextCannotExceedTrainingLimit() {
        XCTAssertThrowsError(try GLM52CompactDSACapacityPolicy().plan(
            currentCapacity: 0,
            requiredRows: 1,
            logicalContext: 1_048_577
        )) { error in
            XCTAssertEqual(
                error as? GLM52CompactDSAError,
                .logicalContextExceedsModel(requested: 1_048_577, maximum: 1_048_576)
            )
        }
    }
}
