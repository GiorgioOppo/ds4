import XCTest
import DS4Core
@testable import DS4Metal

final class GLM52ExpertStreamPlannerTests: XCTestCase {
    private let expertCount: UInt64 = 16
    private let selection: [UInt32] = [7, 0, 15, 3, 8, 1, 14, 2]

    private func descriptor(name: String,
                            type: UInt32,
                            dims: [UInt64],
                            offset: UInt64,
                            bytes: UInt64? = nil) -> GLM52WeightDescriptor {
        GLM52WeightDescriptor(
            name: name,
            type: type,
            dims: dims,
            absOffset: offset,
            bytes: bytes ?? GGUF.tensorNBytes(
                type: type,
                elements: dims.reduce(1, *))!
        )
    }

    private func weights(gateType: UInt32,
                         downType: UInt32 = GLM52TensorSchema.q6_K,
                         gateDims: [UInt64]? = nil,
                         upDims: [UInt64]? = nil,
                         downDims: [UInt64]? = nil,
                         gateBytes: UInt64? = nil,
                         gateOffset: UInt64 = 4_096) -> GLM52RoutedExpertWeights {
        let gd = gateDims ?? [512, 256, expertCount]
        let ud = upDims ?? gd
        let dd = downDims ?? [256, 512, expertCount]
        let gate = descriptor(
            name: "blk.3.ffn_gate_exps.weight",
            type: gateType,
            dims: gd,
            offset: gateOffset,
            bytes: gateBytes
        )
        let upOffset = gateOffset + gate.bytes + 4_096
        let up = descriptor(
            name: "blk.3.ffn_up_exps.weight",
            type: gateType,
            dims: ud,
            offset: upOffset
        )
        let down = descriptor(
            name: "blk.3.ffn_down_exps.weight",
            type: downType,
            dims: dd,
            offset: upOffset + up.bytes + 4_096
        )
        return GLM52RoutedExpertWeights(gate: gate, up: up, down: down)
    }

    func testIQ2Q2AndQ4GateUpWithQ6DownUseExactQuantBlocks() throws {
        let gateTypes: [UInt32] = [
            GLM52TensorSchema.iq2_XXS,
            GLM52TensorSchema.q2_K,
            GLM52TensorSchema.q4_K,
        ]

        for type in gateTypes {
            let planner = try GLM52ExpertStreamPlanner(
                layer: 3, weights: weights(gateType: type))
            let info = try XCTUnwrap(GGUF.typeInfo(type))
            let expectedGateRow = 2 * UInt64(info.blockBytes)
            let expectedGateExpert = expectedGateRow * 256
            let expectedDownExpert = UInt64(210 * 512)

            XCTAssertEqual(planner.gateLayout.blockElements, 256)
            XCTAssertEqual(planner.gateLayout.rowBytes, expectedGateRow)
            XCTAssertEqual(planner.gateLayout.expertBytes, expectedGateExpert)
            XCTAssertEqual(planner.upLayout.expertBytes, expectedGateExpert)
            XCTAssertEqual(planner.downLayout.blockBytes, 210)
            XCTAssertEqual(planner.downLayout.expertBytes, expectedDownExpert)

            let plan = try planner.plan(selectedExperts: selection)
            XCTAssertEqual(plan.expertIDs, selection)
            XCTAssertEqual(plan.experts.count, 8)
            XCTAssertEqual(plan.ranges.count, 24)
            XCTAssertEqual(
                plan.totalBytes,
                UInt64(8) * (2 * expectedGateExpert + expectedDownExpert)
            )

            for read in plan.experts {
                XCTAssertEqual(
                    read.gate.absoluteOffset,
                    planner.gateLayout.descriptor.absOffset
                        + UInt64(read.expertID) * expectedGateExpert
                )
                XCTAssertEqual(read.gate.byteCount, expectedGateExpert)
                XCTAssertEqual(read.up.byteCount, expectedGateExpert)
                XCTAssertEqual(read.down.byteCount, expectedDownExpert)
                XCTAssertEqual(read.ranges.map(\.projection), [.gate, .up, .down])
            }
        }
    }

    func testAdjacentExpertsRemainDistinctContiguousSlices() throws {
        let planner = try GLM52ExpertStreamPlanner(
            layer: 9,
            weights: weights(gateType: GLM52TensorSchema.q4_K)
        )
        let plan = try planner.plan(selectedExperts: Array(0..<8))

        XCTAssertEqual(plan.experts.count, 8)
        XCTAssertEqual(plan.ranges.count, 24)
        for index in 1..<plan.experts.count {
            let previous = plan.experts[index - 1]
            let current = plan.experts[index]
            XCTAssertEqual(
                previous.gate.absoluteOffset + previous.gate.byteCount,
                current.gate.absoluteOffset
            )
            XCTAssertEqual(
                previous.up.absoluteOffset + previous.up.byteCount,
                current.up.absoluteOffset
            )
            XCTAssertEqual(
                previous.down.absoluteOffset + previous.down.byteCount,
                current.down.absoluteOffset
            )
            XCTAssertNotEqual(previous.gate.expertID, current.gate.expertID)
        }
    }

    func testSelectionRejectsWrongCountDuplicatesAndOutOfRangeIDs() throws {
        let planner = try GLM52ExpertStreamPlanner(
            layer: 3,
            weights: weights(gateType: GLM52TensorSchema.iq2_XXS)
        )

        // Shorter plans are legitimate (the per-expert provider fetches one
        // record at a time); empty and over-width plans are refused.
        XCTAssertEqual(try planner.plan(selectedExperts: [0, 1]).experts.count, 2)
        XCTAssertThrowsError(try planner.plan(selectedExperts: [])) { error in
            XCTAssertEqual(
                error as? GLM52ExpertStreamPlannerError,
                .wrongSelectionCount(expected: 8, got: 0)
            )
        }
        XCTAssertThrowsError(try planner.plan(
            selectedExperts: [0, 1, 2, 3, 4, 5, 6, 7, 8])) { error in
            XCTAssertEqual(
                error as? GLM52ExpertStreamPlannerError,
                .wrongSelectionCount(expected: 8, got: 9)
            )
        }

        XCTAssertThrowsError(
            try planner.plan(selectedExperts: [0, 1, 2, 3, 4, 5, 6, 6])
        ) { error in
            XCTAssertEqual(
                error as? GLM52ExpertStreamPlannerError,
                .duplicateExpert(6)
            )
        }

        XCTAssertThrowsError(
            try planner.plan(selectedExperts: [0, 1, 2, 3, 4, 5, 6, 16])
        ) { error in
            XCTAssertEqual(
                error as? GLM52ExpertStreamPlannerError,
                .expertOutOfRange(16, expertCount: expertCount)
            )
        }
    }

    func testPlannerOnlyAcceptsExecutedMoELayers() throws {
        let routed = weights(gateType: GLM52TensorSchema.q2_K)

        for layer in [-1, 0, 2, 78, 79] {
            XCTAssertThrowsError(try GLM52ExpertStreamPlanner(
                layer: layer,
                weights: routed
            )) { error in
                XCTAssertEqual(
                    error as? GLM52ExpertStreamPlannerError,
                    .invalidLayer(layer)
                )
            }
        }

        XCTAssertNoThrow(try GLM52ExpertStreamPlanner(layer: 3, weights: routed))
        XCTAssertNoThrow(try GLM52ExpertStreamPlanner(layer: 77, weights: routed))
    }

    func testRejectsUnalignedUnknownAndMismatchedQuantizedLayouts() throws {
        XCTAssertThrowsError(try GLM52ExpertStreamPlanner(
            layer: 3,
            weights: weights(
                gateType: GLM52TensorSchema.q4_K,
                gateDims: [384, 256, expertCount],
                upDims: [384, 256, expertCount],
                downDims: [256, 384, expertCount]
            )
        )) { error in
            guard case .unalignedInnerDimension(let name, 384, 256) =
                    error as? GLM52ExpertStreamPlannerError else {
                return XCTFail("unexpected error: \(error)")
            }
            XCTAssertTrue(name.contains("gate"))
        }

        XCTAssertThrowsError(try GLM52ExpertStreamPlanner(
            layer: 3,
            weights: weights(gateType: GLM52TensorSchema.q6_K)
        )) { error in
            guard case .unsupportedQuantization(let name, GLM52TensorSchema.q6_K) =
                    error as? GLM52ExpertStreamPlannerError else {
                return XCTFail("unexpected error: \(error)")
            }
            XCTAssertTrue(name.contains("gate"))
        }

        XCTAssertThrowsError(try GLM52ExpertStreamPlanner(
            layer: 3,
            weights: weights(
                gateType: GLM52TensorSchema.q2_K,
                upDims: [512, 512, expertCount],
                downDims: [256, 512, expertCount]
            )
        )) { error in
            guard case .mismatchedProjectionGeometry =
                    error as? GLM52ExpertStreamPlannerError else {
                return XCTFail("unexpected error: \(error)")
            }
        }
    }

    func testRejectsDirectoryByteMismatchAndOffsetOverflow() throws {
        let valid = weights(gateType: GLM52TensorSchema.q2_K)
        let truncated = GLM52RoutedExpertWeights(
            gate: GLM52WeightDescriptor(
                name: valid.gate.name,
                type: valid.gate.type,
                dims: valid.gate.dims,
                absOffset: valid.gate.absOffset,
                bytes: valid.gate.bytes - 1
            ),
            up: valid.up,
            down: valid.down
        )
        XCTAssertThrowsError(try GLM52ExpertStreamPlanner(
            layer: 3, weights: truncated
        )) { error in
            guard case .byteCountMismatch(let name, let expected, let got) =
                    error as? GLM52ExpertStreamPlannerError else {
                return XCTFail("unexpected error: \(error)")
            }
            XCTAssertEqual(name, valid.gate.name)
            XCTAssertEqual(expected, valid.gate.bytes)
            XCTAssertEqual(got, valid.gate.bytes - 1)
        }

        let overflowing = GLM52RoutedExpertWeights(
            gate: GLM52WeightDescriptor(
                name: valid.gate.name,
                type: valid.gate.type,
                dims: valid.gate.dims,
                absOffset: UInt64.max - valid.gate.bytes + 1,
                bytes: valid.gate.bytes
            ),
            up: valid.up,
            down: valid.down
        )
        XCTAssertThrowsError(try GLM52ExpertStreamPlanner(
            layer: 3, weights: overflowing
        )) { error in
            guard case .arithmeticOverflow(let name) =
                    error as? GLM52ExpertStreamPlannerError else {
                return XCTFail("unexpected error: \(error)")
            }
            XCTAssertTrue(name.contains("gate"))
        }
    }
}
