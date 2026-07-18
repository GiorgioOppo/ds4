import XCTest
import DS4Core
@testable import DS4Metal

/// Byte-faithfulness of the GLM 5.2 payload reader against synthetic files:
/// every read must return exactly the bytes the plan pointed at, and every
/// bound violation must surface as a typed error before any byte moves.
final class GLM52PayloadReaderTests: XCTestCase {
    private let expertCount: UInt64 = 16
    private let selection: [UInt32] = [7, 0, 15, 3, 8, 1, 14, 2]

    // MARK: - Fixture

    /// Deterministic non-repeating-ish byte pattern so any offset mistake
    /// changes the compared bytes.
    private func patternByte(_ i: Int) -> UInt8 {
        UInt8(truncatingIfNeeded: i &* 31 &+ (i >> 8) &+ 7)
    }

    private func descriptor(name: String,
                            type: UInt32,
                            dims: [UInt64],
                            offset: UInt64) -> GLM52WeightDescriptor {
        GLM52WeightDescriptor(
            name: name,
            type: type,
            dims: dims,
            absOffset: offset,
            bytes: GGUF.tensorNBytes(type: type, elements: dims.reduce(1, *))!
        )
    }

    /// Small but block-legal routed geometry: gate/up Q4_K [256, 256, 16],
    /// down Q6_K [256, 256, 16] — inner dims stay multiples of the 256-element
    /// quant blocks, total file ~2 MiB.
    private func routedWeights() -> GLM52RoutedExpertWeights {
        let gate = descriptor(
            name: "blk.3.ffn_gate_exps.weight",
            type: GLM52TensorSchema.q4_K,
            dims: [256, 256, expertCount],
            offset: 4_096
        )
        let up = descriptor(
            name: "blk.3.ffn_up_exps.weight",
            type: GLM52TensorSchema.q4_K,
            dims: [256, 256, expertCount],
            offset: gate.absOffset + gate.bytes + 4_096
        )
        let down = descriptor(
            name: "blk.3.ffn_down_exps.weight",
            type: GLM52TensorSchema.q6_K,
            dims: [256, 256, expertCount],
            offset: up.absOffset + up.bytes + 4_096
        )
        return GLM52RoutedExpertWeights(gate: gate, up: up, down: down)
    }

    private func fileEnd(_ weights: GLM52RoutedExpertWeights) -> Int {
        Int(weights.down.absOffset + weights.down.bytes)
    }

    /// Write `byteCount` pattern bytes to a fresh temp file, removed on teardown.
    private func writePatternFile(byteCount: Int) throws -> String {
        let url = FileManager.default.temporaryDirectory
            .appendingPathComponent("glm52-payload-\(UUID().uuidString).bin")
        var data = Data(count: byteCount)
        data.withUnsafeMutableBytes { raw in
            for i in 0..<byteCount { raw[i] = patternByte(i) }
        }
        try data.write(to: url)
        addTeardownBlock {
            try? FileManager.default.removeItem(at: url)
        }
        return url.path
    }

    private func expectedBytes(offset: UInt64, count: UInt64) -> [UInt8] {
        (0..<Int(count)).map { patternByte(Int(offset) + $0) }
    }

    // MARK: - Single descriptor reads

    func testSingleDescriptorReadIsByteFaithful() throws {
        let weights = routedWeights()
        let path = try writePatternFile(byteCount: fileEnd(weights))
        let reader = try GLM52PayloadReader(path: path)

        XCTAssertEqual(reader.fileSize, UInt64(fileEnd(weights)))
        let payload = try reader.bytes(of: weights.gate)
        XCTAssertEqual(payload.count, Int(weights.gate.bytes))
        XCTAssertEqual(
            payload,
            expectedBytes(offset: weights.gate.absOffset, count: weights.gate.bytes)
        )
    }

    func testSingleDescriptorReadRejectsWrongDestinationSize() throws {
        let weights = routedWeights()
        let path = try writePatternFile(byteCount: fileEnd(weights))
        let reader = try GLM52PayloadReader(path: path)

        var short = [UInt8](repeating: 0, count: Int(weights.gate.bytes) - 1)
        try short.withUnsafeMutableBytes { buffer in
            XCTAssertThrowsError(try reader.read(weights.gate, into: buffer)) {
                XCTAssertEqual(
                    $0 as? GLM52PayloadReaderError,
                    .destinationSizeMismatch(expected: weights.gate.bytes,
                                             got: Int(weights.gate.bytes) - 1)
                )
            }
        }
    }

    func testDescriptorBeyondFileEndIsRejectedBeforeReading() throws {
        let weights = routedWeights()
        // Truncate the file 1 byte before the down tensor's payload end.
        let path = try writePatternFile(byteCount: fileEnd(weights) - 1)
        let reader = try GLM52PayloadReader(path: path)

        XCTAssertThrowsError(try reader.bytes(of: weights.down)) {
            XCTAssertEqual(
                $0 as? GLM52PayloadReaderError,
                .rangeOutsideFile(
                    name: weights.down.name,
                    end: weights.down.absOffset + weights.down.bytes,
                    fileSize: UInt64(fileEnd(weights) - 1))
            )
        }
    }

    func testMissingFileIsRejectedAtOpen() {
        XCTAssertThrowsError(
            try GLM52PayloadReader(path: "/nonexistent/glm52-payload.gguf")
        ) {
            guard case .cannotOpen = $0 as? GLM52PayloadReaderError else {
                return XCTFail("expected cannotOpen, got \($0)")
            }
        }
    }

    // MARK: - Expert stream plan reads

    func testExpertPlanReadPacksRecordsInRankOrder() throws {
        let weights = routedWeights()
        let path = try writePatternFile(byteCount: fileEnd(weights))
        let reader = try GLM52PayloadReader(path: path)
        let planner = try GLM52ExpertStreamPlanner(layer: 3, weights: weights)
        let plan = try planner.plan(selectedExperts: selection)

        let layout = try reader.packedLayout(of: plan)
        XCTAssertEqual(layout.expertCount, selection.count)
        XCTAssertEqual(layout.gateBytes, Int(planner.gateLayout.expertBytes))
        XCTAssertEqual(layout.upBytes, Int(planner.upLayout.expertBytes))
        XCTAssertEqual(layout.downBytes, Int(planner.downLayout.expertBytes))
        XCTAssertEqual(layout.upOffset, layout.gateBytes)
        XCTAssertEqual(layout.downOffset, layout.gateBytes + layout.upBytes)
        XCTAssertEqual(UInt64(layout.totalBytes), plan.totalBytes)

        var packed = [UInt8](repeating: 0, count: layout.totalBytes)
        let returned = try packed.withUnsafeMutableBytes {
            try reader.read(plan: plan, into: $0)
        }
        XCTAssertEqual(returned, layout)

        for (rank, expert) in plan.experts.enumerated() {
            let record = layout.recordOffset(rank: rank)
            let slots: [(GLM52ExpertByteRange, Int, Int)] = [
                (expert.gate, record + layout.gateOffset, layout.gateBytes),
                (expert.up, record + layout.upOffset, layout.upBytes),
                (expert.down, record + layout.downOffset, layout.downBytes),
            ]
            for (range, start, count) in slots {
                XCTAssertEqual(
                    Array(packed[start..<start + count]),
                    expectedBytes(offset: range.absoluteOffset,
                                  count: range.byteCount),
                    "rank \(rank) \(range.projection) bytes diverge"
                )
            }
        }

        // The serial path must be byte-identical to the concurrent one.
        var serial = [UInt8](repeating: 0, count: layout.totalBytes)
        _ = try serial.withUnsafeMutableBytes {
            try reader.read(plan: plan, into: $0, concurrent: false)
        }
        XCTAssertEqual(serial, packed)
    }

    func testExpertPlanReadRejectsWrongDestinationSize() throws {
        let weights = routedWeights()
        let path = try writePatternFile(byteCount: fileEnd(weights))
        let reader = try GLM52PayloadReader(path: path)
        let plan = try GLM52ExpertStreamPlanner(layer: 3, weights: weights)
            .plan(selectedExperts: selection)
        let layout = try reader.packedLayout(of: plan)

        var oversized = [UInt8](repeating: 0, count: layout.totalBytes + 1)
        try oversized.withUnsafeMutableBytes { buffer in
            XCTAssertThrowsError(try reader.read(plan: plan, into: buffer)) {
                XCTAssertEqual(
                    $0 as? GLM52PayloadReaderError,
                    .destinationSizeMismatch(expected: UInt64(layout.totalBytes),
                                             got: layout.totalBytes + 1)
                )
            }
        }
    }

    func testExpertPlanReadRejectsTruncatedFile() throws {
        let weights = routedWeights()
        let path = try writePatternFile(byteCount: fileEnd(weights) - 1)
        let reader = try GLM52PayloadReader(path: path)
        let plan = try GLM52ExpertStreamPlanner(layer: 3, weights: weights)
            .plan(selectedExperts: selection)
        let layout = try reader.packedLayout(of: plan)

        var packed = [UInt8](repeating: 0, count: layout.totalBytes)
        try packed.withUnsafeMutableBytes { buffer in
            XCTAssertThrowsError(try reader.read(plan: plan, into: buffer)) {
                guard case .rangeOutsideFile = $0 as? GLM52PayloadReaderError else {
                    return XCTFail("expected rangeOutsideFile, got \($0)")
                }
            }
        }
    }

    func testHandBuiltNonUniformPlanIsRejected() throws {
        let weights = routedWeights()
        let path = try writePatternFile(byteCount: fileEnd(weights))
        let reader = try GLM52PayloadReader(path: path)
        let plan = try GLM52ExpertStreamPlanner(layer: 3, weights: weights)
            .plan(selectedExperts: selection)

        var experts = plan.experts
        let last = experts.removeLast()
        let shrunkDown = GLM52ExpertByteRange(
            projection: .down,
            tensorName: last.down.tensorName,
            tensorType: last.down.tensorType,
            expertID: last.down.expertID,
            absoluteOffset: last.down.absoluteOffset,
            byteCount: last.down.byteCount - 1
        )
        experts.append(GLM52ExpertReadPlan(
            expertID: last.expertID,
            gate: last.gate,
            up: last.up,
            down: shrunkDown,
            totalBytes: last.totalBytes - 1
        ))
        let nonUniform = GLM52ExpertStreamPlan(
            layer: plan.layer,
            experts: experts,
            totalBytes: plan.totalBytes - 1
        )

        XCTAssertThrowsError(try reader.packedLayout(of: nonUniform)) {
            XCTAssertEqual(
                $0 as? GLM52PayloadReaderError,
                .nonUniformPlan(layer: plan.layer)
            )
        }
    }

    func testEmptyPlanIsRejected() throws {
        let weights = routedWeights()
        let path = try writePatternFile(byteCount: fileEnd(weights))
        let reader = try GLM52PayloadReader(path: path)

        let empty = GLM52ExpertStreamPlan(layer: 3, experts: [], totalBytes: 0)
        XCTAssertThrowsError(try reader.packedLayout(of: empty)) {
            XCTAssertEqual(
                $0 as? GLM52PayloadReaderError,
                .emptyPlan(layer: 3)
            )
        }
    }
}
