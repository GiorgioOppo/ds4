import XCTest
import DS4Core
@testable import DS4Metal

final class GLM52TensorSchemaTests: XCTestCase {
    private func records(for range: ClosedRange<Int>,
                         includeGlobals: Bool = false) throws -> [GLM52TensorRecord] {
        var requirements: [GLM52TensorRequirement] = includeGlobals
            ? GLM52TensorSchema.globalRequirements()
            : []
        for layer in range {
            requirements += try GLM52TensorSchema.layerRequirements(layer)
        }
        return requirements.map { requirement in
            GLM52TensorRecord(
                name: requirement.name,
                type: requirement.acceptedTypes.sorted().first!,
                dimensions: requirement.dimensions
            )
        }
    }

    private func replacing(_ name: String, in records: [GLM52TensorRecord],
                           type: UInt32? = nil,
                           dimensions: [UInt64]? = nil) -> [GLM52TensorRecord] {
        records.map { record in
            guard record.name == name else { return record }
            return GLM52TensorRecord(
                name: name,
                type: type ?? record.type,
                dimensions: dimensions ?? record.dimensions
            )
        }
    }

    func testSchemaEncodesDenseMoEAndNextNBoundaries() throws {
        let dense = try GLM52TensorSchema.layerRequirements(2)
        XCTAssertEqual(dense.count, 18)
        XCTAssertTrue(dense.contains { $0.name == "blk.2.ffn_gate.weight" })
        XCTAssertFalse(dense.contains { $0.name == "blk.2.ffn_gate_exps.weight" })

        let routed = try GLM52TensorSchema.layerRequirements(3)
        XCTAssertEqual(routed.count, 23)
        XCTAssertTrue(routed.contains { $0.name == "blk.3.ffn_gate_exps.weight" })
        XCTAssertFalse(routed.contains { $0.name.contains("nextn") })

        let nextN = try GLM52TensorSchema.layerRequirements(78)
        XCTAssertEqual(nextN.count, 27)
        XCTAssertTrue(nextN.contains { requirement in
            requirement.name == "blk.78.nextn.eh_proj.weight"
                && requirement.dimensions == [12_288, 6_144]
        })
    }

    func testCanonicalFullDirectoryPasses() throws {
        let all = try records(for: 0...78, includeGlobals: true)
        XCTAssertEqual(all.count, 1_809)
        XCTAssertNoThrow(try GLM52TensorSchema.validate(records: all))
    }

    func testDuplicateTensorNameIsRejected() throws {
        var local = try records(for: 0...0)
        local.append(local[0])

        XCTAssertThrowsError(try GLM52TensorSchema.validate(
            records: local,
            layerRange: 0...0,
            requireTokenEmbedding: false,
            requireOutput: false
        )) { error in
            XCTAssertEqual(
                error as? GLM52TensorSchemaError,
                .duplicateTensor(local[0].name)
            )
        }
    }

    func testDistributedLayerRangeCanOmitGlobalTensors() throws {
        let local = try records(for: 12...14)
        XCTAssertNoThrow(try GLM52TensorSchema.validate(
            records: local,
            layerRange: 12...14,
            requireTokenEmbedding: false,
            requireOutput: false
        ))
    }

    func testRejectsWrongDenseControlPrecisionOrExtent() throws {
        let name = "blk.0.attn_q_a.weight"
        var local = try records(for: 0...0)
        local = replacing(name, in: local, type: GLM52TensorSchema.q4_K)

        XCTAssertThrowsError(try GLM52TensorSchema.validate(
            records: local,
            layerRange: 0...0,
            requireTokenEmbedding: false,
            requireOutput: false
        )) { error in
            guard case GLM52TensorSchemaError.layout(let gotName, _, _, let types, let dims) = error else {
                return XCTFail("unexpected error: \(error)")
            }
            XCTAssertEqual(gotName, name)
            XCTAssertEqual(types, [GLM52TensorSchema.q8_0])
            XCTAssertEqual(dims, [6_144, 2_048])
        }
    }

    func testGateAndUpMustUseTheSameSupportedQuantization() throws {
        var local = try records(for: 3...3)
        local = replacing("blk.3.ffn_gate_exps.weight", in: local,
                          type: GLM52TensorSchema.q4_K)
        local = replacing("blk.3.ffn_up_exps.weight", in: local,
                          type: GLM52TensorSchema.q5_K)

        XCTAssertThrowsError(try GLM52TensorSchema.validate(
            records: local,
            layerRange: 3...3,
            requireTokenEmbedding: false,
            requireOutput: false
        )) { error in
            XCTAssertEqual(
                error as? GLM52TensorSchemaError,
                .gateUpQuantizationMismatch(
                    layer: 3,
                    gateType: GLM52TensorSchema.q4_K,
                    upType: GLM52TensorSchema.q5_K
                )
            )
        }
    }

    func testAcceptsQ5GateUpWithQ6DownButRejectsQ6Gate() throws {
        var local = try records(for: 3...3)
        local = replacing("blk.3.ffn_gate_exps.weight", in: local,
                          type: GLM52TensorSchema.q5_K)
        local = replacing("blk.3.ffn_up_exps.weight", in: local,
                          type: GLM52TensorSchema.q5_K)
        local = replacing("blk.3.ffn_down_exps.weight", in: local,
                          type: GLM52TensorSchema.q6_K)
        XCTAssertNoThrow(try GLM52TensorSchema.validate(
            records: local,
            layerRange: 3...3,
            requireTokenEmbedding: false,
            requireOutput: false
        ))

        local = replacing("blk.3.ffn_gate_exps.weight", in: local,
                          type: GLM52TensorSchema.q6_K)
        local = replacing("blk.3.ffn_up_exps.weight", in: local,
                          type: GLM52TensorSchema.q6_K)
        XCTAssertThrowsError(try GLM52TensorSchema.validate(
            records: local,
            layerRange: 3...3,
            requireTokenEmbedding: false,
            requireOutput: false
        ))
    }

    func testPartialOutputHeadIsRejectedEvenWhenOutputIsOptional() throws {
        var globals = GLM52TensorSchema.globalRequirements().map { requirement in
            GLM52TensorRecord(
                name: requirement.name,
                type: requirement.acceptedTypes.first!,
                dimensions: requirement.dimensions
            )
        }
        globals.removeAll { $0.name == "output.weight" }
        globals += try records(for: 0...0)

        XCTAssertThrowsError(try GLM52TensorSchema.validate(
            records: globals,
            layerRange: 0...0,
            requireOutput: false
        )) { error in
            XCTAssertEqual(error as? GLM52TensorSchemaError, .partialOutputHead)
        }
    }
}
