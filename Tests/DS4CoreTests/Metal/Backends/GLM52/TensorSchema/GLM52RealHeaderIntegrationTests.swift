import XCTest
import DS4Core
@testable import DS4Metal

/// Opt-in contract check against a sparse copy of a real GLM 5.2 GGUF.
///
/// The fixture needs only the complete GGUF header and tensor directory; it can
/// be extended sparsely to the original byte count, so this test validates the
/// published metadata, tokenizer tables and all 1,809 tensor descriptors without
/// downloading or reading hundreds of gigabytes of tensor payload.
final class GLM52RealHeaderIntegrationTests: XCTestCase {
    func testAntirezIQ2XXSHeaderContractWhenFixtureIsProvided() throws {
        guard let path = ProcessInfo.processInfo.environment["DS4_GLM52_SPARSE_GGUF"],
              !path.isEmpty else {
            throw XCTSkip("set DS4_GLM52_SPARSE_GGUF to an exact-size sparse GGUF fixture")
        }

        let model = try GGUFModel(path: path, metalMapping: false, prefetchCPU: false)
        let configuration = try GLM52Configuration(model: model)
        let tokenizer = try GLM52Tokenizer(model: model)

        XCTAssertEqual(model.n_tensors, 1_809)
        XCTAssertEqual(model.n_kv, 66)
        XCTAssertEqual(configuration.shape, .v5_2)
        XCTAssertEqual(tokenizer.nVocab, 154_880)
        XCTAssertEqual(tokenizer.special.mask, 154_822)
        XCTAssertEqual(tokenizer.special.startOfPrompt, 154_824)
        XCTAssertEqual(tokenizer.special.endOfText, 154_820)
        XCTAssertNil(tokenizer.tokenID("<|tool|>"))

        try GLM52TensorSchema.validate(model: model)

        let weights = try GLM52WeightMap(model: model)
        XCTAssertEqual(weights.layerCount, 79)
        XCTAssertEqual(
            try weights.global(.tokenEmbedding).name,
            "token_embd.weight"
        )
        XCTAssertEqual(
            try weights.layer(0, .denseGate).name,
            "blk.0.ffn_gate.weight"
        )
        XCTAssertEqual(
            try weights.layer(77, .routedDown).name,
            "blk.77.ffn_down_exps.weight"
        )
        XCTAssertThrowsError(try weights.layer(0, .routedGate))

        let planner = try GLM52ExpertStreamPlanner(weightMap: weights, layer: 3)
        let plan = try planner.plan(selectedExperts: [0, 1, 2, 3, 4, 5, 6, 7])
        XCTAssertEqual(plan.experts.count, 8)
        XCTAssertEqual(plan.ranges.count, 24)
        XCTAssertTrue(plan.totalBytes > 0)

        // Payload reader on the exact-size sparse fixture: the whole validated
        // directory must fit the file, and a real top-8 plan must execute end
        // to end (sparse holes read as zeros — the point here is bounds and
        // packing, not weight content).
        let reader = try GLM52PayloadReader(path: path, weightMap: weights)
        XCTAssertEqual(reader.fileSize, 211_075_856_448)
        let layout = try reader.packedLayout(of: plan)
        XCTAssertEqual(UInt64(layout.totalBytes), plan.totalBytes)
        var packed = [UInt8](repeating: 1, count: layout.totalBytes)
        try packed.withUnsafeMutableBytes {
            try reader.read(plan: plan, into: $0)
        }

        let outputNorm = try reader.bytes(of: weights.global(.outputNorm))
        XCTAssertEqual(outputNorm.count, 6_144 * 4)
    }
}
