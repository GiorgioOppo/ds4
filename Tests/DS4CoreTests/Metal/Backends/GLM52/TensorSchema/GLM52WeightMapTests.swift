import XCTest
@testable import DS4Metal

final class GLM52WeightMapTests: XCTestCase {
    func testTypedKeysProduceCanonicalGGUFNames() {
        XCTAssertEqual(GLM52GlobalTensorKey.tokenEmbedding.rawValue,
                       "token_embd.weight")
        XCTAssertEqual(GLM52GlobalTensorKey.output.rawValue, "output.weight")
        XCTAssertEqual(GLM52LayerTensorKey.attentionQueryA.name(layer: 12),
                       "blk.12.attn_q_a.weight")
        XCTAssertEqual(GLM52LayerTensorKey.routedGate.name(layer: 3),
                       "blk.3.ffn_gate_exps.weight")
        XCTAssertEqual(GLM52LayerTensorKey.nextNEmbeddingProjection.name(layer: 78),
                       "blk.78.nextn.eh_proj.weight")
    }

    func testDescriptorRetainsOnlyTensorDirectoryFields() {
        let descriptor = GLM52WeightDescriptor(
            name: "blk.3.ffn_gate_exps.weight",
            type: 16,
            dims: [6_144, 2_048, 256],
            absOffset: 12_345_678,
            bytes: 4_325_376_000
        )

        XCTAssertEqual(descriptor.name, "blk.3.ffn_gate_exps.weight")
        XCTAssertEqual(descriptor.type, 16)
        XCTAssertEqual(descriptor.dims, [6_144, 2_048, 256])
        XCTAssertEqual(descriptor.absOffset, 12_345_678)
        XCTAssertEqual(descriptor.bytes, 4_325_376_000)
    }
}
