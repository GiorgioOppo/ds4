import XCTest
@testable import DS4Metal

final class LagunaWeightMapTests: XCTestCase {
    func testTypedKeysProduceCanonicalGGUFNames() {
        XCTAssertEqual(LagunaGlobalTensorKey.tokenEmbedding.rawValue,
                       "token_embd.weight")
        XCTAssertEqual(LagunaGlobalTensorKey.output.rawValue, "output.weight")
        XCTAssertEqual(LagunaLayerTensorKey.attentionGate.name(layer: 12),
                       "blk.12.attn_gate.weight")
        XCTAssertEqual(LagunaLayerTensorKey.routerBias.name(layer: 3),
                       "blk.3.exp_probs_b.bias")
        XCTAssertEqual(LagunaLayerTensorKey.sharedDown.name(layer: 47),
                       "blk.47.ffn_down_shexp.weight")
        // The complete per-layer contract: 9 attention/norm tensors plus the
        // dense trio and the 8 MoE tensors.
        XCTAssertEqual(LagunaLayerTensorKey.allCases.count, 20)
    }

    func testDescriptorRetainsOnlyTensorDirectoryFields() {
        let descriptor = LagunaWeightDescriptor(
            name: "blk.3.ffn_gate_exps.weight",
            type: 12,
            dims: [3_072, 1_024, 256],
            absOffset: 12_345_678,
            bytes: 452_984_832
        )

        XCTAssertEqual(descriptor.name, "blk.3.ffn_gate_exps.weight")
        XCTAssertEqual(descriptor.type, 12)
        XCTAssertEqual(descriptor.dims, [3_072, 1_024, 256])
        XCTAssertEqual(descriptor.absOffset, 12_345_678)
        XCTAssertEqual(descriptor.bytes, 452_984_832)
    }
}
