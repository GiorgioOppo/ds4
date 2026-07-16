import XCTest
import Foundation
import DS4Core
@testable import DS4Metal

/// Stage C: validates the DSV4 tensor naming scheme against the REAL GGUF model
/// (tensor-table parse only — lazy mmap, no 164GB load). Confirms every layer-0
/// and output tensor exists with the expected dtype, so the GGUF weight loader
/// can map them into LayerWeights/OutputHeadWeights.
final class GGUFWeightMapTests: XCTestCase {
    static let ggufPath = "/Users/oppog/Downloads/ds4-main/gguf/DeepSeek-V4-Flash-Q4KExperts-F16HC-F16Compressor-F16Indexer-Q8Attn-Q8Shared-Q8Out-chat-v2-imatrix.gguf"

    func testTensorNamingMatchesRealModel() throws {
        try XCTSkipUnless(FileManager.default.fileExists(atPath: Self.ggufPath),
                          "full GGUF model not present")
        let model = try GGUFModel(path: Self.ggufPath, metalMapping: false, prefetchCPU: false)

        // All layer-0 dense-path tensors + output tensors must be present.
        var names = DSV4Shape.layerTensorNames(0)
        names += DSV4Shape.outputTensorNames
        for n in names {
            XCTAssertNotNil(model.findTensor(n), "missing GGUF tensor: \(n)")
        }

        // Confirm key dtype assumptions baked into the graph wiring.
        func dtype(_ n: String) -> String? { model.findTensor(n)?.typeName }
        XCTAssertEqual(dtype("blk.0.ffn_gate_exps.weight"), "q4_k", "experts should be Q4_K")
        XCTAssertEqual(dtype("blk.0.ffn_up_exps.weight"), "q4_k")
        XCTAssertEqual(dtype("blk.0.ffn_down_exps.weight"), "q4_k")
        XCTAssertNotNil(dtype("blk.0.attn_output_a.weight"), "attn output is low-rank (a)")
        XCTAssertNotNil(dtype("blk.0.attn_output_b.weight"), "attn output is low-rank (b)")

        // Report a few shapes/dtypes for the record.
        for n in ["blk.0.hc_attn_fn.weight", "blk.0.attn_q_a.weight", "blk.0.attn_q_b.weight",
                  "blk.0.attn_kv.weight", "blk.0.attn_output_a.weight", "blk.0.attn_output_b.weight",
                  "blk.0.ffn_gate_exps.weight", "blk.0.ffn_gate_shexp.weight",
                  "token_embd.weight", "output.weight", "output_hc_fn.weight"] {
            if let t = model.findTensor(n) {
                print("  \(n): \(t.typeName) dims=\(t.dims)")
            }
        }
        // Layer count sanity: blk.42 should exist, blk.43 should not.
        XCTAssertNotNil(model.findTensor("blk.\(DSV4Shape.nLayer - 1).attn_norm.weight"))
        XCTAssertNil(model.findTensor("blk.\(DSV4Shape.nLayer).attn_norm.weight"))
    }
}
