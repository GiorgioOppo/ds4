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

    func testNativeAProjQ4SelectsQ4KernelsFromTensorTypes() throws {
        var writer = try GGUFWriter()
        let q4Names = [
            "attn_q_a.weight", "attn_q_b.weight", "attn_kv.weight",
            "attn_output_a.weight", "attn_output_b.weight",
        ]
        let q8Names = [
            "ffn_gate_shexp.weight", "ffn_up_shexp.weight",
            "ffn_down_shexp.weight",
        ]
        for name in q4Names {
            writer.add(.init(name: "blk.0.\(name)", dims: [256], type: 12,
                             data: Data(count: 144)))
        }
        for name in q8Names {
            writer.add(.init(name: "blk.0.\(name)", dims: [256], type: 8,
                             data: Data(count: 272)))
        }
        let url = FileManager.default.temporaryDirectory
            .appendingPathComponent("aproj-q4-\(UUID().uuidString).gguf")
        try writer.write(to: url.path)
        defer { try? FileManager.default.removeItem(at: url) }

        let model = try GGUFModel(path: url.path, metalMapping: false)
        let quant = GGUFWeights.denseQuantization(model, layer: 0)
        XCTAssertTrue(quant.qAIsQ4)
        XCTAssertTrue(quant.qBIsQ4)
        XCTAssertTrue(quant.kvIsQ4)
        XCTAssertTrue(quant.outputAIsQ4)
        XCTAssertTrue(quant.outputBIsQ4)
        XCTAssertFalse(quant.sharedGateIsQ4)
        XCTAssertFalse(quant.sharedUpIsQ4)
        XCTAssertFalse(quant.sharedDownIsQ4)
    }

    func testAProjQ8KeepsQ8KernelBinding() throws {
        var writer = try GGUFWriter()
        for name in ["attn_q_a.weight", "attn_q_b.weight", "attn_kv.weight",
                     "attn_output_a.weight", "attn_output_b.weight"] {
            writer.add(.init(name: "blk.0.\(name)", dims: [256], type: 8,
                             data: Data(count: 272)))
        }
        let url = FileManager.default.temporaryDirectory
            .appendingPathComponent("aproj-q8-\(UUID().uuidString).gguf")
        try writer.write(to: url.path)
        defer { try? FileManager.default.removeItem(at: url) }

        let model = try GGUFModel(path: url.path, metalMapping: false)
        let quant = GGUFWeights.denseQuantization(model, layer: 0)
        XCTAssertFalse(quant.qAIsQ4)
        XCTAssertFalse(quant.qBIsQ4)
        XCTAssertFalse(quant.kvIsQ4)
        XCTAssertFalse(quant.outputAIsQ4)
        XCTAssertFalse(quant.outputBIsQ4)
    }
}
