import XCTest
import Foundation
import DS4Core
@testable import DS4Metal

/// Stage C: validates the GGUF weight loader reads tensor bytes correctly from
/// the real model's mmap into GPUTensors (small F32 tensors + a scalar — no large
/// load). Confirms GGUFWeights.tensor / scalarF32 are byte-faithful.
final class GGUFLoaderTests: XCTestCase {
    static let ggufPath = "/Users/oppog/Downloads/ds4-main/gguf/DeepSeek-V4-Flash-Q4KExperts-F16HC-F16Compressor-F16Indexer-Q8Attn-Q8Shared-Q8Out-chat-v2-imatrix.gguf"
    static let metalDir = "/Users/oppog/Downloads/ds4-main/DS4-gui/metal"

    func testLoadSmallTensors() throws {
        try XCTSkipUnless(FileManager.default.fileExists(atPath: Self.ggufPath), "GGUF not present")
        try XCTSkipUnless(FileManager.default.fileExists(atPath: Self.metalDir + "/norm.metal"), "kernels absent")
        let rt: MetalRuntime
        do { rt = try MetalRuntime(metalDir: Self.metalDir) } catch { throw XCTSkip("Metal unavailable: \(error)") }
        let model = try GGUFModel(path: Self.ggufPath, metalMapping: false, prefetchCPU: false)

        // attn_norm.weight is F32 [nEmbd]. Load and compare to raw mmap floats.
        let name = "blk.0.attn_norm.weight"
        guard let t = model.findTensor(name) else { return XCTFail("missing \(name)") }
        XCTAssertEqual(t.typeName, "f32")
        let n = Int(t.elements)
        let gt = try GGUFWeights.tensor(rt, model, name)
        let got = gt.floatArray(n)
        let raw = (model.mapBase + Int(t.absOffset)).bindMemory(to: Float.self, capacity: n)
        for i in stride(from: 0, to: n, by: 137) {
            XCTAssertEqual(got[i], raw[i], "attn_norm mismatch at \(i)")
        }
        XCTAssertEqual(gt.byteLength, Int(t.bytes))

        // output_hc_scale.weight scalar.
        let sc = try GGUFWeights.scalarF32(model, "output_hc_scale.weight")
        XCTAssertTrue(sc.isFinite, "output_hc_scale not finite: \(sc)")
        print("  output_hc_scale = \(sc); attn_norm[0..3] = \(Array(got.prefix(3)))")
    }
}
