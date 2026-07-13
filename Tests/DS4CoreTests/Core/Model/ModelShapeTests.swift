import XCTest
@testable import DS4Core

final class ModelShapeTests: XCTestCase {

    static let modelPath = "/Users/oppog/Downloads/ds4-main/gguf/DeepSeek-V4-Flash-Q4KExperts-F16HC-F16Compressor-F16Indexer-Q8Attn-Q8Shared-Q8Out-chat-v2-imatrix.gguf"

    // MARK: Compression-ratio formula (port of ds4_expected_layer_compress_ratio)

    func testFlashCompressRatioFormula() {
        let s = ModelShape.flash
        XCTAssertEqual(s.expectedCompressRatio(layer: 0), 0)
        XCTAssertEqual(s.expectedCompressRatio(layer: 1), 0)
        XCTAssertEqual(s.expectedCompressRatio(layer: 2), 4)
        XCTAssertEqual(s.expectedCompressRatio(layer: 3), 128)
        XCTAssertEqual(s.expectedCompressRatio(layer: 4), 4)
        XCTAssertEqual(s.expectedCompressRatio(layer: 5), 128)
        XCTAssertEqual(s.expectedCompressRatio(layer: 42), 4)
    }

    func testProCompressRatioFormula() {
        let s = ModelShape.pro
        XCTAssertEqual(s.expectedCompressRatio(layer: 0), 128)
        XCTAssertEqual(s.expectedCompressRatio(layer: 1), 128)
        XCTAssertEqual(s.expectedCompressRatio(layer: 2), 4)
        XCTAssertEqual(s.expectedCompressRatio(layer: 3), 128)
        XCTAssertEqual(s.expectedCompressRatio(layer: 60), 4)
    }

    // MARK: Real model — cross-checked against `ds4 --inspect`

    func testRealModelConfig() throws {
        try XCTSkipUnless(FileManager.default.fileExists(atPath: Self.modelPath), "real GGUF not present")
        let model = try GGUFModel(path: Self.modelPath, metalMapping: false)
        let cfg = try ModelConfig(model: model)

        // Shape selection + fields (ground truth from ./ds4 --inspect).
        XCTAssertEqual(cfg.shape.variant, .flash)
        XCTAssertEqual(cfg.shape.name, "DeepSeek V4 Flash")
        XCTAssertEqual(cfg.shape.nLayer, 43)
        XCTAssertEqual(cfg.shape.nEmbd, 4096)
        XCTAssertEqual(cfg.shape.nVocab, 129280)
        XCTAssertEqual(cfg.shape.nHead, 64)
        XCTAssertEqual(cfg.shape.nHeadKV, 1)
        XCTAssertEqual(cfg.shape.nHeadDim, 512)
        XCTAssertEqual(cfg.shape.nExpert, 256)
        XCTAssertEqual(cfg.shape.nExpertUsed, 6)
        XCTAssertEqual(cfg.shape.nSWA, 128)
        XCTAssertEqual(cfg.shape.nIndexerHead, 64)
        XCTAssertEqual(cfg.shape.nIndexerHeadDim, 128)
        XCTAssertEqual(cfg.shape.nIndexerTopK, 512)

        // Per-layer compress ratios must equal the formula for all 43 layers.
        XCTAssertEqual(cfg.compressRatios.count, 43)
        for il in 0..<43 {
            XCTAssertEqual(cfg.compressRatios[il],
                           cfg.shape.expectedCompressRatio(layer: UInt32(il)),
                           "compress ratio mismatch at layer \(il)")
        }

        // SwiGLU clamp: 43 entries, all 10.0.
        XCTAssertEqual(cfg.swigluClampExp.count, 43)
        XCTAssertTrue(cfg.swigluClampExp.allSatisfy { $0 == 10.0 })

        // RoPE parameters.
        XCTAssertEqual(cfg.ropeFreqBase, 10000.0)
        XCTAssertEqual(cfg.ropeScaleFactor, 16.0)
        XCTAssertEqual(cfg.ropeYarnBetaFast, 32.0)
        XCTAssertEqual(cfg.ropeYarnBetaSlow, 1.0)
        XCTAssertEqual(cfg.compressRopeFreqBase, 160000.0)
        XCTAssertEqual(cfg.ropeOrigCtx, 65536)
    }
}
