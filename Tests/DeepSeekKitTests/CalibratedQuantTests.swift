import XCTest
@testable import DeepSeekKit

/// Test del modulo `CalibratedQuant`. Copre:
/// - `.rtn` produce esattamente lo stesso output di
///   `quantizeBF16ToInt8` esistente (no-op wrapper).
/// - `.awq` produce un output diverso da RTN (verifica che il
///   smoothing per-canale stia effettivamente cambiando i pesi
///   quantizzati e/o le scales).
/// - `.smoothQuant` e `.gptq` throwano `QuantNotImplemented`.
/// - `ActivationObserver` accumula correttamente per-channel absmax.
final class CalibratedQuantTests: XCTestCase {

    private func makeTempBF16File(outDim: Int, inDim: Int, seed: UInt64)
        throws -> URL
    {
        let xs = randomBF16Bytes(count: outDim * inDim, seed: seed)
        let url = FileManager.default.temporaryDirectory
            .appendingPathComponent("calib-quant-\(UUID().uuidString).bin")
        try xs.write(to: url, options: .atomic)
        return url
    }

    func testRTNMatchesExistingFunction() throws {
        let outDim = 16, inDim = 256
        let url = try makeTempBF16File(outDim: outDim, inDim: inDim, seed: 11)
        defer { try? FileManager.default.removeItem(at: url) }

        let baseline = try quantizeBF16ToInt8(srcURL: url, srcOffset: 0,
                                              outDim: outDim, inDim: inDim)
        let viaCalib = try quantizeBF16ToInt8Calibrated(srcURL: url, srcOffset: 0,
                                                         outDim: outDim, inDim: inDim,
                                                         method: .rtn, stats: nil)
        XCTAssertEqual(baseline.weight, viaCalib.weight,
                       "RTN wrapper deve essere byte-identico a quantizeBF16ToInt8")
        XCTAssertEqual(baseline.scale, viaCalib.scale,
                       "RTN wrapper deve produrre le stesse scales")
    }

    func testAWQDiffersFromRTNWhenChannelsAreUnbalanced() throws {
        let outDim = 8, inDim = 256
        let url = try makeTempBF16File(outDim: outDim, inDim: inDim, seed: 23)
        defer { try? FileManager.default.removeItem(at: url) }

        // Construct activation stats con uno squilibrio per-canale forte:
        // metà canali con absmax=10, metà con absmax=0.1. AWQ con alpha=0.5
        // dovrebbe alzare lo scale dei canali "rumorosi" e abbassare i
        // "tranquilli", risultando in un quant differente da RTN.
        var actStats = [Float](repeating: 0.1, count: inDim)
        for c in stride(from: 0, to: inDim, by: 2) { actStats[c] = 10.0 }
        let stats = CalibrationStats(perChannelAbsMax: actStats,
                                      observedTokens: 1000)

        let rtn = try quantizeBF16ToInt8Calibrated(srcURL: url, srcOffset: 0,
                                                    outDim: outDim, inDim: inDim,
                                                    method: .rtn, stats: nil)
        let awq = try quantizeBF16ToInt8Calibrated(srcURL: url, srcOffset: 0,
                                                    outDim: outDim, inDim: inDim,
                                                    method: .awq, stats: stats,
                                                    awqAlpha: 0.5)

        // Sanity: shapes uguali.
        XCTAssertEqual(rtn.weight.count, awq.weight.count)
        XCTAssertEqual(rtn.scale.count, awq.scale.count)
        // Output deve differire perché AWQ smussa i pesi prima della
        // quant: almeno qualche byte deve cambiare.
        XCTAssertNotEqual(rtn.weight, awq.weight,
                          "AWQ con squilibrio per-canale deve produrre pesi diversi da RTN")
    }

    func testAWQRequiresStats() throws {
        let outDim = 4, inDim = 128
        let url = try makeTempBF16File(outDim: outDim, inDim: inDim, seed: 99)
        defer { try? FileManager.default.removeItem(at: url) }

        XCTAssertThrowsError(
            try quantizeBF16ToInt8Calibrated(srcURL: url, srcOffset: 0,
                                              outDim: outDim, inDim: inDim,
                                              method: .awq, stats: nil)
        )
    }

    func testSmoothQuantAndGPTQAreNotImplemented() throws {
        let outDim = 4, inDim = 128
        let url = try makeTempBF16File(outDim: outDim, inDim: inDim, seed: 88)
        defer { try? FileManager.default.removeItem(at: url) }
        let stats = CalibrationStats(
            perChannelAbsMax: [Float](repeating: 1, count: inDim))

        XCTAssertThrowsError(
            try quantizeBF16ToInt8Calibrated(srcURL: url, srcOffset: 0,
                                              outDim: outDim, inDim: inDim,
                                              method: .smoothQuant, stats: stats)
        ) { error in
            XCTAssertTrue(error is QuantNotImplemented)
        }

        XCTAssertThrowsError(
            try quantizeBF16ToInt8Calibrated(srcURL: url, srcOffset: 0,
                                              outDim: outDim, inDim: inDim,
                                              method: .gptq, stats: stats)
        ) { error in
            XCTAssertTrue(error is QuantNotImplemented)
        }
    }

    // MARK: - ActivationObserver

    func testObserverAccumulatesPerChannelAbsMax() {
        let observer = ActivationObserver()
        let inDim = 4
        let rows = 3
        // x = [[1,-2,3,0], [-5,0,1,4], [0,0,0,0]]
        let x: [Float] = [1, -2, 3, 0,
                          -5, 0, 1, 4,
                          0, 0, 0, 0]
        x.withUnsafeBufferPointer { buf in
            observer.recordActivation("layer.0",
                                       buf.baseAddress!,
                                       rows: rows,
                                       inDim: inDim)
        }
        let stats = observer.finalize(for: "layer.0")
        XCTAssertNotNil(stats)
        XCTAssertEqual(stats?.perChannelAbsMax, [5, 2, 3, 4])
        XCTAssertEqual(stats?.observedTokens, rows)
    }

    func testObserverReturnsNilForUnknownLayer() {
        let observer = ActivationObserver()
        XCTAssertNil(observer.finalize(for: "unknown.layer"))
    }

    // MARK: - helpers

    private func randomBF16Bytes(count: Int, seed: UInt64) -> Data {
        var state = seed | 1
        var bytes = Data(count: count * 2)
        bytes.withUnsafeMutableBytes { raw in
            let ptr = raw.bindMemory(to: UInt16.self).baseAddress!
            for i in 0..<count {
                state = state &* 6364136223846793005 &+ 1442695040888963407
                let frac = Float(Double(state >> 11) / Double(1 << 53))
                let v = (frac - 0.5) * 2.0
                let f32bits = v.bitPattern
                // BF16 round-to-nearest-even.
                let rounded = f32bits &+ ((f32bits >> 16) & 1) &+ 0x7FFF
                ptr[i] = UInt16(truncatingIfNeeded: rounded >> 16)
            }
        }
        return bytes
    }
}
