import XCTest
import Foundation
import DS4Core
@testable import DS4Metal

/// Device-free coverage of the engine's CPU helpers plus an opt-in
/// real-weights smoke test.
final class LagunaResidentModelTests: XCTestCase {
    func testRuntimeKnobsAreClampedToSupportedRanges() {
        XCTAssertEqual(LagunaResidentModel.resolvedActiveExperts(nil), 10)
        XCTAssertEqual(LagunaResidentModel.resolvedActiveExperts(0), 1)
        XCTAssertEqual(LagunaResidentModel.resolvedActiveExperts(6), 6)
        XCTAssertEqual(LagunaResidentModel.resolvedActiveExperts(99), 10)

        XCTAssertEqual(LagunaResidentModel.resolvedPrefillChunk(nil), 256)
        XCTAssertEqual(LagunaResidentModel.resolvedPrefillChunk(0), 1)
        XCTAssertEqual(LagunaResidentModel.resolvedPrefillChunk(2_048),
                       1_024)
        XCTAssertEqual(LagunaResidentModel.resolvedPreadSplit(0), 1)
        XCTAssertEqual(LagunaResidentModel.resolvedPreadSplit(99), 8)
        XCTAssertEqual(LagunaResidentModel.resolvedSimdgroups(nil), 4)
        XCTAssertEqual(LagunaResidentModel.resolvedSimdgroups(99), 8)
        XCTAssertEqual(
            LagunaResidentModel.resolvedLongAttentionBlockSize(nil), 16)
        XCTAssertEqual(
            LagunaResidentModel.resolvedLongAttentionBlockSize(7), 8)
        XCTAssertEqual(
            LagunaResidentModel.resolvedLongAttentionBlockSize(80), 64)
        XCTAssertEqual(
            LagunaResidentModel.resolvedLongAttentionTopBlocks(nil), 32)
        XCTAssertEqual(
            LagunaResidentModel.resolvedLongAttentionTopBlocks(0), 1)
        XCTAssertEqual(
            LagunaResidentModel.resolvedLongAttentionTopBlocks(999), 128)
        XCTAssertEqual(
            LagunaResidentModel.resolvedLongAttentionRecentTokens(nil), 512)
        XCTAssertEqual(
            LagunaResidentModel.resolvedLongAttentionRecentTokens(1), 128)
    }

    func testQ8RowDequantizationMatchesTheEncoder() throws {
        // Encode two known rows with the shared Q8_0 encoder, then read row 1
        // back through the engine's embedding-row dequantizer.
        let width = 64
        let rows: [[Float]] = [
            (0..<width).map { Float($0) / 17 - 1.5 },
            (0..<width).map { Float(width - $0) / 23 + 0.25 },
        ]
        let rowBytes = QuantEncode.rowSize(type: 8, columns: width)
        var payload = [UInt8](repeating: 0, count: rowBytes * rows.count)
        let flattened = rows.flatMap { $0 }
        let written = flattened.withUnsafeBufferPointer { source in
            payload.withUnsafeMutableBufferPointer { destination in
                QuantEncode.quantizeChunk(
                    type: 8,
                    src: source.baseAddress!,
                    dst: UnsafeMutableRawPointer(destination.baseAddress!),
                    start: 0,
                    rows: rows.count,
                    columns: width,
                    imatrix: nil
                )
            }
        }
        XCTAssertEqual(written, payload.count)

        let decoded = try payload.withUnsafeBytes {
            try LagunaResidentModel.dequantizeQ8Row(
                base: $0.baseAddress!, row: 1, rowCount: rows.count,
                width: width
            )
        }
        // Q8_0 stores an F16 scale and int8 quants: reconstruction is close,
        // not exact.
        for i in 0..<width {
            XCTAssertEqual(decoded[i], rows[1][i],
                           accuracy: 0.02 + abs(rows[1][i]) * 0.02, "col \(i)")
        }
    }

    func testQ8RowDequantizationRejectsOutOfRangeRows() {
        // Kept as an explicit do/catch inside the pointer scope: `try` in an
        // XCTAssertThrowsError autoclosure makes the compiler infer a throwing
        // closure for the rethrows `withUnsafeBytes`.
        let bytes = [UInt8](repeating: 0, count: 34)
        let rejected = bytes.withUnsafeBytes { raw -> Bool in
            do {
                _ = try LagunaResidentModel.dequantizeQ8Row(
                    base: raw.baseAddress!, row: 1, rowCount: 1, width: 32
                )
                return false
            } catch {
                return true
            }
        }
        XCTAssertTrue(rejected, "row 1 of a 1-row table must be rejected")
    }

    /// Opt-in smoke test on real weights: set DS4_LAGUNA_GGUF to the official
    /// Q4_K_M file. Loads a truncated stack, runs two decode steps and checks
    /// the logits are finite and the position advances. Numerical parity
    /// against the reference C engine is the separate gate documented in
    /// PORTING-GAPS.
    func testRealWeightsSmokeIfProvided() throws {
        guard let path = ProcessInfo.processInfo.environment["DS4_LAGUNA_GGUF"],
              FileManager.default.fileExists(atPath: path) else {
            throw XCTSkip("set DS4_LAGUNA_GGUF to run the Laguna engine smoke test")
        }
        let runtime: MetalRuntime
        do { runtime = try MetalRuntime() }
        catch { throw XCTSkip("Metal unavailable: \(error)") }

        var options = LagunaResidentModelOptions()
        options.cacheCapacity = 64
        options.layerCount = Int(
            ProcessInfo.processInfo.environment["DS4_LAGUNA_LAYERS"] ?? ""
        ) ?? 2
        let engine = try LagunaResidentModel(runtime: runtime, path: path,
                                             options: options)
        XCTAssertGreaterThan(engine.loadedLayerCount, 0)

        let first = try engine.forwardNext(1)
        XCTAssertEqual(first.count, 100_352)
        XCTAssertTrue(first.allSatisfy(\.isFinite))
        XCTAssertEqual(engine.position, 1)

        let second = try engine.forwardNext(2)
        XCTAssertTrue(second.allSatisfy(\.isFinite))
        XCTAssertEqual(engine.position, 2)

        // Native disk-KV round trip: restoring the exact F16 planes must
        // reproduce the next-token logits bit-for-bit.
        let checkpoint = FileManager.default.temporaryDirectory
            .appendingPathComponent("laguna-engine-\(UUID().uuidString).lkv")
        defer { try? FileManager.default.removeItem(at: checkpoint) }
        try engine.saveKVCheckpoint(to: checkpoint, tokens: [1, 2])
        let expected = try engine.forwardNext(3)
        engine.resetContext(releaseExcessKV: true)
        XCTAssertEqual(
            try engine.restoreKVCheckpoint(from: checkpoint), [1, 2])
        let restored = try engine.forwardNext(3)
        XCTAssertEqual(restored, expected)
    }
}
