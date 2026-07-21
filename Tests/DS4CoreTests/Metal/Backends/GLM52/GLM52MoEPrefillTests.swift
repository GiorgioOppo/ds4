import DS4Core
import XCTest
@testable import DS4Metal

/// Fase B del prefill multi-token contro il percorso legacy
/// per-applicazione: su record Q8_0 sintetici (d=1, dequant esatta) i due
/// percorsi devono coincidere BIT-PER-BIT — stessa matematica per token,
/// stessa associatività delle somme — e con l'oracle CPU entro tolleranza.
final class GLM52MoEPrefillTests: XCTestCase {
    private let hiddenWidth = 64
    private let inputWidth = 128
    private let expertCount = 3
    private let tokenCount = 5

    private func makeRuntime() throws -> MetalRuntime {
        do { return try MetalRuntime() }
        catch MetalError.noDevice { throw XCTSkip("Metal device unavailable") }
    }

    private func q8Row(_ values: [Int8]) -> [UInt8] {
        precondition(values.count % 32 == 0)
        var bytes: [UInt8] = []
        for block in stride(from: 0, to: values.count, by: 32) {
            withUnsafeBytes(of: Half.bits(1.0).littleEndian) {
                bytes.append(contentsOf: $0)
            }
            for value in values[block..<block + 32] {
                bytes.append(UInt8(bitPattern: value))
            }
        }
        return bytes
    }

    private func silu(_ x: Float) -> Float { x / (1 + exp(-x)) }

    private struct Fixture {
        var staged: GLM52StagedExpertSelection
        var applications: [(slot: Int, token: Int, weight: Float)]
        var gates: [[Int8]]
        var ups: [[Int8]]
        var downs: [[Int8]]
        var ffnIn: [Float]
        var hidden0: [Float]
    }

    private func makeFixture(runtime: MetalRuntime) throws -> Fixture {
        var state: UInt64 = 11
        func nextInt8() -> Int8 {
            state = state &* 6_364_136_223_846_793_005
                &+ 1_442_695_040_888_963_407
            return Int8(truncatingIfNeeded: Int8(
                clamping: Int(truncatingIfNeeded: state >> 40) % 5 - 2))
        }
        var gates: [[Int8]] = []
        var ups: [[Int8]] = []
        var downs: [[Int8]] = []
        var records: [UInt8] = []
        var offsets: [Int] = []
        let gateBytes = hiddenWidth * (inputWidth / 32 * 34)
        for _ in 0..<expertCount {
            offsets.append(records.count)
            let gate = (0..<hiddenWidth * inputWidth).map { _ in nextInt8() }
            let up = (0..<hiddenWidth * inputWidth).map { _ in nextInt8() }
            let down = (0..<inputWidth * hiddenWidth).map { _ in nextInt8() }
            gates.append(gate); ups.append(up); downs.append(down)
            for row in 0..<hiddenWidth {
                records += q8Row(Array(
                    gate[row * inputWidth..<(row + 1) * inputWidth]))
            }
            for row in 0..<hiddenWidth {
                records += q8Row(Array(
                    up[row * inputWidth..<(row + 1) * inputWidth]))
            }
            for row in 0..<inputWidth {
                records += q8Row(Array(
                    down[row * hiddenWidth..<(row + 1) * hiddenWidth]))
            }
        }
        // Piani per token: ffnIn e hidden di partenza distinti per token.
        var ffnIn: [Float] = []
        var hidden0: [Float] = []
        for token in 0..<tokenCount {
            ffnIn += (0..<inputWidth).map {
                Float(($0 + token * 3) % 7) * 0.25 - 0.75
            }
            hidden0 += (0..<inputWidth).map {
                Float(($0 + token) % 5) * 0.1
            }
        }
        // Expert-major come la fase A: esperto 0 su token {0,1,2,4},
        // esperto 1 su {1,3}, esperto 2 su {0,2,3,4} — pesi distinti.
        let applications: [(slot: Int, token: Int, weight: Float)] = [
            (0, 0, 0.50), (0, 1, 0.40), (0, 2, 0.30), (0, 4, 0.20),
            (1, 1, 0.35), (1, 3, 0.25),
            (2, 0, 0.15), (2, 2, 0.45), (2, 3, 0.10), (2, 4, 0.55),
        ]
        let staged = GLM52StagedExpertSelection(
            buffer: try runtime.glm52GraphBuffer(records),
            recordOffsets: offsets,
            gateBytes: gateBytes, upBytes: gateBytes,
            downBytes: inputWidth * (hiddenWidth / 32 * 34),
            gateUpType: GLM52TensorSchema.q8_0,
            downType: GLM52TensorSchema.q8_0)
        return Fixture(staged: staged, applications: applications,
                       gates: gates, ups: ups, downs: downs,
                       ffnIn: ffnIn, hidden0: hidden0)
    }

    private func runPath(runtime: MetalRuntime, fixture: Fixture,
                         legacy: Bool) throws -> [Float] {
        let geometry = GLM52DecodeGeometry(
            layer: GLM52LayerGeometry(
                embeddingWidth: inputWidth, headCount: 2, kvLoraRank: 16,
                ropeDimension: 8, valueDimension: 12, denseHiddenWidth: 64,
                expertHiddenWidth: hiddenWidth, expertsUsed: 2),
            qLoraRank: 24, nopeDimension: 24,
            indexerHeadCount: 4, indexerHeadDimension: 16,
            indexerRotationDimension: 8, indexerTopK: 4)
        let scratch = try GLM52DecodeScratch(
            runtime: runtime, geometry: geometry, scoreCapacity: 8)
        let hiddenAll = try runtime.glm52GraphBuffer(fixture.hidden0)
        let ffnInAll = try runtime.glm52GraphBuffer(fixture.ffnIn)
        if legacy {
            try runtime.glm52ApplyRoutedExpertsLegacy(
                staged: fixture.staged,
                applications: fixture.applications,
                hiddenAll: hiddenAll, ffnInAll: ffnInAll, scratch: scratch,
                embeddingWidth: inputWidth, expertHiddenWidth: hiddenWidth)
        } else {
            try runtime.glm52ApplyRoutedExperts(
                staged: fixture.staged,
                applications: fixture.applications,
                hiddenAll: hiddenAll, ffnInAll: ffnInAll, scratch: scratch,
                embeddingWidth: inputWidth, expertHiddenWidth: hiddenWidth)
        }
        return runtime.glm52GraphReadback(hiddenAll,
                                          count: tokenCount * inputWidth)
    }

    func testMultiTokenPathMatchesLegacyBitExact() throws {
        let runtime = try makeRuntime()
        let fixture = try makeFixture(runtime: runtime)
        let legacy = try runPath(runtime: runtime, fixture: fixture,
                                 legacy: true)
        let multi = try runPath(runtime: runtime, fixture: fixture,
                                legacy: false)
        XCTAssertEqual(legacy.count, multi.count)
        for i in 0..<legacy.count {
            XCTAssertEqual(legacy[i].bitPattern, multi[i].bitPattern,
                           "hidden[\(i)] diverge dal percorso legacy")
        }
    }

    func testMultiWaveSplitStaysBitExact() throws {
        let runtime = try makeRuntime()
        let fixture = try makeFixture(runtime: runtime)
        let legacy = try runPath(runtime: runtime, fixture: fixture,
                                 legacy: true)
        let saved = GLM52PrefillMoEDispatch.waveCap
        defer { GLM52PrefillMoEDispatch.waveCap = saved }
        // Cap 3: spezza dentro l'esperto 0 e dentro l'esperto 2.
        GLM52PrefillMoEDispatch.waveCap = 3
        let multi = try runPath(runtime: runtime, fixture: fixture,
                                legacy: false)
        for i in 0..<legacy.count {
            XCTAssertEqual(legacy[i].bitPattern, multi[i].bitPattern,
                           "hidden[\(i)] diverge con wave spezzate")
        }
    }

    func testMultiTokenPathMatchesCPUOracle() throws {
        let runtime = try makeRuntime()
        let fixture = try makeFixture(runtime: runtime)
        let gpu = try runPath(runtime: runtime, fixture: fixture,
                              legacy: false)
        var expected = fixture.hidden0
        for application in fixture.applications {
            let e = application.slot
            let x = Array(fixture.ffnIn[
                application.token * inputWidth..<(application.token + 1)
                    * inputWidth])
            var mid = [Float](repeating: 0, count: hiddenWidth)
            for r in 0..<hiddenWidth {
                var g: Float = 0, u: Float = 0
                for k in 0..<inputWidth {
                    g += Float(fixture.gates[e][r * inputWidth + k]) * x[k]
                    u += Float(fixture.ups[e][r * inputWidth + k]) * x[k]
                }
                mid[r] = silu(g) * u * application.weight
            }
            for r in 0..<inputWidth {
                var acc: Float = 0
                for k in 0..<hiddenWidth {
                    acc += Float(fixture.downs[e][r * hiddenWidth + k])
                        * mid[k]
                }
                expected[application.token * inputWidth + r] += acc
            }
        }
        for i in 0..<expected.count {
            XCTAssertEqual(gpu[i], expected[i], accuracy: 5e-2,
                           "hidden[\(i)] fuori tolleranza dall'oracle")
        }
    }
}
