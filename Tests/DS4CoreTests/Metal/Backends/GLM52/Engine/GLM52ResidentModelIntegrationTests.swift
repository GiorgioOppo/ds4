import XCTest
import DS4Core
@testable import DS4Metal

/// Opt-in engine smoke over a real (or exact-size sparse) GLM 5.2 GGUF: load
/// the three leading DENSE layers — Q8_0 end to end, no routed experts — as
/// a resident stack, run a real prefill token by token and a short greedy
/// decode. This proves loader, resident upload, cache growth, embedding-row
/// reads and the generation loop against the published tensor layout; it is
/// NOT a quality gate (a truncated stack has no meaningful logits — the
/// full-model logits parity is the roadmap's separate, binding gate).
final class GLM52ResidentModelIntegrationTests: XCTestCase {
    func testDenseLayerEngineSmokeOnRealLayout() throws {
        guard let path = ProcessInfo.processInfo
                  .environment["DS4_GLM52_SPARSE_GGUF"], !path.isEmpty else {
            throw XCTSkip("set DS4_GLM52_SPARSE_GGUF to a real or exact-size "
                          + "sparse GLM 5.2 GGUF")
        }
        let runtime: MetalRuntime
        do { runtime = try MetalRuntime() }
        catch MetalError.noDevice { throw XCTSkip("Metal device unavailable") }

        let model = try GLM52ResidentModel(
            runtime: runtime, path: path,
            options: GLM52ResidentModelOptions(
                layerCount: 3, cacheCapacity: 16, expertSlotCount: 16))
        XCTAssertEqual(model.loadedLayerCount, 3)
        XCTAssertEqual(model.configuration.shape, .v5_2)

        let embedded = try model.embeddingRow(9_333)
        XCTAssertEqual(embedded.count, 6_144)

        let logits = try model.prefill([9_333, 21, 42])
        XCTAssertEqual(logits.count, 154_880)
        XCTAssertEqual(model.position, 3)
        XCTAssertTrue(logits.allSatisfy(\.isFinite),
                      "prefill produced non-finite logits")

        // Prompt token (position 4) + one step for the second sample: the
        // final sampled token is never fed back, so the position lands at 5.
        let generated = try model.generateGreedy(
            prompt: [7], maxNewTokens: 2, endTokens: [])
        XCTAssertEqual(generated.count, 2)
        XCTAssertEqual(model.position, 5)

        XCTAssertThrowsError(try model.embeddingRow(-1))
        XCTAssertThrowsError(try model.embeddingRow(154_880))
    }

    /// Streamed-vs-resident equivalence on REAL weights: the same 4-layer
    /// stack once fully resident and once with the sparse layer streamed
    /// from SSD through the double-buffered slots must produce the same
    /// logits — streaming is a memory strategy, never a numeric one.
    func testStreamedTailMatchesResidentOnRealLayout() throws {
        guard let path = ProcessInfo.processInfo
                  .environment["DS4_GLM52_SPARSE_GGUF"], !path.isEmpty else {
            throw XCTSkip("set DS4_GLM52_SPARSE_GGUF to a real or exact-size "
                          + "sparse GLM 5.2 GGUF")
        }
        let runtime: MetalRuntime
        do { runtime = try MetalRuntime() }
        catch MetalError.noDevice { throw XCTSkip("Metal device unavailable") }

        let resident = try GLM52ResidentModel(
            runtime: runtime, path: path,
            options: GLM52ResidentModelOptions(
                layerCount: 4, cacheCapacity: 8))
        let streamed = try GLM52ResidentModel(
            runtime: runtime, path: path,
            options: GLM52ResidentModelOptions(
                layerCount: 4, cacheCapacity: 8, residentLayerCount: 3))

        for token in [Int32(154_822), 9_333, 21] {
            let a = try resident.forwardNext(token)
            let b = try streamed.forwardNext(token)
            XCTAssertEqual(a.count, b.count)
            for i in 0..<a.count {
                XCTAssertEqual(a[i], b[i], accuracy: 1e-4 + abs(a[i]) * 1e-4,
                               "streamed logits diverge at \(i)")
            }
        }
    }

    /// Layer-major batched prefill vs the token-by-token path on REAL
    /// weights: identical kernels per (layer, token) cell in the same causal
    /// order — the logits must match. Batch is an I/O strategy, never a
    /// numeric one.
    func testBatchedPrefillMatchesSequentialOnRealLayout() throws {
        guard let path = ProcessInfo.processInfo
                  .environment["DS4_GLM52_SPARSE_GGUF"], !path.isEmpty else {
            throw XCTSkip("set DS4_GLM52_SPARSE_GGUF to a real or exact-size "
                          + "sparse GLM 5.2 GGUF")
        }
        let runtime: MetalRuntime
        do { runtime = try MetalRuntime() }
        catch MetalError.noDevice { throw XCTSkip("Metal device unavailable") }

        let options = GLM52ResidentModelOptions(
            layerCount: 4, cacheCapacity: 8)
        let sequential = try GLM52ResidentModel(
            runtime: runtime, path: path, options: options)
        let batched = try GLM52ResidentModel(
            runtime: runtime, path: path, options: options)
        let prompt: [Int32] = [154_822, 9_333, 21]

        var sequentialLogits: [Float] = []
        for token in prompt {
            sequentialLogits = try sequential.forwardNext(token)
        }
        let batchedLogits = try batched.prefill(prompt)

        XCTAssertEqual(batched.position, prompt.count)
        XCTAssertEqual(batchedLogits.count, sequentialLogits.count)
        for i in 0..<batchedLogits.count {
            XCTAssertEqual(batchedLogits[i], sequentialLogits[i],
                           accuracy: 1e-4 + abs(sequentialLogits[i]) * 1e-4,
                           "batched prefill diverges at \(i)")
        }
    }
}
