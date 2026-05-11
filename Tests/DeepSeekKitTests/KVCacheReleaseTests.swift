import XCTest
import Metal
@testable import DeepSeekKit

/// Lifecycle tests for the lazy-realloc KV state on `Compressor` (and,
/// transitively, on `MLA`). Verifies that `releaseState()` drops the
/// underlying `MTLBuffer` strong reference and that the next forward
/// call re-allocates a fresh buffer.
final class KVCacheReleaseTests: XCTestCase {

    /// `releaseState()` nils all three state properties (`kvState`,
    /// `scoreState`, and the `kvCache` alias). This validates the
    /// API surface without exercising any kernels.
    func testCompressorReleaseStateClearsProperties() {
        let comp = Self.makeCompressor(ratio: 128, B: 1)
        XCTAssertNotNil(comp.kvState)
        XCTAssertNotNil(comp.scoreState)

        comp.releaseState()
        XCTAssertNil(comp.kvState, "releaseState() should drop kvState")
        XCTAssertNil(comp.scoreState, "releaseState() should drop scoreState")
        XCTAssertNil(comp.kvCache, "releaseState() should drop kvCache alias")
    }

    /// After `releaseState()`, the next decode call must lazy-reallocate
    /// the state buffers. We pick `startPos = 1` with `ratio = 128` so
    /// `shouldEmit = (1+1) % 128 == 0` is false: the decode path runs
    /// `ensureKVState()` / `ensureScoreState()` and the row blit, then
    /// returns nil — avoiding the heavier pooling/softmax path which
    /// would require `rope` to be wired up. The buffer-identity check
    /// proves we got a fresh allocation, not the original one.
    func testCompressorReleaseStateThenDecodeReallocates() throws {
        let ratio = 128
        let B = 1
        let comp = Self.makeCompressor(ratio: ratio, B: B)
        guard let firstKvBuffer = comp.kvState?.buffer,
              let firstScoreBuffer = comp.scoreState?.buffer else {
            return XCTFail("Compressor should start with allocated state")
        }

        comp.releaseState()
        XCTAssertNil(comp.kvState)
        XCTAssertNil(comp.scoreState)

        let dim = ModelConfig().dim
        let x = Tensor.empty(shape: [B, 1, dim], dtype: .f32)
        let cmd = Device.shared.queue.makeCommandBuffer()!
        let result = comp(x, startPos: 1, in: cmd)
        cmd.commit(); cmd.waitUntilCompleted()
        XCTAssertNil(result, "decode at startPos=1, ratio=128 should not emit")

        XCTAssertNotNil(comp.kvState, "ensureKVState() should have re-allocated")
        XCTAssertNotNil(comp.scoreState, "ensureScoreState() should have re-allocated")
        XCTAssertFalse(comp.kvState!.buffer === firstKvBuffer,
                       "kvState must point to a freshly-allocated MTLBuffer")
        XCTAssertFalse(comp.scoreState!.buffer === firstScoreBuffer,
                       "scoreState must point to a freshly-allocated MTLBuffer")
    }

    // MARK: - helpers

    private static func makeCompressor(ratio: Int, B: Int) -> Compressor {
        let cfg = ModelConfig()
        let dim = cfg.dim
        let headDim = 8
        // overlap is derived from ratio==4 inside Compressor.init; for
        // ratio=128 it's false so coff=1.
        let coff = 1
        let coffHeadDim = coff * headDim

        let ape = Tensor.empty(shape: [ratio, coffHeadDim], dtype: .f32)
        let wkvT = Tensor.empty(shape: [coffHeadDim, dim], dtype: .f32)
        let wgateT = Tensor.empty(shape: [coffHeadDim, dim], dtype: .f32)
        let normW = Tensor.empty(shape: [headDim], dtype: .f32)
        let kvState = Tensor.empty(shape: [B, ratio, coffHeadDim], dtype: .f32)
        let scoreState = Tensor.empty(shape: [B, ratio, coffHeadDim], dtype: .f32)

        let wkv = Linear(inFeatures: dim, outFeatures: coffHeadDim, weight: wkvT, scale: nil)
        let wgate = Linear(inFeatures: dim, outFeatures: coffHeadDim, weight: wgateT, scale: nil)
        let norm = RMSNorm(weight: normW, eps: cfg.normEps)

        return Compressor(config: cfg, compressRatio: ratio, headDim: headDim,
                          rotate: false,
                          ape: ape, wkv: wkv, wgate: wgate, norm: norm,
                          kvState: kvState, scoreState: scoreState)
    }
}
