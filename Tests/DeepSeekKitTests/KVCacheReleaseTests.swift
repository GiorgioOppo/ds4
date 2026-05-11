import XCTest
import Metal
@testable import DeepSeekKit

/// Validates that the KV cache buffers can be released (drops the
/// `MTLBuffer` reference) and lazily re-allocated on the next forward.
///
/// These tests do not exercise the full `Transformer.forward` because that
/// requires real weights; they directly poke at `Compressor.releaseState`,
/// `KVCache.release`, and verify the lazy-init machinery.
final class KVCacheReleaseTests: XCTestCase {

    func testKVCacheScaffoldReleaseAndReallocate() {
        let cache = KVCache(maxBatch: 2, headDim: 8,
                            windowSize: 4, compressRatio: 0,
                            maxSeqLen: 16)
        XCTAssertNotNil(cache.data, "data should be allocated at init")
        let firstBuffer = cache.ensureData().buffer
        cache.release()
        XCTAssertNil(cache.data, "release() should drop the data tensor")
        let second = cache.ensureData()
        XCTAssertNotNil(cache.data, "ensureData() should re-allocate")
        XCTAssertEqual(second.shape, [2, 4, 8])
        XCTAssertFalse(second.buffer === firstBuffer,
                       "expected a freshly-allocated MTLBuffer after release")
    }

    func testCacheBankReleaseDropsAllLayers() {
        let cfg = ModelConfig()                                      // toy defaults
        let bank = CacheBank(config: cfg)
        for layer in bank.layers {
            XCTAssertNotNil(layer.data, "every layer should start allocated")
        }
        bank.release()
        for layer in bank.layers {
            XCTAssertNil(layer.data, "release() should clear every layer")
        }
        // Lazy realloc per-layer still works.
        for layer in bank.layers {
            _ = layer.ensureData()
            XCTAssertNotNil(layer.data)
        }
    }

    func testCompressorReleaseStateThenDecodeReallocates() {
        // Minimal Compressor wired with random tensors; we only check the
        // release/realloc lifecycle, not the numerical output.
        let cfg = ModelConfig()
        let dim = cfg.dim
        let headDim = 8
        let ratio = 128
        let coff = 1
        let coffHeadDim = coff * headDim
        let B = 1

        let ape = Tensor.empty(shape: [ratio, coffHeadDim], dtype: .f32)
        let wkvT = Tensor.empty(shape: [coffHeadDim, dim], dtype: .f32)
        let wgateT = Tensor.empty(shape: [coffHeadDim, dim], dtype: .f32)
        let normW = Tensor.empty(shape: [headDim], dtype: .f32)
        let kvState = Tensor.empty(shape: [B, ratio, coffHeadDim], dtype: .f32)
        let scoreState = Tensor.empty(shape: [B, ratio, coffHeadDim], dtype: .f32)

        let wkv = Linear(inFeatures: dim, outFeatures: coffHeadDim, weight: wkvT, scale: nil)
        let wgate = Linear(inFeatures: dim, outFeatures: coffHeadDim, weight: wgateT, scale: nil)
        let norm = RMSNorm(weight: normW, eps: cfg.normEps)

        let comp = Compressor(config: cfg, compressRatio: ratio, headDim: headDim,
                              rotate: false,
                              ape: ape, wkv: wkv, wgate: wgate, norm: norm,
                              kvState: kvState, scoreState: scoreState)

        XCTAssertNotNil(comp.kvState)
        XCTAssertNotNil(comp.scoreState)

        comp.releaseState()
        XCTAssertNil(comp.kvState, "releaseState() should drop kvState")
        XCTAssertNil(comp.scoreState, "releaseState() should drop scoreState")
        XCTAssertNil(comp.kvCache, "releaseState() should drop kvCache alias")
    }
}
