import Foundation
import Metal

/// Top-level DeepSeek-V4 model. The constructor below is the assembly entry
/// point — populate the layers from a `SafeTensorsFile` once the weight
/// naming convention is confirmed.
public final class DeepSeekV4 {
    public let config: ModelConfig
    public let embedTokens: Tensor          // [vocab, hidden]
    public let layers: [DecoderLayer]
    public let finalNorm: RMSNorm
    public let lmHead: Linear               // weight tied to embedTokens unless config says otherwise

    public init(config: ModelConfig,
                embedTokens: Tensor,
                layers: [DecoderLayer],
                finalNorm: RMSNorm,
                lmHead: Linear) {
        self.config = config
        self.embedTokens = embedTokens
        self.layers = layers
        self.finalNorm = finalNorm
        self.lmHead = lmHead
    }

    /// Forward a single new token. Returns logits of shape [1, vocab].
    public func step(tokenId: Int, cache: CacheBank) -> Tensor {
        let cmd = Device.shared.queue.makeCommandBuffer()!
        var x = embedRow(tokenId)
        for (i, layer) in layers.enumerated() {
            x = layer(x, cache: cache.layers[i], in: cmd)
        }
        let h = finalNorm(x, in: cmd)
        let logits = lmHead(h, in: cmd)
        cmd.commit(); cmd.waitUntilCompleted()
        return logits
    }

    private func embedRow(_ tokenId: Int) -> Tensor {
        // Copy one embedding row into a fresh f32 buffer.
        let dim = config.hiddenSize
        let dst = Tensor.empty(shape: [1, dim], dtype: .f32)
        let cmd = Device.shared.queue.makeCommandBuffer()!
        let blit = cmd.makeBlitCommandEncoder()!
        let srcOffset = embedTokens.offset + tokenId * dim * (embedTokens.dtype.bitsPerElement / 8)
        let bytes = dim * (embedTokens.dtype.bitsPerElement / 8)
        if embedTokens.dtype == .f32 {
            blit.copy(from: embedTokens.buffer, sourceOffset: srcOffset,
                      to: dst.buffer, destinationOffset: 0, size: bytes)
        } else {
            // For non-f32 embedding tables we need a dequant kernel; fall back to
            // host conversion for now so the pipeline runs end-to-end.
            blit.endEncoding()
            cmd.commit(); cmd.waitUntilCompleted()
            let row = embedTokens.toFloatArray()
            let start = tokenId * dim
            let slice = Array(row[start..<start+dim])
            slice.withUnsafeBufferPointer { p in
                memcpy(dst.buffer.contents(), p.baseAddress, dim * MemoryLayout<Float>.size)
            }
            return dst
        }
        blit.endEncoding()
        cmd.commit(); cmd.waitUntilCompleted()
        return dst
    }
}
