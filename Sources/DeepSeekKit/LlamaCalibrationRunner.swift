import Foundation
import Metal

/// Drives a calibration pass on a `LlamaModel` to populate
/// `ActivationObserver` / `HessianObserver` stats for each Linear
/// whose input we can cleanly intercept (TODO §1 follow-up).
///
/// Why a separate runner instead of a hook inside `Linear.forward`:
/// the GPU's lazy execution model means the bytes of the input
/// tensor aren't materialized until the command buffer commits, so
/// reading them mid-forward requires a `commit + waitUntilCompleted`
/// which silently breaks the caller's batched encoding pattern.
/// Doing the synchronization here, at known sites in the per-layer
/// forward sequence, keeps the cost out of the normal-inference
/// `Linear.callAsFunction` (the calibration runner is a strict
/// offline tool).
///
/// Observed sites per layer:
///   - `blk.{L}.attn_q`, `attn_k`, `attn_v`  → input is `attnNorm(x)`
///   - `blk.{L}.ffn_gate`, `ffn_up`          → input is `ffnNorm(h)`
///
/// Not yet observed (documented limitation):
///   - `blk.{L}.attn_output` (wO) — input is the attention output,
///     internal to `StandardMHA.callAsFunction`. Calibrating it
///     would need either a refactor that exposes the attention
///     output as an intermediate Tensor, or a runner that
///     reimplements the attention pipeline inline. Defer to a
///     follow-up if the wO weight quality turns out to matter.
///   - `blk.{L}.ffn_down` — input is `silu(gate) * up`, internal
///     to `SwiGLU.callAsFunction`. Same fix.
///   - `output` (lm head) — single matmul, the calibration corpus
///     loop already covers everything that flows in here.
///
/// Use:
///
///   let actObs = ActivationObserver()
///   let hessObs = HessianObserver()
///   let runner = LlamaCalibrationRunner(
///       model: model, tokenizer: tokenizer,
///       activation: actObs, hessian: hessObs)
///   for sample in corpus {
///       runner.observe(sample)
///   }
///   let stats = actObs.finalize(for: "blk.0.attn_q")!
///   let hess  = hessObs.finalize(for: "blk.0.attn_q")!
///   // → feed to quantizeBF16ToInt8Calibrated(method: .awq, stats: stats)
///   //   or gptqQuantizeBF16ToInt8(hessian: hess.hessian, ...)
public final class LlamaCalibrationRunner {
    public let model: LlamaModel
    public let tokenizer: any Tokenizer
    public let activationObserver: ActivationObserver?
    public let hessianObserver: HessianObserver?

    /// Cap on the number of tokens per single `observe(_:)` call.
    /// Hessian accumulation costs O(rows × inDim²) per call (the
    /// dgemm rank update), so a single 100k-token sample would
    /// stall the calibration loop forever. The cap chops oversized
    /// inputs into batches of this size internally.
    public var maxTokensPerBatch: Int = 1024

    public init(model: LlamaModel,
                tokenizer: any Tokenizer,
                activation: ActivationObserver? = nil,
                hessian: HessianObserver? = nil)
    {
        self.model = model
        self.tokenizer = tokenizer
        self.activationObserver = activation
        self.hessianObserver = hessian
    }

    /// Tokenize `text` and run a single forward pass with the
    /// observers attached. The model's KV cache is **released**
    /// at the end so the next call starts cold — calibration
    /// samples are independent and the previous KV state would
    /// just bias the stats.
    public func observe(_ text: String) {
        let ids = tokenizer.encode(text)
        guard !ids.isEmpty else { return }
        // Chop to maxTokensPerBatch so big samples don't blow up
        // the Hessian dgemm time.
        var offset = 0
        while offset < ids.count {
            let end = min(offset + maxTokensPerBatch, ids.count)
            let chunk = Array(ids[offset..<end])
            observeChunk(chunk)
            offset = end
            // Each chunk is treated as an independent batch — drop
            // KV state so positional context doesn't carry into
            // the next chunk's stats.
            model.releaseCache()
        }
    }

    /// One forward pass over `tokenChunk`, observing the Linear
    /// inputs at the documented sites.
    private func observeChunk(_ tokenChunk: [Int]) {
        let flatIds = tokenChunk.map(Int32.init)
        var cmd = Device.shared.queue.makeCommandBuffer()!
        let S = tokenChunk.count

        // 1. Embedding
        let embedded = model.embed.lookup(flatIds, in: cmd)
        var x = embedded.reshape([1, S, model.config.hiddenSize])

        // 2. Per-layer: observe at the documented sites, then run
        //    the layer normally.
        for layerId in 0..<model.config.nLayers {
            let layer = model.layers[layerId]
            x = observeLayer(x, layerId: layerId, layer: layer,
                              in: &cmd)
        }

        // 3. Final norm + LM head — same as LlamaModel.forward but
        //    we don't bother reading the logits; the forward is only
        //    here to make sure nothing weird happens with the
        //    cmd-pump pattern.
        let normed = model.norm(x, in: cmd)
        let dim = model.config.hiddenSize
        let lastTok = Tensor.empty(shape: [1, dim], dtype: .f32)
        let blit = cmd.makeBlitCommandEncoder()!
        let bytesPerRow = dim * MemoryLayout<Float>.size
        let src = (S - 1) * bytesPerRow
        blit.copy(from: normed.buffer,
                   sourceOffset: normed.offset + src,
                   to: lastTok.buffer,
                   destinationOffset: 0,
                   size: bytesPerRow)
        blit.endEncoding()
        _ = model.lmHead(lastTok, in: cmd)
        cmd.commit()
        cmd.waitUntilCompleted()
    }

    private func observeLayer(_ x: Tensor,
                               layerId: Int,
                               layer: LlamaDecoderLayer,
                               in cmd: inout MTLCommandBuffer) -> Tensor
    {
        let prefix = "blk.\(layerId)"

        // h = x + attn(attnNorm(x))
        let normedA = layer.attnNorm(x, in: cmd)
        // Sync so normedA has real bytes, then observe.
        cmd.commit(); cmd.waitUntilCompleted()
        feedObservers("\(prefix).attn_q", normedA)
        feedObservers("\(prefix).attn_k", normedA)
        feedObservers("\(prefix).attn_v", normedA)
        cmd = Device.shared.queue.makeCommandBuffer()!

        let attnOut = layer.attn(normedA, startPos: 0, in: &cmd)
        Elementwise.addInPlace(attnOut, x, in: cmd)

        // y = h + ffn(ffnNorm(h))
        let normedF = layer.ffnNorm(attnOut, in: cmd)
        cmd.commit(); cmd.waitUntilCompleted()
        feedObservers("\(prefix).ffn_gate", normedF)
        feedObservers("\(prefix).ffn_up", normedF)
        cmd = Device.shared.queue.makeCommandBuffer()!

        let ffnOut = layer.ffn(normedF, in: cmd)
        Elementwise.addInPlace(ffnOut, attnOut, in: cmd)
        return ffnOut
    }

    /// Push the contents of `t` to whichever observers are
    /// attached. Caller is responsible for the GPU sync — by the
    /// time we get here, `t.buffer` must contain materialized
    /// bytes.
    private func feedObservers(_ name: String, _ t: Tensor) {
        precondition(t.dtype == .f32,
                      "LlamaCalibrationRunner expects f32 activations (got \(t.dtype))")
        let inDim = t.shape.last!
        let rows = t.count / inDim
        let ptr = t.buffer.contents()
            .advanced(by: t.offset)
            .bindMemory(to: Float.self, capacity: t.count)
        activationObserver?.recordActivation(
            name, ptr, rows: rows, inDim: inDim)
        hessianObserver?.recordBatch(
            name, ptr, rows: rows, inDim: inDim)
    }
}
