import Foundation
import Metal

/// Calibration runner for the DeepSeek-V4 (MLA + MoE)
/// architecture (TODO §1 follow-up). Mirror of
/// `LlamaCalibrationRunner` but for `Transformer` instead of
/// `LlamaModel`: walks the calibration corpus through
/// `Transformer.forward`, sets `Block.preAttnObserver` /
/// `preFfnObserver` so every block emits its post-norm activation
/// to the observers we attach, and tags those activations with
/// the V4-side tensor names so the stats line up with what the
/// converter sees post-rename.
///
/// Observed sites per block `L`:
///
///   - `yNorm` (input to MLA's wq_a / wkv) →
///       `layers.{L}.attn.wq_a`
///       `layers.{L}.attn.wkv`
///       `layers.{L}.attn.wkv_a`
///     The three names tag the same observation — different
///     HF→V4 converter outputs use different ones (some preserve
///     `wkv_a`/`wkv_b`, some collapse to `wkv`); we emit all
///     synonyms so the converter's per-tensor lookup hits.
///
///   - `yNorm2` (input to FFN expert gates / up projections,
///     pre-routing) →
///       `layers.{L}.ffn.experts.{E}.w1` for every E in
///                                        `[0, nRoutedExperts)`
///       `layers.{L}.ffn.experts.{E}.w3` for every E
///     Pre-routing means every expert gets the same per-channel
///     stats — a documented approximation (router-aware
///     calibration would need to instrument `MoEFFN` per-token,
///     substantial follow-up).
///     Also tags `layers.{L}.ffn.shared_experts.w1` / `w3` when
///     the config carries `nSharedExperts > 0`.
///
/// Not observed (documented limitation):
///   - `attn.wq_b` (qNorm output), `attn.wo_a`, `attn.wo_b`
///     (attention output). Internal to MLA; surfacing them would
///     need similar hook plumbing inside `MLA.callAsFunction`.
///   - Expert `w2` (down projection). Its input is the SwiGLU
///     body output, internal to `MoEFFN`.
///
/// The hooks rotate `cmd` (via `inout`) after each
/// `commit + waitUntilCompleted`, so `Transformer.forward`'s
/// per-block buffer-pump pattern keeps working.
public final class V4CalibrationRunner {
    public let model: Transformer
    public let tokenizer: any Tokenizer
    public let activationObserver: ActivationObserver?
    public let hessianObserver: HessianObserver?

    /// Cap on tokens per single `observe(_:)` batch (default 1024).
    /// Hessian accumulation is O(rows · inDim²) so a giant sample
    /// would dominate the calibration sweep.
    public var maxTokensPerBatch: Int = 1024

    public init(model: Transformer,
                tokenizer: any Tokenizer,
                activation: ActivationObserver? = nil,
                hessian: HessianObserver? = nil)
    {
        self.model = model
        self.tokenizer = tokenizer
        self.activationObserver = activation
        self.hessianObserver = hessian
    }

    /// Tokenize + observe one calibration sample. Chops oversized
    /// inputs into `maxTokensPerBatch` chunks and releases the KV
    /// cache between chunks so positional context doesn't bias the
    /// stats.
    public func observe(_ text: String) {
        let ids = tokenizer.encode(text)
        guard !ids.isEmpty else { return }
        var offset = 0
        while offset < ids.count {
            let end = min(offset + maxTokensPerBatch, ids.count)
            let chunk = Array(ids[offset..<end])
            observeChunk(chunk)
            offset = end
            // KV cache lives inside MLA; drop it via the block
            // wrappers so each chunk starts cold.
            for block in model.layers {
                block.attn.releaseCache()
            }
        }
    }

    private func observeChunk(_ tokenChunk: [Int]) {
        // Install hooks onto every Block before running forward.
        // The hook captures `self.activationObserver` /
        // `self.hessianObserver` to push observations to.
        let nExperts = model.config.nRoutedExperts
        let nShared = model.config.nSharedExperts
        let actObs = self.activationObserver
        let hessObs = self.hessianObserver

        for (layerId, block) in model.layers.enumerated() {
            block.preAttnObserver = { yNorm, cmd in
                cmd.commit(); cmd.waitUntilCompleted()
                let inDim = yNorm.shape.last!
                let rows = yNorm.count / inDim
                let ptr = yNorm.buffer.contents()
                    .advanced(by: yNorm.offset)
                    .bindMemory(to: Float.self, capacity: yNorm.count)
                for name in ["layers.\(layerId).attn.wq_a",
                              "layers.\(layerId).attn.wkv",
                              "layers.\(layerId).attn.wkv_a"] {
                    actObs?.recordActivation(name, ptr,
                                              rows: rows, inDim: inDim)
                    hessObs?.recordBatch(name, ptr,
                                          rows: rows, inDim: inDim)
                }
                cmd = Device.shared.queue.makeCommandBuffer()!
            }
            block.preFfnObserver = { yNorm2, cmd in
                cmd.commit(); cmd.waitUntilCompleted()
                let inDim = yNorm2.shape.last!
                let rows = yNorm2.count / inDim
                let ptr = yNorm2.buffer.contents()
                    .advanced(by: yNorm2.offset)
                    .bindMemory(to: Float.self, capacity: yNorm2.count)
                // One per expert, plus optional shared experts.
                // Pre-routing approximation — every expert gets the
                // same per-channel statistics. See the class doc
                // for the caveat.
                for e in 0..<nExperts {
                    for proj in ["w1", "w3"] {
                        let name = "layers.\(layerId).ffn.experts.\(e).\(proj)"
                        actObs?.recordActivation(name, ptr,
                                                  rows: rows, inDim: inDim)
                        hessObs?.recordBatch(name, ptr,
                                              rows: rows, inDim: inDim)
                    }
                }
                if nShared > 0 {
                    for proj in ["w1", "w3"] {
                        let name = "layers.\(layerId).ffn.shared_experts.\(proj)"
                        actObs?.recordActivation(name, ptr,
                                                  rows: rows, inDim: inDim)
                        hessObs?.recordBatch(name, ptr,
                                              rows: rows, inDim: inDim)
                    }
                }
                cmd = Device.shared.queue.makeCommandBuffer()!
            }
        }

        // Run forward. Returns logits but we discard them; the
        // observers caught what we wanted on the way through.
        _ = model.forward(inputIds: [tokenChunk], startPos: 0)

        // Detach hooks so the next chunk's observers don't double-
        // attach (and so the blocks return to their normal
        // zero-overhead state when the runner is done).
        for block in model.layers {
            block.preAttnObserver = nil
            block.preFfnObserver = nil
        }
    }

    /// Enumeration of every tag name the runner will emit, in the
    /// order the converter would encounter them. Useful for the
    /// CLI to drive the on-disk output without re-walking the
    /// observer's internal map.
    public func tagNames() -> [String] {
        var out: [String] = []
        let nExperts = model.config.nRoutedExperts
        let nShared = model.config.nSharedExperts
        for L in 0..<model.config.nLayers {
            out.append("layers.\(L).attn.wq_a")
            out.append("layers.\(L).attn.wkv")
            out.append("layers.\(L).attn.wkv_a")
            for e in 0..<nExperts {
                out.append("layers.\(L).ffn.experts.\(e).w1")
                out.append("layers.\(L).ffn.experts.\(e).w3")
            }
            if nShared > 0 {
                out.append("layers.\(L).ffn.shared_experts.w1")
                out.append("layers.\(L).ffn.shared_experts.w3")
            }
        }
        return out
    }
}
