import Foundation
import Metal
import DS4Core

extension DenseStreamer {
    /// The LayerWeights fields that are streamed (the "big" set of
    /// layerMappedDense; the small norm/scale tensors live in the skeleton).
    enum Field: Int, CaseIterable {
        case hcAttnFn, qA, qB, kvW, attnOutA, attnOut, hcFfnFn,
             sharedGate, sharedUp, sharedDown, routerW,
             compKv, compGate, idxQB, idxProj, idxKv, idxGate

        var tensorName: String {
            switch self {
            case .hcAttnFn: return "hc_attn_fn.weight"
            case .qA: return "attn_q_a.weight"
            case .qB: return "attn_q_b.weight"
            case .kvW: return "attn_kv.weight"
            case .attnOutA: return "attn_output_a.weight"
            case .attnOut: return "attn_output_b.weight"
            case .hcFfnFn: return "hc_ffn_fn.weight"
            case .sharedGate: return "ffn_gate_shexp.weight"
            case .sharedUp: return "ffn_up_shexp.weight"
            case .sharedDown: return "ffn_down_shexp.weight"
            case .routerW: return "ffn_gate_inp.weight"
            case .compKv: return "attn_compressor_kv.weight"
            case .compGate: return "attn_compressor_gate.weight"
            case .idxQB: return "indexer.attn_q_b.weight"
            case .idxProj: return "indexer.proj.weight"
            case .idxKv: return "indexer_compressor_kv.weight"
            case .idxGate: return "indexer_compressor_gate.weight"
            }
        }
    }

    /// One streamed tensor: where it lives in the GGUF and where it lands in a
    /// staging slot. Layouts differ per layer (ratio-0/4/128 layers carry
    /// different compressor/indexer tensors), so each layer has its own plan.
    struct Entry {
        let field: Field
        let fileOffset: Int
        let bytes: Int
        let stageOffset: Int
    }

    /// Background load in flight: bytes land in `slot`, completion via `sem`.
    final class Pending: @unchecked Sendable {
        let layer: Int
        let slot: Int
        let sem = DispatchSemaphore(value: 0)
        var error: Error?
        init(layer: Int, slot: Int) { self.layer = layer; self.slot = slot }
    }
}
