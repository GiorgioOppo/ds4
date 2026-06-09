import Foundation
import DS4Core

// Stage C: load DeepSeek-V4 weights from a GGUF model into GPUTensors. Each
// tensor's raw bytes (mmap'd in place) are copied into a shared Metal buffer.
// NOTE: copying every layer at full scale exceeds 16GB — real use needs Stage D
// streaming (load/evict per layer). This loader is the per-tensor primitive +
// per-layer/output assembly used by both the all-resident (>=64GB) and the
// streaming paths.

public enum GGUFWeights {
    public enum LoadError: Error, CustomStringConvertible {
        case missing(String)
        case message(String)
        public var description: String {
            switch self {
            case .missing(let n): return "GGUF tensor missing: \(n)"
            case .message(let m): return m
            }
        }
    }

    /// Load a single named tensor's bytes into a GPUTensor.
    public static func tensor(_ rt: MetalRuntime, _ model: GGUFModel, _ name: String) throws -> GPUTensor {
        guard let t = model.findTensor(name) else { throw LoadError.missing(name) }
        let ptr = model.mapBase + Int(t.absOffset)
        return try GPUTensor.raw(rt, ptr: ptr, byteLength: Int(t.bytes), elementCount: Int(t.elements))
    }

    /// Read a single F32 scalar tensor's first value (e.g. output_hc_scale).
    public static func scalarF32(_ model: GGUFModel, _ name: String) throws -> Float {
        guard let t = model.findTensor(name) else { throw LoadError.missing(name) }
        return (model.mapBase + Int(t.absOffset)).loadUnaligned(as: Float.self)
    }

    /// Assemble one decode layer's weights from blk.<il>.* tensors. With
    /// `loadExperts: false` the three 256-expert tensors are left as tiny dummies
    /// (the expert-cache path gathers only the 6 selected experts on demand after
    /// routing — saves loading ~3.6GB of experts per layer).
    /// Detect the routed-expert quant scheme + router precision from the GGUF, so
    /// DSV4Dims dispatches the right MoE kernels. Reads the first layer that has
    /// experts; falls back to the Q4_K + Q8-router default for anything unknown.
    public static func detectMoEQuant(_ model: GGUFModel) -> (gate: MoEQuant, up: MoEQuant, down: MoEQuant, routerF16: Bool) {
        var il = 0
        while il < 128 && model.findTensor("blk.\(il).ffn_gate_exps.weight") == nil { il += 1 }
        let p = "blk.\(il)."
        func q(_ s: String) -> MoEQuant {
            if let t = model.findTensor(p + s), let m = MoEQuant.from(ggufType: t.type) { return m }
            return .q4_K
        }
        let routerF16 = model.findTensor(p + "ffn_gate_inp.weight")?.type == 1   // 1 = f16
        return (q("ffn_gate_exps.weight"), q("ffn_up_exps.weight"), q("ffn_down_exps.weight"), routerF16)
    }

    public static func layer(_ rt: MetalRuntime, _ model: GGUFModel, _ il: Int, loadExperts: Bool = true) throws -> LayerWeights {
        let p = "blk.\(il)."
        func T(_ s: String) throws -> GPUTensor { try tensor(rt, model, p + s) }
        // Optional: present only on compressed layers (ratio!=0). nil on 0,1.
        func optT(_ s: String) throws -> GPUTensor? {
            model.findTensor(p + s) == nil ? nil : try tensor(rt, model, p + s)
        }
        let dummy = try GPUTensor.zerosBytes(rt, byteLength: 1)
        return LayerWeights(
            hcAttnFn: try T("hc_attn_fn.weight"), attnScale: try T("hc_attn_scale.weight"),
            attnBase: try T("hc_attn_base.weight"), attnNorm: try T("attn_norm.weight"),
            qA: try T("attn_q_a.weight"), qANorm: try T("attn_q_a_norm.weight"), qB: try T("attn_q_b.weight"),
            kvW: try T("attn_kv.weight"), kvNorm: try T("attn_kv_a_norm.weight"),
            attnSinks: try T("attn_sinks.weight"),
            attnOutA: try T("attn_output_a.weight"), attnOut: try T("attn_output_b.weight"), // low-rank a + b
            hcFfnFn: try T("hc_ffn_fn.weight"), ffnScale: try T("hc_ffn_scale.weight"),
            ffnBase: try T("hc_ffn_base.weight"), ffnNorm: try T("ffn_norm.weight"),
            sharedGate: try T("ffn_gate_shexp.weight"), sharedUp: try T("ffn_up_shexp.weight"),
            sharedDown: try T("ffn_down_shexp.weight"), routerW: try T("ffn_gate_inp.weight"),
            expGate: loadExperts ? try T("ffn_gate_exps.weight") : dummy,
            expUp: loadExperts ? try T("ffn_up_exps.weight") : dummy,
            expDown: loadExperts ? try T("ffn_down_exps.weight") : dummy,
            compKv: try optT("attn_compressor_kv.weight"), compGate: try optT("attn_compressor_gate.weight"),
            compApe: try optT("attn_compressor_ape.weight"), compNorm: try optT("attn_compressor_norm.weight"))
    }

    /// Build a layer with its routed-expert tensors as NO-COPY mmap views over the
    /// full expert weight (all 256 experts), instead of gathering the 6 selected.
    /// mul_mv_id then reads only the selected rows by their REAL ids (s.selected),
    /// and the OS page cache serves/caches the touched pages across tokens — no
    /// per-token re-gather, no RAM copy. Dense weights are still copied (small).
    /// Requires model opened with metalMapping:true (MAP_SHARED).
    public static func layerMappedExperts(_ rt: MetalRuntime, _ model: GGUFModel, _ il: Int) throws -> LayerWeights {
        var w = try layer(rt, model, il, loadExperts: false)   // dense copied; experts = dummy
        let p = "blk.\(il)."
        w.expGate = try mappedTensor(rt, model, p + "ffn_gate_exps.weight")
        w.expUp   = try mappedTensor(rt, model, p + "ffn_up_exps.weight")
        w.expDown = try mappedTensor(rt, model, p + "ffn_down_exps.weight")
        return w
    }

    /// Like `layer` but the BIG matmul-read weights are NO-COPY mmap views (resident
    /// via the OS page cache, single copy, evictable) instead of copied into Metal
    /// buffers. Only the small weights read by non-byteOffset-aware kernels (norms,
    /// hc scale/base, sinks, compressor APE/norm) are copied. Experts are loaded
    /// separately (gather). This is the C `--ssd-streaming` memory model: ~8GB of
    /// non-routed weights resident as evictable file pages, NOT 8GB of dirty copies.
    /// Requires model opened metalMapping:true and byteOffset-aware matmul encode-forms.
    public static func layerMappedDense(_ rt: MetalRuntime, _ model: GGUFModel, _ il: Int) throws -> LayerWeights {
        let p = "blk.\(il)."
        func M(_ s: String) throws -> GPUTensor { try mappedTensor(rt, model, p + s) }   // no-copy big weight
        func T(_ s: String) throws -> GPUTensor { try tensor(rt, model, p + s) }          // copy small weight
        func optM(_ s: String) throws -> GPUTensor? { model.findTensor(p + s) == nil ? nil : try mappedTensor(rt, model, p + s) }
        func optT(_ s: String) throws -> GPUTensor? { model.findTensor(p + s) == nil ? nil : try tensor(rt, model, p + s) }
        let dummy = try GPUTensor.zerosBytes(rt, byteLength: 1)
        return LayerWeights(
            hcAttnFn: try M("hc_attn_fn.weight"), attnScale: try T("hc_attn_scale.weight"),
            attnBase: try T("hc_attn_base.weight"), attnNorm: try T("attn_norm.weight"),
            qA: try M("attn_q_a.weight"), qANorm: try T("attn_q_a_norm.weight"), qB: try M("attn_q_b.weight"),
            kvW: try M("attn_kv.weight"), kvNorm: try T("attn_kv_a_norm.weight"),
            attnSinks: try T("attn_sinks.weight"),
            attnOutA: try M("attn_output_a.weight"), attnOut: try M("attn_output_b.weight"),
            hcFfnFn: try M("hc_ffn_fn.weight"), ffnScale: try T("hc_ffn_scale.weight"),
            ffnBase: try T("hc_ffn_base.weight"), ffnNorm: try T("ffn_norm.weight"),
            sharedGate: try M("ffn_gate_shexp.weight"), sharedUp: try M("ffn_up_shexp.weight"),
            sharedDown: try M("ffn_down_shexp.weight"), routerW: try M("ffn_gate_inp.weight"),
            expGate: dummy, expUp: dummy, expDown: dummy,   // experts gathered separately
            compKv: try optM("attn_compressor_kv.weight"), compGate: try optM("attn_compressor_gate.weight"),
            compApe: try optT("attn_compressor_ape.weight"), compNorm: try optT("attn_compressor_norm.weight"))
    }

    /// Output head + embedding with the big tensors (embed F16, output Q8, output_hc_fn
    /// F16) as NO-COPY mmap views; the small norm/scale/base are copied.
    public static func outputHeadMapped(_ rt: MetalRuntime, _ model: GGUFModel) throws -> (embed: GPUTensor, head: OutputHeadWeights) {
        let embed = try mappedTensor(rt, model, "token_embd.weight")
        let head = OutputHeadWeights(
            hcFn: try mappedTensor(rt, model, "output_hc_fn.weight"),
            hcScaleScalar: try scalarF32(model, "output_hc_scale.weight"),
            hcBase: try tensor(rt, model, "output_hc_base.weight"),
            norm: try tensor(rt, model, "output_norm.weight"),
            head: try mappedTensor(rt, model, "output.weight"))
        return (embed, head)
    }

    /// No-copy mmap GPUTensor over a whole GGUF tensor's bytes.
    static func mappedTensor(_ rt: MetalRuntime, _ model: GGUFModel, _ name: String) throws -> GPUTensor {
        guard let t = model.findTensor(name) else { throw LoadError.missing(name) }
        let ptr = model.mapBase + Int(t.absOffset)
        return try GPUTensor.mappedNoCopy(rt, ptr: ptr, byteLength: Int(t.bytes), elementCount: Int(t.bytes))
    }

    /// Expert-cache: pack ONLY the `ids` selected experts of a Q4_K MoE tensor
    /// (ffn_*_exps, layout [inDim, outRows, nExpert]) from the mmap into a small
    /// K-expert buffer, so streaming loads ~K/256 of the expert weight per layer.
    /// Call moeMatvecQ4K with ids remapped to 0..<K against the returned tensor.
    public static func gatherExperts(_ rt: MetalRuntime, _ model: GGUFModel, _ name: String,
                                     ids: [Int32], inDim: Int, outRows: Int) throws -> GPUTensor {
        guard let t = model.findTensor(name) else { throw LoadError.missing(name) }
        // Per-expert byte size from the tensor's actual GGUF block layout (q4_K=144,
        // q2_K=84, iq2_xxs=66 per 256 elems) — NOT hardcoded Q4_K.
        guard let info = GGUF.typeInfo(t.type), info.blockElems == 256 else {
            throw LoadError.message("gatherExperts: \(name) has unsupported expert type \(t.typeName)")
        }
        let expertBytes = outRows * (inDim / 256) * Int(info.blockBytes)
        let base = model.mapBase + Int(t.absOffset)
        var packed = [UInt8](repeating: 0, count: ids.count * expertBytes)
        packed.withUnsafeMutableBytes { dst in
            for (i, e) in ids.enumerated() {
                memcpy(dst.baseAddress! + i * expertBytes, base + Int(e) * expertBytes, expertBytes)
            }
        }
        return try GPUTensor.bytes(rt, packed, elementCount: ids.count * expertBytes)
    }

    /// Copy ONE expert's slab from the mmap into `dst` at `slot * expertBytes`
    /// (the ExpertSlotCache fill primitive; dst is a shared-storage pool tensor).
    public static func copyExpert(_ model: GGUFModel, _ name: String, id: Int32,
                                  expertBytes: Int, into dst: GPUTensor, slot: Int) throws {
        guard let t = model.findTensor(name) else { throw LoadError.missing(name) }
        let src = model.mapBase + Int(t.absOffset) + Int(id) * expertBytes
        memcpy(dst.buffer.contents().advanced(by: dst.byteOffset + slot * expertBytes),
               src, expertBytes)
    }

    /// Assemble the output-head weights + embedding table.
    public static func outputHead(_ rt: MetalRuntime, _ model: GGUFModel) throws -> (embed: GPUTensor, head: OutputHeadWeights) {
        let embed = try tensor(rt, model, "token_embd.weight")
        let head = OutputHeadWeights(
            hcFn: try tensor(rt, model, "output_hc_fn.weight"),
            hcScaleScalar: try scalarF32(model, "output_hc_scale.weight"),
            hcBase: try tensor(rt, model, "output_hc_base.weight"),
            norm: try tensor(rt, model, "output_norm.weight"),
            head: try tensor(rt, model, "output.weight"))
        return (embed, head)
    }
}
