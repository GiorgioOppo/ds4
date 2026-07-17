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

    /// Bind-time validation of the routed expert tensors, port of the C checks
    /// the loader enforces before ever dispatching a kernel (ds4.c:3227-3231,
    /// 3626-3631): the quant type must be one of the closed set the kernels
    /// implement (iq2_xxs / q2_K / q4_K — exactly `MoEQuant.from`), gate and up
    /// must share a type, and the declared dims must match the shape. Without
    /// this an unknown type silently fell back to the q4_K kernel: garbage
    /// logits instead of a load error. Geometry-driven (dims, layer count and
    /// hash-layer count come from the instance, not from Flash-only statics).
    public static func validateRuntimeLayout(_ model: GGUFModel, geometry: DSV4RuntimeGeometry) throws {
        let dims = geometry.dims
        for il in 0..<geometry.nLayers {
            let p = "blk.\(il)."
            guard let g = model.findTensor(p + "ffn_gate_exps.weight") else { continue }   // dense layer
            guard let u = model.findTensor(p + "ffn_up_exps.weight"),
                  let dn = model.findTensor(p + "ffn_down_exps.weight") else {
                throw LoadError.missing("\(p)ffn_up_exps/ffn_down_exps.weight")
            }
            for t in [g, u, dn] where MoEQuant.from(ggufType: t.type) == nil {
                throw LoadError.message("\(p): expected a routed expert quant type "
                                        + "(iq2_xxs/q2_K/q4_K), got \(t.typeName)")
            }
            if g.type != u.type {
                throw LoadError.message("\(p): ffn_gate_exps and ffn_up_exps quant types differ "
                                        + "(\(g.typeName) vs \(u.typeName))")
            }
            let gu: [UInt64] = [UInt64(dims.nEmbd), UInt64(dims.expertFfn), UInt64(dims.nExperts)]
            let dd: [UInt64] = [UInt64(dims.expertFfn), UInt64(dims.nEmbd), UInt64(dims.nExperts)]
            if g.dims != gu || u.dims != gu {
                throw LoadError.message("\(p): routed gate/up dims \(g.dims)/\(u.dims), expected \(gu)")
            }
            if dn.dims != dd {
                throw LoadError.message("\(p): routed down dims \(dn.dims), expected \(dd)")
            }
            // Hash routing table: REQUIRED on the first n_hash_layer layers
            // (required_tensorf, ds4.c:4064), I32 [n_expert_used x n_vocab]
            // (ds4.c:3637). Without it those layers would silently fall back
            // to top-k routing — a numerics divergence, not a load error.
            if il < geometry.nHashLayer {
                guard let t = model.findTensor(p + "ffn_gate_tid2eid.weight") else {
                    throw LoadError.missing("\(p)ffn_gate_tid2eid.weight (hash-routed layer)")
                }
                guard t.type == 26, t.dims == [UInt64(dims.k), UInt64(dims.vocab)] else {
                    throw LoadError.message("\(p)ffn_gate_tid2eid.weight: type \(t.typeName) dims \(t.dims), "
                                            + "expected i32 [\(dims.k), \(dims.vocab)]")
                }
            }
            // Selection bias: optional, but when present it must be F32 [n_expert]
            // (tensor_expect_optional, ds4.c:3625) — the finalize kernel reads
            // exactly nExperts floats from it.
            if let b = model.findTensor(p + "exp_probs_b.bias") {
                guard b.type == 0, b.dims == [UInt64(dims.nExperts)] else {
                    throw LoadError.message("\(p)exp_probs_b.bias: type \(b.typeName) dims \(b.dims), "
                                            + "expected f32 [\(dims.nExperts)]")
                }
            }
        }
    }

    public static func layer(_ rt: MetalRuntime, _ model: GGUFModel, _ il: Int, loadExperts: Bool = true) throws -> LayerWeights {
        let p = "blk.\(il)."
        func T(_ s: String) throws -> GPUTensor { try tensor(rt, model, p + s) }
        // Optional: present only on compressed layers (ratio!=0). nil on 0,1.
        func optT(_ s: String) throws -> GPUTensor? {
            model.findTensor(p + s) == nil ? nil : try tensor(rt, model, p + s)
        }
        let dummy = try GPUTensor.zerosBytes(rt, byteLength: 1)
        var w = LayerWeights(
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
        try loadIndexer(&w, model, il, big: optT, small: optT)
        try loadRouterExtras(&w, rt, model, il, mapped: false)
        setExpertQuant(&w, model, il)   // per-layer routed-expert quant (mixed-precision)
        return w
    }

    /// Set `w`'s per-layer routed-expert quant from the GGUF tensor types
    /// (mixed-precision support). MUST be called from EVERY LayerWeights builder
    /// (`layer`, `layerMappedDense`) so `decodeExperts` dispatches the right kernel:
    /// it reads `w.*Quant`, not the model-global `DSV4Dims` quant. The exps tensors
    /// exist in the GGUF even when not loaded as GPUTensors (streaming); unknown or
    /// missing types keep the `.q4_K` default (matches the global fallback).
    static func setExpertQuant(_ w: inout LayerWeights, _ model: GGUFModel, _ il: Int) {
        let p = "blk.\(il)."
        if let g = model.findTensor(p + "ffn_gate_exps.weight").flatMap({ MoEQuant.from(ggufType: $0.type) }) { w.gateQuant = g }
        if let u = model.findTensor(p + "ffn_up_exps.weight").flatMap({ MoEQuant.from(ggufType: $0.type) }) { w.upQuant = u }
        if let dn = model.findTensor(p + "ffn_down_exps.weight").flatMap({ MoEQuant.from(ggufType: $0.type) }) { w.downQuant = dn }
    }

    /// Number of routed layers whose expert quant differs from the model-global
    /// class (the first routed layer = `detectMoEQuant`). >0 ⇒ a mixed-precision
    /// GGUF: those layers decode through the per-layer kernel and bypass the
    /// (single-size-class) expert slot-cache, reading experts via the mmap gather.
    public static func mixedPrecisionLayerCount(_ model: GGUFModel, nLayers: Int) -> Int {
        let cls = detectMoEQuant(model)
        func q(_ p: String, _ s: String) -> MoEQuant? {
            model.findTensor(p + s).flatMap { MoEQuant.from(ggufType: $0.type) }
        }
        var n = 0
        for il in 0..<nLayers {
            let p = "blk.\(il)."
            guard model.findTensor(p + "ffn_gate_exps.weight") != nil else { continue }   // dense layer
            if q(p, "ffn_gate_exps.weight") != cls.gate
                || q(p, "ffn_up_exps.weight") != cls.up
                || q(p, "ffn_down_exps.weight") != cls.down { n += 1 }
        }
        return n
    }

    /// Hash routing table + router selection bias (ds4.c: ffn_gate_tid2eid is
    /// REQUIRED on the first n_hash_layer layers, exp_probs_b.bias is optional
    /// everywhere). Layout validated like layer_hash_selected_experts: I32,
    /// 2 dims, dim[0] == 6 (experts per token). `mapped` puts the ~3 MB table
    /// behind a no-copy mmap view; the tiny bias is always copied.
    static func loadRouterExtras(_ w: inout LayerWeights, _ rt: MetalRuntime, _ model: GGUFModel,
                                 _ il: Int, mapped: Bool) throws {
        let p = "blk.\(il)."
        if let t = model.findTensor(p + "ffn_gate_tid2eid.weight") {
            guard t.type == 26, t.dims.count == 2, t.dims[0] == 6, t.dims[1] > 0 else {
                throw LoadError.message("\(p)ffn_gate_tid2eid.weight has an unexpected layout "
                                        + "(type \(t.typeName), dims \(t.dims))")
            }
            w.tid2eid = mapped ? try mappedTensor(rt, model, p + "ffn_gate_tid2eid.weight")
                               : try tensor(rt, model, p + "ffn_gate_tid2eid.weight")
            w.tid2eidRows = Int(t.dims[1])
        }
        if model.findTensor(p + "exp_probs_b.bias") != nil {
            w.expBias = try tensor(rt, model, p + "exp_probs_b.bias")
        }
    }

    /// CPU-side hash-layer expert selection (layer_hash_selected_experts): row
    /// min(token, rows-1) of blk.<il>.ffn_gate_tid2eid.weight, read straight
    /// from the mmap. This is what makes the hash layers' expert I/O
    /// PREFETCHABLE: their selection depends only on the token id, known before
    /// any GPU work. nil on non-hash layers (tensor absent) or bad layout.
    public static func hashSelectedIds(_ model: GGUFModel, _ il: Int, token: Int) -> [Int32]? {
        guard token >= 0,
              let t = model.findTensor("blk.\(il).ffn_gate_tid2eid.weight"),
              t.type == 26, t.dims.count == 2, t.dims[0] == 6, t.dims[1] > 0 else { return nil }
        let row = min(token, Int(t.dims[1]) - 1)
        let base = model.mapBase + Int(t.absOffset) + row * 6 * 4
        return (0..<6).map { base.loadUnaligned(fromByteOffset: $0 * 4, as: Int32.self) }
    }

    /// NSA indexer tensors (DSA; present only on ratio-4 layers — all optional).
    /// `big` loads the two large projections + compressor kv/gate (mmap no-copy on
    /// the streaming path, copy otherwise); `small` the tiny APE/norm.
    static func loadIndexer(_ w: inout LayerWeights, _ model: GGUFModel, _ il: Int,
                            big: (String) throws -> GPUTensor?,
                            small: (String) throws -> GPUTensor?) rethrows {
        w.idxQB = try big("indexer.attn_q_b.weight")
        w.idxProj = try big("indexer.proj.weight")
        w.idxKv = try big("indexer_compressor_kv.weight")
        w.idxGate = try big("indexer_compressor_gate.weight")
        w.idxApe = try small("indexer_compressor_ape.weight")
        w.idxNorm = try small("indexer_compressor_norm.weight")
        w.idxQBF16 = model.findTensor("blk.\(il).indexer.attn_q_b.weight")?.type == 1   // 1 = f16
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
        var w = LayerWeights(
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
        try loadIndexer(&w, model, il, big: optM, small: optT)
        try loadRouterExtras(&w, rt, model, il, mapped: true)
        setExpertQuant(&w, model, il)   // per-layer routed-expert quant (mixed-precision)
        return w
    }

    /// SMALL-ONLY skeleton for the dense-STREAMING path (DS4_DENSE_STREAM): the
    /// tiny norm/scale/base/sinks tensors are COPIED resident once (they are the
    /// ones read by non-byteOffset-aware kernels), every BIG dense field is a
    /// 1-byte dummy that DenseStreamer swaps for a staging-slot view per call.
    /// Field split identical to layerMappedDense — same weights, different home.
    static func layerSmallSkeleton(_ rt: MetalRuntime, _ model: GGUFModel, _ il: Int) throws -> LayerWeights {
        let p = "blk.\(il)."
        func T(_ s: String) throws -> GPUTensor { try tensor(rt, model, p + s) }
        func optT(_ s: String) throws -> GPUTensor? { model.findTensor(p + s) == nil ? nil : try tensor(rt, model, p + s) }
        let dummy = try GPUTensor.zerosBytes(rt, byteLength: 1)
        var w = LayerWeights(
            hcAttnFn: dummy, attnScale: try T("hc_attn_scale.weight"),
            attnBase: try T("hc_attn_base.weight"), attnNorm: try T("attn_norm.weight"),
            qA: dummy, qANorm: try T("attn_q_a_norm.weight"), qB: dummy,
            kvW: dummy, kvNorm: try T("attn_kv_a_norm.weight"),
            attnSinks: try T("attn_sinks.weight"),
            attnOutA: dummy, attnOut: dummy,
            hcFfnFn: dummy, ffnScale: try T("hc_ffn_scale.weight"),
            ffnBase: try T("hc_ffn_base.weight"), ffnNorm: try T("ffn_norm.weight"),
            sharedGate: dummy, sharedUp: dummy,
            sharedDown: dummy, routerW: dummy,
            expGate: dummy, expUp: dummy, expDown: dummy,   // experts gathered separately
            compKv: nil, compGate: nil,
            compApe: try optT("attn_compressor_ape.weight"), compNorm: try optT("attn_compressor_norm.weight"))
        // Indexer: only the SMALL ape/norm are loaded here; the big projections
        // are streamed (DenseStreamer fills them when the layer has them).
        try loadIndexer(&w, model, il, big: { _ in nil }, small: optT)
        // Hash table + bias stay RESIDENT (copied): the router needs them every
        // token and the 3-layer table is ~9 MB total — not worth streaming.
        try loadRouterExtras(&w, rt, model, il, mapped: false)
        setExpertQuant(&w, model, il)   // per-layer routed-expert quant (mixed-precision)
        return w
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

    /// DS4_WILLNEED_EXPERTS=0 disables EVERY expert readahead hint (adviseRange /
    /// adviseExpert / gatherLayerExperts' batched prefetch) — the measurement
    /// baseline. Default ON: the hints target only actually-selected slabs.
    static let willNeedExperts = ProcessInfo.processInfo.environment["DS4_WILLNEED_EXPERTS"] != "0"

    /// Kick off async readahead of an mmap range: page-aligned madvise(WILLNEED).
    /// Advising ALL the slabs of a gather before the first copy lets the NVMe
    /// work on every region concurrently, instead of one page fault at a time.
    static func adviseRange(_ ptr: UnsafeRawPointer, _ bytes: Int) {
        let page = Int(getpagesize())
        let addr = UInt(bitPattern: ptr)
        let aligned = addr & ~UInt(page - 1)
        guard let p = UnsafeMutableRawPointer(bitPattern: aligned) else { return }
        madvise(p, Int(addr - aligned) + bytes, MADV_WILLNEED)
    }

    /// Readahead hint for ONE expert's slab of an ffn_*_exps tensor (see
    /// adviseRange). Out-of-bounds ids are ignored — this is only a hint.
    public static func adviseExpert(_ model: GGUFModel, _ name: String, id: Int32,
                                    expertBytes: Int) {
        guard willNeedExperts else { return }
        guard let t = model.findTensor(name), id >= 0,
              (Int(id) + 1) * expertBytes <= Int(t.bytes) else { return }
        adviseRange(model.mapBase + Int(t.absOffset) + Int(id) * expertBytes, expertBytes)
    }

    /// pread the full range (loops on short reads). Thread-safe on a shared fd
    /// (explicit offsets, no shared cursor). Returns false on I/O error.
    static func preadFull(_ fd: Int32, into dst: UnsafeMutableRawPointer,
                          bytes: Int, offset: Int) -> Bool {
        var done = 0
        while done < bytes {
            let n = pread(fd, dst + done, bytes - done, off_t(offset + done))
            if n <= 0 { return false }
            done += n
        }
        return true
    }

    /// DS4_PREAD_SPLIT: pread CONCORRENTI per slab nel fill diretto (F_NOCACHE)
    /// della slot-cache. I miss del decode sono pochi per layer (~2-3 × 3 slab
    /// ⇒ coda NVMe ~6-9 richieste), ma il disco rende il suo tetto solo a ~24
    /// richieste in volo (il probe DS4_DIAG "random parallelo"): spezzare ogni
    /// slab in N range DISGIUNTI letti in parallelo alza la profondità di coda
    /// a parità di byte. 1 (default) = una pread per slab, percorso storico.
    static let preadSplit = max(1, min(8, ProcessInfo.processInfo.environment["DS4_PREAD_SPLIT"].flatMap(Int.init) ?? 1))

    /// preadFull spezzata in `parts` range disgiunti letti CONCORRENTEMENTE
    /// (stesso fd: pread ha l'offset esplicito, niente cursore condiviso).
    /// Confini dei range allineati a 16 KB così il F_NOCACHE non rilegge la
    /// stessa pagina da due job. Sotto i 64 KB (o parts=1) delega a preadFull.
    static func preadFullSplit(_ fd: Int32, into dst: UnsafeMutableRawPointer,
                               bytes: Int, offset: Int, parts: Int) -> Bool {
        guard parts > 1, bytes > (64 << 10) else {
            return preadFull(fd, into: dst, bytes: bytes, offset: offset)
        }
        let align = 16 << 10
        let chunk = ((bytes + parts - 1) / parts + align - 1) / align * align
        let n = (bytes + chunk - 1) / chunk
        // nonisolated(unsafe): ogni iterazione scrive un range DISGIUNTO di
        // dst; il flag d'errore e' protetto dal lock (stesso pattern di
        // gatherExperts qui sotto).
        nonisolated(unsafe) let dstBase = dst
        let failed = NSLock()
        nonisolated(unsafe) var anyFailure = false
        DispatchQueue.concurrentPerform(iterations: n) { i in
            let off = i * chunk
            let len = min(chunk, bytes - off)
            if !preadFull(fd, into: dstBase + off, bytes: len, offset: offset + off) {
                failed.lock(); anyFailure = true; failed.unlock()
            }
        }
        return !anyFailure
    }

    /// Expert-cache: pack ONLY the `ids` selected experts of a Q4_K MoE tensor
    /// (ffn_*_exps, layout [inDim, outRows, nExpert]) from the mmap into a small
    /// K-expert buffer, so streaming loads ~K/256 of the expert weight per layer.
    /// Call moeMatvecQ4K with ids remapped to 0..<K against the returned tensor.
    /// The slabs are madvise'd up front and copied CONCURRENTLY straight into the
    /// shared Metal buffer (queue depth ~= ids.count on the SSD, single copy).
    /// With `uncachedFD` (DS4_EXPERT_PREAD) the slabs are pread() DIRECT from
    /// disk instead — zero page-cache footprint, so the expert churn stops
    /// evicting the dense weights on tight-RAM machines. Same bytes either way.
    public static func gatherExperts(_ rt: MetalRuntime, _ model: GGUFModel, _ name: String,
                                     ids: [Int32], inDim: Int, outRows: Int,
                                     uncachedFD: Int32? = nil) throws -> GPUTensor {
        guard let t = model.findTensor(name) else { throw LoadError.missing(name) }
        // Per-expert byte size from the tensor's actual GGUF block layout (q4_K=144,
        // q2_K=84, iq2_xxs=66 per 256 elems) — NOT hardcoded Q4_K.
        guard let info = GGUF.typeInfo(t.type), info.blockElems == 256 else {
            throw LoadError.message("gatherExperts: \(name) has unsupported expert type \(t.typeName)")
        }
        let expertBytes = outRows * (inDim / 256) * Int(info.blockBytes)
        for e in ids where e < 0 || (Int(e) + 1) * expertBytes > Int(t.bytes) {
            throw LoadError.message("gatherExperts: \(name) expert \(e) outside tensor bounds")
        }
        let dst = try GPUTensor.uninitializedBytes(rt, byteLength: ids.count * expertBytes,
                                                   elementCount: ids.count * expertBytes)
        // nonisolated(unsafe): each concurrent iteration writes a DISJOINT slab
        // of dstBase (offset i*expertBytes) and reads only immutable state; the
        // failure flag is guarded by its lock. Safe in practice — explicit
        // opt-out from strict concurrency for the pointer/flag captures.
        nonisolated(unsafe) let dstBase = dst.buffer.contents()
        if let fd = uncachedFD {
            let absBase = Int(t.absOffset)
            let failed = NSLock()
            nonisolated(unsafe) var anyFailure = false
            DispatchQueue.concurrentPerform(iterations: ids.count) { i in
                if !preadFull(fd, into: dstBase + i * expertBytes, bytes: expertBytes,
                              offset: absBase + Int(ids[i]) * expertBytes) {
                    failed.lock(); anyFailure = true; failed.unlock()
                }
            }
            if anyFailure { throw LoadError.message("gatherExperts: pread failed on \(name)") }
            return dst
        }
        nonisolated(unsafe) let base = model.mapBase + Int(t.absOffset)
        if willNeedExperts {
            for e in ids { adviseRange(base + Int(e) * expertBytes, expertBytes) }
        }
        DispatchQueue.concurrentPerform(iterations: ids.count) { i in
            memcpy(dstBase + i * expertBytes, base + Int(ids[i]) * expertBytes, expertBytes)
        }
        return dst
    }

    /// Byte ranges (offset-from-mapBase, length) of the SELECTED experts' slabs in
    /// one routed tensor — used to POSIX_MADV_WILLNEED them before the gather copy.
    static func expertRanges(_ model: GGUFModel, _ name: String, ids: [Int32],
                             inDim: Int, outRows: Int) -> [(offset: UInt64, bytes: UInt64)] {
        guard let t = model.findTensor(name), let info = GGUF.typeInfo(t.type), info.blockElems == 256 else { return [] }
        let expertBytes = UInt64(outRows * (inDim / 256) * Int(info.blockBytes))
        return ids.map { (offset: UInt64(t.absOffset) + UInt64($0) * expertBytes, bytes: expertBytes) }
    }

    /// Gather the selected experts (gate/up/down, packed) for layer `il`. With
    /// `willNeed`, first hint the OS to read the exact slabs we're about to copy
    /// (POSIX_MADV_WILLNEED) so the cold SSD readahead is front-loaded and batched
    /// instead of paid as ~one minor fault per page inside the memcpy. The hint
    /// targets the ACTUALLY-selected experts (non-speculative) — unlike a usage-prior
    /// prefetch it never reads experts we won't use, and on warm pages it's a cheap
    /// no-op. Advisory only: cannot change numerics. Opt-in via DS4_WILLNEED_EXPERTS.
    public static func gatherLayerExperts(_ rt: MetalRuntime, _ model: GGUFModel, _ il: Int,
                                          ids: [Int32], dims: DSV4Dims, willNeed: Bool,
                                          uncachedFD: Int32? = nil) throws
        -> (GPUTensor, GPUTensor, GPUTensor) {
        let gn = "blk.\(il).ffn_gate_exps.weight"
        let un = "blk.\(il).ffn_up_exps.weight"
        let dnn = "blk.\(il).ffn_down_exps.weight"
        if willNeed && uncachedFD == nil {   // madvise is a page-cache hint: pointless with direct pread
            var ranges = expertRanges(model, gn, ids: ids, inDim: dims.nEmbd, outRows: dims.expertFfn)
            ranges += expertRanges(model, un, ids: ids, inDim: dims.nEmbd, outRows: dims.expertFfn)
            ranges += expertRanges(model, dnn, ids: ids, inDim: dims.expertFfn, outRows: dims.nEmbd)
            GGUFModel.prefetch(base: model.mapBase, ranges: ranges)
        }
        let g = try gatherExperts(rt, model, gn, ids: ids, inDim: dims.nEmbd, outRows: dims.expertFfn, uncachedFD: uncachedFD)
        let u = try gatherExperts(rt, model, un, ids: ids, inDim: dims.nEmbd, outRows: dims.expertFfn, uncachedFD: uncachedFD)
        let dn = try gatherExperts(rt, model, dnn, ids: ids, inDim: dims.expertFfn, outRows: dims.nEmbd, uncachedFD: uncachedFD)
        return (g, u, dn)
    }

    /// Copy ONE expert's slab from the mmap into `dst` at `slot * expertBytes`
    /// (the ExpertSlotCache fill primitive; dst is a shared-storage pool tensor).
    /// With `uncachedFD` the slab is pread() DIRECT from disk (F_NOCACHE): zero
    /// page-cache footprint — see gatherExperts.
    /// `slotStride`: byte fra uno slot e il successivo nel pool — nil = packing
    /// stretto (expertBytes); il pool INTERLEAVED passa la dimensione del
    /// record gate|up|down e `dst` è la vista del proprio slab (byteOffset).
    public static func copyExpert(_ model: GGUFModel, _ name: String, id: Int32,
                                  expertBytes: Int, into dst: GPUTensor, slot: Int,
                                  uncachedFD: Int32? = nil, slotStride: Int? = nil) throws {
        let stride = slotStride ?? expertBytes
        guard let t = model.findTensor(name) else { throw LoadError.missing(name) }
        guard id >= 0, (Int(id) + 1) * expertBytes <= Int(t.bytes) else {
            throw LoadError.message("copyExpert: \(name) expert \(id) outside tensor bounds")
        }
        guard dst.byteOffset + slot * stride + expertBytes <= dst.buffer.length else {
            throw LoadError.message("copyExpert: slot \(slot) outside pool buffer")
        }
        let dstPtr = dst.buffer.contents().advanced(by: dst.byteOffset + slot * stride)
        if let fd = uncachedFD {
            guard preadFullSplit(fd, into: dstPtr, bytes: expertBytes,
                                 offset: Int(t.absOffset) + Int(id) * expertBytes,
                                 parts: preadSplit) else {
                throw LoadError.message("copyExpert: pread failed on \(name) expert \(id)")
            }
            return
        }
        let src = model.mapBase + Int(t.absOffset) + Int(id) * expertBytes
        memcpy(dstPtr, src, expertBytes)
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
