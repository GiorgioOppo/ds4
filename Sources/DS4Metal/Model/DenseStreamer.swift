import Foundation
import Metal
import DS4Core

// DS4_DENSE_STREAM: double-buffered streaming of the per-layer DENSE weights.
//
// The dense weights are ~145 MB/layer (~6.2 GB/token on Flash — q_b, output_a/b
// alone are ~107 MB/layer): they cannot stay resident next to the expert cache
// on a 16 GB machine, so route/attn ends up re-reading them through a degraded
// page cache / swap at ~2 GB/s (the measured 2.4 s/token "compute").
//
// Unlike the routed experts, the dense weights are PERFECTLY predictable:
// layer i+1 always follows layer i. So instead of hoping 6 GB stays cached,
// stream them like the C engine streams layers — but with explicit pread +
// F_NOCACHE into a two-slot staging ring, kicked one layer AHEAD so the SSD
// read of layer i+1 overlaps the GPU compute of layer i:
//
//   GPU:  [ compute layer i   ][ compute layer i+1 ] …
//   SSD:  [ read dense i+1    ][ read dense i+2    ] …
//
// RAM cost: 2 slots × max-layer ≈ 300 MB (instead of ~6.2 GB resident), and
// zero page-cache footprint (F_NOCACHE). Same bytes → identical numerics.
//
// Concurrency contract: `weights(_:)` is called from the DECODE thread only
// (the layerProvider), one layer at a time; by the time layer i's provider is
// called, runLayer(i-1) has committed AND waited all its command buffers, so
// the slot holding layer i-1 is GPU-free and can be overwritten with i+1.
// The background loader touches only the file descriptor and the target slot,
// and hands completion back through a semaphore (happens-before for the bytes).
public final class DenseStreamer: @unchecked Sendable {
    /// The LayerWeights fields that are streamed (the "big" set of
    /// layerMappedDense; the small norm/scale tensors live in the skeleton).
    private enum Field: Int, CaseIterable {
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
    private struct Entry {
        let field: Field
        let fileOffset: Int
        let bytes: Int
        let stageOffset: Int
    }

    /// Background load in flight: bytes land in `slot`, completion via `sem`.
    private final class Pending: @unchecked Sendable {
        let layer: Int
        let slot: Int
        let sem = DispatchSemaphore(value: 0)
        var error: Error?
        init(layer: Int, slot: Int) { self.layer = layer; self.slot = slot }
    }

    private let fd: Int32
    private let layers: Range<Int>
    private var entries: [Int: [Entry]] = [:]       // layer -> read/stage plan
    private var skeleton: [Int: LayerWeights] = [:] // layer -> small-resident fields
    private let slots: [MTLBuffer]                  // 2 staging slots
    private var slotLayer = [-1, -1]                // slot -> layer currently staged
    private var pending: Pending?                   // decode-thread-owned

    /// Total bytes streamed per full pass over `layers` (diagnostics).
    public private(set) var bytesPerPass = 0

    public init(rt: MetalRuntime, model: GGUFModel, layers: Range<Int>, lockResident: Bool = false) throws {
        guard let fd = model.uncachedFD() else {
            throw GGUFWeights.LoadError.message("DenseStreamer: cannot open F_NOCACHE descriptor")
        }
        self.fd = fd
        self.layers = layers
        // Per-layer plan: pack this layer's big tensors back-to-back (4 KB
        // aligned — safe for setBuffer offsets and friendly to F_NOCACHE).
        let align = 4096
        var maxSlot = 1
        for il in layers {
            var plan: [Entry] = []
            var off = 0
            for f in Field.allCases {
                guard let t = model.findTensor("blk.\(il).\(f.tensorName)") else { continue }
                plan.append(Entry(field: f, fileOffset: Int(t.absOffset),
                                  bytes: Int(t.bytes), stageOffset: off))
                off += (Int(t.bytes) + align - 1) / align * align
            }
            entries[il] = plan
            skeleton[il] = try GGUFWeights.layerSmallSkeleton(rt, model, il)
            bytesPerPass += plan.reduce(0) { $0 + $1.bytes }
            maxSlot = max(maxSlot, off)
        }
        guard let a = rt.device.makeBuffer(length: maxSlot, options: .storageModeShared),
              let b = rt.device.makeBuffer(length: maxSlot, options: .storageModeShared) else {
            throw MetalError.bufferAlloc
        }
        slots = [a, b]
        if lockResident {
            // DS4_MLOCK: the staging ring is rewritten every ~70 ms — pin it so
            // the memory compressor never touches it. Best-effort.
            _ = mlock(a.contents(), maxSlot)
            _ = mlock(b.contents(), maxSlot)
        }
    }

    /// LayerProvider entry point (DECODE thread only). Returns layer `il`'s
    /// weights with the big fields as views into a ready staging slot, then
    /// kicks the background read of the NEXT layer into the other slot.
    public func weights(_ il: Int) throws -> LayerWeights {
        let slot: Int
        if let p = pending, p.layer == il {
            pending = nil
            p.sem.wait()                        // usually already signalled (read ran during layer i-1)
            if let e = p.error { throw e }
            slotLayer[p.slot] = il
            slot = p.slot
        } else if let s = slotLayer.firstIndex(of: il) {
            slot = s                            // already staged (e.g. retry after an error)
        } else {
            // Cold start or out-of-order request: drain any in-flight load,
            // then read synchronously into the least-recently-used slot.
            if let p = pending { pending = nil; p.sem.wait() }
            slot = slotLayer[0] == il - 1 ? 1 : 0
            try load(il, into: slot)
            slotLayer[slot] = il
        }
        // Kick the next layer of the pass into the OTHER slot: its previous
        // occupant's GPU work completed before this call (runLayer waits its
        // command buffers), so the CPU can overwrite it while the GPU runs `il`.
        let next = il + 1 < layers.upperBound ? il + 1 : layers.lowerBound
        let other = 1 - slot
        if next != il, slotLayer[other] != next, pending == nil {
            let p = Pending(layer: next, slot: other)
            slotLayer[other] = -1               // being overwritten
            pending = p
            DispatchQueue.global(qos: .userInitiated).async {
                do { try self.load(next, into: p.slot) } catch { p.error = error }
                p.sem.signal()
            }
        }
        return makeWeights(il, slot: slot)
    }

    /// pread every tensor of layer `il` into `slot`, all slabs CONCURRENTLY
    /// (10-17 reads of 2-36 MB — real queue depth for the NVMe). F_NOCACHE:
    /// zero page-cache footprint. Runs on the caller's thread.
    private func load(_ il: Int, into slot: Int) throws {
        guard let plan = entries[il] else {
            throw GGUFWeights.LoadError.message("DenseStreamer: layer \(il) outside streamed range")
        }
        let base = slots[slot].contents()
        let lock = NSLock()
        var failed = false
        DispatchQueue.concurrentPerform(iterations: plan.count) { i in
            let e = plan[i]
            if !GGUFWeights.preadFull(fd, into: base + e.stageOffset, bytes: e.bytes, offset: e.fileOffset) {
                lock.lock(); failed = true; lock.unlock()
            }
        }
        if failed {
            throw GGUFWeights.LoadError.message("DenseStreamer: pread failed on layer \(il)")
        }
    }

    /// Skeleton (small resident fields) + staging views for the big ones.
    private func makeWeights(_ il: Int, slot: Int) -> LayerWeights {
        var w = skeleton[il]!
        let buf = slots[slot]
        for e in entries[il]! {
            let t = GPUTensor(buffer: buf, byteLength: e.bytes, count: e.bytes, byteOffset: e.stageOffset)
            switch e.field {
            case .hcAttnFn: w.hcAttnFn = t
            case .qA: w.qA = t
            case .qB: w.qB = t
            case .kvW: w.kvW = t
            case .attnOutA: w.attnOutA = t
            case .attnOut: w.attnOut = t
            case .hcFfnFn: w.hcFfnFn = t
            case .sharedGate: w.sharedGate = t
            case .sharedUp: w.sharedUp = t
            case .sharedDown: w.sharedDown = t
            case .routerW: w.routerW = t
            case .compKv: w.compKv = t
            case .compGate: w.compGate = t
            case .idxQB: w.idxQB = t
            case .idxProj: w.idxProj = t
            case .idxKv: w.idxKv = t
            case .idxGate: w.idxGate = t
            }
        }
        return w
    }
}
