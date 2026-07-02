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

    public init(rt: MetalRuntime, model: GGUFModel, layers: Range<Int>, lockResident: Bool = false,
                q4Dense: Bool = false) throws {
        guard let fd = model.uncachedFD() else {
            throw GGUFWeights.LoadError.message("DenseStreamer: cannot open F_NOCACHE descriptor")
        }
        self.fd = fd
        self.layers = layers
        // Per-layer plan: pack this layer's big tensors back-to-back (4 KB
        // aligned — safe for setBuffer offsets and friendly to F_NOCACHE).
        let align = 4096
        var maxSlot = 1
        // DS4_DENSE_Q4: the three giant attention projections are requantized
        // Q8_0 → Q4_K ONCE at load and kept resident (~1.4 GB, locked with
        // DS4_MLOCK): half the bytes, RAM-speed reads, and they leave the
        // per-token stream entirely. Intentionally lossy; everything else is
        // untouched. The jobs are collected here and run in PARALLEL below —
        // 129 independent (layer, tensor) requants saturate every core, ~6-8×
        // faster load than converting layer by layer.
        var q4Jobs: [(il: Int, f: Field, t: GGUFModel.Tensor)] = []
        LoadProgress.shared.begin("Preparazione layer densi…", from: 0.08, to: 0.30, units: layers.count)
        for il in layers {
            var plan: [Entry] = []
            var off = 0
            let w = try GGUFWeights.layerSmallSkeleton(rt, model, il)
            LoadProgress.shared.advance()
            for f in Field.allCases {
                guard let t = model.findTensor("blk.\(il).\(f.tensorName)") else { continue }
                if q4Dense, f == .qB || f == .attnOut || f == .attnOutA,
                   let info = GGUF.typeInfo(t.type), info.name == "q8_0",
                   Int(t.elements) % 256 == 0 {
                    q4Jobs.append((il: il, f: f, t: t))
                    continue
                }
                plan.append(Entry(field: f, fileOffset: Int(t.absOffset),
                                  bytes: Int(t.bytes), stageOffset: off))
                off += (Int(t.bytes) + align - 1) / align * align
            }
            entries[il] = plan
            skeleton[il] = w
            bytesPerPass += plan.reduce(0) { $0 + $1.bytes }
            maxSlot = max(maxSlot, off)
        }
        if !q4Jobs.isEmpty {
            // Requant CACHE: the converted Q4 tensors are persisted next to the
            // model (<gguf>.q4dense, ~1.4 GB) — the first load pays the requant
            // once, every later load preads the cache in ~0.5 s. Invalidated by
            // model size / job-list mismatches; write failures are ignored.
            // Cache location: next to the model by default (demo/CLI), or in
            // DS4_Q4_CACHE_DIR when set — the SANDBOXED app can't write next
            // to a picker-selected file, so it points this at Application
            // Support. READING tries both places (a cache produced by the demo
            // next to the GGUF gets picked up and PROMOTED into the primary
            // location, so demo and app share one conversion when possible).
            let sibling = model.path + ".q4dense"
            var cachePath = sibling
            if let dir = ProcessInfo.processInfo.environment["DS4_Q4_CACHE_DIR"], !dir.isEmpty {
                try? FileManager.default.createDirectory(atPath: dir, withIntermediateDirectories: true)
                cachePath = dir + "/" + (model.path as NSString).lastPathComponent + ".q4dense"
            }
            var converted: [GPUTensor]
            LoadProgress.shared.begin("Lettura cache Q4…", from: 0.32, to: 0.92, units: q4Jobs.count)
            if let cached = Self.loadQ4Cache(rt, path: cachePath, modelSize: Int(model.size), jobs: q4Jobs) {
                converted = cached
            } else if cachePath != sibling,
                      let cached = Self.loadQ4Cache(rt, path: sibling, modelSize: Int(model.size), jobs: q4Jobs) {
                converted = cached
                LoadProgress.shared.set(0.92, "Copia cache Q4…")
                Self.writeQ4Cache(path: cachePath, modelSize: Int(model.size), jobs: q4Jobs, tensors: cached)
            } else {
                LoadProgress.shared.begin("Riquantizzazione Q4 (solo il primo avvio)…",
                                          from: 0.32, to: 0.88, units: q4Jobs.count)
                var fresh = [GPUTensor?](repeating: nil, count: q4Jobs.count)
                let lock = NSLock()
                var firstError: Error?
                try fresh.withUnsafeMutableBufferPointer { out in
                    DispatchQueue.concurrentPerform(iterations: q4Jobs.count) { i in
                        do {
                            out[i] = try Self.requantQ4(rt, model, fd: fd, tensor: q4Jobs[i].t)
                            LoadProgress.shared.advance()
                        } catch {
                            lock.lock()
                            if firstError == nil { firstError = error }
                            lock.unlock()
                        }
                    }
                    if let e = firstError { throw e }
                }
                converted = try fresh.enumerated().map { i, t in
                    guard let t else {
                        throw GGUFWeights.LoadError.message("DenseStreamer: requant failed on layer \(q4Jobs[i].il)")
                    }
                    return t
                }
                LoadProgress.shared.set(0.90, "Scrittura cache Q4…")
                Self.writeQ4Cache(path: cachePath, modelSize: Int(model.size), jobs: q4Jobs, tensors: converted)
            }
            for (i, job) in q4Jobs.enumerated() {
                let q4 = converted[i]
                if lockResident { q4.lockResident() }
                switch job.f {
                case .qB: skeleton[job.il]!.qB = q4; skeleton[job.il]!.qBQ4 = true
                case .attnOut: skeleton[job.il]!.attnOut = q4; skeleton[job.il]!.attnOutQ4 = true
                default: skeleton[job.il]!.attnOutA = q4; skeleton[job.il]!.attnOutAQ4 = true
                }
            }
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

    // MARK: Q4 requant cache (<gguf>.q4dense)
    //
    // Layout: "DSQ4" | version u32 | modelSize u64 | jobCount u32
    //         then per job: il u32 | field u32 | bytes u64 | fileOffset u64
    //         then the Q4_K blobs (4 KB aligned).
    private static let q4CacheMagic: UInt32 = 0x34515344   // "DSQ4" little-endian

    /// Engine-side stderr log (the demo shows it inline; the app captures
    /// stderr into the engine log). The cache is infrastructure the user can't
    /// see failing otherwise.
    private static func logQ4(_ s: String) {
        FileHandle.standardError.write(Data(("DS4 q4cache: " + s + "\n").utf8))
    }

    /// Load the requant cache if it exists and matches the job list exactly.
    /// Reads through F_NOCACHE straight into the resident buffers (~0.5 s for
    /// ~1.4 GB) — the requant is paid only on the very first load.
    private static func loadQ4Cache(_ rt: MetalRuntime, path: String, modelSize: Int,
                                    jobs: [(il: Int, f: Field, t: GGUFModel.Tensor)]) -> [GPUTensor]? {
        let cfd = open(path, O_RDONLY)
        guard cfd >= 0 else {
            logQ4("assente (\(String(cString: strerror(errno)))) — riquantizzo: \(path)")
            return nil
        }
        defer { close(cfd) }
        _ = fcntl(cfd, F_NOCACHE, 1)
        let headBytes = 20 + jobs.count * 24
        var head = [UInt8](repeating: 0, count: headBytes)
        let okHead = head.withUnsafeMutableBytes {
            GGUFWeights.preadFull(cfd, into: $0.baseAddress!, bytes: headBytes, offset: 0)
        }
        guard okHead else { return nil }
        func u32(_ o: Int) -> UInt32 { head.withUnsafeBytes { $0.loadUnaligned(fromByteOffset: o, as: UInt32.self) } }
        func u64(_ o: Int) -> UInt64 { head.withUnsafeBytes { $0.loadUnaligned(fromByteOffset: o, as: UInt64.self) } }
        guard u32(0) == q4CacheMagic, u32(4) == 1,
              u64(8) == UInt64(modelSize), u32(16) == UInt32(jobs.count) else {
            logQ4("incompatibile (header/modello diversi) — riquantizzo: \(path)")
            return nil
        }
        var records: [(bytes: Int, offset: Int)] = []
        for (i, job) in jobs.enumerated() {
            let o = 20 + i * 24
            let expected = Int(job.t.elements) / 256 * 144
            guard u32(o) == UInt32(job.il), u32(o + 4) == UInt32(job.f.rawValue),
                  u64(o + 8) == UInt64(expected) else { return nil }
            records.append((bytes: expected, offset: Int(u64(o + 16))))
        }
        var out = [GPUTensor?](repeating: nil, count: jobs.count)
        var failed = false
        let lock = NSLock()
        out.withUnsafeMutableBufferPointer { buf in
            DispatchQueue.concurrentPerform(iterations: records.count) { i in
                guard let t = try? GPUTensor.uninitializedBytes(rt, byteLength: records[i].bytes,
                                                                elementCount: records[i].bytes),
                      GGUFWeights.preadFull(cfd, into: t.buffer.contents(),
                                            bytes: records[i].bytes, offset: records[i].offset) else {
                    lock.lock(); failed = true; lock.unlock()
                    return
                }
                buf[i] = t
                LoadProgress.shared.advance()
            }
        }
        guard !failed, out.compactMap({ $0 }).count == jobs.count else {
            logQ4("lettura fallita — riquantizzo: \(path)")
            return nil
        }
        logQ4("caricata (\(jobs.count) tensori): \(path)")
        return out.map { $0! }
    }

    /// Persist the converted tensors (best-effort: failures leave no cache and
    /// the next load simply requantizes again).
    private static func writeQ4Cache(path: String, modelSize: Int,
                                     jobs: [(il: Int, f: Field, t: GGUFModel.Tensor)],
                                     tensors: [GPUTensor]) {
        let align = 4096
        let headBytes = 20 + jobs.count * 24
        var offset = (headBytes + align - 1) / align * align
        var head = Data(capacity: headBytes)
        func put32(_ v: UInt32) { withUnsafeBytes(of: v.littleEndian) { head.append(contentsOf: $0) } }
        func put64(_ v: UInt64) { withUnsafeBytes(of: v.littleEndian) { head.append(contentsOf: $0) } }
        put32(q4CacheMagic); put32(1); put64(UInt64(modelSize)); put32(UInt32(jobs.count))
        var offsets: [Int] = []
        for (i, job) in jobs.enumerated() {
            put32(UInt32(job.il)); put32(UInt32(job.f.rawValue))
            put64(UInt64(tensors[i].byteLength)); put64(UInt64(offset))
            offsets.append(offset)
            offset += (tensors[i].byteLength + align - 1) / align * align
        }
        guard FileManager.default.createFile(atPath: path, contents: nil),
              let fh = FileHandle(forWritingAtPath: path) else {
            logQ4("SCRITTURA FALLITA (permessi/percorso?): \(path)")
            return
        }
        defer { try? fh.close() }
        do {
            try fh.write(contentsOf: head)
            for (i, t) in tensors.enumerated() {
                try fh.seek(toOffset: UInt64(offsets[i]))
                try fh.write(contentsOf: Data(bytesNoCopy: t.buffer.contents(),
                                              count: t.byteLength, deallocator: .none))
            }
            logQ4("scritta (\(tensors.count) tensori, \(offset / 1_048_576) MB): \(path)")
        } catch {
            logQ4("SCRITTURA FALLITA (\(error)) — file rimosso: \(path)")
            try? FileManager.default.removeItem(atPath: path)   // never leave a torn cache
        }
    }

    /// Q8_0 → Q4_K requant of one tensor into a resident buffer (DS4_DENSE_Q4).
    /// Returns nil when the tensor is not Q8_0 (or not superblock-shaped): the
    /// caller keeps streaming it unchanged. Reads the source bytes through the
    /// F_NOCACHE descriptor (no page-cache footprint at load either).
    private static func requantQ4(_ rt: MetalRuntime, _ model: GGUFModel, fd: Int32,
                                  tensor t: GGUFModel.Tensor) throws -> GPUTensor? {
        guard let info = GGUF.typeInfo(t.type), info.name == "q8_0" else { return nil }
        let elems = Int(t.elements)
        guard elems % 256 == 0 else { return nil }
        var q8 = [UInt8](repeating: 0, count: Int(t.bytes))
        let ok = q8.withUnsafeMutableBytes {
            GGUFWeights.preadFull(fd, into: $0.baseAddress!, bytes: $0.count, offset: Int(t.absOffset))
        }
        guard ok else {
            throw GGUFWeights.LoadError.message("DenseStreamer: pread failed requantizing \(t.name)")
        }
        var f32 = [Float](repeating: 0, count: elems)
        q8.withUnsafeBytes { src in
            f32.withUnsafeMutableBufferPointer { dst in
                Quantize.dequantQ8_0(src.baseAddress!, count: elems, into: dst.baseAddress!)
            }
        }
        let outBytes = elems / 256 * 144
        let gpu = try GPUTensor.uninitializedBytes(rt, byteLength: outBytes, elementCount: outBytes)
        f32.withUnsafeBufferPointer {
            Quantize.quantizeQ4_K($0.baseAddress!, count: elems, into: gpu.buffer.contents())
        }
        return gpu
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
