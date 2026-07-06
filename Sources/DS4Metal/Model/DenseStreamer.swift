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
// F_NOCACHE into a staging ring, kicked ahead so the SSD read of the next
// layer(s) overlaps the GPU compute of layer i (DS4_DENSE_AHEAD, default 1;
// =2 keeps i+1 AND i+2 in flight so the disk never idles inside a layer):
//
//   GPU:  [ compute layer i   ][ compute layer i+1 ] …
//   SSD:  [ read dense i+1    ][ read dense i+2    ] …
//
// RAM cost: 2 slots × max-layer ≈ 300 MB (instead of ~6.2 GB resident), and
// zero page-cache footprint (F_NOCACHE). Same bytes → identical numerics.
//
// Concurrency contract: `weights(_:)` is called from the DECODE thread only
// (the layerProvider), one layer at a time. With the ASYNC routed FFN
// (DS4_ASYNC_FFN) layer i-1's FFN command buffer may still be in flight when
// layer i's provider refills a slot — that is safe because the routed FFN
// reads NO streamed dense slab (experts pool + scratch only; the shared-FFN
// cb that does read the staged slabs is waited inside its own runLayer, and
// the route cb commits synchronously). If the routed FFN ever gains a dense
// read, this contract must be revisited. The background loader touches only
// the file descriptor and the target slot, and hands completion back through
// a semaphore (happens-before for the bytes).
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
    /// Read-ahead depth (DS4_DENSE_AHEAD, default 1 = the classic 2-slot ring).
    /// 2 keeps layers i+1 AND i+2 in flight while the GPU computes i: when a
    /// layer's read finishes before its compute, the SSD starts the next one
    /// instead of idling. Costs one extra staging slot (~max-layer bytes).
    private let ahead: Int
    private let slots: [MTLBuffer]                  // ahead+1 staging slots
    private var slotLayer: [Int]                    // slot -> layer currently staged (-1 = none/being written)
    private var pending: [Pending] = []             // decode-thread-owned (≤ ahead in flight)

    /// Total bytes streamed per full pass over `layers` (diagnostics).
    public private(set) var bytesPerPass = 0

    /// `skipIndexerScoring`: don't stage the indexer SCORING projections
    /// (indexer.attn_q_b + indexer.proj). They are read ONLY by the top-K
    /// relevance scoring (DecodeLayer step 3.5), which the caller has proven
    /// can never activate at this context size — so streaming them is pure
    /// wasted SSD bandwidth (~360 MB/token on Flash). The indexer COMPRESSOR
    /// pair (indexer_compressor_kv/gate) keeps streaming (unless `residentComp`
    /// pins it): its recurrent state updates every token and must stay
    /// coherent (KV snapshots export it).
    ///
    /// `residentComp`: load the four NSA compressor projections
    /// (attn_compressor_kv/gate + indexer_compressor_kv/gate) into RESIDENT
    /// buffers ONCE instead of re-streaming them every token. They are small
    /// per layer (F16, ~20 MB on a ratio-4 layer) but exist on 41 of 43 layers
    /// — ~0.6 GB/token of SSD stream traded for ~0.6 GB of resident RAM.
    /// Same bytes → identical numerics.
    public init(rt: MetalRuntime, model: GGUFModel, layers: Range<Int>, lockResident: Bool = false,
                q4Dense: Bool = false, skipIndexerScoring: Bool = false,
                residentComp: Bool = false) throws {
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
        // DS4_SHARED_Q4 (opt-in, needs q4Dense): also requantize the shared-expert
        // FFN projections — their Q8 slabs leave the per-token stream entirely,
        // freeing disk bandwidth for the expert gather. Lossy like the attn trio.
        let sharedQ4 = q4Dense && ProcessInfo.processInfo.environment["DS4_SHARED_Q4"] == "1"
        var scoringSkipped = 0                  // bytes/pass NOT staged (diagnostics)
        // DS4_RESIDENT_COMP: NSA compressor projections diverted to resident
        // buffers (loaded once after the plan is built, like the Q4 jobs).
        var compJobs: [(il: Int, f: Field, t: GGUFModel.Tensor)] = []
        LoadProgress.shared.begin("Preparazione layer densi…", from: 0.08, to: 0.30, units: layers.count)
        for il in layers {
            var plan: [Entry] = []
            var off = 0
            let w = try GGUFWeights.layerSmallSkeleton(rt, model, il)
            LoadProgress.shared.advance()
            for f in Field.allCases {
                guard let t = model.findTensor("blk.\(il).\(f.tensorName)") else { continue }
                if skipIndexerScoring, f == .idxQB || f == .idxProj {
                    // Not staged: w.idxQB/idxProj stay nil and the decoder's
                    // scoring gate (hasIdxScoring) keeps the top-K path off.
                    scoringSkipped += Int(t.bytes)
                    continue
                }
                if residentComp, f == .compKv || f == .compGate || f == .idxKv || f == .idxGate {
                    compJobs.append((il: il, f: f, t: t))
                    continue
                }
                let attnQ4Field = f == .qB || f == .attnOut || f == .attnOutA
                let sharedQ4Field = sharedQ4 && (f == .sharedGate || f == .sharedUp || f == .sharedDown)
                if q4Dense, attnQ4Field || sharedQ4Field,
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
        if scoringSkipped > 0 {
            FileHandle.standardError.write(Data(
                ("DS4 dense-stream: indexer top-k mai attivo a questo contesto — " +
                 "salto lo staging di indexer.attn_q_b/proj (\(scoringSkipped / (1 << 20)) MB/token in meno dal disco)\n").utf8))
        }
        if !compJobs.isEmpty {
            // One synchronous pread per projection (~0.6 GB total ≈ 0.1-0.2 s
            // at load): small enough not to need the Q4 jobs' parallelism or a
            // sidecar cache — the bytes are copied VERBATIM, no conversion.
            LoadProgress.shared.set(0.31, "Compressori NSA residenti…")
            var residentBytes = 0
            for job in compJobs {
                let bytes = Int(job.t.bytes)
                guard let buf = rt.device.makeBuffer(length: bytes, options: .storageModeShared) else {
                    throw MetalError.bufferAlloc
                }
                guard GGUFWeights.preadFull(fd, into: buf.contents(), bytes: bytes,
                                            offset: Int(job.t.absOffset)) else {
                    throw GGUFWeights.LoadError.message(
                        "DenseStreamer: pread failed on blk.\(job.il).\(job.f.tensorName)")
                }
                if lockResident { _ = mlock(buf.contents(), bytes) }
                let tensor = GPUTensor(buffer: buf, byteLength: bytes, count: bytes, byteOffset: 0)
                switch job.f {
                case .compKv: skeleton[job.il]!.compKv = tensor
                case .compGate: skeleton[job.il]!.compGate = tensor
                case .idxKv: skeleton[job.il]!.idxKv = tensor
                case .idxGate: skeleton[job.il]!.idxGate = tensor
                default: break
                }
                residentBytes += bytes
            }
            FileHandle.standardError.write(Data(
                ("DS4 dense-stream: proiezioni compressori NSA residenti — " +
                 "\(residentBytes / (1 << 20)) MB wired, altrettanti MB/token in meno dal disco\n").utf8))
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
            // Content fingerprints of the source tensors (first 4 KB each): the
            // cache must match the MODEL BYTES, not just its size and layout.
            let hashes = q4Jobs.map { Self.sourceHash(fd: fd, tensor: $0.t) }
            LoadProgress.shared.begin("Lettura cache Q4…", from: 0.32, to: 0.92, units: q4Jobs.count)
            if let cached = Self.loadQ4Cache(rt, path: cachePath, modelSize: Int(model.size),
                                             jobs: q4Jobs, hashes: hashes) {
                converted = cached
            } else if cachePath != sibling,
                      let cached = Self.loadQ4Cache(rt, path: sibling, modelSize: Int(model.size),
                                                    jobs: q4Jobs, hashes: hashes) {
                converted = cached
                LoadProgress.shared.set(0.92, "Copia cache Q4…")
                Self.writeQ4Cache(path: cachePath, modelSize: Int(model.size), jobs: q4Jobs,
                                  tensors: cached, hashes: hashes)
            } else {
                LoadProgress.shared.begin("Riquantizzazione Q4 (solo il primo avvio)…",
                                          from: 0.32, to: 0.88, units: q4Jobs.count)
                var fresh = [GPUTensor?](repeating: nil, count: q4Jobs.count)
                let lock = NSLock()
                // nonisolated(unsafe): ogni iterazione scrive SOLO out[i] (indici
                // disgiunti), jobs/rt/model sono letti e basta, l'errore e'
                // protetto dal lock.
                nonisolated(unsafe) var firstError: Error?
                nonisolated(unsafe) let jobs = q4Jobs
                nonisolated(unsafe) let rtRef = rt
                nonisolated(unsafe) let modelRef = model
                try fresh.withUnsafeMutableBufferPointer { out in
                    nonisolated(unsafe) let outBase = out.baseAddress!
                    DispatchQueue.concurrentPerform(iterations: jobs.count) { i in
                        do {
                            outBase[i] = try Self.requantQ4(rtRef, modelRef, fd: fd, tensor: jobs[i].t)
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
                Self.writeQ4Cache(path: cachePath, modelSize: Int(model.size), jobs: q4Jobs,
                                  tensors: converted, hashes: hashes)
            }
            for (i, job) in q4Jobs.enumerated() {
                let q4 = converted[i]
                if lockResident { q4.lockResident() }
                switch job.f {
                case .qB: skeleton[job.il]!.qB = q4; skeleton[job.il]!.qBQ4 = true
                case .attnOut: skeleton[job.il]!.attnOut = q4; skeleton[job.il]!.attnOutQ4 = true
                case .attnOutA: skeleton[job.il]!.attnOutA = q4; skeleton[job.il]!.attnOutAQ4 = true
                case .sharedGate: skeleton[job.il]!.sharedGate = q4; skeleton[job.il]!.sharedGateQ4 = true
                case .sharedUp: skeleton[job.il]!.sharedUp = q4; skeleton[job.il]!.sharedUpQ4 = true
                case .sharedDown: skeleton[job.il]!.sharedDown = q4; skeleton[job.il]!.sharedDownQ4 = true
                default: break
                }
            }
        }
        let aheadEnv = ProcessInfo.processInfo.environment["DS4_DENSE_AHEAD"].flatMap(Int.init) ?? 1
        self.ahead = min(max(1, aheadEnv), max(1, min(3, layers.count - 1)))
        var made: [MTLBuffer] = []
        for _ in 0...ahead {
            guard let b = rt.device.makeBuffer(length: maxSlot, options: .storageModeShared) else {
                throw MetalError.bufferAlloc
            }
            made.append(b)
        }
        slots = made
        slotLayer = Array(repeating: -1, count: made.count)
        if lockResident {
            // DS4_MLOCK: the staging ring is rewritten every ~70 ms — pin it so
            // the memory compressor never touches it. Best-effort.
            for b in made { _ = mlock(b.contents(), maxSlot) }
        }
    }

    /// The next `count` layers of the pass after `il` (wrapping to the range
    /// start), i.e. the layers whose staging slots must not be overwritten.
    private func upcoming(after il: Int, count: Int) -> [Int] {
        var out: [Int] = []
        var next = il
        for _ in 0..<count {
            next = next + 1 < layers.upperBound ? next + 1 : layers.lowerBound
            if next == il { break }
            out.append(next)
        }
        return out
    }

    /// LayerProvider entry point (DECODE thread only). Returns layer `il`'s
    /// weights with the big fields as views into a ready staging slot, then
    /// keeps the read-ahead pipeline `ahead` layers deep: with the default 1
    /// this is the classic 2-slot ring (read i+1 while computing i); with
    /// DS4_DENSE_AHEAD=2 the SSD moves on to i+2 as soon as i+1 lands instead
    /// of idling for the rest of layer i's compute. Overwrite safety: a slot is
    /// reused only when its occupant is neither the current layer nor one of
    /// the next `ahead` layers; the only async cb that can still be in flight
    /// here (the routed FFN) reads no staged slab — see the class contract.
    public func weights(_ il: Int) throws -> LayerWeights {
        let slot: Int
        if let pi = pending.firstIndex(where: { $0.layer == il }) {
            let p = pending.remove(at: pi)
            p.sem.wait()                        // usually already signalled (read ran during layer i-1)
            if let e = p.error { throw e }
            slotLayer[p.slot] = il
            slot = p.slot
        } else if let s = slotLayer.firstIndex(of: il) {
            slot = s                            // already staged (e.g. retry after an error)
        } else {
            // Cold start or out-of-order request: drain every in-flight load,
            // then read synchronously into a slot we won't need imminently.
            for p in pending {
                p.sem.wait()
                if p.error == nil { slotLayer[p.slot] = p.layer }
            }
            pending.removeAll()
            let wanted = Set(upcoming(after: il, count: ahead))
            slot = (0..<slots.count).first { !wanted.contains(slotLayer[$0]) } ?? 0
            try load(il, into: slot)
            slotLayer[slot] = il
        }
        // Top up the pipeline: for each of the next `ahead` layers not already
        // staged or in flight, start a background read into a reusable slot.
        let wanted = Set(upcoming(after: il, count: ahead) + [il])
        for next in upcoming(after: il, count: ahead) {
            if slotLayer.contains(next) || pending.contains(where: { $0.layer == next }) { continue }
            guard let free = (0..<slots.count).first(where: { s in
                s != slot && !wanted.contains(slotLayer[s]) && !pending.contains(where: { $0.slot == s })
            }) else { break }
            let p = Pending(layer: next, slot: free)
            slotLayer[free] = -1                // being overwritten
            pending.append(p)
            DispatchQueue.global(qos: .userInitiated).async {
                do { try self.load(next, into: p.slot) } catch { p.error = error }
                p.sem.signal()
            }
        }
        return makeWeights(il, slot: slot)
    }

    // MARK: Q4 requant cache (<gguf>.q4dense)
    //
    // Layout (v2): "DSQ4" | version u32 | modelSize u64 | jobCount u32
    //   then per job: il u32 | field u32 | q4Bytes u64 | srcHash u64 | blobOffset u64
    //   then the Q4_K blobs (4 KB aligned). `srcHash` fingerprints the SOURCE
    //   tensor's first 4 KB, so replacing the GGUF with a same-size revision
    //   invalidates the cache (size+shape alone could not tell them apart).
    //   v1 caches (no srcHash) are rejected and re-requantized once.
    private static let q4CacheMagic: UInt32 = 0x34515344   // "DSQ4" little-endian
    private static let q4CacheVersion: UInt32 = 2

    /// FNV-1a over the first 4 KB of the source tensor (via the F_NOCACHE fd):
    /// a cheap content fingerprint for cache validation. 0 on read failure —
    /// which then simply mismatches and forces a requant.
    private static func sourceHash(fd: Int32, tensor t: GGUFModel.Tensor) -> UInt64 {
        let n = min(4096, Int(t.bytes))
        var buf = [UInt8](repeating: 0, count: max(1, n))
        let ok = buf.withUnsafeMutableBytes {
            GGUFWeights.preadFull(fd, into: $0.baseAddress!, bytes: n, offset: Int(t.absOffset))
        }
        guard ok else { return 0 }
        var h: UInt64 = 0xcbf2_9ce4_8422_2325
        for b in buf { h = (h ^ UInt64(b)) &* 0x1_0000_0000_01b3 }
        return h
    }

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
                                    jobs: [(il: Int, f: Field, t: GGUFModel.Tensor)],
                                    hashes: [UInt64]) -> [GPUTensor]? {
        let cfd = open(path, O_RDONLY)
        guard cfd >= 0 else {
            logQ4("assente (\(String(cString: strerror(errno)))) — riquantizzo: \(path)")
            return nil
        }
        defer { close(cfd) }
        _ = fcntl(cfd, F_NOCACHE, 1)
        let headBytes = 20 + jobs.count * 32
        var head = [UInt8](repeating: 0, count: headBytes)
        let okHead = head.withUnsafeMutableBytes {
            GGUFWeights.preadFull(cfd, into: $0.baseAddress!, bytes: headBytes, offset: 0)
        }
        guard okHead else {
            logQ4("header illeggibile — riquantizzo: \(path)")
            return nil
        }
        func u32(_ o: Int) -> UInt32 { head.withUnsafeBytes { $0.loadUnaligned(fromByteOffset: o, as: UInt32.self) } }
        func u64(_ o: Int) -> UInt64 { head.withUnsafeBytes { $0.loadUnaligned(fromByteOffset: o, as: UInt64.self) } }
        guard u32(0) == q4CacheMagic, u32(4) == q4CacheVersion,
              u64(8) == UInt64(modelSize), u32(16) == UInt32(jobs.count) else {
            logQ4("incompatibile (versione/modello diversi) — riquantizzo: \(path)")
            return nil
        }
        var records: [(bytes: Int, offset: Int)] = []
        for (i, job) in jobs.enumerated() {
            let o = 20 + i * 32
            let expected = Int(job.t.elements) / 256 * 144
            guard u32(o) == UInt32(job.il), u32(o + 4) == UInt32(job.f.rawValue),
                  u64(o + 8) == UInt64(expected), u64(o + 16) == hashes[i] else {
                logQ4("incompatibile (tensore \(job.t.name) diverso) — riquantizzo: \(path)")
                return nil
            }
            records.append((bytes: expected, offset: Int(u64(o + 24))))
        }
        var out = [GPUTensor?](repeating: nil, count: jobs.count)
        let lock = NSLock()
        // nonisolated(unsafe): scritture su indici DISGIUNTI (buf[i]), records
        // e rt in sola lettura, flag di errore protetto dal lock.
        nonisolated(unsafe) var failed = false
        nonisolated(unsafe) let recs = records
        nonisolated(unsafe) let rtRef = rt
        out.withUnsafeMutableBufferPointer { buf in
            nonisolated(unsafe) let bufBase = buf.baseAddress!
            DispatchQueue.concurrentPerform(iterations: recs.count) { i in
                guard let t = try? GPUTensor.uninitializedBytes(rtRef, byteLength: recs[i].bytes,
                                                                elementCount: recs[i].bytes),
                      GGUFWeights.preadFull(cfd, into: t.buffer.contents(),
                                            bytes: recs[i].bytes, offset: recs[i].offset) else {
                    lock.lock(); failed = true; lock.unlock()
                    return
                }
                bufBase[i] = t
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
    /// the next load simply requantizes again). Written to a .tmp sibling and
    /// renamed into place, so a crash mid-write can never leave a torn cache
    /// under the real name (and a concurrent reader sees old-or-new, never half).
    private static func writeQ4Cache(path: String, modelSize: Int,
                                     jobs: [(il: Int, f: Field, t: GGUFModel.Tensor)],
                                     tensors: [GPUTensor], hashes: [UInt64]) {
        let align = 4096
        let headBytes = 20 + jobs.count * 32
        var offset = (headBytes + align - 1) / align * align
        var head = Data(capacity: headBytes)
        func put32(_ v: UInt32) { withUnsafeBytes(of: v.littleEndian) { head.append(contentsOf: $0) } }
        func put64(_ v: UInt64) { withUnsafeBytes(of: v.littleEndian) { head.append(contentsOf: $0) } }
        put32(q4CacheMagic); put32(q4CacheVersion); put64(UInt64(modelSize)); put32(UInt32(jobs.count))
        var offsets: [Int] = []
        for (i, job) in jobs.enumerated() {
            put32(UInt32(job.il)); put32(UInt32(job.f.rawValue))
            put64(UInt64(tensors[i].byteLength)); put64(hashes[i]); put64(UInt64(offset))
            offsets.append(offset)
            offset += (tensors[i].byteLength + align - 1) / align * align
        }
        let tmp = path + ".tmp"
        guard FileManager.default.createFile(atPath: tmp, contents: nil),
              let fh = FileHandle(forWritingAtPath: tmp) else {
            logQ4("SCRITTURA FALLITA (permessi/percorso?): \(tmp)")
            return
        }
        do {
            try fh.write(contentsOf: head)
            for (i, t) in tensors.enumerated() {
                try fh.seek(toOffset: UInt64(offsets[i]))
                try fh.write(contentsOf: Data(bytesNoCopy: t.buffer.contents(),
                                              count: t.byteLength, deallocator: .none))
            }
            try fh.close()
            try? FileManager.default.removeItem(atPath: path)
            try FileManager.default.moveItem(atPath: tmp, toPath: path)
            logQ4("scritta (\(tensors.count) tensori, \(offset / 1_048_576) MB): \(path)")
        } catch {
            try? fh.close()
            logQ4("SCRITTURA FALLITA (\(error)) — file rimosso: \(tmp)")
            try? FileManager.default.removeItem(atPath: tmp)   // never leave a torn cache
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
        let outBytes = elems / 256 * 144
        let gpu = try GPUTensor.uninitializedBytes(rt, byteLength: outBytes, elementCount: outBytes)
        // Convert per 256-element superblock (8 Q8_0 blocks of 34 B = 272 B in,
        // one 144 B Q4_K superblock out) with a 1 KB scratch, instead of
        // materializing the whole tensor in f32 (up to ~150 MB per job — ×N
        // concurrent requant workers that was a multi-GB load-time spike).
        // Blocks are independent, so the numerics are identical.
        let nsb = elems / 256
        var scratch = [Float](repeating: 0, count: 256)
        q8.withUnsafeBytes { src in
            let base = src.baseAddress!
            let dst = gpu.buffer.contents()
            scratch.withUnsafeMutableBufferPointer { s in
                for sb in 0..<nsb {
                    Quantize.dequantQ8_0(base + sb * 272, count: 256, into: s.baseAddress!)
                    Quantize.quantizeQ4_K(s.baseAddress!, count: 256, into: dst + sb * 144)
                }
            }
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
        // nonisolated(unsafe): ogni pread scrive un range DISGIUNTO dello slot
        // (stageOffset per-tensore); il flag di errore e' protetto dal lock.
        nonisolated(unsafe) let base = slots[slot].contents()
        let lock = NSLock()
        nonisolated(unsafe) var failed = false
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
