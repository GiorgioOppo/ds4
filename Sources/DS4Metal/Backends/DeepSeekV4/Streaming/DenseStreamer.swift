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
    let fd: Int32
    private let device: MTLDevice
    private let lockIndexerResident: Bool
    let layers: Range<Int>
    var entries: [Int: [Entry]] = [:]       // layer -> read/stage plan
    var skeleton: [Int: LayerWeights] = [:] // layer -> small-resident fields
    /// Scoring projections removed from the per-token staging plan while
    /// DS4_LAZY_IDX is active. Their GGUF locations cost only a few bytes of
    /// metadata until the live context crosses the sparse-indexer frontier.
    private var lazyIndexerJobs: [(il: Int, f: Field, t: GGUFModel.Tensor)] = []
    private let lazyIndexerLock = NSLock()
    private var indexerScoringActivated = false
    /// Read-ahead depth (DS4_DENSE_AHEAD, default 1 = the classic 2-slot ring).
    /// 2 keeps layers i+1 AND i+2 in flight while the GPU computes i: when a
    /// layer's read finishes before its compute, the SSD starts the next one
    /// instead of idling. Costs one extra staging slot (~max-layer bytes).
    let ahead: Int
    /// Exact allocation added by each increment of `DS4_DENSE_AHEAD` for this
    /// model/configuration. The ring owns `ahead + 1` equally sized buffers.
    public let stagingBytesPerAheadSlot: Int
    let slots: [MTLBuffer]                  // ahead+1 staging slots
    var slotLayer: [Int]                    // slot -> layer currently staged (-1 = none/being written)
    var pending: [Pending] = []             // decode-thread-owned (≤ ahead in flight)

    /// Total bytes streamed per full pass over `layers` (diagnostics).
    public private(set) var bytesPerPass = 0

    /// `lazyIndexerScoring`: remove the indexer SCORING projections
    /// (indexer.attn_q_b + indexer.proj) from the per-token stream. They are
    /// read only by top-K relevance scoring, so `activateIndexerScoring()`
    /// copies them once into resident buffers when the live context actually
    /// reaches that path. Until then their RAM and SSD cost is zero. The
    /// indexer COMPRESSOR pair (indexer_compressor_kv/gate) keeps streaming
    /// (unless `residentComp` pins it): its recurrent state updates every token
    /// and must stay coherent (KV snapshots export it).
    ///
    /// `residentComp`: load the four NSA compressor projections
    /// (attn_compressor_kv/gate + indexer_compressor_kv/gate) into RESIDENT
    /// buffers ONCE instead of re-streaming them every token. They are small
    /// per layer (F16, ~20 MB on a ratio-4 layer) but exist on 41 of 43 layers
    /// — ~0.6 GB/token of SSD stream traded for ~0.6 GB of resident RAM.
    /// Same bytes → identical numerics.
    public init(rt: MetalRuntime, model: GGUFModel, layers: Range<Int>, lockResident: Bool = false,
                q4Dense: Bool = false, lazyIndexerScoring: Bool = false,
                residentComp: Bool = false) throws {
        guard let fd = model.uncachedFD() else {
            throw GGUFWeights.LoadError.message("DenseStreamer: cannot open F_NOCACHE descriptor")
        }
        self.fd = fd
        self.device = rt.device
        self.lockIndexerResident = lockResident
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
        // AProjQ4 checkpoints already carry the same projections as canonical
        // Q4_K bytes.  With the Q4 residency knobs enabled, copy those bytes
        // directly once: no requantization and no .q4dense sidecar.
        var nativeQ4Jobs: [(il: Int, f: Field, t: GGUFModel.Tensor)] = []
        // DS4_SHARED_Q4 (opt-in, needs q4Dense): also requantize the shared-expert
        // FFN projections — their Q8 slabs leave the per-token stream entirely,
        // freeing disk bandwidth for the expert gather. Lossy like the attn trio.
        let sharedQ4 = q4Dense && ProcessInfo.processInfo.environment["DS4_SHARED_Q4"] == "1"
        // Experimental: halve resident compressor traffic/RAM by converting
        // their original F16 projections to Q8_0 once. Kept behind an explicit
        // knob because this changes model numerics.
        let compQ8 = residentComp && ProcessInfo.processInfo.environment["DS4_COMP_Q8"] == "1"
        // DS4_QKV_Q4 (opt-in, needs q4Dense): also requantize q_a and kv — the
        // last mid-size Q8 attention slabs still streamed (~16 MB/layer:
        // ~0.7 GB/token off the stream for ~0.35 GB resident). Same cache and
        // checkpoint machinery: a cache built WITHOUT this knob stays valid and
        // only the new tensors are requantized (records match per key). Lossy
        // like the others — A/B before adopting as default.
        let qkvQ4 = q4Dense && ProcessInfo.processInfo.environment["DS4_QKV_Q4"] == "1"
        var scoringDeferred = 0                 // bytes/pass NOT staged (diagnostics)
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
                if lazyIndexerScoring, f == .idxQB || f == .idxProj {
                    // Never put scoring tensors in a staging slot: retain only
                    // their GGUF jobs for the one-shot resident activation.
                    lazyIndexerJobs.append((il: il, f: f, t: t))
                    scoringDeferred += Int(t.bytes)
                    continue
                }
                if residentComp, f == .compKv || f == .compGate || f == .idxKv || f == .idxGate {
                    compJobs.append((il: il, f: f, t: t))
                    continue
                }
                let attnQ4Field = f == .qB || f == .attnOut || f == .attnOutA
                let sharedQ4Field = sharedQ4 && (f == .sharedGate || f == .sharedUp || f == .sharedDown)
                let qkvQ4Field = qkvQ4 && (f == .qA || f == .kvW)
                if q4Dense, attnQ4Field || sharedQ4Field || qkvQ4Field,
                   Int(t.elements) % 256 == 0 {
                    if t.type == 8 { // q8_0: create/load the requant cache
                        q4Jobs.append((il: il, f: f, t: t))
                        continue
                    }
                    if t.type == 12 { // q4_K: already in the executable layout
                        nativeQ4Jobs.append((il: il, f: f, t: t))
                        continue
                    }
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
        if scoringDeferred > 0 {
            DS4Log.info("lazy-idx",
                "indexer.attn_q_b/proj fuori dal piano per-token — " +
                "\(scoringDeferred / (1 << 20)) MB/token in meno dal disco; " +
                "caricamento residente una sola volta alla soglia live")
        }
        if !compJobs.isEmpty {
            if compQ8 {
                try Self.loadQ8Compressors(rt: rt, model: model, fd: fd, jobs: compJobs,
                                           skeleton: &skeleton, lockResident: lockResident)
            } else {
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
                DS4Log.info("dense-stream",
                    "proiezioni compressori NSA residenti — " +
                    "\(residentBytes / (1 << 20)) MB wired, altrettanti MB/token in meno dal disco")
            }
        }
        if !nativeQ4Jobs.isEmpty {
            LoadProgress.shared.begin("Caricamento proiezioni Q4 native…",
                                      from: 0.32, to: 0.90,
                                      units: nativeQ4Jobs.count)
            var loaded = [GPUTensor?](repeating: nil, count: nativeQ4Jobs.count)
            let jobs = nativeQ4Jobs
            let deviceRef = rt.device
            let failureLock = NSLock()
            nonisolated(unsafe) var firstError: Error?
            try loaded.withUnsafeMutableBufferPointer { output in
                nonisolated(unsafe) let outputBase = output.baseAddress!
                DispatchQueue.concurrentPerform(iterations: jobs.count) { index in
                    let job = jobs[index]
                    do {
                        let bytes = Int(job.t.bytes)
                        guard let buffer = deviceRef.makeBuffer(
                            length: max(1, bytes), options: .storageModeShared)
                        else { throw MetalError.bufferAlloc }
                        guard GGUFWeights.preadFull(
                            fd, into: buffer.contents(), bytes: bytes,
                            offset: Int(job.t.absOffset))
                        else {
                            throw GGUFWeights.LoadError.message(
                                "DenseStreamer: pread Q4 nativo fallito su \(job.t.name)")
                        }
                        outputBase[index] = GPUTensor(
                            buffer: buffer, byteLength: bytes,
                            count: Int(job.t.elements))
                        LoadProgress.shared.advance()
                    } catch {
                        failureLock.lock()
                        if firstError == nil { firstError = error }
                        failureLock.unlock()
                    }
                }
                if let firstError { throw firstError }
            }
            var residentBytes = 0
            for (index, job) in jobs.enumerated() {
                guard let tensor = loaded[index] else {
                    throw GGUFWeights.LoadError.message(
                        "DenseStreamer: proiezione Q4 nativa mancante: \(job.t.name)")
                }
                if lockResident { tensor.lockResident() }
                switch job.f {
                case .qB: skeleton[job.il]!.qB = tensor
                case .attnOut: skeleton[job.il]!.attnOut = tensor
                case .attnOutA: skeleton[job.il]!.attnOutA = tensor
                case .qA: skeleton[job.il]!.qA = tensor
                case .kvW: skeleton[job.il]!.kvW = tensor
                case .sharedGate: skeleton[job.il]!.sharedGate = tensor
                case .sharedUp: skeleton[job.il]!.sharedUp = tensor
                case .sharedDown: skeleton[job.il]!.sharedDown = tensor
                default: break
                }
                residentBytes += tensor.byteLength
            }
            DS4Log.info("q4-native",
                "\(nativeQ4Jobs.count) proiezioni Q4_K lette direttamente dal GGUF — "
                + "\(residentBytes / (1 << 20)) MB residenti, nessuna cache .q4dense")
        }
        if !q4Jobs.isEmpty {
            // Requant CACHE: the converted Q4 tensors are persisted next to the
            // model (<gguf>.q4dense, ~1.4 GB) — the first load pays the requant
            // once, every later load preads the cache in ~0.5 s. Invalidated by
            // model size / job-list mismatches; write failures are ignored.
            // Partial CHECKPOINTS (same format, fewer records) are persisted
            // between requant batches: a first load interrupted mid-requant
            // resumes from the completed tensors instead of restarting.
            // Cache location: next to the model by default (demo/CLI), or in
            // DS4_Q4_CACHE_DIR when set — the SANDBOXED app can't write next
            // to a picker-selected file, so it points this at Application
            // Support. READING tries both places (a cache produced by the demo
            // next to the GGUF gets picked up and PROMOTED into the primary
            // location, so demo and app share one conversion when possible).
            let sibling = model.path + ".q4dense"
            var cachePath = sibling
            if let dir = ProcessInfo.processInfo.environment["DS4_Q4_CACHE_DIR"], !dir.isEmpty {
                // Loud failure: if this silently failed, every later write
                // would fail with an unhelpful "permessi/percorso?" and every
                // launch would re-pay the requant.
                do { try FileManager.default.createDirectory(atPath: dir, withIntermediateDirectories: true) }
                catch { Self.logQ4("creazione cartella cache FALLITA (\(error.localizedDescription)): \(dir)") }
                cachePath = dir + "/" + (model.path as NSString).lastPathComponent + ".q4dense"
            }
            // Distributed SLICE (layer range ≠ full model): reads are served by
            // the FULL cache too (loadQ4Cache matches records by key), but a
            // slice requant must never overwrite the full cache with a subset —
            // the next full run would reject it and re-requant everything, in
            // ping-pong. Slice writes go to a per-range name.
            let partial = layers.lowerBound > 0 ||
                model.findTensor("blk.\(layers.upperBound).attn_q_a.weight") != nil
            let sliceSuffix = ".L\(layers.lowerBound)-\(layers.upperBound - 1)"
            let writeCachePath = partial ? cachePath + sliceSuffix : cachePath
            var readCandidates = [cachePath]
            if partial { readCandidates.append(cachePath + sliceSuffix) }
            if cachePath != sibling {
                readCandidates.append(sibling)
                if partial { readCandidates.append(sibling + sliceSuffix) }
            }
            var converted: [GPUTensor]?
            var loadedFrom: String?
            // Content fingerprints of the source tensors (first 4 KB each): the
            // cache must match the MODEL BYTES, not just its size and layout.
            let hashes = q4Jobs.map { Self.sourceHash(fd: fd, tensor: $0.t) }
            LoadProgress.shared.begin("Lettura cache Q4…", from: 0.32, to: 0.92, units: q4Jobs.count)
            // Seed dalla cache migliore: una cache COMPLETA salta il requant;
            // un CHECKPOINT (cache parziale scritta da un load interrotto a
            // metà requant — force quit, crash, riavvio) pre-riempie ciò che
            // ha e solo i tensori mancanti vengono riconvertiti. Prima di
            // questo, un'interruzione perdeva TUTTO il lavoro: il .q4dense
            // non compariva mai e "solo il primo avvio" diventava ogni avvio.
            var seed: [GPUTensor?]?
            for cand in readCandidates {
                guard let cached = Self.loadQ4Cache(rt, path: cand, modelSize: Int(model.size),
                                                    jobs: q4Jobs, hashes: hashes) else { continue }
                if !cached.contains(where: { $0 == nil }) {
                    converted = cached.map { $0! }; loadedFrom = cand; break
                }
                if (seed?.compactMap({ $0 }).count ?? 0) < cached.compactMap({ $0 }).count {
                    seed = cached
                }
            }
            if let cached = converted, let src = loadedFrom, src != cachePath, !partial {
                // Promotion into the primary location (demo cache → app dir):
                // FULL caches only — a slice copy under the primary name would
                // be exactly the overwrite this block exists to prevent.
                LoadProgress.shared.set(0.92, "Copia cache Q4…")
                Self.writeQ4Cache(path: cachePath, modelSize: Int(model.size), jobs: q4Jobs,
                                  tensors: cached, hashes: hashes)
            }
            if converted == nil {
                var fresh = seed ?? [GPUTensor?](repeating: nil, count: q4Jobs.count)
                let missing = fresh.indices.filter { fresh[$0] == nil }
                // Barra in MB di sorgente Q8, non in tensori: un tensore può
                // richiedere minuti e con l'avanzamento per-job la barra
                // restava FERMA per tutto il primo batch — indistinguibile da
                // un blocco (e l'utente chiudeva l'app a forza). requantQ4
                // avanza di un'unità ogni ~1 MB convertito.
                let totalMB = max(1, missing.reduce(0) { $0 + Int(q4Jobs[$1].t.bytes) } >> 20)
                LoadProgress.shared.begin("Riquantizzazione Q4 (solo il primo avvio)…",
                                          from: 0.32, to: 0.88, units: totalMB)
                Self.logQ4("riquantizzo \(missing.count) di \(q4Jobs.count) tensori (~\(totalMB) MB Q8) " +
                           "con checkpoint periodici in: \(writeCachePath)")
                // Build non ottimizzata (Xcode Run in Debug, `swift build` senza
                // -c release): il quantizzatore scalare diventa 20-50× più lento
                // — ORE invece di minuti, con la CPU al 100% e la barra quasi
                // ferma. È la prima causa di "bloccato sulla riquantizzazione":
                // dillo forte nel log invece di lasciarlo indovinare.
                var debugBuild = false
                assert({ debugBuild = true; return true }())
                if debugBuild {
                    Self.logQ4("ATTENZIONE: build di DEBUG (-Onone) — la riquantizzazione può richiedere " +
                               "ORE invece di minuti. Compila in Release (`make app` oppure Xcode con " +
                               "configurazione Release) per pagarla una volta sola in pochi minuti.")
                }
                // PREFLIGHT di scrittura: crea SUBITO il file cache (header
                // valido, zero record) prima di pagare la conversione. Se la
                // scrittura non è possibile (permessi, spazio, percorso) lo si
                // scopre ORA con il motivo nel log — non dopo minuti/ore di
                // CPU; e il file visibile in cartella è la prova che i
                // checkpoint hanno dove atterrare.
                if Self.writeQ4Cache(path: writeCachePath, modelSize: Int(model.size),
                                     jobs: [], tensors: [], hashes: []) {
                    Self.logQ4("preflight: file cache creato subito, i checkpoint lo riempiranno: \(writeCachePath)")
                } else {
                    Self.logQ4("preflight FALLITO (motivo nella riga sopra): la cache NON potrà essere " +
                               "salvata — il requant procede solo in memoria e si ripeterà al prossimo avvio")
                }
                let lock = NSLock()
                // Ogni iterazione scrive SOLO out[i] (indici disgiunti),
                // jobs/rt/model sono letti e basta, l'errore e' protetto dal
                // lock. L'array di job e' Sendable; i riferimenti runtime e
                // model mantengono l'annotazione unsafe dedicata.
                let jobs = q4Jobs
                nonisolated(unsafe) let rtRef = rt
                nonisolated(unsafe) let modelRef = model
                // A BATCH con checkpoint: dopo ogni batch la cache parziale
                // viene persistita (stesso formato, meno record), così un load
                // ucciso a metà riparte dai soli tensori mancanti invece che
                // da zero. L'ordine di q4Jobs interfoglia layer e tipi di
                // tensore, quindi i batch restano bilanciati. 16 tensori
                // (~550 MB sorgente): il PRIMO checkpoint arriva presto — è la
                // prova visibile che il file viene scritto davvero.
                let batchSize = 16
                var doneCount = q4Jobs.count - missing.count
                for start in stride(from: 0, to: missing.count, by: batchSize) {
                    let batch = Array(missing[start..<min(start + batchSize, missing.count)])
                    nonisolated(unsafe) var firstError: Error?
                    try fresh.withUnsafeMutableBufferPointer { out in
                        nonisolated(unsafe) let outBase = out.baseAddress!
                        DispatchQueue.concurrentPerform(iterations: batch.count) { bi in
                            let i = batch[bi]
                            do {
                                outBase[i] = try Self.requantQ4(rtRef, modelRef, fd: fd, tensor: jobs[i].t)
                            } catch {
                                lock.lock()
                                if firstError == nil { firstError = error }
                                lock.unlock()
                            }
                        }
                        if let e = firstError { throw e }
                    }
                    doneCount += batch.count
                    if start + batchSize < missing.count {
                        var js: [(il: Int, f: Field, t: GGUFModel.Tensor)] = []
                        var ts: [GPUTensor] = []
                        var hs: [UInt64] = []
                        for (i, t) in fresh.enumerated() {
                            guard let t else { continue }
                            js.append(q4Jobs[i]); ts.append(t); hs.append(hashes[i])
                        }
                        Self.writeQ4Cache(path: writeCachePath, modelSize: Int(model.size),
                                          jobs: js, tensors: ts, hashes: hs)
                        Self.logQ4("checkpoint requant: \(doneCount)/\(q4Jobs.count) tensori")
                    }
                }
                let freshConverted = try fresh.enumerated().map { i, t -> GPUTensor in
                    guard let t else {
                        throw GGUFWeights.LoadError.message("DenseStreamer: requant failed on layer \(q4Jobs[i].il)")
                    }
                    return t
                }
                converted = freshConverted
                LoadProgress.shared.set(0.90, "Scrittura cache Q4…")
                Self.writeQ4Cache(path: writeCachePath, modelSize: Int(model.size), jobs: q4Jobs,
                                  tensors: freshConverted, hashes: hashes)
            }
            guard let q4Tensors = converted else {
                throw GGUFWeights.LoadError.message("DenseStreamer: cache Q4 non disponibile dopo il load")
            }
            for (i, job) in q4Jobs.enumerated() {
                let q4 = q4Tensors[i]
                if lockResident { q4.lockResident() }
                switch job.f {
                case .qB: skeleton[job.il]!.qB = q4; skeleton[job.il]!.qBQ4 = true
                case .attnOut: skeleton[job.il]!.attnOut = q4; skeleton[job.il]!.attnOutQ4 = true
                case .attnOutA: skeleton[job.il]!.attnOutA = q4; skeleton[job.il]!.attnOutAQ4 = true
                case .qA: skeleton[job.il]!.qA = q4; skeleton[job.il]!.qAQ4 = true
                case .kvW: skeleton[job.il]!.kvW = q4; skeleton[job.il]!.kvQ4 = true
                case .sharedGate: skeleton[job.il]!.sharedGate = q4; skeleton[job.il]!.sharedGateQ4 = true
                case .sharedUp: skeleton[job.il]!.sharedUp = q4; skeleton[job.il]!.sharedUpQ4 = true
                case .sharedDown: skeleton[job.il]!.sharedDown = q4; skeleton[job.il]!.sharedDownQ4 = true
                default: break
                }
            }
        }
        let aheadEnv = ProcessInfo.processInfo.environment["DS4_DENSE_AHEAD"].flatMap(Int.init) ?? 1
        self.ahead = min(max(1, aheadEnv), max(1, min(3, layers.count - 1)))
        self.stagingBytesPerAheadSlot = maxSlot
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

    /// Materialize every deferred indexer scoring projection exactly once.
    ///
    /// The decoder invokes this before asking the layer provider for the first
    /// sparse-indexer token. Holding the lock across the one-shot read makes the
    /// API idempotent even if a future caller races activation; a failed read
    /// installs nothing and remains retryable. `weights(_:)` and activation are
    /// otherwise decode-thread operations, while pending dense reads touch only
    /// staging slots and are independent from `skeleton`.
    public func activateIndexerScoring() throws {
        lazyIndexerLock.lock()
        defer { lazyIndexerLock.unlock() }
        guard !indexerScoringActivated else { return }
        guard !lazyIndexerJobs.isEmpty else {
            indexerScoringActivated = true
            return
        }

        var loaded: [(job: (il: Int, f: Field, t: GGUFModel.Tensor), tensor: GPUTensor)] = []
        loaded.reserveCapacity(lazyIndexerJobs.count)
        for job in lazyIndexerJobs {
            let bytes = Int(job.t.bytes)
            guard let buffer = device.makeBuffer(length: max(1, bytes), options: .storageModeShared) else {
                throw MetalError.bufferAlloc
            }
            loaded.append((job: job,
                           tensor: GPUTensor(buffer: buffer, byteLength: bytes,
                                             count: Int(job.t.elements))))
        }

        // Independent destination buffers let the NVMe service the one-time
        // activation at queue depth, instead of serializing dozens of slabs.
        let failureLock = NSLock()
        nonisolated(unsafe) var failedTensor: String?
        nonisolated(unsafe) let reads = loaded
        DispatchQueue.concurrentPerform(iterations: reads.count) { i in
            let item = reads[i]
            if !GGUFWeights.preadFull(fd, into: item.tensor.buffer.contents(),
                                      bytes: item.tensor.byteLength,
                                      offset: Int(item.job.t.absOffset)) {
                failureLock.lock()
                if failedTensor == nil { failedTensor = item.job.t.name }
                failureLock.unlock()
            }
        }
        if let failedTensor {
            throw GGUFWeights.LoadError.message(
                "DenseStreamer: pread lazy indexer failed on \(failedTensor)")
        }

        var residentBytes = 0
        for item in loaded {
            guard var w = skeleton[item.job.il] else {
                throw GGUFWeights.LoadError.message(
                    "DenseStreamer: missing skeleton for lazy indexer layer \(item.job.il)")
            }
            switch item.job.f {
            case .idxQB: w.idxQB = item.tensor
            case .idxProj: w.idxProj = item.tensor
            default: continue
            }
            if lockIndexerResident { _ = item.tensor.lockResident() }
            skeleton[item.job.il] = w
            residentBytes += item.tensor.byteLength
        }
        indexerScoringActivated = true
        lazyIndexerJobs.removeAll(keepingCapacity: false)
        DS4Log.info("lazy-idx",
            "scoring attivato — \(residentBytes / (1 << 20)) MB " +
            "caricati una volta in buffer residenti" +
            (lockIndexerResident ? " (mlock richiesto)" : ""))
    }
}
