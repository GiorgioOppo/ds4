import Foundation
import DS4Core

// DS4_EXPERT_BUNDLE=1: sidecar file (<gguf>.expbundle) that repacks the routed
// experts so each expert's gate|up|down slabs are CONTIGUOUS (4 KB-aligned
// records, ordered by layer then expert id).
//
// Why: a slot-cache miss reads the three slabs of ONE expert. In the GGUF they
// live in three different tensors — three ~2 MB reads scattered across the
// file. Measured on the 2-bit Flash: the gather runs at ~49% of the SSD's
// parallel ceiling. In the bundle the same bytes are adjacent: the three
// concurrent preads form one ~7 MB sequential burst per miss.
//
// Same bytes, same numerics — only the on-disk LAYOUT changes. The bundle
// duplicates the expert region of the model on disk (~ everything but the
// dense weights), so it is OPT-IN and skipped when free space is short.
// Built once next to the model (.tmp + rename, torn files impossible);
// validated by size/geometry + per-layer content fingerprints; any failure
// logs and falls back to the plain GGUF reads.
public final class ExpertBundle: @unchecked Sendable {
    private let fd: Int32
    private let layers: Range<Int>
    private let nExpert: Int
    private let gateBytes: Int, upBytes: Int, downBytes: Int
    private let dataBase: Int
    private let record: Int              // aligned gate+up+down record stride
    /// One-shot runtime PROOF in the engine log that misses are actually being
    /// served from the sidecar — "caricato" only proves the file validated.
    private let useLock = NSLock()
    private var usedOnce = false

    private func noteUse() {
        useLock.lock(); let first = !usedOnce; usedOnce = true; useLock.unlock()
        if first { Self.log("in uso: primo esperto servito dal sidecar") }
    }

    private static let magic: UInt32 = 0x4245_5344   // "DSEB" little-endian
    private static let version: UInt32 = 1
    private static let align = 4096

    private static func log(_ s: String) {
        FileHandle.standardError.write(Data(("DS4 expbundle: " + s + "\n").utf8))
    }

    deinit { close(fd) }

    private init(fd: Int32, layers: Range<Int>, nExpert: Int,
                 gateBytes: Int, upBytes: Int, downBytes: Int, dataBase: Int, record: Int) {
        self.fd = fd; self.layers = layers; self.nExpert = nExpert
        self.gateBytes = gateBytes; self.upBytes = upBytes; self.downBytes = downBytes
        self.dataBase = dataBase; self.record = record
    }

    // MARK: Layout

    private static func headerBytes(layerCount: Int) -> Int { 56 + layerCount * 8 }

    private static func recordStride(_ gate: Int, _ up: Int, _ down: Int) -> Int {
        (gate + up + down + align - 1) / align * align
    }

    /// FNV-1a over the first 4 KB of a layer's gate-experts tensor (source fd):
    /// the bundle must match the MODEL BYTES, not just its size and shape.
    private static func layerHash(fd: Int32, model: GGUFModel, layer: Int) -> UInt64 {
        guard let t = model.findTensor("blk.\(layer).ffn_gate_exps.weight") else { return 0 }
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

    // MARK: Open / build

    /// Open a valid sidecar, or build it (one-time, several minutes: it copies
    /// the whole expert region). Returns nil on ANY failure — callers keep the
    /// plain GGUF read path, so the bundle can never break inference.
    public static func openOrBuild(model: GGUFModel, layers: Range<Int>, nExpert: Int,
                                   gateBytes: Int, upBytes: Int, downBytes: Int) -> ExpertBundle? {
        guard nExpert > 0, gateBytes > 0, upBytes > 0, downBytes > 0 else { return nil }
        // Every layer in range must carry the three expert tensors sized as
        // expected — models with dense-only layers are not bundled (yet).
        for il in layers {
            for (name, bytes) in [("ffn_gate_exps.weight", gateBytes),
                                  ("ffn_up_exps.weight", upBytes),
                                  ("ffn_down_exps.weight", downBytes)] {
                guard let t = model.findTensor("blk.\(il).\(name)"),
                      Int(t.bytes) == bytes * nExpert else {
                    log("layer \(il) senza tensori esperti attesi — bundle non applicabile")
                    return nil
                }
            }
        }
        // NOTE: uncachedFD() is the model's SHARED, memoized F_NOCACHE descriptor
        // (closed only in the model's deinit) — it must NOT be closed here, or
        // every later pread on it (dense streamer hashes, q4 requant, expert
        // pread) fails at load.
        guard let srcFD = model.uncachedFD() else {
            log("descrittore F_NOCACHE non disponibile — salto")
            return nil
        }
        let hashes = layers.map { layerHash(fd: srcFD, model: model, layer: $0) }
        // Location: next to the model by default (demo/CLI). The SANDBOXED app
        // can neither read nor write siblings of a picker-selected file, so it
        // sets DS4_BUNDLE_DIR (Application Support): READING tries both places
        // (a bundle built by the demo next to the GGUF is reused when the
        // sandbox allows it — 72 GB are NOT copied around), building goes to
        // the directory when set.
        let sibling = model.path + ".expbundle"
        var candidates = [sibling]
        var buildPath = sibling
        if let dir = ProcessInfo.processInfo.environment["DS4_BUNDLE_DIR"], !dir.isEmpty {
            try? FileManager.default.createDirectory(atPath: dir, withIntermediateDirectories: true)
            let inDir = dir + "/" + (model.path as NSString).lastPathComponent + ".expbundle"
            candidates.append(inDir)
            buildPath = inDir
        }
        for path in candidates {
            if let b = openExisting(path: path, modelSize: Int(model.size), layers: layers, nExpert: nExpert,
                                    gateBytes: gateBytes, upBytes: upBytes, downBytes: downBytes, hashes: hashes) {
                log("caricato: \(path)")
                return b
            }
        }
        // Spell out WHERE we looked: "the file is there!" mysteries are always a
        // path/name mismatch, and this line settles them from the engine log.
        log("nessun bundle valido trovato in: " + candidates.joined(separator: "  |  "))
        return build(path: buildPath, model: model, srcFD: srcFD, layers: layers, nExpert: nExpert,
                     gateBytes: gateBytes, upBytes: upBytes, downBytes: downBytes, hashes: hashes)
    }

    /// Validate + open an existing bundle. nil = absent or mismatched.
    private static func openExisting(path: String, modelSize: Int, layers: Range<Int>, nExpert: Int,
                                     gateBytes: Int, upBytes: Int, downBytes: Int,
                                     hashes: [UInt64]) -> ExpertBundle? {
        let fd = open(path, O_RDONLY)
        guard fd >= 0 else { return nil }
        _ = fcntl(fd, F_NOCACHE, 1)
        let hb = headerBytes(layerCount: layers.count)
        var head = [UInt8](repeating: 0, count: hb)
        let okHead = head.withUnsafeMutableBytes {
            GGUFWeights.preadFull(fd, into: $0.baseAddress!, bytes: hb, offset: 0)
        }
        func u32(_ o: Int) -> UInt32 { head.withUnsafeBytes { $0.loadUnaligned(fromByteOffset: o, as: UInt32.self) } }
        func u64(_ o: Int) -> UInt64 { head.withUnsafeBytes { $0.loadUnaligned(fromByteOffset: o, as: UInt64.self) } }
        guard okHead,
              u32(0) == magic, u32(4) == version, u64(8) == UInt64(modelSize),
              u32(16) == UInt32(layers.lowerBound), u32(20) == UInt32(layers.count),
              u32(24) == UInt32(nExpert),
              u64(32) == UInt64(gateBytes), u64(40) == UInt64(upBytes), u64(48) == UInt64(downBytes) else {
            close(fd)
            log("incompatibile (header/modello diversi) — ricostruisco: \(path)")
            return nil
        }
        for (i, h) in hashes.enumerated() where u64(56 + i * 8) != h {
            close(fd)
            log("incompatibile (layer \(layers.lowerBound + i) diverso) — ricostruisco: \(path)")
            return nil
        }
        let rec = recordStride(gateBytes, upBytes, downBytes)
        let base = (hb + align - 1) / align * align
        var st = stat()
        guard fstat(fd, &st) == 0,
              Int(st.st_size) >= base + layers.count * nExpert * rec else {
            close(fd)
            log("troncato — ricostruisco: \(path)")
            return nil
        }
        return ExpertBundle(fd: fd, layers: layers, nExpert: nExpert,
                            gateBytes: gateBytes, upBytes: upBytes, downBytes: downBytes,
                            dataBase: base, record: rec)
    }

    /// One-time build: stream every expert's three slabs from the GGUF into
    /// contiguous records (.tmp + rename). Refuses when free disk is short.
    private static func build(path: String, model: GGUFModel, srcFD: Int32,
                              layers: Range<Int>, nExpert: Int,
                              gateBytes: Int, upBytes: Int, downBytes: Int,
                              hashes: [UInt64]) -> ExpertBundle? {
        let rec = recordStride(gateBytes, upBytes, downBytes)
        let hb = headerBytes(layerCount: layers.count)
        let base = (hb + align - 1) / align * align
        let totalBytes = base + layers.count * nExpert * rec
        let free = (try? FileManager.default.attributesOfFileSystem(forPath: (path as NSString).deletingLastPathComponent))?[.systemFreeSize] as? Int ?? 0
        guard free > totalBytes + (1 << 30) else {
            log("spazio disco insufficiente (~\(totalBytes >> 30) GB richiesti) — salto")
            return nil
        }
        log("costruzione (una tantum): \(layers.count) layer × \(nExpert) esperti, ~\(totalBytes >> 30) GB → \(path)")
        LoadProgress.shared.begin("Costruzione expert-bundle (una tantum)…", from: 0.05, to: 0.28,
                                  units: layers.count)
        var head = Data(capacity: hb)
        func put32(_ v: UInt32) { withUnsafeBytes(of: v.littleEndian) { head.append(contentsOf: $0) } }
        func put64(_ v: UInt64) { withUnsafeBytes(of: v.littleEndian) { head.append(contentsOf: $0) } }
        put32(magic); put32(version); put64(UInt64(model.size))
        put32(UInt32(layers.lowerBound)); put32(UInt32(layers.count))
        put32(UInt32(nExpert)); put32(0)
        put64(UInt64(gateBytes)); put64(UInt64(upBytes)); put64(UInt64(downBytes))
        for h in hashes { put64(h) }

        let tmp = path + ".tmp"
        guard FileManager.default.createFile(atPath: tmp, contents: nil),
              let fh = FileHandle(forWritingAtPath: tmp) else {
            log("SCRITTURA FALLITA (permessi/percorso?): \(tmp)")
            return nil
        }
        let buf = UnsafeMutableRawPointer.allocate(byteCount: rec, alignment: align)
        defer { buf.deallocate() }
        memset(buf, 0, rec)   // record padding stays zero
        do {
            try fh.write(contentsOf: head)
            try fh.write(contentsOf: Data(count: base - hb))   // pad to the aligned data base
            for il in layers {
                let gT = model.findTensor("blk.\(il).ffn_gate_exps.weight")!
                let uT = model.findTensor("blk.\(il).ffn_up_exps.weight")!
                let dT = model.findTensor("blk.\(il).ffn_down_exps.weight")!
                for e in 0..<nExpert {
                    // Source reads are sequential WITHIN each tensor (expert e
                    // then e+1) — the build itself runs near sequential speed.
                    guard GGUFWeights.preadFull(srcFD, into: buf, bytes: gateBytes,
                                                offset: Int(gT.absOffset) + e * gateBytes),
                          GGUFWeights.preadFull(srcFD, into: buf + gateBytes, bytes: upBytes,
                                                offset: Int(uT.absOffset) + e * upBytes),
                          GGUFWeights.preadFull(srcFD, into: buf + gateBytes + upBytes, bytes: downBytes,
                                                offset: Int(dT.absOffset) + e * downBytes) else {
                        throw GGUFWeights.LoadError.message("pread sorgente fallita (layer \(il) expert \(e))")
                    }
                    try fh.write(contentsOf: Data(bytesNoCopy: buf, count: rec, deallocator: .none))
                }
                LoadProgress.shared.advance()
            }
            try fh.close()
            try? FileManager.default.removeItem(atPath: path)
            try FileManager.default.moveItem(atPath: tmp, toPath: path)
        } catch {
            try? fh.close()
            log("COSTRUZIONE FALLITA (\(error)) — file rimosso: \(tmp)")
            try? FileManager.default.removeItem(atPath: tmp)
            return nil
        }
        log("scritto (~\(totalBytes >> 30) GB): \(path)")
        return openExisting(path: path, modelSize: Int(model.size), layers: layers, nExpert: nExpert,
                            gateBytes: gateBytes, upBytes: upBytes, downBytes: downBytes, hashes: hashes)
    }

    // MARK: Read path

    /// pread expert (layer, id)'s three slabs into the pool slots. The slabs
    /// are ADJACENT in the sidecar: the three concurrent preads form one ~7 MB
    /// sequential burst instead of three scattered ~2 MB random reads.
    /// Returns false on any error (caller falls back to the GGUF reads).
    public func copyExpert(layer: Int, id: Int32,
                           gateDst: GPUTensor, upDst: GPUTensor, downDst: GPUTensor,
                           slot: Int) -> Bool {
        guard layers.contains(layer), id >= 0, Int(id) < nExpert else { return false }
        let base = dataBase + ((layer - layers.lowerBound) * nExpert + Int(id)) * record
        let jobs: [(fileOff: Int, bytes: Int, dst: GPUTensor)] = [
            (0, gateBytes, gateDst),
            (gateBytes, upBytes, upDst),
            (gateBytes + upBytes, downBytes, downDst)]
        for j in jobs where j.dst.byteOffset + (slot + 1) * j.bytes > j.dst.buffer.length {
            return false
        }
        let lock = NSLock()
        var ok = true
        DispatchQueue.concurrentPerform(iterations: jobs.count) { i in
            let j = jobs[i]
            let p = j.dst.buffer.contents().advanced(by: j.dst.byteOffset + slot * j.bytes)
            if !GGUFWeights.preadFull(fd, into: p, bytes: j.bytes, offset: base + j.fileOff) {
                lock.lock(); ok = false; lock.unlock()
            }
        }
        if ok { noteUse() }
        return ok
    }

    /// Gather the selected `ids` into three freshly packed K-expert tensors
    /// (same shape gatherLayerExperts returns) — used by the batched PREFILL
    /// path too, where each expert of the union becomes one sequential burst.
    /// nil on any error (caller falls back to the GGUF gather).
    public func gatherPacked(_ rt: MetalRuntime, layer: Int,
                             ids: [Int32]) -> (GPUTensor, GPUTensor, GPUTensor)? {
        guard layers.contains(layer), !ids.isEmpty,
              ids.allSatisfy({ $0 >= 0 && Int($0) < nExpert }) else { return nil }
        guard let g = try? GPUTensor.zerosBytes(rt, byteLength: ids.count * gateBytes),
              let u = try? GPUTensor.zerosBytes(rt, byteLength: ids.count * upBytes),
              let dn = try? GPUTensor.zerosBytes(rt, byteLength: ids.count * downBytes) else { return nil }
        let lock = NSLock()
        var ok = true
        DispatchQueue.concurrentPerform(iterations: ids.count) { k in
            if !copyExpert(layer: layer, id: ids[k], gateDst: g, upDst: u, downDst: dn, slot: k) {
                lock.lock(); ok = false; lock.unlock()
            }
        }
        return ok ? (g, u, dn) : nil
    }
}
