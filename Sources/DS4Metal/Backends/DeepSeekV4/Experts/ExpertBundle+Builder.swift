import Foundation
import Metal
import DS4Core

extension ExpertBundle {
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
        // Location. READING tries the sibling first (a bundle built by the
        // demo/CLI next to the GGUF is always reused), then DS4_BUNDLE_DIR.
        // BUILDING goes to DS4_BUNDLE_DIR whenever it is set: the app OWNS its
        // bundle under Application Support (survives model moves, never
        // touches the user's model folder); the demo/CLI (env unset) builds
        // next to the model. To share ONE copy between demo and app, point
        // the demo at the same dir: DS4_BUNDLE_DIR="$HOME/Library/Application
        // Support/DwarfStar/expert-bundle".
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
        // ONE builder at a time per target: the Settings button and a model
        // load must never write the same .tmp concurrently. The second caller
        // proceeds WITHOUT a bundle (the load never fails or queues for
        // minutes because of an optimization).
        buildCoordLock.lock()
        let alreadyBuilding = buildsInFlight.contains(buildPath)
        if !alreadyBuilding { buildsInFlight.insert(buildPath) }
        buildCoordLock.unlock()
        if alreadyBuilding {
            log("costruzione già in corso per: \(buildPath) — si prosegue senza bundle (riprova al prossimo load)")
            return nil
        }
        defer { buildCoordLock.lock(); buildsInFlight.remove(buildPath); buildCoordLock.unlock() }
        return build(path: buildPath, model: model, srcFD: srcFD, layers: layers, nExpert: nExpert,
                     gateBytes: gateBytes, upBytes: upBytes, downBytes: downBytes, hashes: hashes)
    }

    // Guarded by buildCoordLock (every access takes the lock first): safe in
    // practice, hence the nonisolated(unsafe) opt-out from strict concurrency.
    private static let buildCoordLock = NSLock()
    nonisolated(unsafe) private static var buildsInFlight = Set<String>()

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
        return ExpertBundle(fd: fd, path: path, layers: layers, nExpert: nExpert,
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
            // NB: file cancellati con Finder finiscono nel CESTINO e non liberano
            // spazio finché non lo si svuota — il caso classico di questo salto.
            log("spazio disco insufficiente (~\(totalBytes >> 30) GB richiesti, ~\(free >> 30) GB liberi — il Cestino conta!) — salto")
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
}
