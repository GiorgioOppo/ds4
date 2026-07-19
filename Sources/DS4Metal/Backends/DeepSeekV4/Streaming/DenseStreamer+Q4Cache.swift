import Foundation
import Metal
import DS4Core

extension DenseStreamer {
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
    static func sourceHash(fd: Int32, tensor t: GGUFModel.Tensor) -> UInt64 {
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
    static func logQ4(_ s: String) {
        DS4Log.info("q4cache", s)
    }

    /// Load the requant cache if it exists and matches the model. Returns an
    /// array aligned to `jobs`: fully populated for a complete cache, with nil
    /// HOLES when the file is a CHECKPOINT written by an interrupted requant
    /// (the caller requantizes only the holes). nil = file absent/unusable.
    /// Reads through F_NOCACHE straight into the resident buffers (~0.5 s for
    /// ~1.4 GB) — the requant is paid only on the very first load.
    static func loadQ4Cache(_ rt: MetalRuntime, path: String, modelSize: Int,
                                    jobs: [(il: Int, f: Field, t: GGUFModel.Tensor)],
                                    hashes: [UInt64]) -> [GPUTensor?]? {
        let cfd = open(path, O_RDONLY)
        guard cfd >= 0 else {
            logQ4("assente (\(String(cString: strerror(errno)))) — riquantizzo: \(path)")
            return nil
        }
        defer { close(cfd) }
        _ = fcntl(cfd, F_NOCACHE, 1)
        var fixed = [UInt8](repeating: 0, count: 20)
        let okFixed = fixed.withUnsafeMutableBytes {
            GGUFWeights.preadFull(cfd, into: $0.baseAddress!, bytes: 20, offset: 0)
        }
        guard okFixed else {
            logQ4("header illeggibile — riquantizzo: \(path)")
            return nil
        }
        func fu32(_ o: Int) -> UInt32 { fixed.withUnsafeBytes { $0.loadUnaligned(fromByteOffset: o, as: UInt32.self) } }
        func fu64(_ o: Int) -> UInt64 { fixed.withUnsafeBytes { $0.loadUnaligned(fromByteOffset: o, as: UInt64.self) } }
        // The cache may hold a SUPERSET of the jobs (a distributed worker's
        // layer SLICE reusing the full-model cache) or a SUBSET (a checkpoint
        // from an interrupted requant). Records are matched by (layer, field)
        // KEY, not by position or count; bytes and source hash must still
        // agree PER RECORD, so a partially-replaced model invalidates only
        // the tensors that actually changed.
        let cacheCount = Int(fu32(16))
        guard fu32(0) == q4CacheMagic, fu32(4) == q4CacheVersion,
              fu64(8) == UInt64(modelSize), cacheCount >= 0, cacheCount <= 8192 else {
            logQ4("incompatibile (versione/modello/estensione diversi) — riquantizzo: \(path)")
            return nil
        }
        if cacheCount == 0 {
            // Solo il preflight di un requant interrotto subito dopo l'avvio:
            // header valido, nessun tensore ancora scritto.
            logQ4("vuota (preflight, nessun checkpoint ancora) — riquantizzo: \(path)")
            return nil
        }
        var head = [UInt8](repeating: 0, count: cacheCount * 32)
        let okHead = head.withUnsafeMutableBytes {
            GGUFWeights.preadFull(cfd, into: $0.baseAddress!, bytes: cacheCount * 32, offset: 20)
        }
        guard okHead else {
            logQ4("tabella record illeggibile — riquantizzo: \(path)")
            return nil
        }
        func u32(_ o: Int) -> UInt32 { head.withUnsafeBytes { $0.loadUnaligned(fromByteOffset: o, as: UInt32.self) } }
        func u64(_ o: Int) -> UInt64 { head.withUnsafeBytes { $0.loadUnaligned(fromByteOffset: o, as: UInt64.self) } }
        var index: [UInt64: (bytes: UInt64, hash: UInt64, offset: Int)] = [:]
        index.reserveCapacity(cacheCount)
        for i in 0..<cacheCount {
            let o = i * 32
            index[UInt64(u32(o)) << 32 | UInt64(u32(o + 4))] =
                (bytes: u64(o + 8), hash: u64(o + 16), offset: Int(u64(o + 24)))
        }
        var records: [(bytes: Int, offset: Int)?] = []
        var found = 0
        for (i, job) in jobs.enumerated() {
            let expected = Int(job.t.elements) / 256 * 144
            if let r = index[UInt64(job.il) << 32 | UInt64(job.f.rawValue)],
               r.bytes == UInt64(expected), r.hash == hashes[i] {
                records.append((bytes: expected, offset: r.offset))
                found += 1
            } else {
                records.append(nil)
            }
        }
        guard found > 0 else {
            logQ4("incompatibile (nessun tensore in comune) — riquantizzo: \(path)")
            return nil
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
                guard let r = recs[i] else { return }   // hole: requantized by the caller
                guard let t = try? GPUTensor.uninitializedBytes(rtRef, byteLength: r.bytes,
                                                                elementCount: r.bytes),
                      GGUFWeights.preadFull(cfd, into: t.buffer.contents(),
                                            bytes: r.bytes, offset: r.offset) else {
                    lock.lock(); failed = true; lock.unlock()
                    return
                }
                bufBase[i] = t
                LoadProgress.shared.advance()
            }
        }
        guard !failed, out.compactMap({ $0 }).count == found else {
            logQ4("lettura fallita — riquantizzo: \(path)")
            return nil
        }
        if found < jobs.count {
            logQ4("checkpoint parziale (\(found)/\(jobs.count) tensori) — riquantizzo solo i mancanti: \(path)")
        } else {
            logQ4("caricata (\(jobs.count)\(cacheCount > jobs.count ? " di \(cacheCount)" : "") tensori): \(path)")
        }
        return out
    }

    /// Persist the converted tensors (best-effort: failures leave no cache and
    /// the next load simply requantizes again). Written to a .tmp sibling and
    /// renamed into place, so a crash mid-write can never leave a torn cache
    /// under the real name (and a concurrent reader sees old-or-new, never half).
    /// Returns true on success — the requant preflight writes an EMPTY cache
    /// (zero record) up front to prove the path is writable before any work.
    @discardableResult
    static func writeQ4Cache(path: String, modelSize: Int,
                                     jobs: [(il: Int, f: Field, t: GGUFModel.Tensor)],
                                     tensors: [GPUTensor], hashes: [UInt64]) -> Bool {
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
        // `offset` is now the full file size: refuse to start on a near-full
        // disk (a torn 1.4 GB write helps nobody) with a log that names the
        // real cause — the classic case is the app container's volume filled
        // by other caches, and "il Cestino conta" (Finder deletes don't free
        // space until emptied).
        let dir = (path as NSString).deletingLastPathComponent
        let free = (try? FileManager.default.attributesOfFileSystem(forPath: dir))?[.systemFreeSize] as? Int ?? 0
        guard free > offset + (1 << 28) else {
            logQ4("spazio disco insufficiente (~\(offset >> 20) MB richiesti, ~\(free >> 20) MB liberi — il Cestino conta!) — cache non scritta: \(path)")
            return false
        }
        let tmp = path + ".tmp"
        guard FileManager.default.createFile(atPath: tmp, contents: nil),
              let fh = FileHandle(forWritingAtPath: tmp) else {
            logQ4("SCRITTURA FALLITA (cartella \(FileManager.default.fileExists(atPath: dir) ? "presente" : "ASSENTE"), ~\(free >> 20) MB liberi): \(tmp)")
            return false
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
            return true
        } catch {
            try? fh.close()
            logQ4("SCRITTURA FALLITA (\(error)) — file rimosso: \(tmp)")
            try? FileManager.default.removeItem(atPath: tmp)   // never leave a torn cache
            return false
        }
    }

    /// Q8_0 → Q4_K requant of one tensor into a resident buffer (DS4_DENSE_Q4).
    /// Returns nil when the tensor is not Q8_0 (or not superblock-shaped): the
    /// caller keeps streaming it unchanged. Reads the source bytes through the
    /// F_NOCACHE descriptor (no page-cache footprint at load either).
    static func requantQ4(_ rt: MetalRuntime, _ model: GGUFModel, fd: Int32,
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
        // Heartbeat della barra di caricamento: un'unità (~1 MB di Q8
        // sorgente, 3855 superblocchi da 272 B) ogni pochi superblocchi
        // convertiti — il chiamante ha dimensionato la fase in MB. Senza
        // questo la barra restava ferma per l'intero primo batch di tensori
        // (minuti), indistinguibile da un'app bloccata.
        let sbPerMB = max(1, (1 << 20) / 272)
        var scratch = [Float](repeating: 0, count: 256)
        q8.withUnsafeBytes { src in
            let base = src.baseAddress!
            let dst = gpu.buffer.contents()
            scratch.withUnsafeMutableBufferPointer { s in
                for sb in 0..<nsb {
                    Quantize.dequantQ8_0(base + sb * 272, count: 256, into: s.baseAddress!)
                    Quantize.quantizeQ4_K(s.baseAddress!, count: 256, into: dst + sb * 144)
                    if sb % sbPerMB == sbPerMB - 1 { LoadProgress.shared.advance() }
                }
            }
        }
        return gpu
    }
}
