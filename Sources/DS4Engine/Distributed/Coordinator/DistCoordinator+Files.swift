import Foundation
import DS4Core

extension DistCoordinator {
    // MARK: File distribution (the workers receive everything from here)

    /// The distributable file set: the GGUF plus supported derived caches.
    /// Hashes come from the persistent cache; the first run streams the file
    /// once to compute them (logged — it can take minutes on a 100+ GB gguf).
    func buildFileOffer(onLog: @Sendable (String) -> Void) throws -> [DistFileEntry] {
        let ggufName = (config.modelPath as NSString).lastPathComponent
        // v8: la catena di checkpoint viaggia nell'offer (32 B ogni 256 MB —
        // ~9 KB per il gguf) e permette ai worker di riprendere un `.part`
        // interrotto dal l'ultimo blocco verificato, anche tra sessioni.
        guard let (size, _) = DistFileHash.stat(config.modelPath),
              let (sha, chain) = DistFileHash.cachedOrComputeWithChain(path: config.modelPath, onLog: onLog) else {
            throw DistError.sliceGap("cannot read/hash the gguf at \(config.modelPath)")
        }
        var entries = [DistFileEntry(kind: .gguf, name: ggufName, size: size, sha256: sha, chain: chain)]
        // The Q4 requant cache is derived and deterministic: ~1.4 GB on the
        // wire beats minutes of re-requant on every worker.
        if config.useDenseQ4,
           let q4 = findLocalPath(kind: .q4Dense),
           let (qSize, _) = DistFileHash.stat(q4),
           let (qSha, qChain) = DistFileHash.cachedOrComputeWithChain(path: q4, onLog: onLog) {
            entries.append(DistFileEntry(kind: .q4Dense, name: ggufName + ".q4dense",
                                         size: qSize, sha256: qSha, chain: qChain))
        }
        return entries
    }

    /// Stream one offered file to a worker: sequential 4 MB chunks (F_NOCACHE
    /// read — the transfer must not evict the coordinator's hot page cache),
    /// then DONE, then the worker's hash-verified ack.
    func sendFile(_ entry: DistFileEntry, index: Int, to conn: DistConnection,
                          peer: Peer, from resumeOffset: UInt64 = 0,
                          onLog: @Sendable (String) -> Void) async throws {
        guard let path = entry.kind == .gguf ? config.modelPath : findLocalPath(kind: entry.kind) else {
            throw DistError.sliceGap("offered file \(entry.name) no longer found locally")
        }
        // v8 resume: il worker ha convalidato il suo .part con la catena di
        // checkpoint e chiede di ripartire da un confine di blocco.
        let start = min(resumeOffset, entry.size)
        if start > 0 {
            onLog("file: riprendo \(entry.name) dal blocco verificato a \(start / 1_048_576) MB "
                  + "(\((entry.size - start) / 1_048_576) MB restanti) → \(peer.host):\(peer.port)\n")
        } else {
            onLog("file: invio \(entry.name) (\(entry.size / 1_048_576) MB) a \(peer.host):\(peer.port)…\n")
        }
        let fd = open(path, O_RDONLY)
        guard fd >= 0 else { throw DistError.sliceGap("cannot open \(path)") }
        defer { close(fd) }
        _ = fcntl(fd, F_NOCACHE, 1)
        guard lseek(fd, off_t(start), SEEK_SET) == off_t(start) else {
            throw DistError.sliceGap("seek to \(start) failed on \(path)")
        }

        var offset: UInt64 = start
        var buf = [UInt8](repeating: 0, count: Dist.fileChunkBytes)
        let t0 = Date()
        var lastLogged: UInt64 = start
        while offset < entry.size {
            try Task.checkCancellation()
            let n = buf.withUnsafeMutableBytes { read(fd, $0.baseAddress, Dist.fileChunkBytes) }
            guard n > 0 else { throw DistError.sliceGap("read failed at \(offset) of \(path)") }
            let chunk = DistFileChunk(index: index, offset: offset, data: Data(buf[0..<n]))
            try await conn.sendFrame(.fileChunk, chunk.encoded())
            offset += UInt64(n)
            if offset - lastLogged >= 1_073_741_824 {          // progress every GB
                lastLogged = offset
                let mbps = Double(offset - start) / 1_048_576 / max(0.001, Date().timeIntervalSince(t0))
                onLog(String(format: "file: %@ %.0f%% (%.0f MB/s)\n", entry.name,
                             Double(offset) / Double(entry.size) * 100, mbps))
            }
        }
        try await conn.sendFrame(.fileDone, DistFileDone(index: index).encoded())
        let (aType, aPayload) = try await readControl(conn)
        guard aType == .fileAck, let ack = DistKV.decodeAck(aPayload) else { throw DistError.badFrame }
        // Nack del trasferimento: RITENTABILE (transferFailed, non remote) —
        // il tentativo successivo rinegozia dalla catena di checkpoint e
        // ritrasmette solo il mancante.
        guard ack.ok else { throw DistError.transferFailed("\(peer.host):\(peer.port): \(ack.message)") }
        onLog(String(format: "file: %@ trasferito in %.0fs\n", entry.name, Date().timeIntervalSince(t0)))
    }

    /// Where a derived file lives on the coordinator: next to the gguf, else
    /// in the app-owned cache directory (same lookup order as the engine).
    private func findLocalPath(kind: DistFileEntry.Kind) -> String? {
        let name = (config.modelPath as NSString).lastPathComponent
        let (ext, dirEnv): (String, String)
        switch kind {
        case .expertBundle: (ext, dirEnv) = (".expbundle", "DS4_BUNDLE_DIR")
        case .q4Dense:      (ext, dirEnv) = (".q4dense", "DS4_Q4_CACHE_DIR")
        case .gguf:         return config.modelPath
        }
        var candidates = [config.modelPath + ext]
        if let dir = ProcessInfo.processInfo.environment[dirEnv], !dir.isEmpty {
            candidates.append(dir + "/" + name + ext)
        }
        return candidates.first { FileManager.default.fileExists(atPath: $0) }
    }
}
