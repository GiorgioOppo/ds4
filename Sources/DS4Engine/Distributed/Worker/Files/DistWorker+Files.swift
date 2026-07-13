import Foundation
import CryptoKit
import DS4Core

extension DistWorker {
    /// One coordinator-streamed file being received: sequential chunks written
    /// to a `.part` in the managed store while SHA-256 accumulates INLINE — the
    /// final verification is a compare, never a re-read of tens of GB.
    final class IncomingFile {
        enum Outcome { case success(String), failure(String) }
        let entry: DistFileEntry
        let index: Int
        private let onLog: @Sendable (String) -> Void
        private let tmpURL: URL
        private let finalURL: URL
        private var fd: Int32
        private var hasher = SHA256()
        private var received: UInt64 = 0
        private var lastLogged: UInt64 = 0
        private let t0 = Date()

        /// `resumeFrom`: riprende un `.part` esistente da un confine di blocco
        /// GIÀ CONVALIDATO dal chiamante con la catena di checkpoint (v8). Il
        /// file viene troncato lì, l'hash dell'intero file viene RI-PRIMATO
        /// rileggendo il prefisso (la SHA-256 finale deve coprire tutti i
        /// byte), e la ricezione continua da quell'offset. Con 0 (o su
        /// qualunque errore di ripresa) si riparte da zero.
        init(entry: DistFileEntry, index: Int, resumeFrom: UInt64 = 0,
             onLog: @escaping @Sendable (String) -> Void) {
            self.entry = entry; self.index = index; self.onLog = onLog
            let store = DistFileStore.shared
            try? FileManager.default.createDirectory(at: store.directory,
                                                     withIntermediateDirectories: true)
            self.finalURL = store.url(for: entry.name)
            self.tmpURL = finalURL.appendingPathExtension("part")
            if resumeFrom > 0 {
                self.fd = open(tmpURL.path, O_WRONLY, 0o644)
                var primed = false
                if fd >= 0, ftruncate(fd, off_t(resumeFrom)) == 0,
                   lseek(fd, off_t(resumeFrom), SEEK_SET) == off_t(resumeFrom),
                   Self.prime(hasher: &hasher, path: tmpURL.path, upTo: resumeFrom) {
                    received = resumeFrom
                    primed = true
                    onLog("file: \(entry.name) — riprendo dal checkpoint a \(resumeFrom / 1_048_576) MB\n")
                }
                if !primed {
                    // MAI ributtare il .part con O_TRUNC qui: il coordinatore
                    // sta già trasmettendo da `resumeFrom` (promesso nel
                    // fileNeed), quindi "ricominciare da zero" non può comunque
                    // riuscire e brucerebbe GB di progresso convalidato per un
                    // singhiozzo di IO locale. fd = -1: gli append falliscono,
                    // il DONE fa nack (transferFailed, RITENTABILE lato
                    // coordinatore) e il tentativo successivo rinegozia dalla
                    // catena di checkpoint con il .part intatto.
                    if fd >= 0 { close(fd) }
                    self.fd = -1
                    onLog("file: \(entry.name) — ripresa non riuscita (IO locale): tentativo annullato, .part conservato\n")
                }
            } else {
                self.fd = open(tmpURL.path, O_CREAT | O_TRUNC | O_WRONLY, 0o644)
            }
            if fd >= 0 { _ = fcntl(fd, F_NOCACHE, 1) }   // the transfer must not evict hot pages
        }

        /// Rileggi il prefisso [0, upTo) dentro `hasher` (8 MB per read).
        private static func prime(hasher: inout SHA256, path: String, upTo: UInt64) -> Bool {
            let rfd = open(path, O_RDONLY)
            guard rfd >= 0 else { return false }
            defer { close(rfd) }
            _ = fcntl(rfd, F_NOCACHE, 1)
            var remaining = upTo
            var buf = [UInt8](repeating: 0, count: 8 * 1024 * 1024)
            while remaining > 0 {
                let want = Int(min(UInt64(buf.count), remaining))
                let n = buf.withUnsafeMutableBytes { read(rfd, $0.baseAddress, want) }
                guard n > 0 else { return false }
                buf.withUnsafeBytes { hasher.update(bufferPointer: UnsafeRawBufferPointer(rebasing: $0[0..<n])) }
                remaining -= UInt64(n)
            }
            return true
        }

        /// Append one sequential chunk; false on disorder/overflow/IO error.
        func append(_ chunk: DistFileChunk) -> Bool {
            guard fd >= 0, chunk.offset == received,
                  received + UInt64(chunk.data.count) <= entry.size else { return false }
            let ok = chunk.data.withUnsafeBytes { buf -> Bool in
                var off = 0
                while off < buf.count {
                    let w = write(fd, buf.baseAddress! + off, buf.count - off)
                    if w <= 0 { return false }
                    off += w
                }
                return true
            }
            guard ok else { return false }
            chunk.data.withUnsafeBytes { hasher.update(bufferPointer: $0) }
            received += UInt64(chunk.data.count)
            if received - lastLogged >= 1_073_741_824 {          // progress every GB
                lastLogged = received
                let mbps = Double(received) / 1_048_576 / max(0.001, Date().timeIntervalSince(t0))
                onLog(String(format: "file: %@ %.0f%% (%.0f MB/s)\n", entry.name,
                             Double(received) / Double(entry.size) * 100, mbps))
            }
            return true
        }

        /// Verify size + hash, move into place, record in the store manifest.
        func finalize() -> Outcome {
            guard fd >= 0 else { return .failure("could not open \(tmpURL.path)") }
            close(fd); fd = -1
            // v8: il .part resta su disco anche in fallimento — la prossima
            // offerta lo convalida con la catena di checkpoint e riprende dal
            // prefisso buono invece di ritrasferire tutto.
            guard received == entry.size else {
                return .failure("incomplete: \(received)/\(entry.size) bytes (.part conservato per la ripresa)")
            }
            let digest = Data(hasher.finalize())
            guard digest == entry.sha256 else {
                return .failure("sha256 mismatch (got \(digest.hexString.prefix(16))…) (.part conservato per la ripresa)")
            }
            unlink(finalURL.path)
            guard rename(tmpURL.path, finalURL.path) == 0 else {
                // Il .part è COMPLETO e verificato: si conserva — alla prossima
                // offerta la verifica dell'hash intero lo promuove sul posto.
                return .failure("rename failed: errno \(errno) (.part conservato)")
            }
            DistFileStore.shared.remember(name: entry.name, size: entry.size, sha256: digest)
            return .success(finalURL.path)
        }

        /// Chiudi TENENDO il `.part` (v8): alla prossima offerta — anche in
        /// una sessione futura — la catena di checkpoint convalida il prefisso
        /// buono e la ricezione riprende da lì invece che da zero. Anche un
        /// contenuto sospetto (chunk fuori ordine, hash finale sbagliato) si
        /// conserva: sarà la verifica a catena a decidere cosa salvare.
        func suspend() {
            if fd >= 0 { close(fd); fd = -1 }
        }
    }

    /// v8, passo 1: verifica del FILE INTERO. Un `.part` che copre già tutta
    /// la size (trasferimento caduto proprio sul finale, o DONE mai arrivato)
    /// si convalida con la SHA-256 dell'intero file e si PROMUOVE sul posto —
    /// zero byte ritrasferiti, zero round-trip. Solo se l'hash intero non
    /// torna (o il .part è più corto) si passa alla verifica per blocchi
    /// (resumePoint) per trovare fin dove il trasferimento era arrivato.
    func promoteCompletePart(_ entry: DistFileEntry,
                                     onLog: @Sendable (String) -> Void) -> String? {
        let store = DistFileStore.shared
        let finalURL = store.url(for: entry.name)
        let partURL = finalURL.appendingPathExtension("part")
        guard let attrs = try? FileManager.default.attributesOfItem(atPath: partURL.path),
              let partSize = (attrs[.size] as? NSNumber)?.uint64Value,
              partSize == entry.size else { return nil }
        onLog("file: \(entry.name) — .part completo, verifico l'hash dell'intero file…\n")
        guard let digest = DistFileHash.compute(path: partURL.path), digest == entry.sha256 else {
            onLog("file: \(entry.name) — hash intero NON corrisponde: cerco l'ultimo blocco buono\n")
            return nil
        }
        unlink(finalURL.path)
        guard rename(partURL.path, finalURL.path) == 0 else { return nil }
        store.remember(name: entry.name, size: entry.size, sha256: digest)
        onLog("file: \(entry.name) — verificato e promosso senza ritrasferimento\n")
        return finalURL.path
    }

    /// v8, passo 2: quanto del `.part` locale (trasferimento interrotto, anche
    /// di una sessione passata) è FIDATO. Convalida i blocchi INTERI con la
    /// catena di hash dell'offerta — chain[k] = SHA256(chain[k-1] ‖ SHA256(blocco k)),
    /// ogni anello impegna tutto il prefisso — tronca il file al primo blocco
    /// che non torna e restituisce l'offset di ripresa (multiplo del
    /// checkpoint; 0 = nulla da salvare). La coda parziale oltre l'ultimo
    /// checkpoint non è verificabile e si ritrasferisce.
    func resumePoint(_ entry: DistFileEntry, onLog: @Sendable (String) -> Void) -> UInt64 {
        guard !entry.chain.isEmpty else { return 0 }
        let partPath = DistFileStore.shared.url(for: entry.name).appendingPathExtension("part").path
        guard let attrs = try? FileManager.default.attributesOfItem(atPath: partPath),
              let partSize = (attrs[.size] as? NSNumber)?.uint64Value, partSize > 0 else { return 0 }
        let cp = Dist.fileCheckpointBytes
        let fullBlocks = Int(min(partSize / cp, UInt64(entry.chain.count)))
        var good = 0
        if fullBlocks > 0 {
            let fd = open(partPath, O_RDONLY)
            guard fd >= 0 else { return 0 }
            defer { close(fd) }
            _ = fcntl(fd, F_NOCACHE, 1)
            var prev: Data?
            var buf = [UInt8](repeating: 0, count: 8 * 1024 * 1024)
            outer: for k in 0..<fullBlocks {
                var blockHasher = SHA256()
                var remaining = cp
                while remaining > 0 {
                    let want = Int(min(UInt64(buf.count), remaining))
                    let n = buf.withUnsafeMutableBytes { read(fd, $0.baseAddress, want) }
                    guard n > 0 else { break outer }
                    buf.withUnsafeBytes {
                        blockHasher.update(bufferPointer: UnsafeRawBufferPointer(rebasing: $0[0..<n]))
                    }
                    remaining -= UInt64(n)
                }
                var link = SHA256()
                if let p = prev { link.update(data: p) }
                link.update(data: Data(blockHasher.finalize()))
                let digest = Data(link.finalize())
                guard digest == entry.chain[k] else {
                    onLog("file: \(entry.name) — blocco \(k) non verificato, scarto la coda\n")
                    break
                }
                prev = digest
                good = k + 1
            }
        }
        let resume = UInt64(good) * cp
        // Tronca ORA la parte non verificata: il fileNeed promette esattamente
        // questo offset e la ricezione (IncomingFile) riparte da lì.
        if resume < partSize {
            let wfd = open(partPath, O_WRONLY)
            if wfd >= 0 { _ = ftruncate(wfd, off_t(resume)); close(wfd) }
        }
        return min(resume, entry.size)
    }

    /// Resolve one offered file WITHOUT transferring: the managed store first
    /// (manifest hash recorded at reception), then same-named local files
    /// (size pre-filter, then the cached-or-computed hash must match — a
    /// matching local gguf costs one full hash the first time, then it's a
    /// manifest lookup).
    func resolveOffered(_ entry: DistFileEntry) -> String? {
        let store = DistFileStore.shared
        if store.has(name: entry.name, size: entry.size, sha256: entry.sha256) {
            return store.url(for: entry.name).path
        }
        let name = DistFileStore.sanitize(entry.name)
        let hintDir = (config.localModelPath as NSString).deletingLastPathComponent
        var candidates = [hintDir + "/" + name]
        if (config.localModelPath as NSString).lastPathComponent == name {
            candidates.append(config.localModelPath)
        }
        for path in candidates where FileManager.default.fileExists(atPath: path) {
            guard let (size, _) = DistFileHash.stat(path), size == entry.size else { continue }
            if DistFileHash.cachedOrCompute(path: path, onLog: onLog) == entry.sha256 {
                return path
            }
        }
        return nil
    }
}
