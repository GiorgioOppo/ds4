import Foundation
import CryptoKit
import DS4Core
@preconcurrency import Network

/// A distributed WORKER: starts IDLE (listening, no model loaded) and receives
/// EVERYTHING from the COORDINATOR — the files (gguf + sidecar, via the
/// hash-verified FILE OFFER/transfer; the huge copy happens once, afterwards
/// the managed-store manifest answers in milliseconds) and the job (ASSIGN:
/// context size, cache budgets, bundle on/off, layer slice). On ASSIGN the
/// worker loads (or reuses) its slice engine and replies READY; then it serves
/// WORK frames (run the slice, or the output head when flagged) and answers
/// with RESULT frames.
public final class DistWorker: @unchecked Sendable {
    public struct Config: Sendable {
        public var port: UInt16
        /// Local gguf hint: the worker tries the coordinator's path verbatim
        /// first, then a file with the assigned NAME next to this hint, then
        /// the hint itself when the filename matches (per-Mac disk layouts).
        public var localModelPath: String
        public init(port: UInt16, localModelPath: String) {
            self.port = port; self.localModelPath = localModelPath
        }
    }

    private let config: Config
    private let onLog: @Sendable (String) -> Void
    private let queue = DispatchQueue(label: "ds4.dist.worker")
    private let gate = DistGate()
    private var listener: NWListener?

    /// The coordinator-defined job this worker currently serves.
    struct Assignment: Equatable, Sendable {
        var resolvedModelPath: String
        var contextSize: Int
        var expertCacheSlots: Int
        var useExpertBundle: Bool
        var useDenseQ4: Bool
        var layerStart: Int
        var layerEnd: Int
        var hasOutput: Bool
    }

    /// One coordinator-streamed file being received: sequential chunks written
    /// to a `.part` in the managed store while SHA-256 accumulates INLINE — the
    /// final verification is a compare, never a re-read of tens of GB.
    private final class IncomingFile {
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
    private func promoteCompletePart(_ entry: DistFileEntry,
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
    private func resumePoint(_ entry: DistFileEntry, onLog: @Sendable (String) -> Void) -> UInt64 {
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
    private func resolveOffered(_ entry: DistFileEntry) -> String? {
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

    /// Engine + assignment, set by ASSIGN. Guarded by `stateLock`: WORK frames
    /// can arrive on a different connection (forwarding) than the assigning one.
    private let stateLock = NSLock()
    private var engine: DistEngine?
    private var assignment: Assignment?
    private var loadingAssignment = false
    /// L'assegnazione del load in corso (v8): un retry del coordinatore con la
    /// STESSA assegnazione si aggancia al load invece di ricevere "busy".
    private var pendingAssignment: Assignment?
    /// Expert parallelism (Fase B): lo shard verticale di esperti, alternativo
    /// all'assegnazione a layer (un worker fa l'uno o l'altro).
    private var expertShard: ExpertShardEngine?
    private var loadingShard = false

    private func currentShard() -> ExpertShardEngine? {
        stateLock.lock(); defer { stateLock.unlock() }
        return expertShard
    }
    private func claimShardLoad() -> Bool {
        stateLock.lock(); defer { stateLock.unlock() }
        if loadingShard { return false }
        loadingShard = true
        expertShard = nil            // libera il vecchio shard PRIMA del nuovo load
        return true
    }
    private func commitShard(_ s: ExpertShardEngine?) {
        stateLock.lock(); expertShard = s; loadingShard = false; stateLock.unlock()
    }
    /// Where this shard persists its usage imatrix (slice-keyed: counts are
    /// collected only for the owned layers). nil until assigned.
    private var usageFile: URL?

    /// App Support home for worker-side per-shard state (usage + disk KV),
    /// keyed by model AND slice: a checkpoint holds only the shard's layers,
    /// so a changed slice must never see another slice's entries.
    static func shardStateDirectory(modelName: String, layerStart: Int, layerEnd: Int) -> URL {
        FileManager.default.urls(for: .applicationSupportDirectory, in: .userDomainMask)[0]
            .appendingPathComponent("DwarfStar/dist-worker/\(modelName)-\(layerStart)-\(layerEnd)",
                                    isDirectory: true)
    }

    /// Persist the usage collected so far (cheap JSON; called between turns).
    private func persistUsage() {
        stateLock.lock()
        let file = usageFile
        let data = engine?.usageData()
        stateLock.unlock()
        guard let file, let data else { return }
        try? FileManager.default.createDirectory(at: file.deletingLastPathComponent(),
                                                 withIntermediateDirectories: true)
        try? data.write(to: file)
    }

    /// One TURN at a time, enforced at the session level (NOT per connection:
    /// in forwarding mode a worker legitimately holds one connection from the
    /// coordinator AND one from the previous worker). A pos==0 chunk ADOPTS its
    /// session id as current; any chunk from a different session is refused
    /// with an ERROR frame — a competing coordinator fails loudly instead of
    /// silently resetting the active turn's KV shard.
    private let sessionLock = NSLock()
    private var currentSession: UInt32?

    private func admit(_ work: DistWork) -> Bool {
        sessionLock.lock(); defer { sessionLock.unlock() }
        // turnStart (not pos==0): with KV reuse/restore a turn may begin
        // mid-context, and it is still the legitimate start of a new turn.
        if work.flags.contains(.turnStart) { currentSession = work.session; return true }
        return currentSession == work.session
    }

    public init(config: Config, onLog: @escaping @Sendable (String) -> Void) {
        self.config = config
        self.onLog = onLog
    }

    public func start() throws {
        let params = NWParameters.tcp
        params.allowLocalEndpointReuse = true
        guard let port = NWEndpoint.Port(rawValue: config.port) else { throw DistError.badPort }
        let l = try NWListener(using: params, on: port)
        l.stateUpdateHandler = { [onLog, config] state in
            switch state {
            case .ready: onLog("worker in ascolto su :\(config.port) — in attesa dell'assegnazione dal coordinatore\n")
            case .failed(let e): onLog("worker listener failed: \(e)\n")
            default: break
            }
        }
        l.newConnectionHandler = { [weak self] c in self?.accept(c) }
        l.start(queue: queue)
        listener = l
    }

    public func stop() {
        persistUsage()               // keep what this shard learned across sessions
        listener?.cancel(); listener = nil
        stateLock.lock(); engine = nil; assignment = nil; usageFile = nil; stateLock.unlock()
    }

    /// Resolve the gguf named by an ASSIGN to a LOCAL file: the coordinator's
    /// path verbatim (shared disk layouts), else a file with the same name in
    /// the local hint's directory, else the hint itself when the name matches.
    static func resolveModelPath(requestedPath: String, modelName: String,
                                 localHint: String,
                                 exists: (String) -> Bool = { FileManager.default.fileExists(atPath: $0) })
        -> String? {
        if !requestedPath.isEmpty, exists(requestedPath) { return requestedPath }
        if !modelName.isEmpty {
            let sibling = ((localHint as NSString).deletingLastPathComponent as NSString)
                .appendingPathComponent(modelName)
            if exists(sibling) { return sibling }
            if (localHint as NSString).lastPathComponent == modelName, exists(localHint) {
                return localHint
            }
        }
        return nil
    }

    private func accept(_ c: NWConnection) {
        c.start(queue: queue)
        let conn = DistConnection(c)
        Task { [weak self] in await self?.serve(conn) }
    }

    /// Snapshot for HELLO/READY: the active assignment, or the idle state.
    private func helloPayload() -> DistHello {
        stateLock.lock(); defer { stateLock.unlock() }
        if let engine, let a = assignment {
            return DistHello(modelName: engine.modelName, layerStart: a.layerStart,
                             layerEnd: a.layerEnd, hasOutput: a.hasOutput,
                             nLayers: engine.nLayers, contextSize: a.contextSize)
        }
        return .idle(localModelName: (config.localModelPath as NSString).lastPathComponent,
                     nLayers: DistEngine.modelLayers)
    }

    /// Current (engine, assignment) if ready to serve WORK; nil while idle/loading.
    private func activeEngine() -> (engine: DistEngine, assignment: Assignment)? {
        stateLock.lock(); defer { stateLock.unlock() }
        guard let engine, let assignment, !loadingAssignment else { return nil }
        return (engine, assignment)
    }

    private func serve(_ conn: DistConnection) async {
        onLog("connessione in ingresso\n")
        // Outbound connections (next-hop worker / coordinator return), per session.
        var downstream: [String: DistConnection] = [:]
        defer { for c in downstream.values { c.cancel() } }

        // `expectHello`: next-hop workers greet new connections with a HELLO frame
        // (consume it once); the coordinator's return listener does not.
        func outbound(_ host: String, _ port: UInt16, expectHello: Bool) async throws -> DistConnection {
            let key = "\(host):\(port)"
            if let c = downstream[key] { return c }
            let c = try await DistConnection.connect(host: host, port: port, queue: queue,
                                                     readyTimeout: 10, onState: onLog)
            if expectHello { _ = try await c.readFrame() }
            downstream[key] = c
            return c
        }

        // File distribution state for THIS connection: the offer's entries,
        // where each resolved locally, and the file currently being received.
        var offerEntries: [DistFileEntry] = []
        var resolvedFiles: [String: String] = [:]      // sanitized name → local path
        var incoming: IncomingFile?
        var pendingResume: [Int: UInt64] = [:]         // v8: offer index → offset di ripresa promesso
        var transferError: String?                     // first failure, reported at fileDone

        do {
            try await conn.sendFrame(.hello, helloPayload().encoded())

            while true {
                let (type, payload) = try await conn.readFrame()
                if type == .fileOffer {
                    guard let offer = DistFileOffer.decode(payload) else { continue }
                    // Una nuova offerta azzera lo stato di trasferimento della
                    // connessione: un incoming/errore rimasti da un'offerta
                    // precedente non devono interferire (e l'incoming va
                    // sospeso PRIMA che resumePoint tronchi il suo .part).
                    incoming?.suspend(); incoming = nil
                    transferError = nil
                    offerEntries = offer.entries
                    var needs: [Int] = []
                    var offsets: [UInt64] = []
                    pendingResume.removeAll()
                    for (i, entry) in offer.entries.enumerated() {
                        if let local = resolveOffered(entry) {
                            resolvedFiles[DistFileStore.sanitize(entry.name)] = local
                            onLog("file: \(entry.name) già presente (hash ok) — \(local)\n")
                        } else if let promoted = promoteCompletePart(entry, onLog: onLog) {
                            // v8 passo 1: .part a size piena e hash INTERO valido
                            // → promosso sul posto, niente da trasferire.
                            resolvedFiles[DistFileStore.sanitize(entry.name)] = promoted
                        } else {
                            // v8 passo 2: hash intero fallito o .part parziale —
                            // la catena di checkpoint dice fin dove il
                            // trasferimento era arrivato e si riprende da lì.
                            let resume = resumePoint(entry, onLog: onLog)
                            needs.append(i)
                            offsets.append(resume)
                            pendingResume[i] = resume
                            if resume > 0 {
                                onLog("file: \(entry.name) parziale — riprendo da \(resume / 1_048_576) di \(entry.size / 1_048_576) MB\n")
                            } else {
                                onLog("file: \(entry.name) mancante (\(entry.size / 1_048_576) MB) — richiedo il trasferimento\n")
                            }
                        }
                    }
                    try await conn.sendFrame(.fileNeed, DistFileNeed(indices: needs, offsets: offsets).encoded())
                    continue
                }
                if type == .fileChunk {
                    // Failures are remembered and reported ONCE at fileDone —
                    // the coordinator only reads the ack there, and one bad
                    // chunk must not pile an ack per remaining chunk.
                    guard let chunk = DistFileChunk.decode(payload),
                          chunk.index >= 0, chunk.index < offerEntries.count else {
                        transferError = transferError ?? "malformed fileChunk frame"
                        continue
                    }
                    // Il primo chunk di un file deve arrivare ESATTAMENTE
                    // all'offset di ripresa dichiarato nel fileNeed (0 senza .part).
                    if incoming == nil, transferError == nil,
                       chunk.offset == (pendingResume[chunk.index] ?? 0) {
                        incoming = IncomingFile(entry: offerEntries[chunk.index],
                                                index: chunk.index,
                                                resumeFrom: pendingResume[chunk.index] ?? 0,
                                                onLog: onLog)
                    }
                    if let file = incoming, file.index == chunk.index, file.append(chunk) {
                        continue
                    }
                    incoming?.suspend(); incoming = nil
                    transferError = transferError ?? "out-of-order or unexpected chunk"
                    continue
                }
                if type == .fileDone {
                    defer { transferError = nil }
                    // v8, caso limite: il .part copriva GIÀ tutto il file (catena
                    // verificata) → il coordinatore non manda alcun chunk e il
                    // DONE arriva senza un IncomingFile: crealo qui a offset
                    // pieno, così finalize() ne verifica hash e lo promuove.
                    if incoming == nil, transferError == nil,
                       let done = DistFileDone.decode(payload),
                       done.index >= 0, done.index < offerEntries.count,
                       let resume = pendingResume[done.index],
                       resume == offerEntries[done.index].size {
                        incoming = IncomingFile(entry: offerEntries[done.index], index: done.index,
                                                resumeFrom: resume, onLog: onLog)
                    }
                    guard let done = DistFileDone.decode(payload),
                          let file = incoming, file.index == done.index, transferError == nil else {
                        incoming?.suspend(); incoming = nil
                        try await conn.sendFrame(.fileAck, DistKV.encodeAck(
                            ok: false, message: transferError ?? "DONE without a matching transfer"))
                        continue
                    }
                    incoming = nil
                    switch file.finalize() {
                    case .success(let path):
                        resolvedFiles[DistFileStore.sanitize(file.entry.name)] = path
                        onLog("file: \(file.entry.name) ricevuto e verificato\n")
                        try await conn.sendFrame(.fileAck, DistKV.encodeAck(ok: true))
                    case .failure(let reason):
                        onLog("file: \(file.entry.name) SCARTATO: \(reason)\n")
                        try await conn.sendFrame(.fileAck, DistKV.encodeAck(ok: false, message: reason))
                    }
                    continue
                }
                if type == .assign {
                    try await handleAssign(payload, on: conn, resolvedFiles: resolvedFiles)
                    continue
                }
                if type == .expertAssign {
                    try await handleExpertAssign(payload, on: conn, resolvedFiles: resolvedFiles)
                    continue
                }
                if type == .expertWork {
                    guard let req = DistExpertWork.decode(payload) else {
                        try await conn.sendFrame(.error, Data("malformed EXPERT WORK frame".utf8))
                        continue
                    }
                    guard let shard = currentShard() else {
                        try await conn.sendFrame(.error,
                                                 Data("worker not ready: no expert shard loaded (send EXPERT ASSIGN first)".utf8))
                        continue
                    }
                    do {
                        // Sincrona (gather SSD + un cb GPU, ~ms): una richiesta
                        // alla volta per connessione — il parallelismo del
                        // verticale è TRA i worker.
                        let sum = try shard.partial(req)
                        try await conn.sendFrame(.expertSum, sum.encoded())
                    } catch {
                        try await conn.sendFrame(.error, Data("expertWork(seq \(req.seq)): \(error)".utf8))
                    }
                    continue
                }
                if type == .kvQuery || type == .kvRestore || type == .kvSave {
                    try await handleKV(type, payload, on: conn)
                    continue
                }
                guard type == .work, let work = DistWork.decode(payload) else { continue }

                guard let (engine, active) = activeEngine() else {
                    let msg = "worker not ready: no assignment loaded (send ASSIGN first)"
                    onLog(msg + "\n")
                    try await conn.sendFrame(.error, Data(msg.utf8))
                    continue
                }
                // Validate the chunk BEFORE touching the engine: sizes come from
                // the network, and a mismatch would otherwise crash the process
                // (out-of-bounds slicing) or touch KV the shard never allocated.
                let stateLen = engine.hcStateCount
                let n = work.nTokens
                guard n >= 1, work.hc.count == n * stateLen,
                      work.layerStart == active.layerStart, work.layerEnd == active.layerEnd,
                      work.pos >= 0, work.pos + n <= active.contextSize else {
                    let msg = "invalid WORK frame: nTokens=\(work.nTokens) hc=\(work.hc.count) "
                        + "(state \(stateLen)) layers \(work.layerStart)...\(work.layerEnd) "
                        + "(assigned \(active.layerStart)...\(active.layerEnd)) pos \(work.pos)"
                    onLog(msg + "\n")
                    try await conn.sendFrame(.error, Data(msg.utf8))
                    continue
                }
                guard admit(work) else {
                    let msg = "refused WORK for session \(work.session): another turn is active on this worker"
                    onLog(msg + "\n")
                    try await conn.sendFrame(.error, Data(msg.utf8))
                    continue
                }
                // A new turn boundary: persist the usage the PREVIOUS turn
                // accumulated (cheap JSON, same cadence as the local engine).
                if work.flags.contains(.turnStart) { persistUsage() }

                // Serialize compute: one chunk at a time against the shard.
                // The chunk's hc holds nTokens states; split, run, re-concat.
                let outStates: [[Float]] = try await gate.run {
                    var hcs: [[Float]] = []
                    hcs.reserveCapacity(n)
                    for i in 0..<n { hcs.append(Array(work.hc[i*stateLen..<(i+1)*stateLen])) }
                    return try engine.forwardSliceBatch(hcs: hcs, posBase: work.pos,
                                                        start: work.layerStart, end: work.layerEnd,
                                                        tokens: work.tokenIds.isEmpty ? nil
                                                            : work.tokenIds.map(Int.init))
                }
                if work.pos == 0, let outHC = outStates.first {
                    func nrm(_ a: [Float]) -> Float { (a.reduce(0) { $0 + $1 * $1 }).squareRoot() }
                    let inHC = Array(work.hc[0..<stateLen])
                    onLog(String(format: "diag: layer %d…%d  |in|=%.2f  |out|=%.2f\n",
                                 work.layerStart, work.layerEnd, nrm(inHC), nrm(outHC)))
                }

                let isTerminal = work.route.isEmpty || work.routeIndex >= work.route.count - 1
                if isTerminal {
                    // Terminal hop: produce logits for the chunk's LAST token if asked,
                    // else hidden states (relay) / a bare ack (forwarding flow control).
                    // Every result ECHOES the work's session id (stale-reply guard).
                    let result: DistResult
                    if work.flags.contains(.outputLogits) {
                        result = DistResult(session: work.session, kind: .logits, bits: 32,
                                            values: try engine.head(hc: outStates[n-1]))
                    } else if work.route.isEmpty {
                        result = DistResult(session: work.session, kind: .hidden, bits: work.hcBits,
                                            values: outStates.flatMap { $0 })
                    } else {
                        result = DistResult(session: work.session, kind: .ack, bits: 32, values: [])
                    }
                    if work.route.isEmpty {
                        try await conn.sendFrame(.result, result.encoded())     // relay: reply upstream
                    } else {
                        let back = try await outbound(work.returnHost, work.returnPort, expectHello: false)
                        try await back.sendFrame(.result, result.encoded())     // forwarding: reply to coordinator
                    }
                } else {
                    // Forward the chunk to the next hop in the route.
                    let nextIdx = work.routeIndex + 1
                    let next = work.route[nextIdx]
                    let fwd = DistWork(session: work.session, pos: work.pos, nTokens: n,
                                       layerStart: next.layerStart, layerEnd: next.layerEnd,
                                       flags: work.flags, hcBits: work.hcBits,
                                       route: work.route, routeIndex: nextIdx,
                                       returnHost: work.returnHost, returnPort: work.returnPort,
                                       hc: outStates.flatMap { $0 }, tokenIds: work.tokenIds)
                    let c = try await outbound(next.host, next.port, expectHello: true)
                    try await c.sendFrame(.work, fwd.encoded())
                }
            }
        } catch {
            incoming?.suspend()               // KEEP the half-received .part: the
                                              // next offer resumes it via the
                                              // checkpoint chain (v8)
            onLog("sessione chiusa: \(error)\n")
            conn.cancel()
        }
    }

    /// Handle an ASSIGN: validate, resolve the gguf (files distributed by the
    /// coordinator first, then local fallbacks), load (or reuse) the slice
    /// engine, reply READY — or an ERROR frame with the reason.
    private func handleAssign(_ payload: Data, on conn: DistConnection,
                              resolvedFiles: [String: String]) async throws {
        guard let assign = DistAssign.decode(payload) else {
            try await conn.sendFrame(.error, Data("malformed ASSIGN frame".utf8))
            return
        }
        guard assign.layerStart >= 0, assign.layerStart <= assign.layerEnd,
              assign.layerEnd < DistEngine.modelLayers, assign.contextSize > 0 else {
            let msg = "invalid ASSIGN: layers \(assign.layerStart)...\(assign.layerEnd) "
                + "(model has \(DistEngine.modelLayers)), context \(assign.contextSize)"
            onLog(msg + "\n")
            try await conn.sendFrame(.error, Data(msg.utf8))
            return
        }
        // The coordinator distributes the files: whatever the OFFER resolved
        // (received into the managed store, or hash-matched locally) wins;
        // the plain path fallback remains for robustness.
        let sanitizedName = DistFileStore.sanitize(assign.modelName)
        guard let resolved = resolvedFiles[sanitizedName]
                ?? Self.resolveModelPath(requestedPath: assign.modelPath,
                                         modelName: assign.modelName,
                                         localHint: config.localModelPath) else {
            let msg = "gguf '\(assign.modelName)' not available on this worker "
                + "(not offered/transferred, and no hash-matching local file)"
            onLog(msg + "\n")
            try await conn.sendFrame(.error, Data(msg.utf8))
            return
        }
        // v9: i knob di PERFORMANCE del coordinatore, applicati PRIMA del
        // load (l'env è letto alla creazione del motore). Doppia whitelist:
        // il filo non deve mai poter impostare variabili arbitrarie qui.
        let allowedKnobs = Set(Dist.perfKnobKeys)
        var appliedKnobs: [String] = []
        for (k, v) in assign.envKnobs where allowedKnobs.contains(k) && v.count <= 256 {
            _ = setenv(k, v, 1)
            appliedKnobs.append("\(k)=\(v)")
        }
        if !appliedKnobs.isEmpty {
            onLog("knob dal coordinatore: " + appliedKnobs.joined(separator: " ") + "\n")
        }
        // Sidecar/derived caches: coordinator-decided. The env is read at
        // ENGINE LOAD; pointing the cache-dir vars at the transferred files'
        // directory covers the case where the gguf resolved elsewhere (the
        // sibling rule would miss them).
        _ = setenv("DS4_EXPERT_BUNDLE", assign.useExpertBundle ? "1" : "0", 1)
        if assign.useExpertBundle,
           let bundlePath = resolvedFiles[sanitizedName + ".expbundle"] {
            _ = setenv("DS4_BUNDLE_DIR", (bundlePath as NSString).deletingLastPathComponent, 1)
        }
        _ = setenv("DS4_DENSE_Q4", assign.useDenseQ4 ? "1" : "0", 1)
        if assign.useDenseQ4,
           let q4Path = resolvedFiles[sanitizedName + ".q4dense"] {
            // The coordinator's requant cache: loaded in ~0.5 s instead of
            // re-requantizing for minutes on this worker.
            _ = setenv("DS4_Q4_CACHE_DIR", (q4Path as NSString).deletingLastPathComponent, 1)
        }
        let wanted = Assignment(resolvedModelPath: resolved, contextSize: assign.contextSize,
                                expertCacheSlots: assign.expertCacheSlots,
                                useExpertBundle: assign.useExpertBundle,
                                useDenseQ4: assign.useDenseQ4,
                                layerStart: assign.layerStart, layerEnd: assign.layerEnd,
                                hasOutput: assign.hasOutput)

        // Locked state transitions live in sync helpers: NSLock is not usable
        // directly inside an async function.
        var claim = claimAssignment(wanted)
        if case .inFlight = claim {
            // v8 retry: la connessione che ha chiesto QUESTO stesso load è
            // caduta e il coordinatore ha riconnesso. Il load prosegue per
            // conto suo: aggancialo — rilancia il progresso e attendi l'esito
            // invece di rispondere "busy" (che il coordinatore tratta come
            // fatale e affosserebbe la route per un blip di rete).
            onLog("assegnazione identica già in caricamento — mi aggancio al load in corso\n")
            let deadline = Date().addingTimeInterval(45 * 60)
            joinLoop: while Date() < deadline {
                try await Task.sleep(nanoseconds: 1_500_000_000)
                let s = LoadProgress.shared.snapshot
                if !s.stage.isEmpty {
                    try? await conn.sendFrame(.progress,
                                              Data(String(format: "%@ (%.0f%%)", s.stage, s.fraction * 100).utf8))
                }
                claim = claimAssignment(wanted)
                if case .inFlight = claim { continue }
                break joinLoop           // reuse (commit), load (il vecchio è fallito: slot nostro) o busy
            }
            if case .inFlight = claim {
                try await conn.sendFrame(.error, Data("load in corso da troppo tempo (timeout join)".utf8))
                return
            }
        }
        switch claim {
        case .reuse(let current):
            applyAncillary(assign, to: current)
            onLog("assegnazione invariata: layer \(wanted.layerStart)...\(wanted.layerEnd) — riuso il motore\n")
            try await conn.sendFrame(.ready, helloPayload().encoded())
            return
        case .busy:
            try await conn.sendFrame(.error, Data("worker busy loading a previous assignment".utf8))
            return
        case .inFlight:
            return                   // impossibile: gestito sopra (timeout incluso)
        case .load:
            break                    // old shard freed, `loadingAssignment` claimed
        }

        onLog("assegnazione: \(assign.modelName) · layer \(wanted.layerStart)...\(wanted.layerEnd)"
              + (wanted.hasOutput ? " +output" : "") + " · ctx \(wanted.contextSize)"
              + (wanted.expertCacheSlots > 0 ? " · \(wanted.expertCacheSlots) slot cache" : "")
              + " — carico il motore…\n")
        let t0 = Date()
        // The load is SILENT for minutes otherwise (mmap, Metal init, resident
        // copies, sidecar/Q4 REQUANT builds on first run): mirror the phase
        // breadcrumbs + LoadProgress into the worker log AND relay them to the
        // coordinator as `progress` frames, so BOTH logs show what is running.
        LoadProgress.shared.reset()
        let report: @Sendable (String) -> Void = { [onLog] text in
            onLog("caricamento: \(text)\n")
        }
        let progressConn = conn
        let poller = Task { [report] in
            var lastStage = ""
            var lastPct = -10.0
            while !Task.isCancelled {
                let s = LoadProgress.shared.snapshot
                let pct = s.fraction * 100
                if !s.stage.isEmpty, s.stage != lastStage || pct - lastPct >= 10 {
                    lastStage = s.stage; lastPct = pct
                    let text = String(format: "%@ (%.0f%%)", s.stage, pct)
                    report(text)
                    try? await progressConn.sendFrame(.progress, Data(text.utf8))
                }
                try? await Task.sleep(nanoseconds: 1_500_000_000)
            }
        }
        defer { poller.cancel() }
        do {
            // Off this connection's task: the load (mmap + Metal init) runs for
            // minutes while frames could arrive. On a GCD thread, NOT a Swift
            // Concurrency cooperative thread: the load fans out everywhere with
            // DispatchQueue.concurrentPerform (Q4 requant, cache reads, expert
            // fills), which from a cooperative thread can degrade to near-SERIAL
            // execution — one core pegged and the requant taking hours instead
            // of minutes. Phase breadcrumbs cover what LoadProgress does not.
            let slots = wanted.expertCacheSlots
            let loaded = try await withCheckedThrowingContinuation { (cont: CheckedContinuation<DistEngine, Error>) in
                DispatchQueue.global(qos: .userInitiated).async { [report] in
                    do {
                        cont.resume(returning: try DistEngine(
                            modelPath: resolved, contextSize: wanted.contextSize,
                            expertCacheSlots: slots > 0 ? slots : nil,
                            kvLayers: wanted.layerStart..<(wanted.layerEnd + 1),
                            onLoadLog: report))
                    } catch {
                        cont.resume(throwing: error)
                    }
                }
            }
            poller.cancel()              // no progress frames after READY is submitted
            commitAssignment(loaded, wanted)
            applyAncillary(assign, to: loaded)
            onLog(String(format: "motore pronto in %.1fs\n", Date().timeIntervalSince(t0)))
            try await conn.sendFrame(.ready, helloPayload().encoded())
        } catch {
            poller.cancel()
            releaseAssignmentClaim()
            let msg = "engine load failed: \(error)"
            onLog(msg + "\n")
            try await conn.sendFrame(.error, Data(msg.utf8))
        }
    }

    // Synchronous locked state transitions for handleAssign (the compiler
    // forbids direct NSLock use in async bodies — a suspension while holding
    // the lock would deadlock).

    private enum AssignmentClaim { case reuse(DistEngine), busy, inFlight, load }

    /// Reuse the loaded engine, report busy, join an identical in-flight load,
    /// or claim the load slot (freeing the old shard FIRST so its memory is
    /// gone before the new one arrives).
    private func claimAssignment(_ wanted: Assignment) -> AssignmentClaim {
        stateLock.lock(); defer { stateLock.unlock() }
        if assignment == wanted, let current = engine { return .reuse(current) }
        if loadingAssignment {
            // v8 retry: se la STESSA assegnazione è già in caricamento (la
            // connessione che l'ha chiesta è caduta a metà load, il retry del
            // coordinatore riconnette), il nuovo tentativo si AGGANCIA al load
            // in corso invece di ricevere "busy" — un blip di rete durante i
            // minuti del load non deve affossare la route.
            return pendingAssignment == wanted ? .inFlight : .busy
        }
        loadingAssignment = true
        pendingAssignment = wanted
        engine = nil
        assignment = nil
        return .load
    }

    private func commitAssignment(_ loaded: DistEngine, _ wanted: Assignment) {
        stateLock.lock()
        engine = loaded
        assignment = wanted
        loadingAssignment = false
        pendingAssignment = nil
        stateLock.unlock()
    }

    private func releaseAssignmentClaim() {
        stateLock.lock(); loadingAssignment = false; pendingAssignment = nil; stateLock.unlock()
    }

    /// EXPERT ASSIGN (Fase B, scissione verticale): carica lo shard di esperti
    /// definito dalla mask. Alternativo all'assegnazione a layer; il modello
    /// arriva dalla stessa distribuzione file v8 (offer/need/resume).
    private func handleExpertAssign(_ payload: Data, on conn: DistConnection,
                                    resolvedFiles: [String: String]) async throws {
        guard let assign = DistExpertAssign.decode(payload) else {
            try await conn.sendFrame(.error, Data("malformed EXPERT ASSIGN frame".utf8))
            return
        }
        let sanitizedName = DistFileStore.sanitize(assign.modelName)
        guard let resolved = resolvedFiles[sanitizedName]
                ?? Self.resolveModelPath(requestedPath: "", modelName: assign.modelName,
                                         localHint: config.localModelPath) else {
            let msg = "gguf '\(assign.modelName)' not available on this worker"
            onLog(msg + "\n")
            try await conn.sendFrame(.error, Data(msg.utf8))
            return
        }
        // Knob di performance del coordinatore (stessa whitelist di handleAssign).
        let allowedKnobs = Set(Dist.perfKnobKeys)
        for (k, v) in assign.envKnobs where allowedKnobs.contains(k) && v.count <= 256 {
            _ = setenv(k, v, 1)
        }
        _ = setenv("DS4_EXPERT_BUNDLE", assign.useExpertBundle ? "1" : "0", 1)
        if assign.useExpertBundle,
           let bundlePath = resolvedFiles[sanitizedName + ".expbundle"] {
            _ = setenv("DS4_BUNDLE_DIR", (bundlePath as NSString).deletingLastPathComponent, 1)
        }
        guard claimShardLoad() else {
            try await conn.sendFrame(.error, Data("worker busy loading an expert shard".utf8))
            return
        }
        onLog("expert-shard: \(assign.modelName) — carico lo shard…\n")
        do {
            let report: @Sendable (String) -> Void = { [onLog] t in onLog("caricamento shard: \(t)\n") }
            let maskData = assign.expertMask
            let slots = assign.expertCacheSlots
            let usageJSON = assign.usageJSON
            // Su thread GCD, non sul pool cooperativo (stessa ragione del load
            // del motore a slice qui sopra: fan-out concurrentPerform pieni).
            let shard = try await withCheckedThrowingContinuation { (cont: CheckedContinuation<ExpertShardEngine, Error>) in
                DispatchQueue.global(qos: .userInitiated).async {
                    do {
                        cont.resume(returning: try ExpertShardEngine(
                            modelPath: resolved, expertMask: maskData,
                            expertCacheSlots: slots, usageJSON: usageJSON,
                            onLoadLog: report))
                    } catch {
                        cont.resume(throwing: error)
                    }
                }
            }
            commitShard(shard)
            onLog("expert-shard pronto: \(shard.ownedCount)/256 esperti\n")
            try await conn.sendFrame(.ready, helloPayload().encoded())
        } catch {
            commitShard(nil)
            onLog("expert-shard: load fallito (\(error))\n")
            try await conn.sendFrame(.error, Data("expert shard load failed: \(error)".utf8))
        }
    }

    /// Apply the parts of an ASSIGN that do NOT require a reload: the usage
    /// imatrix (slot-cache pre-warm) and the disk-KV budget. Runs on every
    /// ASSIGN, including the engine-reuse path.
    private func applyAncillary(_ assign: DistAssign, to engine: DistEngine) {
        let dir = Self.shardStateDirectory(modelName: assign.modelName,
                                           layerStart: assign.layerStart,
                                           layerEnd: assign.layerEnd)
        // Usage: this shard's own refined profile wins (it already contains a
        // coordinator seed from a previous session); else the ASSIGN's blob.
        let file = dir.appendingPathComponent("usage.json")
        if let own = try? Data(contentsOf: file) {
            engine.loadUsage(own)
            onLog("imatrix: profilo locale dello shard caricato\n")
        } else if !assign.usageJSON.isEmpty {
            engine.loadUsage(assign.usageJSON)
            onLog("imatrix: profilo del coordinatore caricato (\(assign.usageJSON.count) byte)\n")
        }
        engine.setDiskKV(directory: assign.diskKVBudgetTokens > 0
                             ? dir.appendingPathComponent("kv", isDirectory: true) : nil,
                         budgetTokens: assign.diskKVBudgetTokens)
        if assign.diskKVBudgetTokens > 0 {
            onLog("disk KV shard: budget \(assign.diskKVBudgetTokens) token\n")
        }
        stateLock.lock(); usageFile = file; stateLock.unlock()
    }

    /// Serve one KV control frame: query stored prefixes, restore an exact
    /// checkpoint, or checkpoint the current shard state.
    private func handleKV(_ type: Dist.MsgType, _ payload: Data, on conn: DistConnection) async throws {
        guard let (engine, _) = activeEngine() else {
            let msg = "worker not ready: no assignment loaded"
            switch type {
            case .kvQuery: try await conn.sendFrame(.kvLengths, DistKV.encodeLengths([]))
            default: try await conn.sendFrame(.kvAck, DistKV.encodeAck(ok: false, message: msg))
            }
            return
        }
        switch type {
        case .kvQuery:
            guard let ids = DistKV.decodeTokens(payload) else {
                try await conn.sendFrame(.kvLengths, DistKV.encodeLengths([]))
                return
            }
            try await conn.sendFrame(.kvLengths,
                                     DistKV.encodeLengths(engine.storedPrefixLengths(of: ids)))
        case .kvRestore:
            guard let ids = DistKV.decodeTokens(payload) else {
                try await conn.sendFrame(.kvAck, DistKV.encodeAck(ok: false, message: "malformed kvRestore"))
                return
            }
            // Through the gate: the restore writes the shard's KV buffers and
            // must not interleave with compute.
            let ok = await gate.run { engine.restoreKV(tokens: ids) }
            if ok { onLog("KV shard ripristinato da disco (\(ids.count) token)\n") }
            try await conn.sendFrame(.kvAck, DistKV.encodeAck(
                ok: ok, message: ok ? "" : "no checkpoint for \(ids.count) tokens"))
        case .kvSave:
            guard let (ids, cold) = DistKV.decodeSave(payload) else {
                try await conn.sendFrame(.kvAck, DistKV.encodeAck(ok: false, message: "malformed kvSave"))
                return
            }
            // Export under the gate (state must hold still); the disk write
            // itself streams in the background (SnapshotBox).
            await gate.run { engine.saveKV(tokens: ids, cold: cold) }
            onLog("KV shard: checkpoint \(ids.count) token avviato\n")
            try await conn.sendFrame(.kvAck, DistKV.encodeAck(ok: true))
        default:
            break
        }
    }
}

/// Serializes async closures (the shard runs one step at a time).
/// Executor su coda GCD seriale (SE-0392): i body fanno fan-out con
/// DispatchQueue.concurrentPerform (gather esperti, slice del decoder), che
/// da un thread del pool cooperativo può degradare al quasi-seriale — vedi
/// InferenceService.engineQueue. Semantica identica (coda seriale), cambia
/// solo il thread di esecuzione.
actor DistGate {
    private nonisolated let queue = DispatchSerialQueue(label: "ds4.dist.gate", qos: .userInitiated)
    nonisolated var unownedExecutor: UnownedSerialExecutor { queue.asUnownedSerialExecutor() }
    func run<T: Sendable>(_ body: @Sendable () throws -> T) async rethrows -> T { try body() }
}
