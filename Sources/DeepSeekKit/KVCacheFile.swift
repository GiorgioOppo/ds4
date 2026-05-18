import Foundation
import Metal
import Darwin

/// Persistent, mmap-backed KV cache storage for a single conversation.
///
/// A `KVCacheFile` owns a file on disk laid out as:
///
///   [ HEADER (4 KiB, fixed) ][ PAYLOAD (page-aligned, variable) ]
///
/// The payload region is exposed as a single `MTLBuffer` over the mmap'd
/// pages (`storageModeShared` + `bytesNoCopy`), and the layers ask for
/// sub-views at known offsets. Apple Silicon's unified memory plus
/// `MADV_RANDOM` lets the kernel keep hot pages resident and page out
/// the rest on memory pressure — no copies on the critical path.
///
/// The header carries the metadata needed for cross-turn reuse (Step 3):
///   - `prefilledTokens`: how far the cache is filled
///   - `historyHash`: 128-bit hash of the messages that produced it
///   - `modelPathHash`: 64-bit hash of `modelDirPath`, so a model swap
///     invalidates the file
///   - `payloadBytes`: declared size of the payload region; mismatch
///     means the model config changed (different layer count, head
///     dim, window, etc.) and the file should be wiped.
///
/// Step 1 only builds and exposes this storage. Steps 2 and 3 plug it
/// into the layers and the inference lifecycle.
public final class KVCacheFile {
    public static let headerBytes: Int = 4096
    public static let magic: UInt32 = 0x4B564331   // 'KVC1' big-endian

    public struct Header {
        public var version: UInt32
        public var payloadBytes: UInt64
        public var prefilledTokens: UInt64
        public var historyHashLow: UInt64
        public var historyHashHigh: UInt64
        public var modelPathHash: UInt64
    }

    public let url: URL
    public let payloadBuffer: MTLBuffer
    public let payloadOffset: Int
    public let payloadBytes: Int

    // Retained so the deallocator can hand them back to the kernel
    // once Metal is done with the buffer.
    private let basePointer: UnsafeMutableRawPointer
    private let totalBytes: Int

    /// Opens (or creates) the file at `url`, sizes it to fit a payload
    /// of `payloadBytes`, mmaps it, and wraps the payload region as a
    /// shared-storage `MTLBuffer`. Each `KVCacheFile` instance owns
    /// the mapping for its lifetime; the `MTLBuffer`'s deallocator
    /// releases the mapping when Metal drops its last reference.
    public init(url: URL,
                 payloadBytes: Int,
                 device: MTLDevice = Device.shared.mtl) throws {
        precondition(payloadBytes > 0, "payloadBytes must be positive")

        let pageSize = Int(getpagesize())
        let alignedPayload = roundUp(payloadBytes, to: pageSize)
        let total = Self.headerBytes + alignedPayload

        // 1) Open + size the file.
        let fd = open(url.path,
                       O_RDWR | O_CREAT,
                       0o644)
        guard fd >= 0 else {
            throw KVCacheFileError.openFailed(errno: errno, path: url.path)
        }
        if ftruncate(fd, off_t(total)) != 0 {
            let e = errno
            close(fd)
            throw KVCacheFileError.truncateFailed(errno: e, path: url.path)
        }

        // 2) Map the whole file. PROT_READ|WRITE + MAP_SHARED so writes
        //    persist; MADV_RANDOM tells the kernel not to prefetch — the
        //    attention kernel reads scattered rows.
        //
        //    Bridge through `Int` so no `UnsafeMutableRawPointer?` ever
        //    appears in this block — every prior attempt to unwrap the
        //    mmap result inside a `guard let` had the type checker
        //    re-widen the binding back to Optional at the use sites.
        //
        //      Int(bitPattern: pointer?)  →  Int   (nil maps to 0)
        //      0  → mmap returned NULL
        //      -1 → mmap returned MAP_FAILED ((void *)-1)
        //      _  → valid mapping; rebuild a non-optional pointer.
        let addr = Int(bitPattern: mmap(nil, total,
                                         PROT_READ | PROT_WRITE,
                                         MAP_SHARED, fd, 0))
        close(fd)  // the mapping holds its own reference
        if addr == 0 || addr == -1 {
            throw KVCacheFileError.mmapFailed(errno: errno, path: url.path)
        }
        // `UnsafeMutableRawPointer(bitPattern:)` is failable only for 0,
        // which we just rejected — force-unwrap is safe and produces a
        // strictly non-optional binding the rest of the function uses.
        let raw: UnsafeMutableRawPointer =
            UnsafeMutableRawPointer(bitPattern: addr)!
        madvise(raw, total, MADV_RANDOM)

        // 3) Wrap as a Metal buffer. Use the full mapping so payload
        //    offsets are simple integer adds; layers will subscript
        //    via `offset:` on `Tensor`.
        let deallocator: (UnsafeMutableRawPointer, Int) -> Void = { ptr, len in
            munmap(ptr, len)
        }
        guard let buf = device.makeBuffer(
            bytesNoCopy: raw,
            length: total,
            options: .storageModeShared,
            deallocator: deallocator)
        else {
            munmap(raw, total)
            throw KVCacheFileError.metalWrapFailed(path: url.path)
        }

        self.url = url
        self.payloadBuffer = buf
        self.payloadOffset = Self.headerBytes
        self.payloadBytes = alignedPayload
        self.basePointer = raw
        self.totalBytes = total

        // 4) Initialise (or validate) the header.
        let h = currentHeader()
        if !isInitialised(h) {
            writeFreshHeader(payloadBytes: UInt64(alignedPayload))
        }
    }

    // ---- header access ----

    /// Reads the current on-disk header. Returns a header with
    /// `version == 0` and zeroed payload size when the file was just
    /// created (uninitialised).
    public func readHeader() -> Header {
        currentHeader()
    }

    /// Stamps a brand-new header into the file. Use this when the
    /// cache contents become invalid (model swap, config change, hash
    /// mismatch) — the payload bytes stay on disk but readers will
    /// treat the cache as empty. Cancella anche il manifest companion
    /// (i token IDs salvati non corrispondono più a niente di valido).
    public func resetHeader(modelPathHash: UInt64) {
        let h = Header(
            version: 1,
            payloadBytes: UInt64(payloadBytes),
            prefilledTokens: 0,
            historyHashLow: 0,
            historyHashHigh: 0,
            modelPathHash: modelPathHash)
        writeHeader(h)
        deleteManifest()
    }

    /// Bumps the prefill checkpoint after a generation step has
    /// extended the cache. Caller must also update the history hash
    /// when the prefilled prefix actually grew.
    public func updateCheckpoint(prefilledTokens: UInt64,
                                  historyHashLow: UInt64,
                                  historyHashHigh: UInt64,
                                  modelPathHash: UInt64) {
        var h = currentHeader()
        h.version = 1
        h.payloadBytes = UInt64(payloadBytes)
        h.prefilledTokens = prefilledTokens
        h.historyHashLow = historyHashLow
        h.historyHashHigh = historyHashHigh
        h.modelPathHash = modelPathHash
        writeHeader(h)
    }

    /// Tronca il prefill checkpoint ad `N` token senza zeroare il
    /// payload. Usato dal prefix-matching: quando il nuovo prompt
    /// condivide solo `N` token col cache salvato, scartiamo
    /// virtualmente quelli oltre `N` senza dover invalidare l'intero
    /// cache. Le pagine del payload oltre `N` saranno sovrascritte
    /// dal prossimo prefill (qualunque sia il loro contenuto attuale).
    public func truncatePrefilledTo(_ newCount: UInt64,
                                      historyHashLow: UInt64,
                                      historyHashHigh: UInt64) {
        var h = currentHeader()
        guard h.version != 0, newCount <= h.prefilledTokens else {
            return
        }
        h.prefilledTokens = newCount
        h.historyHashLow = historyHashLow
        h.historyHashHigh = historyHashHigh
        writeHeader(h)
    }

    // ---- token manifest (companion file) ----

    /// URL del file companion che memorizza la lista dei token IDs
    /// dei `prefilledTokens` del cache. Layout binario:
    ///
    ///   [ u32 magic 'KVM1' ][ u32 version=1 ][ u64 count ][ count × i32 token_ids ]
    ///
    /// File separato dal `KVCacheFile` principale per evitare
    /// vincoli di alignment col payload (Metal buffer page-aligned)
    /// e per permettere update atomico via tmp+rename indipendente.
    public var manifestURL: URL {
        url.appendingPathExtension("manifest")
    }

    /// Magic identifier per il manifest file: 'KVM1' big-endian.
    public static let manifestMagic: UInt32 = 0x4B564D31

    /// Carica la lista dei token IDs dal manifest companion. Ritorna
    /// `nil` se il manifest non esiste, è corrotto, o ha versione
    /// incompatibile. Letto in un colpo solo (file piccolo: 32k token
    /// = 128 KB).
    public func readManifest() -> [Int32]? {
        guard let data = try? Data(contentsOf: manifestURL),
              data.count >= 16 else {
            return nil
        }
        return data.withUnsafeBytes { (raw: UnsafeRawBufferPointer) -> [Int32]? in
            guard let base = raw.baseAddress else { return nil }
            let magic = base.load(fromByteOffset: 0, as: UInt32.self)
            let version = base.load(fromByteOffset: 4, as: UInt32.self)
            let count = base.load(fromByteOffset: 8, as: UInt64.self)
            guard magic == Self.manifestMagic,
                  version == 1,
                  16 + Int(count) * 4 <= data.count
            else { return nil }
            let tokenPtr = base.advanced(by: 16)
                .assumingMemoryBound(to: Int32.self)
            return Array(UnsafeBufferPointer(start: tokenPtr, count: Int(count)))
        }
    }

    /// Scrive (sovrascrive atomicamente) il manifest dei token IDs
    /// che corrispondono al payload attuale. Da chiamare ogni volta
    /// che `prefilledTokens` cambia. Atomic via tmp+rename per non
    /// lasciare il manifest in stato incoerente se il processo crasha
    /// durante la scrittura.
    public func writeManifest(_ tokens: [Int32]) throws {
        var data = Data(capacity: 16 + tokens.count * 4)
        var magic = Self.manifestMagic.littleEndian
        var version = UInt32(1).littleEndian
        var count = UInt64(tokens.count).littleEndian
        withUnsafeBytes(of: &magic)   { data.append(contentsOf: $0) }
        withUnsafeBytes(of: &version) { data.append(contentsOf: $0) }
        withUnsafeBytes(of: &count)   { data.append(contentsOf: $0) }
        tokens.withUnsafeBufferPointer { p in
            let byteCount = p.count * MemoryLayout<Int32>.stride
            let bytePtr = UnsafeRawPointer(p.baseAddress!)
                .assumingMemoryBound(to: UInt8.self)
            data.append(bytePtr, count: byteCount)
        }
        let tmp = manifestURL.appendingPathExtension("tmp")
        try data.write(to: tmp, options: .atomic)
        // .atomic write già garantisce atomicità su un singolo path;
        // ma noi usiamo un companion `.tmp` perché vogliamo che il
        // rename finale sia esplicito (tracciabile in case di crash
        // mid-rename) e perché Foundation potrebbe usare una strategia
        // diversa per file piccoli. Sostituzione atomica via
        // FileManager.replaceItem se il path esiste.
        let fm = FileManager.default
        if fm.fileExists(atPath: manifestURL.path) {
            _ = try fm.replaceItemAt(manifestURL, withItemAt: tmp)
        } else {
            try fm.moveItem(at: tmp, to: manifestURL)
        }
    }

    /// Cancella il manifest. Usato quando il cache viene resettato
    /// (es. `resetHeader`).
    public func deleteManifest() {
        try? FileManager.default.removeItem(at: manifestURL)
    }

    /// Calcola la lunghezza del prefisso comune fra i token salvati
    /// nel manifest e `newTokens`. Ritorna 0 se il manifest non
    /// esiste o se il primo token differisce. O(min(savedCount,
    /// newTokens.count)) — un singolo scan lineare.
    ///
    /// Caso d'uso: chat multi-turn. La nuova turn dell'utente ha
    /// in testa il system prompt + tutti i message precedenti
    /// (identici al cache salvato) e termina con il nuovo user
    /// message + new assistant message-start. Il common prefix è
    /// "tutto fino al new user message" — possiamo skippare il
    /// prefill di quei token e ripartire da lì.
    public func commonPrefixLength(with newTokens: [Int32]) -> Int {
        guard let saved = readManifest() else { return 0 }
        let limit = min(saved.count, newTokens.count)
        var i = 0
        while i < limit && saved[i] == newTokens[i] {
            i += 1
        }
        return i
    }

    /// Outcome del tentativo di resume da cache.
    public enum ResumeOutcome: Equatable {
        /// Nessun manifest, hash mismatch, o common prefix sotto la
        /// soglia minima. Cache da considerare invalida; ripartire
        /// da zero (prefill di tutti i `newTokens`).
        case invalid
        /// Match parziale: i primi `tokens` del nuovo prompt sono
        /// coperti dal cache. Il caller deve fare prefill solo dei
        /// rimanenti `newTokens[tokens..<]`.
        case partial(tokens: Int)
        /// Match esatto: tutti i nuovi token sono già nel cache.
        /// Stato raro (la nuova turn ha esattamente lo stesso prompt
        /// della precedente) ma valido — il caller può saltare il
        /// prefill interamente.
        case full
    }

    /// Tenta il resume da un cache salvato. Combina la validazione
    /// degli hash globali (modelPathHash) + lookup del prefix
    /// matching nel manifest + truncate del checkpoint al common
    /// prefix.
    ///
    /// - Parameter newTokens: la sequenza di token che vogliamo
    ///   processare in questa turn (dal system prompt al new message
    ///   start, incluso).
    /// - Parameter modelPathHash: hash della path del modello
    ///   attuale; deve corrispondere a quanto nel cache.
    /// - Parameter minPrefixTokens: numero minimo di token in
    ///   prefisso comune per considerare il resume utile. Default 16:
    ///   sotto questa soglia il cost di lookup supera il guadagno.
    /// - Returns: outcome che dice quanti token sono già coperti.
    ///   Se `.partial`, il header viene già aggiornato (truncate)
    ///   per riflettere il nuovo prefilledTokens; il caller può
    ///   procedere col prefill del resto.
    public func attemptResume(newTokens: [Int32],
                                modelPathHash: UInt64,
                                minPrefixTokens: Int = 16)
        -> ResumeOutcome
    {
        let h = readHeader()
        guard h.version == 1,
              h.modelPathHash == modelPathHash,
              h.payloadBytes == UInt64(payloadBytes),
              h.prefilledTokens > 0
        else {
            return .invalid
        }
        let common = commonPrefixLength(with: newTokens)
        if common == 0 || common < minPrefixTokens {
            return .invalid
        }
        // Compute new history hash for the truncated prefix using
        // a simple FNV-1a 128-bit-like roll over the first `common`
        // tokens. La caller può riusare il valore al successivo
        // writeManifest per coerenza.
        var hLow: UInt64 = 0xcbf29ce484222325
        var hHigh: UInt64 = 0x84222325cbf29ce4
        for i in 0..<common {
            let t = UInt64(bitPattern: Int64(newTokens[i]))
            hLow ^= t
            hLow = hLow &* 0x100000001b3
            hHigh ^= t.byteSwapped
            hHigh = hHigh &* 0x100000001b3
        }
        if common == newTokens.count {
            truncatePrefilledTo(UInt64(common),
                                 historyHashLow: hLow,
                                 historyHashHigh: hHigh)
            return .full
        }
        truncatePrefilledTo(UInt64(common),
                             historyHashLow: hLow,
                             historyHashHigh: hHigh)
        return .partial(tokens: common)
    }

    // ---- payload access ----

    /// Returns a Metal buffer + byte offset describing a sub-region
    /// of the payload. Layers compose a `Tensor` over this by passing
    /// `buffer:` and `offset:` to `Tensor.init`.
    public func region(offset: Int, length: Int) -> (MTLBuffer, Int) {
        precondition(offset >= 0)
        precondition(length > 0)
        precondition(offset + length <= payloadBytes,
                      "region [\(offset), \(offset+length)) exceeds payload size \(payloadBytes)")
        return (payloadBuffer, payloadOffset + offset)
    }

    /// Costruisce un `Tensor` che vive nel payload del cache file.
    /// Niente alloc — è un wrapper della stessa storage GPU-visibile
    /// del file mmappato. Tutti i write tramite questo tensor finiscono
    /// direttamente nel file su disco (storageModeShared + MAP_SHARED).
    ///
    /// Usato dal wiring fisico cross-restart per allocare Compressor.
    /// kvState/scoreState e Attention.kvCache come slice del KVCacheFile
    /// invece che come MTLBuffer indipendenti. Significa che alla
    /// chiusura dell'app lo stato resta su disco; al riavvio basta
    /// riaprire il file e ricostruire i wrapper Tensor con gli stessi
    /// offset → KV cache automaticamente "restored".
    public func tensor(at region: KVCacheLayout.Region,
                        shape: [Int], dtype: DType) -> Tensor {
        let elementBytes = (dtype.bitsPerElement + 7) / 8
        let expectedBytes = shape.reduce(1, *) * elementBytes
        precondition(expectedBytes <= region.bytes,
                      "tensor shape \(shape) dtype \(dtype) needs " +
                      "\(expectedBytes) bytes but region has \(region.bytes)")
        precondition(region.offset >= 0
                      && region.offset + region.bytes <= payloadBytes,
                      "region [\(region.offset), \(region.offset + region.bytes)) " +
                      "out of payload [0, \(payloadBytes))")
        return Tensor(shape: shape, dtype: dtype,
                       buffer: payloadBuffer,
                       offset: payloadOffset + region.offset)
    }

    // ---- internals ----

    private func currentHeader() -> Header {
        var h = Header(version: 0, payloadBytes: 0,
                        prefilledTokens: 0,
                        historyHashLow: 0, historyHashHigh: 0,
                        modelPathHash: 0)
        // Layout (little-endian on Apple Silicon):
        //   u32 magic | u32 version | u64 payloadBytes
        //   u64 prefilledTokens | u64 hashLow | u64 hashHigh
        //   u64 modelPathHash
        let p = basePointer.assumingMemoryBound(to: UInt8.self)
        let magic = readUInt32(p, 0)
        guard magic == Self.magic else { return h }
        h.version          = readUInt32(p, 4)
        h.payloadBytes     = readUInt64(p, 8)
        h.prefilledTokens  = readUInt64(p, 16)
        h.historyHashLow   = readUInt64(p, 24)
        h.historyHashHigh  = readUInt64(p, 32)
        h.modelPathHash    = readUInt64(p, 40)
        return h
    }

    private func isInitialised(_ h: Header) -> Bool {
        h.version != 0 && h.payloadBytes == UInt64(payloadBytes)
    }

    private func writeFreshHeader(payloadBytes: UInt64) {
        let h = Header(version: 1,
                        payloadBytes: payloadBytes,
                        prefilledTokens: 0,
                        historyHashLow: 0,
                        historyHashHigh: 0,
                        modelPathHash: 0)
        writeHeader(h)
    }

    private func writeHeader(_ h: Header) {
        let p = basePointer.assumingMemoryBound(to: UInt8.self)
        writeUInt32(p, 0,  Self.magic)
        writeUInt32(p, 4,  h.version)
        writeUInt64(p, 8,  h.payloadBytes)
        writeUInt64(p, 16, h.prefilledTokens)
        writeUInt64(p, 24, h.historyHashLow)
        writeUInt64(p, 32, h.historyHashHigh)
        writeUInt64(p, 40, h.modelPathHash)
        // Force the dirty header pages to disk so a crash mid-turn
        // doesn't leave a stale prefill checkpoint pointing at
        // partially-written payload data.
        msync(basePointer, Self.headerBytes, MS_ASYNC)
    }
}

public enum KVCacheFileError: Error, LocalizedError {
    case openFailed(errno: Int32, path: String)
    case truncateFailed(errno: Int32, path: String)
    case mmapFailed(errno: Int32, path: String)
    case metalWrapFailed(path: String)

    public var errorDescription: String? {
        switch self {
        case .openFailed(let e, let p):
            return "open(\(p)) failed: \(String(cString: strerror(e)))"
        case .truncateFailed(let e, let p):
            return "ftruncate(\(p)) failed: \(String(cString: strerror(e)))"
        case .mmapFailed(let e, let p):
            return "mmap(\(p)) failed: \(String(cString: strerror(e)))"
        case .metalWrapFailed(let p):
            return "MTLDevice.makeBuffer(bytesNoCopy:) failed for \(p)"
        }
    }
}

// ---- byte helpers (host endian = little on Apple Silicon) ----

@inline(__always)
private func roundUp(_ x: Int, to multiple: Int) -> Int {
    let r = x % multiple
    return r == 0 ? x : x + (multiple - r)
}

@inline(__always)
private func readUInt32(_ p: UnsafePointer<UInt8>, _ off: Int) -> UInt32 {
    var v: UInt32 = 0
    memcpy(&v, p + off, 4)
    return v
}

@inline(__always)
private func readUInt64(_ p: UnsafePointer<UInt8>, _ off: Int) -> UInt64 {
    var v: UInt64 = 0
    memcpy(&v, p + off, 8)
    return v
}

@inline(__always)
private func writeUInt32(_ p: UnsafeMutablePointer<UInt8>, _ off: Int, _ v: UInt32) {
    var x = v
    memcpy(p + off, &x, 4)
}

@inline(__always)
private func writeUInt64(_ p: UnsafeMutablePointer<UInt8>, _ off: Int, _ v: UInt64) {
    var x = v
    memcpy(p + off, &x, 8)
}
