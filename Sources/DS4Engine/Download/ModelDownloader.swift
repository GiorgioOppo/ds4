import Foundation
import CryptoKit

// Native Swift model downloader — replaces download_model.sh. Resumable HTTP
// download (Range), straight from the Hugging Face `resolve` endpoint into
// <ggufDir>/<file>.part, renamed to <file> on completion. No external script,
// no curl/hf subprocess. (The huge PRO multi-hundred-GB files also work here;
// the old script used the `hf` CLI for those only as a curl-robustness choice.)
//
// Integrity: the `resolve` endpoint 302-redirects to Hugging Face's LFS CDN
// (cdn-lfs*.huggingface.co / CloudFront), whose certificates/keys we do not
// control and which rotate continuously — so TLS public-key pinning would be
// fragile here. The robust, CDN-rotation-proof control is content pinning:
// verify the SHA-256 of the assembled file against a known-good digest (the HF
// LFS oid). This defends against a tampered/corrupted artifact regardless of the
// transport. See `ModelTarget.sha256`.

public struct ModelTarget: Sendable, Identifiable {
    public let id: String          // CLI-style target id (q4-imatrix, …)
    public let file: String        // GGUF filename in the HF repo
    public let approxGB: Int
    public let note: String
    /// Known-good lowercase-hex SHA-256 of the GGUF (the Hugging Face LFS oid).
    /// When set, the downloader verifies the assembled file and refuses a
    /// mismatch (deleting the bad file). nil = unknown: the downloader instead
    /// reports the digest it computed so it can be pinned here. This content pin
    /// is independent of the TLS channel and survives HF CDN key rotation.
    public let sha256: String?
    public init(id: String, file: String, approxGB: Int, note: String, sha256: String? = nil) {
        self.id = id; self.file = file; self.approxGB = approxGB; self.note = note
        self.sha256 = sha256
    }
}

public enum ModelDownloader {
    public static let repo = "antirez/deepseek-v4-gguf"

    public static let targets: [ModelTarget] = [
        // To activate verification, paste the file's SHA-256 (HF LFS oid) into
        // `sha256:` below — e.g. from `shasum -a 256 <file>` after a trusted
        // download, or the repo's LFS pointer (`…/raw/main/<file>` → `oid sha256:`).
        .init(id: "q2-imatrix",
              file: "DeepSeek-V4-Flash-IQ2XXS-w2Q2K-AProjQ8-SExpQ8-OutQ8-chat-v2-imatrix.gguf",
              approxGB: 81, note: "2-bit routed experts — 96/128 GB RAM"),
        .init(id: "q2-q4-imatrix",
              file: "DeepSeek-V4-Flash-Layers37-42Q4KExperts-OtherExpertLayersIQ2XXSGateUp-Q2KDown-AProjQ8-SExpQ8-OutQ8-chat-v2-imatrix-fixed.gguf",
              approxGB: 98, note: "mixed q2/q4 — higher quality, 128 GB"),
        .init(id: "q4-imatrix",
              file: "DeepSeek-V4-Flash-Q4KExperts-F16HC-F16Compressor-F16Indexer-Q8Attn-Q8Shared-Q8Out-chat-v2-imatrix.gguf",
              approxGB: 153, note: "4-bit routed experts — 256 GB RAM"),
        .init(id: "mtp",
              file: "DeepSeek-V4-Flash-MTP-Q4K-Q8_0-F32.gguf",
              approxGB: 4, note: "optional speculative-decoding component"),
        .init(id: "pro-q2-imatrix",
              file: "DeepSeek-V4-Pro-IQ2XXS-w2Q2K-AProjQ8-SExpQ8-OutQ8-Instruct-imatrix.gguf",
              approxGB: 430, note: "PRO q2 single GGUF — 512 GB / 128 GB streaming"),
    ]

    public static func target(_ id: String) -> ModelTarget? { targets.first { $0.id == id } }

    public static func resolveURL(_ file: String) -> URL {
        URL(string: "https://huggingface.co/\(repo)/resolve/main/\(file)")!
    }

    /// Resolve an HF token: explicit > HF_TOKEN env > ~/.cache/huggingface/token.
    public static func resolveToken(_ explicit: String?) -> String? {
        if let t = explicit, !t.isEmpty { return t }
        if let t = ProcessInfo.processInfo.environment["HF_TOKEN"], !t.isEmpty { return t }
        let p = (NSHomeDirectory() as NSString).appendingPathComponent(".cache/huggingface/token")
        if let t = try? String(contentsOfFile: p, encoding: .utf8) {
            let trimmed = t.trimmingCharacters(in: .whitespacesAndNewlines)
            if !trimmed.isEmpty { return trimmed }
        }
        return nil
    }

    public enum DownloadError: Error, CustomStringConvertible {
        case http(Int)
        case checksumMismatch(expected: String, got: String)
        public var description: String {
            switch self {
            case .http(let c): return "HTTP \(c)"
            case .checksumMismatch(let e, let g):
                return "SHA-256 non corrisponde — atteso \(e), ottenuto \(g). Il file scaricato è stato rimosso (possibile corruzione o manomissione)."
            }
        }
    }

    /// Download `target` into `ggufDir` with resume. `onProgress(done, total)` is
    /// called periodically (bytes); `onStatus` carries human-readable phase
    /// messages (integrity verification). Returns the final file URL. Honors Task
    /// cancellation (the .part file is left for a later resume).
    ///
    /// On completion the file's SHA-256 is checked against `target.sha256` when
    /// pinned (mismatch → the file is removed and `checksumMismatch` thrown); when
    /// unpinned, the computed digest is reported via `onStatus` so it can be pinned.
    public static func download(target: ModelTarget, ggufDir: String, token: String? = nil,
                                onProgress: @Sendable (Int64, Int64) -> Void,
                                onStatus: @Sendable (String) -> Void = { _ in }) async throws -> URL {
        try FileManager.default.createDirectory(atPath: ggufDir, withIntermediateDirectories: true)
        let dest = URL(fileURLWithPath: (ggufDir as NSString).appendingPathComponent(target.file))
        let part = URL(fileURLWithPath: dest.path + ".part")
        if FileManager.default.fileExists(atPath: dest.path) {
            onProgress(1, 1)
            // A cached file still gets verified when a digest is pinned (it may be
            // stale/corrupt); no on-the-fly digest available, so this re-reads it.
            try verify(file: dest, target: target, computed: nil, onProgress: onProgress, onStatus: onStatus)
            return dest
        }

        var offset: Int64 = 0
        if let attrs = try? FileManager.default.attributesOfItem(atPath: part.path),
           let sz = attrs[.size] as? Int64 { offset = sz }

        var req = URLRequest(url: resolveURL(target.file))
        if offset > 0 { req.setValue("bytes=\(offset)-", forHTTPHeaderField: "Range") }
        if let tok = resolveToken(token) { req.setValue("Bearer \(tok)", forHTTPHeaderField: "Authorization") }

        let (bytes, response) = try await URLSession.shared.bytes(for: req)
        guard let http = response as? HTTPURLResponse else { throw DownloadError.http(-1) }
        if http.statusCode == 416 { // already complete on disk
            try? FileManager.default.moveItem(at: part, to: dest); onProgress(offset, offset)
            try verify(file: dest, target: target, computed: nil, onProgress: onProgress, onStatus: onStatus)
            return dest
        }
        guard http.statusCode == 200 || http.statusCode == 206 else { throw DownloadError.http(http.statusCode) }
        if http.statusCode == 200 { offset = 0 }  // server ignored Range -> restart

        let total = offset + max(0, http.expectedContentLength)
        if offset == 0 { FileManager.default.createFile(atPath: part.path, contents: nil) }
        let handle = try FileHandle(forWritingTo: part)
        defer { try? handle.close() }
        try handle.seekToEnd()

        // Hash on the fly ONLY for a single full session from byte 0: then the
        // running digest equals the whole-file digest (no extra read). A resumed
        // download (offset > 0) skips this; verification re-reads the file instead.
        let fullSession = (offset == 0)
        var hasher = SHA256()
        var done = offset
        var buf = Data(); buf.reserveCapacity(1 << 20)
        for try await b in bytes {
            buf.append(b)
            if buf.count >= (1 << 20) {
                try handle.write(contentsOf: buf)
                if fullSession { hasher.update(data: buf) }
                done += Int64(buf.count); buf.removeAll(keepingCapacity: true)
                onProgress(done, total)
                try Task.checkCancellation()
            }
        }
        if !buf.isEmpty {
            try handle.write(contentsOf: buf)
            if fullSession { hasher.update(data: buf) }
            done += Int64(buf.count)
        }
        try handle.close()
        try? FileManager.default.removeItem(at: dest)
        try FileManager.default.moveItem(at: part, to: dest)
        onProgress(done, total)

        try verify(file: dest, target: target,
                   computed: fullSession ? hex(hasher.finalize()) : nil,
                   onProgress: onProgress, onStatus: onStatus)
        return dest
    }

    /// Verify `file` against the pinned `target.sha256`, or report the digest when
    /// unpinned. `computed` is the on-the-fly digest from a fresh download (avoids
    /// re-reading the file); pass nil to compute it by streaming from disk.
    /// Throws `checksumMismatch` (and removes the file) on a pinned mismatch.
    private static func verify(file: URL, target: ModelTarget, computed: String?,
                               onProgress: @Sendable (Int64, Int64) -> Void,
                               onStatus: @Sendable (String) -> Void) throws {
        guard let expected = target.sha256, !expected.isEmpty else {
            if let d = computed {
                onStatus("SHA-256: \(d)\nNessun digest atteso configurato per “\(target.id)”: incollalo in ModelTarget.sha256 per attivare la verifica automatica.")
            }
            return
        }
        onStatus("Verifica integrità (SHA-256)…")
        var total: Int64 = 1
        if let attrs = try? FileManager.default.attributesOfItem(atPath: file.path),
           let sz = attrs[.size] as? Int64 { total = sz }
        let got = try computed ?? sha256Hex(of: file, onProgress: { hashed in onProgress(hashed, total) })
        guard got.caseInsensitiveCompare(expected) == .orderedSame else {
            try? FileManager.default.removeItem(at: file)
            throw DownloadError.checksumMismatch(expected: expected.lowercased(), got: got)
        }
        onStatus("✓ SHA-256 verificato")
    }

    /// Stream a file from disk and return its lowercase-hex SHA-256, reporting the
    /// number of bytes hashed via `onProgress`. Honors Task cancellation.
    public static func sha256Hex(of url: URL, chunk: Int = 8 << 20,
                                 onProgress: (Int64) -> Void = { _ in }) throws -> String {
        let handle = try FileHandle(forReadingFrom: url)
        defer { try? handle.close() }
        var hasher = SHA256()
        var hashed: Int64 = 0
        while let data = try handle.read(upToCount: chunk), !data.isEmpty {
            hasher.update(data: data)
            hashed += Int64(data.count)
            onProgress(hashed)
            try Task.checkCancellation()
        }
        return hex(hasher.finalize())
    }

    private static func hex(_ digest: SHA256.Digest) -> String {
        digest.map { String(format: "%02x", $0) }.joined()
    }
}
