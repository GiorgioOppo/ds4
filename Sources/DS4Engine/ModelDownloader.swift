import Foundation

// Native Swift model downloader — replaces download_model.sh. Resumable HTTP
// download (Range), straight from the Hugging Face `resolve` endpoint into
// <ggufDir>/<file>.part, renamed to <file> on completion. No external script,
// no curl/hf subprocess. (The huge PRO multi-hundred-GB files also work here;
// the old script used the `hf` CLI for those only as a curl-robustness choice.)

public struct ModelTarget: Sendable, Identifiable {
    public let id: String          // CLI-style target id (q4-imatrix, …)
    public let file: String        // GGUF filename in the HF repo
    public let approxGB: Int
    public let note: String
    public init(id: String, file: String, approxGB: Int, note: String) {
        self.id = id; self.file = file; self.approxGB = approxGB; self.note = note
    }
}

public enum ModelDownloader {
    public static let repo = "antirez/deepseek-v4-gguf"

    public static let targets: [ModelTarget] = [
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
        public var description: String { switch self { case .http(let c): return "HTTP \(c)" } }
    }

    /// Download `target` into `ggufDir` with resume. `onProgress(done, total)` is
    /// called periodically (bytes). Returns the final file URL. Honors Task
    /// cancellation (the .part file is left for a later resume).
    public static func download(target: ModelTarget, ggufDir: String, token: String? = nil,
                                onProgress: @Sendable (Int64, Int64) -> Void) async throws -> URL {
        try FileManager.default.createDirectory(atPath: ggufDir, withIntermediateDirectories: true)
        let dest = URL(fileURLWithPath: (ggufDir as NSString).appendingPathComponent(target.file))
        let part = URL(fileURLWithPath: dest.path + ".part")
        if FileManager.default.fileExists(atPath: dest.path) { onProgress(1, 1); return dest }

        var offset: Int64 = 0
        if let attrs = try? FileManager.default.attributesOfItem(atPath: part.path),
           let sz = attrs[.size] as? Int64 { offset = sz }

        var req = URLRequest(url: resolveURL(target.file))
        if offset > 0 { req.setValue("bytes=\(offset)-", forHTTPHeaderField: "Range") }
        if let tok = resolveToken(token) { req.setValue("Bearer \(tok)", forHTTPHeaderField: "Authorization") }

        let (bytes, response) = try await URLSession.shared.bytes(for: req)
        guard let http = response as? HTTPURLResponse else { throw DownloadError.http(-1) }
        if http.statusCode == 416 { // already complete on disk
            try? FileManager.default.moveItem(at: part, to: dest); onProgress(offset, offset); return dest
        }
        guard http.statusCode == 200 || http.statusCode == 206 else { throw DownloadError.http(http.statusCode) }
        if http.statusCode == 200 { offset = 0 }  // server ignored Range -> restart

        let total = offset + max(0, http.expectedContentLength)
        if offset == 0 { FileManager.default.createFile(atPath: part.path, contents: nil) }
        let handle = try FileHandle(forWritingTo: part)
        defer { try? handle.close() }
        try handle.seekToEnd()

        var done = offset
        var buf = Data(); buf.reserveCapacity(1 << 20)
        for try await b in bytes {
            buf.append(b)
            if buf.count >= (1 << 20) {
                try handle.write(contentsOf: buf); done += Int64(buf.count); buf.removeAll(keepingCapacity: true)
                onProgress(done, total)
                try Task.checkCancellation()
            }
        }
        if !buf.isEmpty { try handle.write(contentsOf: buf); done += Int64(buf.count) }
        try handle.close()
        try? FileManager.default.removeItem(at: dest)
        try FileManager.default.moveItem(at: part, to: dest)
        onProgress(done, total)
        return dest
    }
}
