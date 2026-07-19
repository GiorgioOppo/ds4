import Foundation
import DS4Core

/// Output CLI del demo (header, righe per-token): testo libero, ma sulla
/// STESSA destinazione e con le stesse proprietà del log unificato.
func log(_ s: String) { DS4Log.raw(s) }

// MARK: - Opt-in A/B logit trace

/// A bounded, post-timing trace used by `scripts/metal_ab.sh`.
///
/// Capturing an Array is copy-on-write, so the hot path only retains the logits
/// storage that the decoder has already materialized.  Hashing and file I/O are
/// deliberately deferred to `finish`: the prefill/decode wall clocks printed by
/// the demo therefore do not include trace serialization.  The hard frame cap
/// prevents an accidentally large `maxNew` from retaining an unbounded number
/// of full-vocabulary vectors (one Flash vector is roughly 0.5 MiB).
final class ABLogitTrace {
    private struct PendingFrame {
        let phase: String
        let step: Int
        let inputToken: Int?
        let logits: [Float]
    }

    private struct Candidate: Codable {
        let token: Int
        let logit: Float
    }

    private struct Frame: Codable {
        let phase: String
        let step: Int
        let inputToken: Int?
        let offsetFloats: Int
        let count: Int
        let finiteCount: Int
        let bitHashFNV1a64: String
        let top3: [Candidate]
    }

    private struct Document: Codable {
        let format: String
        let byteOrder: String
        let floatFormat: String
        let traceLimit: Int
        let capturedFrames: Int
        let generatedTokens: [Int]
        let frames: [Frame]
    }

    let prefix: String
    let frameLimit: Int
    private var pending: [PendingFrame] = []
    private var finished = false

    /// `DS4_AB_TRACE=/tmp/run-name` enables `<prefix>.json` + `<prefix>.f32`.
    /// The default is 9 frames (prefill plus eight decode forwards), clamped to
    /// 64 frames so a diagnostic typo cannot consume hundreds of MiB of RAM.
    static func fromEnvironment() -> ABLogitTrace? {
        let env = ProcessInfo.processInfo.environment
        guard let prefix = env["DS4_AB_TRACE"], !prefix.isEmpty else { return nil }
        let requested = env["DS4_AB_TRACE_FRAMES"].flatMap(Int.init) ?? 9
        return ABLogitTrace(prefix: prefix, frameLimit: max(1, min(64, requested)))
    }

    private init(prefix: String, frameLimit: Int) {
        self.prefix = prefix
        self.frameLimit = frameLimit
        pending.reserveCapacity(frameLimit)
    }

    func capture(phase: String, step: Int, inputToken: Int?, logits: [Float]) {
        guard !finished, pending.count < frameLimit else { return }
        pending.append(PendingFrame(phase: phase, step: step,
                                    inputToken: inputToken, logits: logits))
    }

    /// Serialize the retained vectors after all timed measurements have ended.
    /// The raw file is a concatenation of little-endian Float32 vectors; JSON
    /// offsets are expressed in floats, not bytes.
    func finish(generatedTokens: [Int]) throws {
        guard !finished else { return }
        finished = true

        let prefixURL = URL(fileURLWithPath: prefix)
        let directory = prefixURL.deletingLastPathComponent()
        try FileManager.default.createDirectory(at: directory,
                                                withIntermediateDirectories: true)
        let rawURL = URL(fileURLWithPath: prefix + ".f32")
        let jsonURL = URL(fileURLWithPath: prefix + ".json")
        guard FileManager.default.createFile(atPath: rawURL.path, contents: nil)
                || FileManager.default.fileExists(atPath: rawURL.path) else {
            throw CocoaError(.fileWriteUnknown)
        }
        let raw = try FileHandle(forWritingTo: rawURL)
        defer { try? raw.close() }
        // An explicit output directory may be reused across A/B runs. Opening a
        // FileHandle does not guarantee that stale trailing bytes are removed.
        try raw.truncate(atOffset: 0)

        var offset = 0
        var frames: [Frame] = []
        frames.reserveCapacity(pending.count)
        for item in pending {
            let summary = Self.summarize(item.logits)
            let bytes = item.logits.withUnsafeBytes { ptr in
                Data(bytes: ptr.baseAddress!, count: ptr.count)
            }
            try raw.write(contentsOf: bytes)
            frames.append(Frame(phase: item.phase, step: item.step,
                                inputToken: item.inputToken,
                                offsetFloats: offset, count: item.logits.count,
                                finiteCount: summary.finiteCount,
                                bitHashFNV1a64: summary.hash,
                                top3: summary.top3))
            offset += item.logits.count
        }
        try raw.synchronize()

        let document = Document(format: "ds4-ab-logits-v1",
                                byteOrder: "little", floatFormat: "ieee754-f32",
                                traceLimit: frameLimit, capturedFrames: frames.count,
                                generatedTokens: generatedTokens, frames: frames)
        let encoder = JSONEncoder()
        encoder.outputFormatting = [.prettyPrinted, .sortedKeys]
        let metadata = try encoder.encode(document)
        try metadata.write(to: jsonURL, options: .atomic)

        // Release the potentially multi-megabyte COW arrays immediately rather
        // than waiting for the end of the process diagnostics.
        pending.removeAll(keepingCapacity: false)
    }

    private static func summarize(_ logits: [Float])
        -> (finiteCount: Int, hash: String, top3: [Candidate]) {
        var finiteCount = 0
        var hash: UInt64 = 0xcbf29ce484222325
        var top: [(token: Int, logit: Float)] = []
        top.reserveCapacity(3)

        for (token, value) in logits.enumerated() {
            var bits = value.bitPattern.littleEndian
            withUnsafeBytes(of: &bits) { bytes in
                for byte in bytes {
                    hash ^= UInt64(byte)
                    hash = hash &* 0x100000001b3
                }
            }
            guard value.isFinite else { continue }
            finiteCount += 1
            let insertion = top.firstIndex {
                value > $0.logit || (value == $0.logit && token < $0.token)
            } ?? top.count
            if insertion < 3 {
                top.insert((token, value), at: insertion)
                if top.count > 3 { top.removeLast() }
            }
        }
        return (finiteCount, String(format: "%016llx", hash),
                top.map { Candidate(token: $0.token, logit: $0.logit) })
    }
}
