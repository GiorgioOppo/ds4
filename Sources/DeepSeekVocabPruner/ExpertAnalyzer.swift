import Foundation
import DeepSeekKit
import DeepSeekConverter   // `CancellationToken`

/// Phase 1 of the expert pruner: load the model, walk a calibration
/// corpus through `Transformer.forward`, and aggregate per-(layer,
/// expert) routing counts via the `MoEFFN.routingObserver` hook.
///
/// Heavyweight: needs Metal, the full weight load (mmap is OK), and
/// a KV cache allocation. For an Italian-only deployment the
/// calibration corpus can be small (1-10 MB of representative text)
/// — the dispatch plan converges to a stable usage histogram fast,
/// so a giant corpus is unnecessary.
///
/// Hash-routed layers (the first `nHashLayers`) are still observed
/// — the gate kernel produces score-based logits used for the
/// weight normalisation, and the per-token index lookup goes
/// through `tid2eid`, so the routing pattern shows up in `idxArr`
/// just like the score-routed layers.
public enum ExpertAnalyzer {

    /// Run the analyzer end-to-end.
    ///
    /// - Parameter modelDir: directory of the source checkpoint
    ///   (same as `ExpertPruneSpec.inputDir`). Must contain
    ///   `config.json`, `tokenizer.json`, the safetensors index +
    ///   shards.
    /// - Parameter corpus: file or directory walked recursively for
    ///   `.txt` and `.jsonl` (the latter parsed as one JSON record
    ///   per line with a `text` field).
    /// - Parameter coverage: passed straight to
    ///   `ExpertKeepDecision.build`.
    /// - Parameter minKeptFloor: min experts kept per layer (passed
    ///   to `ExpertKeepDecision.build` after clamping ≥ topK).
    /// - Parameter maxTokensPerBatch: per-`observe(_:)` cap, matches
    ///   the `V4CalibrationRunner` default of 1024.
    /// - Parameter maxCalibrationTokens: optional cap on total
    ///   tokens processed (0 = no cap).
    /// - Parameter alreadyProcessed: file paths to skip on resume.
    /// - Parameter partialUsage: usage grid already accumulated from
    ///   a previous interrupted run.
    /// - Parameter tokensProcessed: tokens counted by previous runs.
    /// - Parameter onFileDone: called after each file finishes so the
    ///   caller can checkpoint progress.
    /// - Parameter onEvent: streaming events (log / scanned).
    public static func analyze(
        modelDir: URL,
        corpus: URL,
        coverage: Double,
        minKeptFloor: Int,
        maxTokensPerBatch: Int = 1024,
        maxCalibrationTokens: Int = 0,
        alreadyProcessed: Set<String> = [],
        partialUsage: [ExpertUsageRow] = [],
        tokensProcessed: Int = 0,
        cancellation: CancellationToken? = nil,
        onFileDone: @escaping (String, Int, [ExpertUsageRow]) -> Void = { _, _, _ in },
        onEvent: @escaping (VocabPruneEvent) -> Void
    ) throws -> ExpertKeepDecision {
        throw NSError(domain: "ExpertAnalyzer", code: 99,
                      userInfo: [NSLocalizedDescriptionKey: "Expert pruning is not supported with the new MLX backend yet."])
    }

    // MARK: - Corpus walker

    private static func collectCorpusFiles(_ url: URL) throws -> [URL] {
        let fm = FileManager.default
        var isDir: ObjCBool = false
        guard fm.fileExists(atPath: url.path, isDirectory: &isDir) else {
            throw NSError(domain: "ExpertAnalyzer", code: 10,
                          userInfo: [NSLocalizedDescriptionKey:
                                      "corpus path does not exist: \(url.path)"])
        }
        if !isDir.boolValue {
            return [url]
        }
        var out: [URL] = []
        if let it = fm.enumerator(at: url,
                                   includingPropertiesForKeys: [.isRegularFileKey]) {
            for case let f as URL in it {
                let ext = f.pathExtension.lowercased()
                if ext == "txt" || ext == "jsonl" { out.append(f) }
            }
        }
        return out.sorted { $0.path < $1.path }
    }

    /// Read a `.txt` (one record per line) or `.jsonl` (one JSON
    /// object with a `text` field per line) into an array of
    /// strings. Lines that fail to parse are skipped silently.
    private static func readRecords(from url: URL) throws -> [String] {
        let ext = url.pathExtension.lowercased()
        let data = try Data(contentsOf: url)
        guard let text = String(data: data, encoding: .utf8) else { return [] }
        let lines = text.split(omittingEmptySubsequences: true,
                                whereSeparator: { $0 == "\n" || $0 == "\r\n" })
        if ext == "jsonl" {
            var out: [String] = []
            out.reserveCapacity(lines.count)
            for ln in lines {
                guard let bytes = ln.data(using: .utf8),
                      let obj = try? JSONSerialization.jsonObject(with: bytes) as? [String: Any],
                      let t = obj["text"] as? String else { continue }
                out.append(t)
            }
            return out
        }
        return lines.map(String.init)
    }
}

/// Thread-safe-by-virtue-of-single-writer counter for routing
/// decisions. The MoEFFN observer fires from inside `Transformer.forward`
/// which is single-threaded per session — the only writer is the
/// command-buffer-driving thread — so we don't need a lock here.
/// The `snapshot()` returns a flat `[ExpertUsageRow]` array suitable
/// for `ExpertKeepDecision.build` and for serialisation.
fileprivate final class ExpertUsageCounter: @unchecked Sendable {
    let nLayers: Int
    let nExperts: Int
    private var grid: [[Int]]

    init(nLayers: Int, nExperts: Int, seed: [ExpertUsageRow]) {
        self.nLayers = nLayers
        self.nExperts = nExperts
        self.grid = Array(repeating: Array(repeating: 0,
                                             count: nExperts),
                            count: nLayers)
        for r in seed {
            guard r.layerId >= 0 && r.layerId < nLayers,
                  r.expertId >= 0 && r.expertId < nExperts else { continue }
            grid[r.layerId][r.expertId] = r.count
        }
    }

    func recordBatch(layerId: Int, indices: [Int32]) {
        guard layerId >= 0 && layerId < nLayers else { return }
        for v in indices {
            let e = Int(v)
            if e >= 0 && e < nExperts {
                grid[layerId][e] += 1
            }
        }
    }

    func snapshot() -> [ExpertUsageRow] {
        var rows: [ExpertUsageRow] = []
        rows.reserveCapacity(nLayers * nExperts)
        for L in 0..<nLayers {
            for E in 0..<nExperts {
                rows.append(ExpertUsageRow(layerId: L,
                                             expertId: E,
                                             count: grid[L][E]))
            }
        }
        return rows
    }
}
