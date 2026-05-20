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

        onEvent(.log("Loading config + tokenizer from \(modelDir.path)…"))
        let configURL = modelDir.appendingPathComponent("config.json")
        let config = try ModelConfig.load(from: configURL)
        let loaded = try TokenizerLoader.load(tokenizerDir: modelDir)
        let tokenizer = loaded.tokenizer

        onEvent(.log("Loading Transformer weights (this may take a while)…"))
        let model = try Transformer.load(config: config, from: modelDir)
        try cancellation?.throwIfCancelled()

        // ---- Wire the routing observer onto every block. ----
        let nLayers = model.config.nLayers
        let nExperts = model.config.nRoutedExperts
        let topK = model.config.nActivatedExperts

        // Use a class so the actor-isolation-free Sendable closure
        // can mutate shared state without struct copy semantics.
        let counter = ExpertUsageCounter(nLayers: nLayers,
                                          nExperts: nExperts,
                                          seed: partialUsage)

        for block in model.layers {
            block.ffn.routingObserver = { layerId, idxArr, _wArr in
                counter.recordBatch(layerId: layerId, indices: idxArr)
            }
        }

        // ---- Walk the corpus. ----
        let files = try collectCorpusFiles(corpus)
        if files.isEmpty {
            throw NSError(domain: "ExpertAnalyzer", code: 1,
                          userInfo: [NSLocalizedDescriptionKey:
                                      "no .txt/.jsonl files found at \(corpus.path)"])
        }
        onEvent(.log("Calibration corpus: \(files.count) file(s). topK=\(topK), " +
                       "nLayers=\(nLayers), nExperts=\(nExperts)."))

        var tokensSeen = tokensProcessed
        var linesSeen = 0
        let cap = maxCalibrationTokens

        for file in files {
            let key = file.standardizedFileURL.path
            if alreadyProcessed.contains(key) {
                onEvent(.log("Skipping (already processed): \(key)"))
                continue
            }
            try cancellation?.throwIfCancelled()

            // Read the file as a sequence of records.
            let records = try readRecords(from: file)
            var tokensInFile = 0
            for line in records {
                if cap > 0 && tokensSeen >= cap { break }
                try cancellation?.throwIfCancelled()

                let ids = tokenizer.encode(line)
                guard !ids.isEmpty else { continue }

                var offset = 0
                while offset < ids.count {
                    if cap > 0 && tokensSeen >= cap { break }
                    let chunkEnd = min(offset + maxTokensPerBatch, ids.count)
                    let chunk = Array(ids[offset..<chunkEnd])
                    _ = model.forward(inputIds: [chunk], startPos: 0)
                    // Drop KV cache between chunks so positional bias
                    // doesn't accumulate (same trick as
                    // V4CalibrationRunner).
                    for block in model.layers {
                        block.attn.releaseCache()
                    }
                    offset = chunkEnd
                    tokensSeen += chunk.count
                    tokensInFile += chunk.count
                    onEvent(.scanned(lines: linesSeen, tokens: tokensSeen))
                }
                linesSeen += 1
            }

            onFileDone(key, tokensInFile, counter.snapshot())
            if cap > 0 && tokensSeen >= cap {
                onEvent(.log("Reached max-calibration-tokens cap (\(cap))."))
                break
            }
        }

        // Detach observers — keep `model` clean if the caller holds
        // onto it (we don't, but be tidy).
        for block in model.layers {
            block.ffn.routingObserver = nil
        }

        let usage = counter.snapshot()
        onEvent(.log("Building decision: coverage=\(coverage), " +
                       "minKeptFloor=\(minKeptFloor) " +
                       "(clamped ≥ nActivatedExperts=\(topK))"))

        let floor = max(minKeptFloor, topK)
        let decision = ExpertKeepDecision.build(
            usage: usage,
            nLayers: nLayers,
            nRoutedExperts: nExperts,
            nActivatedExperts: topK,
            coverage: coverage,
            minKept: floor)
        return decision
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
