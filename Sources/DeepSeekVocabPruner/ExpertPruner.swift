import Foundation
import DeepSeekKit
import DeepSeekConverter   // `CancellationToken`

/// Facade for the expert-prune job. Mirrors `VocabPruner` shape:
/// two phases (analyzer → rewriter) run in sequence, with checkpoint
/// resume and streaming events through a `(VocabPruneEvent) -> Void`
/// closure.
///
/// The two pruners (vocab + expert) share the `VocabPruneEvent` type
/// so a single CLI / UI consumer can subscribe to both phases of a
/// pipeline run via one closure.
public enum ExpertPruner {

    /// Run the expert-prune job end-to-end.
    ///
    /// - If `spec.expertStatsFile` is set, Phase 1 is skipped and the
    ///   decision is rebuilt from the file's `[ExpertUsageRow]` grid
    ///   using `spec.coverage` and `spec.minKeptFloor`.
    /// - Otherwise the analyzer loads the model and walks
    ///   `spec.calibCorpus`.
    ///
    /// `spec.dryRun = true` runs only Phase 1 and writes
    /// `expert_usage.json` for inspection.
    public static func run(
        spec: ExpertPruneSpec,
        cancellation: CancellationToken = CancellationToken(),
        onEvent: @escaping @Sendable (VocabPruneEvent) -> Void
    ) async throws {
        let fm = FileManager.default

        // ---- Resume: load existing checkpoint if compatible. ----
        let specHash = ExpertPruneCheckpoint.computeSpecHash(
            inputDir: spec.inputDir,
            calibCorpus: spec.calibCorpus,
            coverage: spec.coverage,
            minKeptFloor: spec.minKeptFloor)

        var checkpoint: ExpertPruneCheckpoint? = nil
        if spec.resume, let existing = ExpertPruneCheckpoint.load(from: spec.outputDir) {
            if existing.specHash == specHash {
                checkpoint = existing
                onEvent(.log("Expert phase: resuming from checkpoint saved at " +
                              "\(existing.savedAt) (phase=\(existing.phase.rawValue))"))
            } else {
                onEvent(.log("Expert phase: checkpoint found but spec hash mismatch; " +
                              "restarting from scratch."))
                ExpertPruneCheckpoint.delete(from: spec.outputDir)
            }
        } else if !spec.resume {
            ExpertPruneCheckpoint.delete(from: spec.outputDir)
        }

        let store = ExpertCheckpointStore(outputDir: spec.outputDir,
                                            specHash: specHash,
                                            existing: checkpoint)

        // ---- Phase 1: decide which experts to drop. ----
        let decision: ExpertKeepDecision
        if let statsURL = spec.expertStatsFile {
            onEvent(.log("Loading pre-computed expert usage from \(statsURL.path)"))
            let data = try Data(contentsOf: statsURL)
            let usage = try JSONDecoder().decode([ExpertUsageRow].self, from: data)
            let config = try ModelConfig.load(
                from: spec.inputDir.appendingPathComponent("config.json"))
            let floor = max(spec.minKeptFloor, config.nActivatedExperts)
            decision = ExpertKeepDecision.build(
                usage: usage,
                nLayers: config.nLayers,
                nRoutedExperts: config.nRoutedExperts,
                nActivatedExperts: config.nActivatedExperts,
                coverage: spec.coverage,
                minKept: floor)
            onEvent(.expertDecisionReady(decision))
            store.markPhase2(decision: decision)
        } else if let ckpt = checkpoint,
                  let cached = ckpt.decision,
                  ckpt.phase != .analyzer
        {
            decision = cached
            onEvent(.log("Skipping Phase 1 (decision already in checkpoint)."))
            onEvent(.expertDecisionReady(cached))
        } else {
            guard let calibCorpus = spec.calibCorpus else {
                throw NSError(domain: "ExpertPruner", code: 2,
                              userInfo: [NSLocalizedDescriptionKey:
                                          "spec needs either `calibCorpus` or `expertStatsFile`"])
            }
            onEvent(.log("Phase 1: calibrating expert usage on " +
                          "\(calibCorpus.path) (coverage=\(spec.coverage), " +
                          "minKeptFloor=\(spec.minKeptFloor))…"))

            let alreadyProc = Set(checkpoint?.analyzer?.processedFiles ?? [])
            let partialUsage = checkpoint?.analyzer?.partialUsage ?? []
            let tokensProc = checkpoint?.analyzer?.tokensProcessed ?? 0
            if !alreadyProc.isEmpty {
                onEvent(.log("Resume: \(alreadyProc.count) calib file(s) " +
                              "already processed (\(tokensProc) tokens)."))
            }

            decision = try ExpertAnalyzer.analyze(
                modelDir: spec.inputDir,
                corpus: calibCorpus,
                coverage: spec.coverage,
                minKeptFloor: spec.minKeptFloor,
                maxTokensPerBatch: spec.maxTokensPerBatch,
                maxCalibrationTokens: spec.maxCalibrationTokens,
                alreadyProcessed: alreadyProc,
                partialUsage: partialUsage,
                tokensProcessed: tokensProc,
                cancellation: cancellation,
                onFileDone: { path, fileTokens, snapshot in
                    store.recordAnalyzerFile(
                        path: path,
                        fileTokens: fileTokens,
                        snapshot: snapshot)
                },
                onEvent: onEvent)

            onEvent(.expertDecisionReady(decision))
            onEvent(.log("Phase 1 done. dropped=\(decision.totalDropped), " +
                          "kept=\(decision.totalKept) " +
                          "(across \(decision.nLayers) layers, " +
                          "\(decision.nRoutedExperts) experts each)."))
            store.markPhase2(decision: decision)
            try cancellation.throwIfCancelled()
        }

        // Persist the decision as `expert_usage.json` alongside the
        // output so a `--expert-stats <file>` replay or a follow-up
        // dry-run inspection can find it.
        try fm.createDirectory(at: spec.outputDir,
                                withIntermediateDirectories: true)
        let usageURL = spec.outputDir.appendingPathComponent("expert_usage.json")
        let usageData = try JSONEncoder().encode(decision.usage)
        try usageData.write(to: usageURL)
        let decisionURL = spec.outputDir.appendingPathComponent("expert_keep_ids.json")
        let decisionData = try JSONEncoder().encode(decision)
        try decisionData.write(to: decisionURL)

        if spec.dryRun {
            onEvent(.log("Dry-run: skipping rewriter."))
            onEvent(.expertFinished(bytesIn: 0, bytesOut: 0,
                                     totalDropped: decision.totalDropped,
                                     totalKept: decision.totalKept))
            store.cleanup()
            return
        }

        // ---- Phase 2: rewriter. ----
        let alreadyCompleted = Set(checkpoint?.rewriter?.completedShards ?? [])
        if !alreadyCompleted.isEmpty {
            onEvent(.log("Resume Phase 2: \(alreadyCompleted.count) " +
                          "shard(s) already written, skipping."))
        }
        onEvent(.log("Phase 2: rewriting checkpoint to \(spec.outputDir.path)…"))
        let (bytesIn, bytesOut) = try ExpertRewriter.rewrite(
            inputDir: spec.inputDir,
            outputDir: spec.outputDir,
            decision: decision,
            alreadyCompletedShards: alreadyCompleted,
            cancellation: cancellation,
            onShardDone: { name in
                store.recordCompletedShard(name)
            },
            onEvent: onEvent)
        onEvent(.log("Phase 2 done."))
        store.cleanup()

        onEvent(.expertFinished(bytesIn: bytesIn,
                                 bytesOut: bytesOut,
                                 totalDropped: decision.totalDropped,
                                 totalKept: decision.totalKept))
    }
}

// MARK: - Checkpoint store (internal)

fileprivate final class ExpertCheckpointStore: @unchecked Sendable {
    private let outputDir: URL
    private let lock = NSLock()
    private var state: ExpertPruneCheckpoint

    init(outputDir: URL, specHash: String,
         existing: ExpertPruneCheckpoint?)
    {
        self.outputDir = outputDir
        if let existing {
            self.state = existing
        } else {
            self.state = ExpertPruneCheckpoint(
                phase: .analyzer,
                specHash: specHash,
                analyzer: ExpertPruneCheckpoint.AnalyzerState())
        }
    }

    func recordAnalyzerFile(path: String,
                              fileTokens: Int,
                              snapshot: [ExpertUsageRow]) {
        lock.lock(); defer { lock.unlock() }
        var a = state.analyzer ?? ExpertPruneCheckpoint.AnalyzerState()
        if !a.processedFiles.contains(path) {
            a.processedFiles.append(path)
        }
        a.tokensProcessed += fileTokens
        a.partialUsage = snapshot
        state.analyzer = a
        state.savedAt = Date()
        try? state.save(to: outputDir)
    }

    func markPhase2(decision: ExpertKeepDecision) {
        lock.lock(); defer { lock.unlock() }
        state.phase = .rewriter
        state.decision = decision
        state.rewriter = ExpertPruneCheckpoint.RewriterState()
        // Free the partial-usage memory now that the decision is
        // committed.
        state.analyzer = nil
        state.savedAt = Date()
        try? state.save(to: outputDir)
    }

    func recordCompletedShard(_ name: String) {
        lock.lock(); defer { lock.unlock() }
        var r = state.rewriter ?? ExpertPruneCheckpoint.RewriterState()
        if !r.completedShards.contains(name) {
            r.completedShards.append(name)
        }
        state.rewriter = r
        state.savedAt = Date()
        try? state.save(to: outputDir)
    }

    func cleanup() {
        ExpertPruneCheckpoint.delete(from: outputDir)
    }
}
