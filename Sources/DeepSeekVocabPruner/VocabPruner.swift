import Foundation
import DeepSeekKit
import DeepSeekConverter   // `CancellationToken`

/// Facade pubblica del modulo, stile `Converter` in
/// `Sources/DeepSeekConverter/Converter.swift:15-30`.
///
/// Esegue le due fasi (Analyzer → Rewriter) in sequenza, con
/// supporto per cancellation e streaming di eventi via closure.
/// L'unica differenza vs il pattern Converter è che qui NON
/// spawniamo un subprocess: il pruner è puro Swift e gira
/// in-process.
public enum VocabPruner {

    /// Esegue il job di pruning. Se `spec.keepIdsFile` è settato,
    /// Fase 1 viene saltata e la decisione viene letta dal file
    /// (utile per replay deterministico o ispezione manuale).
    public static func run(
        spec: VocabPruneSpec,
        cancellation: CancellationToken = CancellationToken(),
        onEvent: @escaping @Sendable (VocabPruneEvent) -> Void
    ) async throws {
        try await Task.detached(priority: .userInitiated) {
            try Self.runSync(spec: spec,
                              cancellation: cancellation,
                              onEvent: onEvent)
        }.value
    }

    /// Versione sincrona (usata dal CLI che gira già su un
    /// thread dedicato). Non è esposta come `async` perché lo
    /// scheduling è già fatto dal caller.
    public static func runSync(
        spec: VocabPruneSpec,
        cancellation: CancellationToken = CancellationToken(),
        onEvent: @escaping @Sendable (VocabPruneEvent) -> Void
    ) throws {
        let tokenizerURL = spec.inputDir.appendingPathComponent("tokenizer.json")
        guard FileManager.default.fileExists(atPath: tokenizerURL.path) else {
            throw NSError(domain: "VocabPruner", code: 1,
                          userInfo: [NSLocalizedDescriptionKey:
                                      "input dir missing tokenizer.json: \(spec.inputDir.path)"])
        }

        // ---- Resume: carica eventuale checkpoint ----
        let specHash = PruneCheckpoint.computeSpecHash(
            inputDir: spec.inputDir,
            corpus: spec.corpus,
            coverage: spec.coverage)

        var checkpoint: PruneCheckpoint? = nil
        if spec.resume {
            if let existing = PruneCheckpoint.load(from: spec.outputDir) {
                if existing.specHash == specHash {
                    checkpoint = existing
                    onEvent(.log("Resuming from checkpoint saved at " +
                                  "\(existing.savedAt) (phase=\(existing.phase.rawValue))"))
                } else {
                    onEvent(.log("Checkpoint trovato ma spec hash non corrisponde; " +
                                  "ricomincio da zero."))
                    PruneCheckpoint.delete(from: spec.outputDir)
                }
            }
        } else {
            PruneCheckpoint.delete(from: spec.outputDir)
        }

        let store = CheckpointStore(outputDir: spec.outputDir,
                                     specHash: specHash,
                                     existing: checkpoint)

        // ---- Fase 1: KeepDecision ----
        let decision: KeepDecision
        if let keepFile = spec.keepIdsFile {
            onEvent(.log("Loading pre-computed keep_ids from \(keepFile.path)"))
            let data = try Data(contentsOf: keepFile)
            decision = try JSONDecoder().decode(KeepDecision.self, from: data)
            onEvent(.coverage(pct: decision.coveragePct,
                              kept: decision.keepIds.count,
                              total: decision.oldVocabSize))
            onEvent(.decisionReady(decision))
            store.markPhase2(decision: decision)
        } else if let ckpt = checkpoint,
                  let cachedDecision = ckpt.decision,
                  ckpt.phase != .analyzer
        {
            decision = cachedDecision
            onEvent(.log("Skipping Phase 1 (decision già presente nel checkpoint)."))
            onEvent(.coverage(pct: cachedDecision.coveragePct,
                              kept: cachedDecision.keepIds.count,
                              total: cachedDecision.oldVocabSize))
            onEvent(.decisionReady(cachedDecision))
        } else {
            guard let corpus = spec.corpus else {
                throw NSError(domain: "VocabPruner", code: 2,
                              userInfo: [NSLocalizedDescriptionKey:
                                          "spec needs either `corpus` or `keepIdsFile`"])
            }
            onEvent(.log("Phase 1: analyzing corpus at \(corpus.path) " +
                          "with coverage \(spec.coverage) " +
                          "(concurrency=\(spec.concurrency))..."))

            // Inizia da partial state se disponibile.
            let alreadyProc = Set(checkpoint?.analyzer?.processedFiles ?? [])
            let partial = checkpoint?.analyzer?.partialCounts ?? [:]
            if !alreadyProc.isEmpty {
                onEvent(.log("Resume: \(alreadyProc.count) file " +
                              "già processati, ripartenza dal prossimo."))
            }

            // Snapshot di outputDir + cancellation per la closure.
            let dir = spec.outputDir
            decision = try VocabAnalyzer.analyze(
                tokenizerJSON: tokenizerURL,
                corpus: corpus,
                coverage: spec.coverage,
                concurrency: spec.concurrency,
                alreadyProcessed: alreadyProc,
                partialCounts: partial,
                onFileDone: { path, lines, tokens, counts in
                    store.recordAnalyzerFile(
                        path: path, lines: lines,
                        tokens: tokens, counts: counts)
                    _ = dir   // capture intentional; store già lega dir
                },
                onEvent: onEvent)
            onEvent(.decisionReady(decision))
            onEvent(.log("Phase 1 done: kept \(decision.keepIds.count) " +
                          "of \(decision.oldVocabSize) " +
                          "(new vocab_size = \(decision.newVocabSize))"))
            store.markPhase2(decision: decision)

            if cancellation.isCancelled {
                throw NSError(domain: "VocabPruner", code: 99,
                              userInfo: [NSLocalizedDescriptionKey: "cancelled"])
            }
        }

        if spec.dryRun {
            onEvent(.log("Dry-run: skipping output write."))
            onEvent(.finished(bytesIn: 0, bytesOut: 0,
                              vocabIn: decision.oldVocabSize,
                              vocabOut: decision.newVocabSize))
            // Dry-run completato: cleanup del checkpoint
            // (un eventuale resume successivo entrerebbe in Fase 2
            // pensando che ci sia già un decision pronto).
            store.cleanup()
            return
        }

        // Persisti la KeepDecision a fianco dell'output per
        // ispezione/replay.
        try FileManager.default.createDirectory(
            at: spec.outputDir, withIntermediateDirectories: true)
        let decisionURL = spec.outputDir.appendingPathComponent("keep_ids.json")
        let decisionData = try JSONEncoder().encode(decision)
        try decisionData.write(to: decisionURL)

        // ---- Fase 2: Rewriter ----
        let alreadyCompleted = Set(checkpoint?.rewriter?.completedShards ?? [])
        if !alreadyCompleted.isEmpty {
            onEvent(.log("Resume Phase 2: \(alreadyCompleted.count) " +
                          "shard già scritti, salto."))
        }
        onEvent(.log("Phase 2: rewriting checkpoint to \(spec.outputDir.path)..."))
        let (bytesIn, bytesOut) = try VocabRewriter.rewrite(
            inputDir: spec.inputDir,
            outputDir: spec.outputDir,
            decision: decision,
            alreadyCompletedShards: alreadyCompleted,
            onShardDone: { name in
                store.recordCompletedShard(name)
            },
            onEvent: onEvent)
        onEvent(.log("Phase 2 done."))

        // Job completato: rimuovi il checkpoint.
        store.cleanup()

        onEvent(.finished(bytesIn: bytesIn,
                          bytesOut: bytesOut,
                          vocabIn: decision.oldVocabSize,
                          vocabOut: decision.newVocabSize))
    }
}

// MARK: - Checkpoint store (internal)

/// Wrapper class che accumula lo stato del `PruneCheckpoint` in
/// memoria e lo persiste atomicamente ad ogni mutazione. NSLock
/// per gli aggiornamenti da `onFileDone` chiamato da più thread
/// dell'analyzer parallelo.
fileprivate final class CheckpointStore: @unchecked Sendable {
    private let outputDir: URL
    private let lock = NSLock()
    private var state: PruneCheckpoint

    init(outputDir: URL, specHash: String, existing: PruneCheckpoint?) {
        self.outputDir = outputDir
        if let existing {
            self.state = existing
        } else {
            self.state = PruneCheckpoint(
                phase: .analyzer,
                specHash: specHash,
                analyzer: PruneCheckpoint.AnalyzerState())
        }
    }

    func recordAnalyzerFile(path: String,
                              lines: Int,
                              tokens: Int,
                              counts: [Int: Int]) {
        lock.lock(); defer { lock.unlock() }
        var a = state.analyzer ?? PruneCheckpoint.AnalyzerState()
        a.processedFiles.append(path)
        a.linesScanned += lines
        a.tokensScanned += tokens
        for (k, v) in counts {
            a.partialCounts[k, default: 0] += v
        }
        state.analyzer = a
        state.savedAt = Date()
        try? state.save(to: outputDir)
    }

    func markPhase2(decision: KeepDecision) {
        lock.lock(); defer { lock.unlock() }
        state.phase = .rewriter
        state.decision = decision
        state.rewriter = PruneCheckpoint.RewriterState()
        // Libera la memoria del count map ora che la decision è
        // pronta.
        state.analyzer = nil
        state.savedAt = Date()
        try? state.save(to: outputDir)
    }

    func recordCompletedShard(_ name: String) {
        lock.lock(); defer { lock.unlock() }
        var r = state.rewriter ?? PruneCheckpoint.RewriterState()
        if !r.completedShards.contains(name) {
            r.completedShards.append(name)
        }
        state.rewriter = r
        state.savedAt = Date()
        try? state.save(to: outputDir)
    }

    func cleanup() {
        PruneCheckpoint.delete(from: outputDir)
    }
}
