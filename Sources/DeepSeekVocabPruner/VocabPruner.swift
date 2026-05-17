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

        // ---- Fase 1: KeepDecision ----
        let decision: KeepDecision
        if let keepFile = spec.keepIdsFile {
            onEvent(.log("Loading pre-computed keep_ids from \(keepFile.path)"))
            let data = try Data(contentsOf: keepFile)
            decision = try JSONDecoder().decode(KeepDecision.self, from: data)
            onEvent(.coverage(pct: decision.coveragePct,
                              kept: decision.keepIds.count,
                              total: decision.oldVocabSize))
        } else {
            guard let corpus = spec.corpus else {
                throw NSError(domain: "VocabPruner", code: 2,
                              userInfo: [NSLocalizedDescriptionKey:
                                          "spec needs either `corpus` or `keepIdsFile`"])
            }
            onEvent(.log("Phase 1: analyzing corpus at \(corpus.path) " +
                          "with coverage \(spec.coverage)..."))
            decision = try VocabAnalyzer.analyze(
                tokenizerJSON: tokenizerURL,
                corpus: corpus,
                coverage: spec.coverage,
                onEvent: onEvent)
            onEvent(.log("Phase 1 done: kept \(decision.keepIds.count) " +
                          "of \(decision.oldVocabSize) " +
                          "(new vocab_size = \(decision.newVocabSize))"))

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
        onEvent(.log("Phase 2: rewriting checkpoint to \(spec.outputDir.path)..."))
        let (bytesIn, bytesOut) = try VocabRewriter.rewrite(
            inputDir: spec.inputDir,
            outputDir: spec.outputDir,
            decision: decision,
            onEvent: onEvent)
        onEvent(.log("Phase 2 done."))

        onEvent(.finished(bytesIn: bytesIn,
                          bytesOut: bytesOut,
                          vocabIn: decision.oldVocabSize,
                          vocabOut: decision.newVocabSize))
    }
}
