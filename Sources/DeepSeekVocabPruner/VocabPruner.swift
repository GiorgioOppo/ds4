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
            let inFlight = checkpoint?.analyzer?.inFlightFile
            if !alreadyProc.isEmpty {
                onEvent(.log("Resume: \(alreadyProc.count) file " +
                              "già processati, ripartenza dal prossimo."))
            }
            if let infl = inFlight {
                onEvent(.log("Resume intra-file: '\(infl.path)' " +
                              "dalla linea \(infl.lineOffset) " +
                              "(\(infl.tokensInFile) token già contati)."))
            }

            decision = try await VocabAnalyzer.analyze(
                tokenizerJSON: tokenizerURL,
                corpus: corpus,
                coverage: spec.coverage,
                concurrency: spec.concurrency,
                alreadyProcessed: alreadyProc,
                partialCounts: partial,
                inFlightFile: inFlight,
                chunkedFile: checkpoint?.analyzer?.chunkedFile,
                tokenBatchThreshold: 10_000,
                cancellation: cancellation,
                onFileDone: { path, lines, tokens, counts in
                    store.recordAnalyzerFile(
                        path: path,
                        totalLinesInFile: lines,
                        totalTokensInFile: tokens,
                        fullFileCounts: counts)
                },
                onTokenBatch: { path, lineOffset, tokensInFile, _, cumulCounts in
                    store.recordAnalyzerPartial(
                        path: path,
                        newLineOffset: lineOffset,
                        newTokensInFile: tokensInFile,
                        cumulativeCountsForFile: cumulCounts)
                },
                onChunkedFileStart: { path, fileSize, boundaries in
                    store.setupChunkedState(path: path,
                                             fileSize: fileSize,
                                             boundaries: boundaries)
                },
                onChunkDone: { path, chunkIdx, chunkLines, chunkTokens, chunkCounts in
                    store.recordChunkedChunkDone(
                        path: path,
                        chunkIdx: chunkIdx,
                        chunkLines: chunkLines,
                        chunkTokens: chunkTokens,
                        chunkCounts: chunkCounts)
                },
                onEvent: onEvent)
            onEvent(.decisionReady(decision))
            onEvent(.log("Phase 1 done: kept \(decision.keepIds.count) " +
                          "of \(decision.oldVocabSize) " +
                          "(new vocab_size = \(decision.newVocabSize))"))
            store.markPhase2(decision: decision)

            try cancellation.throwIfCancelled()
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
            cancellation: cancellation,
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

    /// File completato. Sposta i `counts` cumulativi del file (incl.
    /// eventuali partial counts del resume) dentro `partialCounts`
    /// aggregato, aggiunge al `processedFiles`, e — se il file era
    /// quello in-flight — pulisce `inFlightFile`. Aggiorna anche
    /// `linesScanned/tokensScanned` con il delta non ancora
    /// contabilizzato (cioè totale del file meno quanto già
    /// riportato dai partial save intra-file).
    func recordAnalyzerFile(path: String,
                              totalLinesInFile: Int,
                              totalTokensInFile: Int,
                              fullFileCounts: [Int: Int]) {
        lock.lock(); defer { lock.unlock() }
        var a = state.analyzer ?? PruneCheckpoint.AnalyzerState()

        // Determina quali counts spostare a partialCounts e i delta
        // delle stats, in base al tipo di stato in-flight per questo
        // file:
        //
        //   - chunked: `fullFileCounts` ricevuti = solo i chunk FRESH
        //     processati in questa run. Il TOTALE del file (FRESH +
        //     resumed) è già aggregato in `chunkedFile.partialCountsForFile`
        //     (aggiornato chunk-per-chunk da recordChunkedChunkDone).
        //     Usiamo direttamente quello per evitare di mancare i
        //     counts dei chunk resumed. linesScanned/tokensScanned
        //     sono già stati aggiornati dai chunk callback → delta=0.
        //
        //   - sequential (inFlight): `fullFileCounts` = TOTALE del
        //     file (resume + fresh, accumulati da onTokenBatch).
        //     linesScanned/tokensScanned aggiornati incrementalmente
        //     da recordAnalyzerPartial → delta = totale - last saved ≈ 0.
        //
        //   - né chunked né inFlight (multi-file parallel, file fresco
        //     in sequential): `fullFileCounts` = TOTALE del file, e
        //     linesScanned/tokensScanned non sono stati aggiornati
        //     incrementalmente per questo file → delta = totale.
        var lineDelta = totalLinesInFile
        var tokenDelta = totalTokensInFile
        var countsToAggregate = fullFileCounts

        if let ck = a.chunkedFile, ck.path == path {
            // Chunked: usa il totale dal chunked state.
            countsToAggregate = ck.partialCountsForFile
            lineDelta -= ck.linesInFile
            tokenDelta -= ck.tokensInFile
            a.chunkedFile = nil
        } else if let infl = a.inFlightFile, infl.path == path {
            // Sequential resume: i counts incremental sono già stati
            // sommati. Sottrai il last-saved offset.
            lineDelta -= infl.lineOffset
            tokenDelta -= infl.tokensInFile
            a.inFlightFile = nil
        }

        if !a.processedFiles.contains(path) {
            a.processedFiles.append(path)
        }
        for (k, v) in countsToAggregate {
            a.partialCounts[k, default: 0] += v
        }
        a.linesScanned += max(0, lineDelta)
        a.tokensScanned += max(0, tokenDelta)
        state.analyzer = a
        state.savedAt = Date()
        try? state.save(to: outputDir)
    }

    /// Save intra-file: aggiorna lo stato del file in-flight
    /// senza spostarlo nei `processedFiles`. `cumulativeCountsForFile`
    /// è il count del file dall'inizio (linea 0) fino a `newLineOffset`
    /// — non un delta. Same per `newTokensInFile`.
    func recordAnalyzerPartial(path: String,
                                 newLineOffset: Int,
                                 newTokensInFile: Int,
                                 cumulativeCountsForFile: [Int: Int]) {
        lock.lock(); defer { lock.unlock() }
        var a = state.analyzer ?? PruneCheckpoint.AnalyzerState()
        let prevLineOffset = a.inFlightFile?.lineOffset ?? 0
        let prevTokensInFile = a.inFlightFile?.tokensInFile ?? 0
        let lineDelta = max(0, newLineOffset - prevLineOffset)
        let tokenDelta = max(0, newTokensInFile - prevTokensInFile)
        a.inFlightFile = PruneCheckpoint.InFlightFile(
            path: path,
            lineOffset: newLineOffset,
            tokensInFile: newTokensInFile,
            partialCountsForFile: cumulativeCountsForFile)
        a.linesScanned += lineDelta
        a.tokensScanned += tokenDelta
        state.analyzer = a
        state.savedAt = Date()
        try? state.save(to: outputDir)
    }

    /// Setup iniziale del `chunkedFile` state per il save per-chunk
    /// in modalità single-file chunked. Chiamato dall'analyzer
    /// PRIMA del dispatch dei chunk; popola path + fileSize +
    /// boundaries per permettere al resume successivo di verificare
    /// che la configurazione sia compatibile.
    ///
    /// Se esiste già un `chunkedFile` con lo stesso path + fileSize
    /// + boundaries, NON viene reset (preserva i chunk completati
    /// dalle run precedenti). Se differiscono o non esiste, crea
    /// uno nuovo vuoto.
    func setupChunkedState(path: String,
                            fileSize: UInt64,
                            boundaries: [Int]) {
        lock.lock(); defer { lock.unlock() }
        var a = state.analyzer ?? PruneCheckpoint.AnalyzerState()
        if let existing = a.chunkedFile,
           existing.path == path,
           existing.fileSize == fileSize,
           existing.boundaries == boundaries
        {
            // Compatibile: preserva il progress esistente.
            return
        }
        a.chunkedFile = PruneCheckpoint.ChunkedFileState(
            path: path,
            fileSize: fileSize,
            boundaries: boundaries)
        // Pulisce eventuale inFlightFile residuo: i due stati sono
        // mutually exclusive per un dato file (sequenziale vs chunked).
        if a.inFlightFile?.path == path {
            a.inFlightFile = nil
        }
        state.analyzer = a
        state.savedAt = Date()
        try? state.save(to: outputDir)
    }

    /// Save di un singolo chunk completato durante il single-file
    /// chunked mode. Chiamato dal thread del chunk; lock-protected
    /// perché più chunk possono completarsi quasi-simultaneamente
    /// (su macchine multi-core con concurrency alta).
    ///
    /// I `chunkCounts` sono i counts del singolo chunk (non
    /// cumulativi del file). Vengono mergiati in
    /// `chunkedFile.partialCountsForFile` per costituire il
    /// cumulativo, e l'indice viene aggiunto a `completedChunks`.
    func recordChunkedChunkDone(path: String,
                                  chunkIdx: Int,
                                  chunkLines: Int,
                                  chunkTokens: Int,
                                  chunkCounts: [Int: Int]) {
        lock.lock(); defer { lock.unlock() }
        var a = state.analyzer ?? PruneCheckpoint.AnalyzerState()
        guard var ck = a.chunkedFile, ck.path == path else {
            // Setup mancante o path mismatch: skip silenziosamente
            // (l'analyzer comunque emetterà il file done a fine
            // processing che ricostruisce lo stato finale).
            return
        }
        if ck.completedChunks.contains(chunkIdx) {
            return  // già registrato — guard against duplicate calls
        }
        ck.completedChunks.append(chunkIdx)
        ck.completedChunks.sort()
        for (k, v) in chunkCounts {
            ck.partialCountsForFile[k, default: 0] += v
        }
        ck.tokensInFile += chunkTokens
        ck.linesInFile += chunkLines
        a.chunkedFile = ck
        a.linesScanned += chunkLines
        a.tokensScanned += chunkTokens
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
