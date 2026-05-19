import Foundation
import DeepSeekKit
import DeepSeekConverter   // `CancellationToken`

/// Preview di un singolo token droppato dall'analyzer. Il content
/// è la stringa byte-encoded come appare nel vocab originale (es.
/// `ĠBuongiorno` per "Buongiorno" preceduto da spazio).
public struct DroppedTokenPreview: Sendable, Codable, Equatable {
    public let id: Int
    public let content: String
    public let count: Int
    public init(id: Int, content: String, count: Int) {
        self.id = id
        self.content = content
        self.count = count
    }
}

/// Decisione di pruning prodotta da `VocabAnalyzer`: chi tiene,
/// chi va via, e la mappatura old→new ID.
public struct KeepDecision: Sendable, Codable {
    /// Set dei `tokenId` (vocab originale) che sopravvivono.
    public let keepIds: [Int]               // serializzato come array per Codable

    /// Mappa `oldId → newId`. Gli ID degli `addedTokens` sono
    /// preservati (nuovo == vecchio). Gli altri sono ricompattati a
    /// partire da 0, saltando i buchi degli addedTokens.
    public let oldToNew: [Int: Int]

    /// Dimensione del nuovo vocab = max(newId) + 1.
    public let newVocabSize: Int

    /// Dimensione del vocab originale (totalVocab).
    public let oldVocabSize: Int

    /// Copertura cumulativa raggiunta (0..1).
    public let coveragePct: Double

    /// Top-N (default 50) dei token che sono stati droppati ma che
    /// AVEVANO una frequenza nel corpus. Utile come "anteprima
    /// dell'impatto" nella UI / dry-run. Non include i token che
    /// sono stati droppati dal force-exclude (script foreign) e che
    /// avevano count 0 nel corpus — quelli sono virtualmente
    /// infiniti e poco informativi.
    public let previewDropped: [DroppedTokenPreview]

    public init(keepIds: [Int],
                oldToNew: [Int: Int],
                newVocabSize: Int,
                oldVocabSize: Int,
                coveragePct: Double,
                previewDropped: [DroppedTokenPreview] = []) {
        self.keepIds = keepIds
        self.oldToNew = oldToNew
        self.newVocabSize = newVocabSize
        self.oldVocabSize = oldVocabSize
        self.coveragePct = coveragePct
        self.previewDropped = previewDropped
    }

    // Codable custom per `[Int: Int]` (Codable di default usa
    // String keys; lo serializziamo come due array paralleli).
    private enum CodingKeys: String, CodingKey {
        case keepIds, oldToNewKeys, oldToNewValues, newVocabSize,
             oldVocabSize, coveragePct, previewDropped
    }

    public init(from decoder: Decoder) throws {
        let c = try decoder.container(keyedBy: CodingKeys.self)
        self.keepIds = try c.decode([Int].self, forKey: .keepIds)
        let ks = try c.decode([Int].self, forKey: .oldToNewKeys)
        let vs = try c.decode([Int].self, forKey: .oldToNewValues)
        var map: [Int: Int] = [:]
        for (k, v) in zip(ks, vs) { map[k] = v }
        self.oldToNew = map
        self.newVocabSize = try c.decode(Int.self, forKey: .newVocabSize)
        self.oldVocabSize = try c.decode(Int.self, forKey: .oldVocabSize)
        self.coveragePct = try c.decode(Double.self, forKey: .coveragePct)
        self.previewDropped = (try? c.decode([DroppedTokenPreview].self,
                                              forKey: .previewDropped)) ?? []
    }

    public func encode(to encoder: Encoder) throws {
        var c = encoder.container(keyedBy: CodingKeys.self)
        try c.encode(keepIds, forKey: .keepIds)
        let sorted = oldToNew.sorted { $0.key < $1.key }
        try c.encode(sorted.map { $0.key }, forKey: .oldToNewKeys)
        try c.encode(sorted.map { $0.value }, forKey: .oldToNewValues)
        try c.encode(newVocabSize, forKey: .newVocabSize)
        try c.encode(oldVocabSize, forKey: .oldVocabSize)
        try c.encode(coveragePct, forKey: .coveragePct)
        try c.encode(previewDropped, forKey: .previewDropped)
    }
}

/// Fase 1: scansiona il corpus, tokenizza, conta le frequenze e
/// decide chi tiene.
public enum VocabAnalyzer {

    // MARK: - Entry point

    /// Esegue l'analisi: tokenizza il corpus, costruisce la curva
    /// cumulativa di copertura, applica i force-include e i
    /// force-exclude, restituisce un `KeepDecision`.
    ///
    /// - Parameter concurrency: numero di thread paralleli per la
    ///   tokenizzazione del corpus. Default 1 (sequenziale). Se >1,
    ///   i file vengono distribuiti via `DispatchQueue.concurrentPerform`;
    ///   `BPETokenizer.encode` viene chiamato da più thread (sicuro
    ///   perché tutte le stored properties sono `let`).
    /// - Parameter alreadyProcessed: file già contati in una run
    ///   precedente (resume). Saltati.
    /// - Parameter partialCounts: count map parziale da una run
    ///   precedente. Punto di partenza dell'aggregato. Usato insieme
    ///   ad `alreadyProcessed` per il resume.
    /// - Parameter onFileDone: callback chiamato dopo aver
    ///   processato un singolo file. Riceve il path del file più le
    ///   stats accumulate; usato per scrivere il checkpoint
    ///   incrementale dal caller.
    public static func analyze(
        tokenizerJSON: URL,
        corpus: URL,
        coverage: Double,
        concurrency: Int = 1,
        alreadyProcessed: Set<String> = [],
        partialCounts: [Int: Int] = [:],
        inFlightFile: PruneCheckpoint.InFlightFile? = nil,
        chunkedFile: PruneCheckpoint.ChunkedFileState? = nil,
        tokenBatchThreshold: Int = 10_000,
        cancellation: CancellationToken? = nil,
        onFileDone: ((String, Int, Int, [Int: Int]) -> Void)? = nil,
        onTokenBatch: ((String, Int, Int, Int, [Int: Int]) -> Void)? = nil,
        onChunkedFileStart: ((String, UInt64, [Int]) -> Void)? = nil,
        onChunkDone: ((String, Int, Int, Int, [Int: Int]) -> Void)? = nil,
        onEvent: @escaping (VocabPruneEvent) -> Void
    ) throws -> KeepDecision {
        precondition(coverage > 0 && coverage <= 1.0,
                     "coverage must be in (0, 1]")
        precondition(concurrency >= 1, "concurrency must be >= 1")
        precondition(tokenBatchThreshold >= 1,
                     "tokenBatchThreshold must be positive")

        // 1) Carica il tokenizer originale.
        let tokenizer = try Self.loadTokenizer(at: tokenizerJSON)
        let totalVocab = max(tokenizer.invVocab.keys.max() ?? 0,
                              tokenizer.invAddedTokens.keys.max() ?? 0) + 1

        // 2) Scan + count: conta le occorrenze di ogni token id.
        let files = try Self.listCorpusFiles(corpus)
            .filter { !alreadyProcessed.contains($0.url.path) }

        // Inizializza con i partial counts da una run precedente.
        var counts = partialCounts
        counts.reserveCapacity(max(counts.count, totalVocab / 4))
        var lines = 0
        var tokens = 0

        // L'in-flight resume è SUPPORTATO SOLO nel path sequenziale.
        // Il path sequenziale skippa le `lineOffset` linee già fatte
        // e parte dai `partialCountsForFile` cumulativi del file.
        //
        // Nei path parallel (single-file chunked o multi-file), il
        // file in-flight verrebbe processato DA CAPO (i thread non
        // sanno gestire un resume parziale), quindi se pre-popolassimo
        // `counts` qui avremmo un DOPPIO CONTEGGIO: una volta dai
        // partialCountsForFile, una seconda dai counts FRESH del file
        // ri-processato.
        //
        // Strategia: pre-popola SOLO nel branch sequenziale. Nei
        // branch parallel, logga un warning e ignora l'inflight
        // (il file verrà processato da capo, i counts saranno
        // corretti). Il `CheckpointStore.recordAnalyzerFile`
        // gestisce poi la pulizia dell'inflight quando il file
        // completa, mantenendo lo stato del checkpoint consistente.

        if concurrency == 1 {
            // Pre-popola per il path sequenziale (gestisce inflight).
            if let infl = inFlightFile {
                Self.mergeCounts(infl.partialCountsForFile, into: &counts)
                tokens += infl.tokensInFile
                lines += infl.lineOffset
            }
            // Path sequenziale: supporta save intra-file ogni
            // `tokenBatchThreshold` token via `onTokenBatch`.
            for file in files {
                try cancellation?.throwIfCancelled()
                let path = file.url.path
                let matchesInFlight = (inFlightFile?.path == path)
                let resumeOffset = matchesInFlight ? (inFlightFile?.lineOffset ?? 0) : 0
                let resumeCounts = matchesInFlight ? (inFlightFile?.partialCountsForFile ?? [:]) : [:]
                let resumeTokens = matchesInFlight ? (inFlightFile?.tokensInFile ?? 0) : 0

                // Stato cumulativo del file durante questa esecuzione.
                var fileCounts = resumeCounts
                var fileTokens = resumeTokens
                var fileLineOffset = resumeOffset

                let (_, _, scannedCounts) = try Self.countTokensInFile(
                    file.url,
                    isJsonl: file.isJsonl,
                    tokenizer: tokenizer,
                    startLineOffset: resumeOffset,
                    tokenBatchThreshold: tokenBatchThreshold,
                    cancellation: cancellation,
                    onTokenBatch: { batchLines, batchTokens, deltaCounts,
                                    totalLinesInFile, _ in
                        fileLineOffset = totalLinesInFile
                        fileTokens += batchTokens
                        Self.mergeCounts(deltaCounts, into: &fileCounts)
                        Self.mergeCounts(deltaCounts, into: &counts)
                        tokens += batchTokens
                        lines += batchLines
                        onTokenBatch?(path, fileLineOffset,
                                      fileTokens, batchTokens, fileCounts)
                        onEvent(.scanned(lines: lines, tokens: tokens))
                    })
                // Edge case: file senza alcun token (vuoto o tutto
                // skippato per JSONL malformato). `onTokenBatch` non
                // viene chiamato, `scannedCounts` è vuoto, gli
                // accumulator stanno a zero. fileCounts == resumeCounts.
                _ = scannedCounts  // ignorato: counts globali già aggiornati nel callback
                onFileDone?(path, fileLineOffset, fileTokens, fileCounts)
            }
        } else if files.count == 1 {
            // Single-file parallel: divide il file in `concurrency`
            // range di byte allineati al newline e li processa in
            // parallel. Necessario perché il path "per-file parallel"
            // qui sotto userebbe 1 sola iterazione (= 1 thread).
            //
            // **Save intra-file** (granularità per-chunk completato):
            // ogni chunk completato triggera un save al checkpoint
            // via `onChunkDone`. Per concurrency=8 e fileSize=5GB,
            // chunkSize=~640MB → save ogni ~30-60s di processing
            // (vs prima: save solo a fine file dopo ~5-10 min).
            //
            // **Resume**: se il checkpoint ha un `chunkedFile` per
            // questo path con gli stessi boundary, i chunk completati
            // vengono saltati e i loro counts (salvati in
            // `partialCountsForFile`) pre-popolati nei counts globali.
            // Se boundary o fileSize differiscono (file modificato o
            // concurrency cambiata), si riparte da capo.
            //
            // **Resume da inFlightFile (sequential)**: ignorato. Se
            // l'utente passa da concurrency=1 a >1, il file viene
            // processato da capo (warning loggato).
            let file = files[0]
            if let infl = inFlightFile, infl.path == file.url.path {
                onEvent(.log("Warning: intra-file checkpoint sequential " +
                              "trovato (\(infl.tokensInFile) token già " +
                              "contati su '\(infl.path)') ma chunked mode " +
                              "non lo supporta. Quel checkpoint verrà " +
                              "ignorato; il file riparte da capo."))
            }
            // Resume da chunkedFile: verifica match path + fileSize
            // + boundaries con quanto sta per essere ricalcolato.
            // Se OK, pre-popola counts globali e skippa i chunk.
            var resumeChunks: Set<Int> = []
            if let ck = chunkedFile, ck.path == file.url.path {
                // Ricalcola fileSize per verifica.
                let fileSize: UInt64
                if let attrs = try? FileManager.default.attributesOfItem(atPath: file.url.path),
                   let sz = attrs[FileAttributeKey.size] as? UInt64 {
                    fileSize = sz
                } else {
                    fileSize = 0
                }
                // Validità del checkpoint round-robin:
                //   - fileSize identico (file non modificato)
                //   - boundaries == [0, fileSize] (formato round-robin
                //     attuale; vecchi checkpoint byte-chunked hanno
                //     più boundary e sono incompatibili)
                //   - max(completedChunks) < concurrency attuale
                //     (l'utente non ha ridotto i thread; un thread idx=7
                //     in 8-thread non ha senso se ora abbiamo 4 thread)
                let isRoundRobin = ck.boundaries.count == 2
                let maxCompleted = ck.completedChunks.max() ?? -1
                if ck.fileSize == fileSize, isRoundRobin,
                   !ck.completedChunks.isEmpty,
                   maxCompleted < concurrency
                {
                    Self.mergeCounts(ck.partialCountsForFile, into: &counts)
                    tokens += ck.tokensInFile
                    lines += ck.linesInFile
                    resumeChunks = Set(ck.completedChunks)
                    onEvent(.log("Resume chunked: \(ck.completedChunks.count) " +
                                  "thread già completati su '\(ck.path)' " +
                                  "(\(ck.tokensInFile) token), processo i restanti."))
                } else if ck.fileSize == fileSize, !isRoundRobin {
                    onEvent(.log("Warning: checkpoint chunked legacy " +
                                  "(byte-based) incompatibile col round-robin " +
                                  "attuale. Riparto da zero per '\(ck.path)'."))
                } else if ck.fileSize == fileSize, maxCompleted >= concurrency {
                    onEvent(.log("Warning: concurrency ridotta " +
                                  "(\(maxCompleted + 1) → \(concurrency)). " +
                                  "Resume non possibile. Riparto da capo."))
                } else if ck.fileSize != fileSize {
                    onEvent(.log("Warning: chunked checkpoint trovato ma " +
                                  "fileSize cambiata (\(ck.fileSize) → " +
                                  "\(fileSize)). Riparto da capo."))
                }
            }
            onEvent(.log("Single-file parallel mode (round-robin): " +
                          "'\(file.url.path)' distribuito su " +
                          "\(concurrency) thread — thread k processa " +
                          "le righe i con i % \(concurrency) == k."))
            let result = try Self.processFileParallel(
                file.url,
                isJsonl: file.isJsonl,
                tokenizer: tokenizer,
                concurrency: concurrency,
                resumeFromCompletedChunks: resumeChunks,
                cancellation: cancellation,
                onBoundariesReady: { fileSize, boundaries in
                    // Setup PRIMA del dispatch: il caller registra
                    // lo state nel checkpoint così
                    // recordChunkedChunkDone (chiamato dai thread
                    // sotto) può trovare il `chunkedFile` valido.
                    onChunkedFileStart?(file.url.path, fileSize, boundaries)
                },
                onChunkDone: { idx, chunkLines, chunkTokens, chunkCounts in
                    // Callback dal thread del chunk. Notifica al
                    // caller (VocabPruner) per save al checkpoint.
                    // Il caller deve gestire concorrenza (lock interno).
                    onChunkDone?(file.url.path, idx,
                                  chunkLines, chunkTokens, chunkCounts)
                },
                onProgress: { liveLines, liveTokens in
                    onEvent(.scanned(lines: lines + liveLines,
                                      tokens: tokens + liveTokens))
                })
            Self.mergeCounts(result.counts, into: &counts)
            lines += result.lines
            tokens += result.tokens
            onFileDone?(file.url.path, result.lines, result.tokens, result.counts)
        } else {
            // Path parallelo per-file: DispatchQueue.concurrentPerform
            // su index file. Ogni thread costruisce il proprio map
            // locale, merge sotto lock.
            //
            // Resume da inFlightFile: il file puntato dall'inflight
            // (se ancora presente in `files`) sarà processato da capo
            // da uno dei thread, stesso meccanismo del branch
            // chunked sopra. Vedi commento.
            if let infl = inFlightFile,
               files.contains(where: { $0.url.path == infl.path }) {
                onEvent(.log("Warning: intra-file checkpoint trovato " +
                              "(\(infl.tokensInFile) token già contati " +
                              "su '\(infl.path)') ma multi-file parallel " +
                              "non supporta il resume parziale. " +
                              "Quel file verrà processato da capo."))
            }
            let lock = NSLock()
            // Snapshot mutable via class wrapper per condividerlo fra
            // thread (il `inout` non funziona attraverso closure
            // concurrentPerform).
            final class Aggregate: @unchecked Sendable {
                var counts: [Int: Int]
                var lines: Int = 0
                var tokens: Int = 0
                init(counts: [Int: Int]) { self.counts = counts }
            }
            let agg = Aggregate(counts: counts)
            var caught: Error? = nil

            DispatchQueue.concurrentPerform(iterations: files.count) { idx in
                // Cancellation check all'inizio di ogni file. I file
                // già in volo non possono essere interrotti a metà
                // (read-then-tokenize è già su un singolo thread del
                // pool), ma evitiamo di iniziarne di nuovi.
                if cancellation?.isCancelled == true { return }
                let file = files[idx]
                do {
                    let (lc, tc, localCounts) = try Self.countTokensInFile(
                        file.url, isJsonl: file.isJsonl, tokenizer: tokenizer,
                        cancellation: cancellation)
                    lock.lock()
                    Self.mergeCounts(localCounts, into: &agg.counts)
                    agg.lines += lc
                    agg.tokens += tc
                    let snapshotLines = agg.lines
                    let snapshotTokens = agg.tokens
                    lock.unlock()
                    onFileDone?(file.url.path, lc, tc, localCounts)
                    onEvent(.scanned(lines: snapshotLines, tokens: snapshotTokens))
                } catch {
                    lock.lock()
                    if caught == nil { caught = error }
                    lock.unlock()
                }
            }
            if let caught { throw caught }
            counts = agg.counts
            lines = agg.lines
            tokens = agg.tokens
        }
        try cancellation?.throwIfCancelled()
        onEvent(.scanned(lines: lines, tokens: tokens))

        // 3) Force-include: addedTokens, byte-level base 256, ASCII,
        //    Latin-Extended (utili anche se non visti nel corpus).
        var forcedKeep = Set<Int>()
        for id in tokenizer.addedTokens.values { forcedKeep.insert(id) }
        for id in tokenizer.invAddedTokens.keys { forcedKeep.insert(id) }
        for id in Self.byteLevelBaseIds(in: tokenizer) { forcedKeep.insert(id) }
        for id in Self.latinAndAsciiIds(in: tokenizer) { forcedKeep.insert(id) }

        // 4) Force-exclude: token che decodificano a script non latini
        //    (CJK, Hangul, Hiragana/Katakana, arabo, ebraico, devanagari,
        //    thai). Anche se vengono visti raramente nel corpus, è
        //    deliberato tagliarli per ridurre la matrice.
        let forceDrop = Self.scriptForeignIds(in: tokenizer)

        // 5) Top-K per copertura cumulativa.
        let sortedByFreq = counts.sorted { $0.value > $1.value }
        let totalCount = sortedByFreq.reduce(0) { $0 + $1.value }
        var cumulative = 0
        var topK = Set<Int>()
        for (id, c) in sortedByFreq {
            if forceDrop.contains(id) { continue }
            topK.insert(id)
            cumulative += c
            if Double(cumulative) >= Double(totalCount) * coverage {
                break
            }
        }

        // 6) Unione finale: forcedKeep ∪ topK, meno forceDrop.
        var finalKeep = forcedKeep.union(topK).subtracting(forceDrop)
        // Edge case: gli addedTokens NON vanno tolti anche se in
        // forceDrop — sono special tokens DeepSeek e non sono mai
        // "script foreign". `forcedKeep.union` li ha già aggiunti,
        // ma `.subtracting(forceDrop)` potrebbe rimuoverne uno per
        // errore se il content per qualche ragione matcha. Lo
        // ripristiniamo esplicitamente.
        for id in tokenizer.invAddedTokens.keys { finalKeep.insert(id) }

        let pct = totalCount > 0
            ? Double(cumulative) / Double(totalCount) : 0
        onEvent(.coverage(pct: pct,
                          kept: finalKeep.count,
                          total: totalVocab))

        // 7) Costruisci la mappa oldToNew. addedTokens preservano il
        //    loro ID (necessario perché il chat template / EncodingDSV4
        //    riferisce gli special token per stringa ma il
        //    SafeTensorsRewriter usa lo stesso ID-slot dell'embedding).
        let oldToNew = Self.buildRemap(
            keep: finalKeep,
            preserveIds: Set(tokenizer.invAddedTokens.keys),
            totalVocab: totalVocab)

        let newVocabSize = (oldToNew.values.max() ?? -1) + 1

        // Top-N dropped per UI / dry-run preview.
        let droppedIds = Set(counts.keys).subtracting(finalKeep)
        let preview: [DroppedTokenPreview] = droppedIds
            .compactMap { id -> DroppedTokenPreview? in
                guard let count = counts[id],
                      let token = tokenizer.invVocab[id] else { return nil }
                return DroppedTokenPreview(id: id, content: token, count: count)
            }
            .sorted { $0.count > $1.count }
            .prefix(50)
            .map { $0 }

        return KeepDecision(
            keepIds: finalKeep.sorted(),
            oldToNew: oldToNew,
            newVocabSize: newVocabSize,
            oldVocabSize: totalVocab,
            coveragePct: pct,
            previewDropped: preview)
    }

    /// Costruisce la mappa oldId→newId. Gli ID in `preserveIds`
    /// mantengono il loro valore originale. Tutti gli altri ID in
    /// `keep` vengono ricompattati a partire da 0, saltando gli slot
    /// occupati dai preserved (così non c'è collisione).
    static func buildRemap(keep: Set<Int>,
                            preserveIds: Set<Int>,
                            totalVocab: Int) -> [Int: Int] {
        var out: [Int: Int] = [:]
        // Step 1: gli ID preservati mantengono il valore.
        for id in preserveIds where keep.contains(id) {
            out[id] = id
        }
        // Step 2: gli ID non preservati vengono assegnati ai primi
        // slot liberi, partendo da 0 e saltando i preservedIds.
        let remaining = keep.subtracting(preserveIds).sorted()
        var nextSlot = 0
        for id in remaining {
            while preserveIds.contains(nextSlot) {
                nextSlot += 1
            }
            out[id] = nextSlot
            nextSlot += 1
        }
        return out
    }

    // MARK: - Loaders

    static func loadTokenizer(at url: URL) throws -> BPETokenizer {
        let data = try Data(contentsOf: url)
        return try BPETokenizer(jsonData: data)
    }

    /// Descrittore di un file del corpus.
    struct CorpusFile: Sendable {
        let url: URL
        let isJsonl: Bool
    }

    /// Elenca i file del corpus senza processarli. Supporta:
    /// - file singolo `.txt` / `.jsonl`,
    /// - directory walkata ricorsivamente per `.txt`/`.jsonl`.
    /// Ritorna l'elenco ordinato per stabilità del resume.
    static func listCorpusFiles(_ root: URL) throws -> [CorpusFile] {
        let fm = FileManager.default
        var isDir: ObjCBool = false
        guard fm.fileExists(atPath: root.path, isDirectory: &isDir) else {
            throw NSError(domain: "VocabAnalyzer", code: 10,
                          userInfo: [NSLocalizedDescriptionKey:
                                      "Corpus path does not exist: \(root.path)"])
        }
        var out: [CorpusFile] = []
        if isDir.boolValue {
            guard let it = fm.enumerator(at: root,
                                          includingPropertiesForKeys: nil,
                                          options: [.skipsHiddenFiles]) else {
                return []
            }
            for case let url as URL in it {
                let ext = url.pathExtension.lowercased()
                if ext == "txt" || ext == "jsonl" {
                    out.append(CorpusFile(url: url, isJsonl: ext == "jsonl"))
                }
            }
        } else {
            let ext = root.pathExtension.lowercased()
            out.append(CorpusFile(url: root, isJsonl: ext == "jsonl"))
        }
        out.sort { $0.url.path < $1.url.path }
        return out
    }

    /// Tokenizza un singolo file e ritorna le statistiche locali.
    /// Non emette eventi — il caller aggrega e sceglie quando
    /// notificare.
    ///
    /// - Parameter startLineOffset: numero di linee da skippare in
    ///   testa al file. Usato per il resume intra-file (continua dal
    ///   punto in cui il batch precedente aveva salvato).
    /// - Parameter tokenBatchThreshold: emette `onTokenBatch` ogni
    ///   ~`threshold` token cumulati nel batch corrente. Usato dal
    ///   caller per scrivere un checkpoint intermedio. Se `nil` o
    ///   il callback è `nil`, non c'è batching e si processa tutto
    ///   il file in un colpo solo.
    /// - Parameter onTokenBatch: callback `(linesInBatch,
    ///   tokensInBatch, deltaCounts, totalLinesSoFar,
    ///   totalTokensSoFar)`. `totalLinesSoFar` include
    ///   `startLineOffset`.
    static func countTokensInFile(
        _ url: URL,
        isJsonl: Bool,
        tokenizer: BPETokenizer,
        startLineOffset: Int = 0,
        tokenBatchThreshold: Int? = nil,
        cancellation: CancellationToken? = nil,
        onTokenBatch: ((Int, Int, [Int: Int], Int, Int) -> Void)? = nil
    ) throws -> (lines: Int, tokens: Int, counts: [Int: Int]) {
        // mmap esplicito: file > qualche GB non vengono caricati in
        // RAM contiguamente, le pagine sono fetched on demand dal
        // kernel. Critico per corpora grandi.
        let data = try Data(contentsOf: url, options: .alwaysMapped)
        var counts: [Int: Int] = [:]
        var lines = startLineOffset
        var tokens = 0

        // Batch accumulators (per il save intra-file).
        var batchCounts: [Int: Int] = [:]
        var batchLines = 0
        var batchTokens = 0

        // Byte-level scan: niente `String(data: 5GB)` + `split` (sync,
        // pre-loop, ignoravano il cancellation per minuti). Iteriamo
        // i bytes direttamente cercando '\n' come delimiter di riga;
        // per ogni riga decodifichiamo solo lo slice piccolo (kB).
        // Identico pattern di `processByteRange`.
        let totalBytes = data.count
        var sinceLastCancelCheck = 0
        var linesSkippedForOffset = 0
        var cancelledDuringScan = false

        // La closure è non-throwing → cancel propagato via flag
        // `cancelledDuringScan`, throw dopo l'uscita. Il batch
        // callback è non-throwing (signature impone Void return)
        // quindi può essere chiamato in-line.
        data.withUnsafeBytes { (raw: UnsafeRawBufferPointer) in
            guard let base = raw.baseAddress?
                    .assumingMemoryBound(to: UInt8.self) else { return }

            var pos = 0
            while pos < totalBytes {
                var lineEnd = pos
                while lineEnd < totalBytes && base[lineEnd] != 0x0A {
                    lineEnd &+= 1
                }

                // Skip startLineOffset linee senza tokenizzare.
                if linesSkippedForOffset < startLineOffset {
                    linesSkippedForOffset &+= 1
                    pos = lineEnd &+ 1
                    continue
                }

                if lineEnd > pos {
                    let lineSlice = data[pos..<lineEnd]
                    let text: String?
                    if isJsonl {
                        if let obj = try? JSONSerialization.jsonObject(with: lineSlice)
                            as? [String: Any],
                           let t = obj["text"] as? String {
                            text = t
                        } else {
                            text = nil
                        }
                    } else {
                        text = String(data: lineSlice, encoding: .utf8)
                    }
                    if let text = text, !text.isEmpty {
                        let ids = tokenizer.encode(text)
                        for id in ids {
                            counts[id, default: 0] += 1
                            batchCounts[id, default: 0] += 1
                            tokens &+= 1
                            batchTokens &+= 1
                        }
                    }
                    lines &+= 1
                    batchLines &+= 1
                }
                pos = lineEnd &+ 1

                // Flush periodico ogni ~threshold token. Path
                // sequenziale only (parallel multi-file non passa
                // threshold). Chiamata in-line — onTokenBatch è
                // non-throwing.
                if let threshold = tokenBatchThreshold,
                   let cb = onTokenBatch,
                   batchTokens >= threshold
                {
                    cb(batchLines, batchTokens, batchCounts, lines, tokens)
                    batchCounts.removeAll(keepingCapacity: true)
                    batchLines = 0
                    batchTokens = 0
                }

                // Cancellation check ogni 1k linee. Niente più
                // pre-processing pesante PRIMA del loop, quindi
                // il check è raggiungibile in pochi ms dal momento
                // del cancel premuto.
                sinceLastCancelCheck &+= 1
                if sinceLastCancelCheck >= 1000 {
                    if cancellation?.isCancelled == true {
                        cancelledDuringScan = true
                        return  // esce dalla closure
                    }
                    sinceLastCancelCheck = 0
                }
            }
        }

        if cancelledDuringScan {
            try cancellation?.throwIfCancelled()
        }

        // Flush finale residuo (può essere < threshold).
        if batchTokens > 0, let cb = onTokenBatch {
            cb(batchLines, batchTokens, batchCounts, lines, tokens)
        }
        return (lines, tokens, counts)
    }

    /// Merge in-place di un count map locale dentro l'aggregato.
    static func mergeCounts(_ local: [Int: Int], into aggregate: inout [Int: Int]) {
        for (k, v) in local {
            aggregate[k, default: 0] += v
        }
    }

    /// Risultato di `processFileParallel`. Include i boundary
    /// effettivamente usati per permettere al caller di salvare il
    /// `ChunkedFileState` nel checkpoint coerente con quanto
    /// processato.
    struct ParallelFileResult {
        let lines: Int
        let tokens: Int
        let counts: [Int: Int]
        let boundaries: [Int]
        let fileSize: UInt64
    }

    /// Tokenizza un SINGOLO file in parallelo dividendolo in
    /// `concurrency` range di byte allineati al newline (`\n`,
    /// 0x0A). Ogni range gira su un thread separato via
    /// `DispatchQueue.concurrentPerform`, accumula counts locali in
    /// un dictionary privato (no lock contention sull'hot path), e
    /// poi il main thread fa il merge finale.
    ///
    /// Usa `Data(contentsOf:options: .alwaysMapped)` → il file
    /// viene mmappato dal kernel, le pagine vengono caricate on
    /// demand. RAM footprint = footprint reale del working set,
    /// non l'intero file.
    ///
    /// **Resume support (single-file chunked checkpoint)**:
    /// - `resumeFromCompletedChunks`: indici di chunk già processati
    ///   in una run precedente. Vengono saltati (e i loro `result`
    ///   restano zero — i loro counts devono già essere in
    ///   `partialCountsForFile` del checkpoint, pre-popolati nei
    ///   counts globali dal caller).
    /// - I boundary vengono ricalcolati a partire da `concurrency`
    ///   corrente; il caller verifica che corrispondano a quelli
    ///   salvati prima di passare `resumeFromCompletedChunks`.
    ///
    /// - Parameter onChunkDone: callback chiamato dopo che un
    ///   thread completa il proprio chunk. Riceve `(chunkIdx,
    ///   lines, tokens, counts)`. Usato dal caller per save al
    ///   checkpoint. Eseguito sul thread del chunk — il caller deve
    ///   gestire concorrenza (lock interno).
    /// - Parameter onProgress: callback con `(liveLines, liveTokens)`
    ///   aggregato; chiamato ogni ~200ms (debounce).
    static func processFileParallel(
        _ url: URL,
        isJsonl: Bool,
        tokenizer: BPETokenizer,
        concurrency: Int,
        resumeFromCompletedChunks: Set<Int> = [],
        cancellation: CancellationToken? = nil,
        onBoundariesReady: ((UInt64, [Int]) -> Void)? = nil,
        onChunkDone: ((Int, Int, Int, [Int: Int]) -> Void)? = nil,
        onProgress: ((Int, Int) -> Void)? = nil
    ) throws -> ParallelFileResult {
        precondition(concurrency >= 1, "concurrency must be >= 1")
        let data = try Data(contentsOf: url, options: .alwaysMapped)
        let fileSize = data.count
        if fileSize == 0 {
            return ParallelFileResult(lines: 0, tokens: 0, counts: [:],
                                       boundaries: [0], fileSize: 0)
        }

        // 1) Round-robin line distribution invece di chunk byte
        // contigui. Ogni thread `k` (0..<nChunks) scanna l'intero
        // file mmap'd ma tokenizza solo le righe `i` tali che
        // `i % nChunks == k`. Bilanciamento perfetto del lavoro
        // anche con righe di dimensione molto eterogenea
        // (es. un dataset misto JSON oggetti grandi + righe brevi).
        //
        // Costo: ogni thread esegue il byte-scan dell'intero file
        // (cercando i '\n' per contare l'indice di riga). Su
        // mmap'd Data zero-copy, lo scan è CPU-bound a ~1 GB/s
        // per thread → ~5s per file da 5GB. La tokenizzazione,
        // 10× più costosa, domina il wall-clock — quindi il
        // byte-scan ripetuto è ammortizzato.
        //
        // Vantaggio rispetto al byte-chunking: niente pre-pass
        // per contare le linee + niente boundary array.
        // `boundaries = [0, fileSize]` resta per backward compat
        // con il checkpoint (il caller setta `chunkedFile.boundaries`
        // ma la logica di resume usa solo `completedChunks`).
        let nChunks = max(1, min(concurrency, fileSize / 1024 + 1))
        let boundaries: [Int] = [0, fileSize]
        let actualChunks = nChunks

        // Notifica al caller i boundary calcolati PRIMA del dispatch
        // dei thread, così il caller può fare setup del checkpoint
        // state ed essere pronto a ricevere gli `onChunkDone`
        // callback. Critico per il save intra-file in chunked mode
        // — senza questo setup, recordChunkedChunkDone non sa dove
        // mettere i counts.
        onBoundariesReady?(UInt64(fileSize), boundaries)

        // 2) Strutture per chunk-local results. Una `AnalyzerChunkSlot`
        // per chunk; ogni thread scrive solo nel proprio slot
        // (nessun lock contention sull'hot path).
        let results: [AnalyzerChunkSlot] = (0..<actualChunks)
            .map { _ in AnalyzerChunkSlot() }
        let lock = NSLock()
        var caught: Error? = nil

        // 3) Progress reporting (debounced).
        let progressLock = NSLock()
        var lastProgressEmit = Date.distantPast

        DispatchQueue.concurrentPerform(iterations: actualChunks) { idx in
            // Cancellation cooperativa: niente nuovi chunk dopo
            // cancel(). Quelli già in volo finiscono (un chunk dura
            // 1-10s a seconda del tokenizer + bytes), poi il controllo
            // ritorna al caller che fa rethrow.
            if cancellation?.isCancelled == true { return }
            // Resume: skippa i chunk già completati in una run
            // precedente (i loro counts sono già nel checkpoint, e
            // il caller li ha pre-popolati nei counts globali).
            if resumeFromCompletedChunks.contains(idx) {
                return
            }
            // Round-robin: ogni thread scanna l'intero file ma
            // tokenizza solo le righe `lineIdx % actualChunks == idx`.
            let r = results[idx]
            do {
                try Self.processByteRange(
                    data: data,
                    start: 0,
                    end: fileSize,
                    isJsonl: isJsonl,
                    tokenizer: tokenizer,
                    chunkResult: r,
                    cancellation: cancellation,
                    lineFilter: { lineIdx in
                        lineIdx % actualChunks == idx
                    },
                    onTokenProgress: {
                        // Debounce: chiama onProgress al massimo ogni
                        // 200ms (evita storm di callback durante
                        // tokenization veloce).
                        guard let cb = onProgress else { return }
                        progressLock.lock()
                        let now = Date()
                        let shouldEmit = now.timeIntervalSince(lastProgressEmit) > 0.2
                        if shouldEmit { lastProgressEmit = now }
                        progressLock.unlock()
                        if shouldEmit {
                            lock.lock()
                            let totalLines = results.reduce(0) { $0 + $1.lines }
                            let totalTokens = results.reduce(0) { $0 + $1.tokens }
                            lock.unlock()
                            cb(totalLines, totalTokens)
                        }
                    })
                // Chunk completato: notifica al caller per save
                // intermedio del checkpoint. Eseguito sul thread del
                // chunk — il caller (CheckpointStore) ha il proprio
                // lock interno.
                onChunkDone?(idx, r.lines, r.tokens, r.counts)
            } catch {
                lock.lock()
                if caught == nil { caught = error }
                lock.unlock()
            }
        }
        if let caught { throw caught }
        try cancellation?.throwIfCancelled()

        // 4) Merge finale dei chunk results (solo i chunk processati
        // in QUESTA run; i resumed sono già nei counts del caller).
        var totalLines = 0
        var totalTokens = 0
        var totalCounts: [Int: Int] = [:]
        let estCap = results.map { $0.counts.count }.max() ?? 0
        totalCounts.reserveCapacity(estCap * 2)
        for r in results {
            totalLines += r.lines
            totalTokens += r.tokens
            for (k, v) in r.counts {
                totalCounts[k, default: 0] += v
            }
        }
        return ParallelFileResult(lines: totalLines,
                                    tokens: totalTokens,
                                    counts: totalCounts,
                                    boundaries: boundaries,
                                    fileSize: UInt64(fileSize))
    }

    /// Trova l'offset del primo `\n` a partire da `offset` (incluso).
    /// Ritorna `data.count` se non trova nulla (fine file).
    /// Usa `withUnsafeBytes` per iterare sui byte senza overhead di
    /// `Data` subscript (che fa bounds check su ogni accesso).
    static func findNewline(after offset: Int, in data: Data) -> Int {
        if offset >= data.count { return data.count }
        return data.withUnsafeBytes { (raw: UnsafeRawBufferPointer) -> Int in
            let base = raw.baseAddress!.assumingMemoryBound(to: UInt8.self)
            let limit = data.count
            var i = offset
            while i < limit {
                if base[i] == 0x0A { return i }
                i &+= 1
            }
            return limit
        }
    }


    /// Processa un range di byte `[start, end)` dentro `data`,
    /// iterando linea per linea e tokenizzando ciascuna. Riempie
    /// `chunkResult.{lines, tokens, counts}` direttamente — nessun
    /// lock necessario perché ogni chunk ha il suo `ChunkResult`
    /// privato.
    ///
    /// L'iterazione è byte-level via `withUnsafeBytes`: niente
    /// `String.split`, niente `Substring` alloc.
    fileprivate static func processByteRange(
        data: Data,
        start: Int,
        end: Int,
        isJsonl: Bool,
        tokenizer: BPETokenizer,
        chunkResult: AnalyzerChunkSlot,
        cancellation: CancellationToken? = nil,
        lineFilter: ((Int) -> Bool)? = nil,
        onTokenProgress: (() -> Void)? = nil
    ) throws {
        var localCounts: [Int: Int] = [:]
        localCounts.reserveCapacity(1024)
        var localLines = 0
        var localTokens = 0

        // Inizializza i contatori del chunk a 0 (nel caso di retry).
        chunkResult.lines = 0
        chunkResult.tokens = 0

        // La closure di `withUnsafeBytes` è non-throwing, quindi non
        // possiamo throw direttamente. Settiamo un flag e lo
        // controlliamo all'uscita.
        var cancelled = false

        // Body non throws (try? JSONSerialization e' interno), quindi
        // niente `try` esterno; il `throws` della funzione resta per
        // futura espansione e per la simmetria con `processFileParallel`.
        data.withUnsafeBytes { (raw: UnsafeRawBufferPointer) in
            let base = raw.baseAddress!.assumingMemoryBound(to: UInt8.self)

            var pos = start
            var lastFlush = 0
            // Indice globale della riga corrente (zero-based dal
            // primo `pos = start`). Usato dal `lineFilter` per il
            // round-robin: il caller passa `{ $0 % N == k }` per
            // selezionare solo le righe del proprio thread.
            var lineIdx = 0
            // Cancel check counter per le righe SKIPPATE dal filter
            // (per il round-robin, ogni thread itera tutte le righe
            // del file ma tokenizza solo le sue: anche le righe
            // skipped contano nel responsiveness del cancel).
            var sinceCheckSkippedLines = 0
            while pos < end {
                // Trova la fine della linea corrente (newline o
                // confine del range).
                var lineEnd = pos
                while lineEnd < end && base[lineEnd] != 0x0A {
                    lineEnd &+= 1
                }

                let shouldProcess = lineFilter?(lineIdx) ?? true

                if shouldProcess && lineEnd > pos {
                    // Estrai la linea come Data slice (zero-copy view).
                    let lineSlice = data[pos..<lineEnd]
                    let text: String?
                    if isJsonl {
                        if let obj = try? JSONSerialization.jsonObject(with: lineSlice) as? [String: Any],
                           let t = obj["text"] as? String {
                            text = t
                        } else {
                            text = nil
                        }
                    } else {
                        text = String(data: lineSlice, encoding: .utf8)
                    }
                    if let text = text, !text.isEmpty {
                        let ids = tokenizer.encode(text)
                        for id in ids {
                            localCounts[id, default: 0] += 1
                            localTokens &+= 1
                        }
                        localLines &+= 1
                    }
                } else if !shouldProcess {
                    // Riga skipped: contiamo nel contatore separato
                    // così il cancel può scattare anche se il thread
                    // sta in fase di "scan-and-skip" (es. round-robin
                    // dove il 7/8 delle righe non sono sue).
                    sinceCheckSkippedLines &+= 1
                    if sinceCheckSkippedLines >= 10_000 {
                        if cancellation?.isCancelled == true {
                            cancelled = true
                            return
                        }
                        sinceCheckSkippedLines = 0
                    }
                }
                pos = lineEnd &+ 1   // skip il \n
                lineIdx &+= 1

                // Flush periodico nel chunkResult + progress callback +
                // cancellation check (~ogni 50k token: bilanciamento tra
                // overhead del check e responsiveness del cancel).
                if localTokens - lastFlush > 50_000 {
                    chunkResult.lines = localLines
                    chunkResult.tokens = localTokens
                    lastFlush = localTokens
                    onTokenProgress?()
                    if cancellation?.isCancelled == true {
                        cancelled = true
                        return  // esce dal `withUnsafeBytes` closure
                    }
                }
            }
        }

        if cancelled {
            try cancellation?.throwIfCancelled()
        }

        // Final flush dei counts del chunk.
        chunkResult.lines = localLines
        chunkResult.tokens = localTokens
        for (k, v) in localCounts {
            chunkResult.counts[k, default: 0] += v
        }
        onTokenProgress?()
    }

    // MARK: - Force-include / force-exclude predicates

    /// Restituisce gli ID dei 256 byte-level base token (mapping
    /// GPT-2 byte→unicode). Necessari per il fallback UTF-8 su
    /// qualunque input.
    static func byteLevelBaseIds(in tok: BPETokenizer) -> Set<Int> {
        let bytes = Self.byteLevelBaseStrings()
        var ids = Set<Int>()
        for s in bytes {
            if let id = tok.vocab[s] {
                ids.insert(id)
            }
        }
        return ids
    }

    /// Token che decodificano (via byteLevelUnicodeToByte) in stringhe
    /// con solo caratteri ASCII / Latin-1 / Latin Extended A/B. Forced
    /// keep — utili a coprire input italiano/europeo anche se rari.
    static func latinAndAsciiIds(in tok: BPETokenizer) -> Set<Int> {
        var ids = Set<Int>()
        for (id, token) in tok.invVocab {
            if tok.invAddedTokens[id] != nil { continue }   // skip special
            guard let decoded = decodeByteLevelToken(token) else { continue }
            if isAllLatinOrAscii(decoded) {
                ids.insert(id)
            }
        }
        return ids
    }

    /// Token che decodificano in stringhe contenenti SCRIPT NON
    /// LATINI (CJK, hangul, kana, arabic, hebrew, devanagari, thai).
    /// Forced drop — taglio aggressivo per ridurre la matrice.
    static func scriptForeignIds(in tok: BPETokenizer) -> Set<Int> {
        var ids = Set<Int>()
        for (id, token) in tok.invVocab {
            if tok.invAddedTokens[id] != nil { continue }   // mai droppare special
            guard let decoded = decodeByteLevelToken(token) else { continue }
            if containsForeignScript(decoded) {
                ids.insert(id)
            }
        }
        return ids
    }

    /// Decode di un token byte-level GPT-2 → stringa UTF-8 reale.
    /// Restituisce nil se il token contiene un character non in
    /// `unicodeToByte` (succede solo per special tokens, già filtrati
    /// dal caller).
    static func decodeByteLevelToken(_ token: String) -> String? {
        let u2b = unicodeToByteMap()
        var bytes: [UInt8] = []
        for ch in token {
            let s = String(ch)
            if let b = u2b[s] {
                bytes.append(b)
            } else {
                // Carattere fuori dalla byteToUnicode map (es. emoji
                // letterale in un added_token). Skip: il caller non
                // dovrebbe arrivarci, ma defenseive.
                return nil
            }
        }
        return String(bytes: bytes, encoding: .utf8)
    }

    /// True se tutti i character sono in ASCII (U+0000..U+007F) o
    /// Latin Extended (U+0080..U+024F) o punteggiatura latina di base
    /// (U+2010..U+206F). Whitespace incluso.
    static func isAllLatinOrAscii(_ s: String) -> Bool {
        for scalar in s.unicodeScalars {
            let v = scalar.value
            if v <= 0x024F { continue }                       // ASCII + Latin
            if (0x2010...0x206F).contains(v) { continue }     // Latin punctuation
            if (0x2070...0x209F).contains(v) { continue }     // sub/superscripts
            return false
        }
        return true
    }

    /// True se la stringa contiene almeno un character in uno dei
    /// range "foreign" che vogliamo eliminare in modalità italiano-only.
    /// Conservativo: se contiene anche UN solo char foreign, il token
    /// va droppato (perché incapsula bytes di quel character).
    static func containsForeignScript(_ s: String) -> Bool {
        for scalar in s.unicodeScalars {
            let v = scalar.value
            // CJK
            if (0x3400...0x4DBF).contains(v) { return true }   // Ext A
            if (0x4E00...0x9FFF).contains(v) { return true }   // Unified Ideographs
            if (0x20000...0x2FA1F).contains(v) { return true } // Ext B-G + supplement
            // Kana
            if (0x3040...0x30FF).contains(v) { return true }   // Hiragana + Katakana
            if (0x31F0...0x31FF).contains(v) { return true }   // Katakana phonetic
            // Hangul
            if (0x1100...0x11FF).contains(v) { return true }   // Hangul Jamo
            if (0x3130...0x318F).contains(v) { return true }   // Compat Jamo
            if (0xAC00...0xD7AF).contains(v) { return true }   // Hangul Syllables
            // Arabic
            if (0x0600...0x06FF).contains(v) { return true }
            if (0x0750...0x077F).contains(v) { return true }
            if (0x08A0...0x08FF).contains(v) { return true }
            if (0xFB50...0xFDFF).contains(v) { return true }
            if (0xFE70...0xFEFF).contains(v) { return true }
            // Hebrew
            if (0x0590...0x05FF).contains(v) { return true }
            // Devanagari
            if (0x0900...0x097F).contains(v) { return true }
            // Thai
            if (0x0E00...0x0E7F).contains(v) { return true }
        }
        return false
    }

    // MARK: - GPT-2 byteToUnicode (replicato da BPETokenizer perché private)

    /// Restituisce le 256 stringhe di base che GPT-2 byte-level usa
    /// per rappresentare ogni byte 0..255. Sono i "single-character"
    /// token che servono come fallback UTF-8.
    static func byteLevelBaseStrings() -> [String] {
        let (b2u, _) = byteUnicodeMaps()
        return (UInt8(0)...UInt8(0xFF)).map { b2u[$0]! }
    }

    static func unicodeToByteMap() -> [String: UInt8] {
        return byteUnicodeMaps().1
    }

    /// Replica deterministica della `makeByteToUnicode` di
    /// `BPETokenizer.swift` (privata in quel file). Cache-able su
    /// chiamate ripetute (per ora ricalcolata — 256 entry, costo
    /// trascurabile).
    static func byteUnicodeMaps() -> ([UInt8: String], [String: UInt8]) {
        var bs: [UInt8] = []
        for b in UInt8(0x21)...UInt8(0x7E) { bs.append(b) }    // ASCII printable
        for b in UInt8(0xA1)...UInt8(0xAC) { bs.append(b) }    // Latin-1 printable
        for b in UInt8(0xAE)...UInt8(0xFF) { bs.append(b) }
        var cs = bs.map { UInt32($0) }
        var n: UInt32 = 0
        for b in 0...0xFF {
            let bb = UInt8(b)
            if !bs.contains(bb) {
                bs.append(bb)
                cs.append(256 + n)
                n += 1
            }
        }
        var b2u: [UInt8: String] = [:]
        var u2b: [String: UInt8] = [:]
        for (b, c) in zip(bs, cs) {
            let s = String(UnicodeScalar(c)!)
            b2u[b] = s
            u2b[s] = b
        }
        return (b2u, u2b)
    }
}

/// Slot per i risultati di un chunk durante il single-file parallel
/// tokenization. Ogni thread riempie il proprio slot (un per chunk);
/// nessun lock necessario perché non c'è condivisione fra slot.
/// `@unchecked Sendable` perché la mutazione è confinata al thread
/// owner — l'invariante è garantita dal pattern `concurrentPerform`.
fileprivate final class AnalyzerChunkSlot: @unchecked Sendable {
    var lines: Int = 0
    var tokens: Int = 0
    var counts: [Int: Int] = [:]
}
