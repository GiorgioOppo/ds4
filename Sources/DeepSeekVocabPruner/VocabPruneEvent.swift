import Foundation

/// Eventi emessi dal `VocabPruner` durante l'esecuzione.
///
/// Stesso pattern di `ConversionEvent` in
/// `Sources/DeepSeekConverter/ConversionProgress.swift`: un consumer
/// (CLI o UI) si abbona via la closure passata a
/// `VocabPruner.run(...)` e aggrega lo stato in un
/// `VocabPruneStatus` se serve.
public enum VocabPruneEvent: Sendable {
    /// Fase 1 — scanning del corpus: numero di linee viste e token
    /// totali contati finora.
    case scanned(lines: Int, tokens: Int)

    /// Fase 1 — risultato della copertura: percentuale raggiunta,
    /// numero di token tenuti, numero di token totali nel
    /// vocabolario originale.
    case coverage(pct: Double, kept: Int, total: Int)

    /// Fase 1 — decisione completa pronta. Pubblicato subito dopo
    /// `coverage` e prima del primo `shardWritten`. Permette alla
    /// UI di leggere `previewDropped` per la tabella "anteprima
    /// dei token tagliati" e di mostrare i count di vocab/merges
    /// senza dover aprire il `keep_ids.json` scritto su disco.
    case decisionReady(KeepDecision)

    /// Fase 2 — uno shard di safetensors scritto. `i` è 1-based.
    case shardWritten(i: Int, total: Int)

    /// Log generico per progressi non strutturati.
    case log(String)

    /// Fine job. Riporta byte input/output e dimensione vocab
    /// pre/post per stimare il risparmio.
    case finished(bytesIn: UInt64, bytesOut: UInt64,
                  vocabIn: Int, vocabOut: Int)
}

/// Snapshot aggregato dello stato del pruning, utile a una UI che
/// vuole renderizzare una progress bar / etichetta senza tenere lo
/// streaming degli eventi.
public struct VocabPruneStatus: Sendable, Equatable {
    public var linesScanned: Int = 0
    public var tokensScanned: Int = 0
    public var coveragePct: Double = 0
    public var keptVocab: Int = 0
    public var totalVocab: Int = 0
    public var shardsWritten: Int = 0
    public var shardsTotal: Int = 0
    public var logLines: [String] = []
    public var finishedAt: Date? = nil
    public var bytesIn: UInt64 = 0
    public var bytesOut: UInt64 = 0

    public init() {}

    public mutating func apply(_ event: VocabPruneEvent) {
        switch event {
        case .scanned(let lines, let tokens):
            linesScanned = lines
            tokensScanned = tokens
        case .coverage(let pct, let kept, let total):
            coveragePct = pct
            keptVocab = kept
            totalVocab = total
        case .decisionReady:
            // La status non tiene il decision (e' pesante); chi lo
            // vuole sottoscrive l'evento direttamente.
            break
        case .shardWritten(let i, let total):
            shardsWritten = i
            shardsTotal = total
        case .log(let line):
            logLines.append(line)
        case .finished(let bIn, let bOut, _, _):
            bytesIn = bIn
            bytesOut = bOut
            finishedAt = Date()
        }
    }
}
