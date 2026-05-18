import Foundation

/// Configurazione di un job di vocab pruning.
///
/// Stile mirror di `QuantizeSpec` in
/// `Sources/DeepSeekConverter/ConversionSpec.swift`. Tutti i campi
/// sono `var` per consentire override programmatici via UI o CLI.
public struct VocabPruneSpec: Sendable {

    /// Directory del checkpoint convertito (output di `converter`).
    /// Deve contenere: `tokenizer.json`, `config.json`,
    /// `model.safetensors.index.json` + N shards `model-*.safetensors`.
    public var inputDir: URL

    /// Directory di destinazione. DEVE essere diversa da `inputDir`
    /// (controllo idempotente in `VocabRewriter`).
    public var outputDir: URL

    /// Sorgente del corpus italiano (o multilingua latino) usato per
    /// stimare la frequenza dei token. Può essere:
    /// - un singolo file `.txt`,
    /// - un singolo file `.jsonl` (campo `text` per riga, JSON-encoded),
    /// - una directory che `VocabAnalyzer` cammina ricorsivamente per
    ///   tutti i `.txt` e `.jsonl`.
    public var corpus: URL?

    /// Soglia di copertura cumulativa per scegliere K. Esempi:
    /// - `0.9995` (default): mantiene i top-K token che insieme
    ///   coprono il 99.95% delle occorrenze nel corpus.
    /// - `0.9999`: copertura quasi totale, vocab risultante più grande.
    public var coverage: Double = 0.9995

    /// Path opzionale a un file `keep_ids.json` pre-computato (formato:
    /// `{"keep_ids": [int], "oldToNew": {"oldId": newId}}`). Se
    /// presente, `VocabAnalyzer` viene saltato e il `corpus` ignorato.
    public var keepIdsFile: URL?

    /// Quando true, esegue solo l'analisi (Fase 1) e stampa la
    /// statistica di copertura, senza scrivere nulla in `outputDir`.
    public var dryRun: Bool = false

    /// Numero di thread paralleli per la Fase 1 (analyzer). 1 =
    /// sequenziale (default). Su corpus grossi (molti file)
    /// scalare a `cpuCount * 2` riduce sensibilmente il tempo.
    /// `BPETokenizer` è thread-safe (tutte le stored properties
    /// `let`, NSRegularExpression thread-safe), quindi il parallel
    /// dispatch è sicuro.
    public var concurrency: Int = 1

    /// Quando true (default), legge il checkpoint da
    /// `<outputDir>/.vocab_pruner_checkpoint.json` se esiste e
    /// coincide col current spec, e riprende da dove si era
    /// interrotto il job precedente. Quando false, cancella il
    /// checkpoint e ricomincia da zero.
    public var resume: Bool = true

    public init(inputDir: URL,
                outputDir: URL,
                corpus: URL? = nil,
                coverage: Double = 0.9995,
                keepIdsFile: URL? = nil,
                dryRun: Bool = false,
                concurrency: Int = 1,
                resume: Bool = true) {
        self.inputDir = inputDir
        self.outputDir = outputDir
        self.corpus = corpus
        self.coverage = coverage
        self.keepIdsFile = keepIdsFile
        self.dryRun = dryRun
        self.concurrency = concurrency
        self.resume = resume
    }
}
