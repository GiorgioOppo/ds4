import Foundation

/// Configurazione di un job di expert pruning. Mirror di
/// `VocabPruneSpec` ma per la Fase B (MoE expert reduction).
///
/// Workflow tipico:
/// 1. Analyzer: carica il modello, runna il forward su `calibCorpus`
///    e accumula i count di routing per `(layerId, expertId)`.
/// 2. Decision: applica la coverage threshold per scegliere quali
///    esperti tenere per layer.
/// 3. Rewriter: scrive il nuovo checkpoint con i tensor degli
///    esperti droppati assenti, le righe corrispondenti del
///    `gate.weight` impostate a large-negative, le entry del
///    `tid2eid` rimappate, e `pruned_experts` aggiunto a
///    `config.json`.
///
/// Replay deterministico: salva `expert_usage.json` dopo Fase 1
/// per ripartire dalla Fase 2 senza ri-caricare il modello.
public struct ExpertPruneSpec: Sendable {

    /// Directory del checkpoint sorgente. Può essere il checkpoint
    /// originale convertito O l'output di un precedente vocab-prune
    /// — il pruner non si cura del fatto che il vocab sia stato
    /// ridotto. DEVE contenere `config.json`,
    /// `model.safetensors.index.json` + shards, e `tokenizer.json`
    /// (per l'analyzer).
    public var inputDir: URL

    /// Directory di destinazione. DEVE essere diversa da `inputDir`.
    /// In pipeline (vocab + expert), questo è settato dal facade
    /// a una directory intermedia o all'output finale a seconda di
    /// quale fase produce il deliverable.
    public var outputDir: URL

    /// Corpus di calibrazione. Stesso formato di `VocabPruneSpec.corpus`
    /// (file .txt / .jsonl o directory ricorsiva). Saltato se è
    /// settato `expertStatsFile`.
    ///
    /// Calibrazione è O(model_forward) per token — molto più costosa
    /// della tokenizzazione. Usa un corpus PIÙ PICCOLO di quello
    /// usato per la vocab phase (consiglio: 1-10 MB di testo
    /// rappresentativo).
    public var calibCorpus: URL?

    /// Coverage threshold per layer: `0.99` significa "tieni i
    /// top-K esperti che insieme coprono il 99% delle routing
    /// assignments osservate in quel layer". Floor: ogni layer
    /// mantiene almeno `max(nActivatedExperts, minKeptFloor)`
    /// esperti.
    public var coverage: Double = 0.99

    /// Floor minimo di esperti vivi per layer (oltre al vincolo
    /// `>= nActivatedExperts`). Default `4` lascia margine al
    /// gate per input fuori-distribuzione. Settato a
    /// `nActivatedExperts` per pruning più aggressivo.
    public var minKeptFloor: Int = 4

    /// Path opzionale a un `expert_usage.json` pre-computato (output
    /// dell'analyzer di una run precedente, o prodotto da uno
    /// script esterno). Se settato, l'analyzer viene saltato e
    /// `calibCorpus` viene ignorato.
    public var expertStatsFile: URL?

    /// Quando true, esegue solo Fase 1 (analyzer) e salva
    /// `expert_usage.json`, senza scrivere il rewriter output.
    public var dryRun: Bool = false

    /// Numero massimo di token processati per singola chiamata di
    /// `runner.observe(_:)`. Default 1024 (come `V4CalibrationRunner`).
    /// Riducilo per limitare il working set della KV cache; alzalo
    /// per ridurre l'overhead di KV-cache release fra chunk.
    public var maxTokensPerBatch: Int = 1024

    /// Cap opzionale sul numero totale di token processati durante
    /// la calibrazione. `0` (default) = nessun cap, processa tutto
    /// il corpus. Utile per smoke-test su corpus enormi.
    public var maxCalibrationTokens: Int = 0

    /// Quando true (default), legge `expert_pruner.json` dalla
    /// `outputDir` se esiste e ricomincia da dove ha lasciato.
    /// Quando false, cancella eventuale checkpoint e ricomincia
    /// da zero.
    public var resume: Bool = true

    public init(inputDir: URL,
                outputDir: URL,
                calibCorpus: URL? = nil,
                coverage: Double = 0.99,
                minKeptFloor: Int = 4,
                expertStatsFile: URL? = nil,
                dryRun: Bool = false,
                maxTokensPerBatch: Int = 1024,
                maxCalibrationTokens: Int = 0,
                resume: Bool = true)
    {
        self.inputDir = inputDir
        self.outputDir = outputDir
        self.calibCorpus = calibCorpus
        self.coverage = coverage
        self.minKeptFloor = minKeptFloor
        self.expertStatsFile = expertStatsFile
        self.dryRun = dryRun
        self.maxTokensPerBatch = maxTokensPerBatch
        self.maxCalibrationTokens = maxCalibrationTokens
        self.resume = resume
    }
}
