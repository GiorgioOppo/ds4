import Foundation
import DeepSeekConverter        // CancellationToken
import DeepSeekVocabPruner

/// State holder per `VocabPrunerSheet`.
///
/// Mirror di `ConvertViewModel` — `@MainActor`, `@Published` per il
/// binding SwiftUI, `Task` + `CancellationToken` per il run async.
/// Locale al sheet: non vive a livello di app.
@MainActor
final class VocabPrunerViewModel: ObservableObject {

    // ---- Form state ----
    @Published var inputDir: URL?
    @Published var outputDir: URL?
    /// Path a file `.txt` / `.jsonl` o a directory walkata
    /// ricorsivamente. Richiesto per la Fase 1 (analyzer).
    @Published var corpus: URL?
    /// Soglia di copertura cumulativa (0..1). Default 0.9995.
    @Published var coverage: Double = 0.9995
    /// Se true, solo Fase 1 + statistica copertura; niente scrittura.
    @Published var dryRun: Bool = false
    /// Numero di thread paralleli per la Fase 1. Default = 80% dei
    /// core attivi (vedi `VocabPruneSpec.defaultConcurrency`). 1 =
    /// sequenziale con save intra-file ogni 10k token.
    @Published var concurrency: Int = VocabPruneSpec.defaultConcurrency
    /// Se true (default), riprende dal checkpoint se compatibile.
    @Published var resumeEnabled: Bool = true

    // ---- Runtime state ----
    @Published var status: VocabPruneStatus = VocabPruneStatus()
    @Published var isRunning: Bool = false
    /// True dal momento in cui l'utente preme Cancel finché il task
    /// background non rilascia (può durare 1-30s perché il cancel
    /// è cooperativo: i thread/chunk già in volo finiscono prima
    /// di rilasciare il controllo). Quando true:
    ///   - il bottone Cancel diventa disabled + label "Cancelling…"
    ///   - la ProgressView mostra il testo di cancel in corso
    ///   - i nuovi `VocabPruneEvent` non aggiornano più `progressFraction`
    ///     così la barra resta visivamente "ferma" al punto di
    ///     cancel invece di continuare ad avanzare
    @Published var isCancelling: Bool = false
    @Published var lastError: String? = nil

    /// Popolato dalla Fase 1 (analyze) appena prima della Fase 2.
    /// La UI lo legge per la tabella "Dropped tokens preview".
    @Published var lastDecision: KeepDecision? = nil

    /// Info su un checkpoint precedente trovato in `outputDir` con
    /// spec compatibile (= se eseguiamo "Start" ora, riprenderemo
    /// invece di ripartire da zero). Aggiornato ad ogni cambio dei
    /// campi che influenzano l'hash dello spec.
    @Published var checkpointInfo: CheckpointInfo? = nil

    struct CheckpointInfo: Equatable {
        let phase: String
        let savedAt: Date
        let analyzerFiles: Int
        let analyzerLines: Int
        let rewriterShards: Int
    }

    /// Riferimento alla history (passato dal sheet). Nuovi record
    /// sono aggiunti su `.finished`.
    weak var history: VocabPruneHistory? = nil

    private var task: Task<Void, Never>? = nil
    private var cancellation: CancellationToken? = nil

    /// True se i campi obbligatori sono compilati e il job non è in
    /// esecuzione. La sheet binda l'attributo `disabled` del bottone
    /// "Start" a `!canStart`.
    var canStart: Bool {
        guard let src = inputDir, let dst = outputDir else { return false }
        guard FileManager.default.fileExists(atPath: src.path) else { return false }
        guard src.standardizedFileURL != dst.standardizedFileURL else { return false }
        guard corpus != nil else { return false }
        return !isRunning
    }

    /// Frazione di avanzamento per la `ProgressView`. Restituisce 0
    /// finché `status.shardsTotal == 0` (UI userà un indicator
    /// indeterminato in quel periodo).
    var progressFraction: Double {
        guard status.shardsTotal > 0 else { return 0 }
        return Double(status.shardsWritten) / Double(status.shardsTotal)
    }

    func start() {
        guard canStart,
              let src = inputDir, let dst = outputDir, let corp = corpus
        else { return }

        let spec = VocabPruneSpec(
            inputDir: src,
            outputDir: dst,
            corpus: corp,
            coverage: coverage,
            keepIdsFile: nil,
            dryRun: dryRun,
            concurrency: concurrency,
            resume: resumeEnabled)

        isRunning = true
        isCancelling = false
        lastError = nil
        status = VocabPruneStatus()
        lastDecision = nil
        let token = CancellationToken()
        self.cancellation = token

        // Capture references for the closure (sendable hop).
        let specCopy = spec
        self.task = Task { [weak self] in
            do {
                try await VocabPruner.run(
                    spec: specCopy,
                    cancellation: token,
                    onEvent: { [weak self] event in
                        Task { @MainActor in
                            self?.handle(event: event, spec: specCopy)
                        }
                    })
                await MainActor.run { [weak self] in
                    self?.isRunning = false
                    self?.isCancelling = false
                }
            } catch {
                await MainActor.run { [weak self] in
                    let wasCancelling = self?.isCancelling ?? false
                    self?.isRunning = false
                    self?.isCancelling = false
                    let cancelMsg = (error as NSError).userInfo[NSLocalizedDescriptionKey] as? String
                    // Se il task è stato terminato da un cancel
                    // esplicito dell'utente, non mostriamo l'errore
                    // come fallimento — è UX previsto. Per gli altri
                    // errori (file I/O, parse, etc.) mostriamo il
                    // messaggio.
                    if wasCancelling {
                        self?.lastError = nil
                        // Reset UI dello stato run per permettere
                        // un nuovo Start pulito. Lo `status.cancelled`
                        // resta true così la sheet può mostrare un
                        // banner "Run cancelled — checkpoint salvato".
                        self?.status = VocabPruneStatus(cancelled: true)
                        self?.lastDecision = nil
                        // Refresh delle info checkpoint così l'utente
                        // vede quanti file/shard sono stati salvati
                        // dal cancel (info pre-flight per il next Start).
                        self?.refreshCheckpointInfo()
                    } else {
                        self?.lastError = (error as? LocalizedError)?.errorDescription
                            ?? cancelMsg
                            ?? error.localizedDescription
                    }
                }
            }
        }
    }

    private func handle(event: VocabPruneEvent, spec: VocabPruneSpec) {
        // Quando l'utente ha cancellato, congeliamo la UI sul punto
        // raggiunto. Gli eventi continuano ad arrivare per qualche
        // secondo (i thread in volo non si interrompono mid-chunk),
        // ma li scartiamo così la progress bar non avanza più e
        // l'utente capisce che il cancel è stato accettato.
        if isCancelling { return }
        status.apply(event)
        switch event {
        case .decisionReady(let decision):
            self.lastDecision = decision
        case .finished(let bytesIn, let bytesOut, let vIn, let vOut):
            history?.add(VocabPruneRecord(
                inputDir: spec.inputDir.path,
                outputDir: spec.outputDir.path,
                corpus: spec.corpus?.path,
                coverage: spec.coverage,
                oldVocabSize: vIn,
                newVocabSize: vOut,
                bytesIn: bytesIn,
                bytesOut: bytesOut,
                dryRun: spec.dryRun))
        default:
            break
        }
    }

    func cancel() {
        // Idempotente: doppi click sul bottone non fanno danni.
        guard isRunning, !isCancelling else { return }
        isCancelling = true
        cancellation?.cancel()
    }

    // MARK: - Checkpoint info

    /// Aggiorna `checkpointInfo` leggendo dal disco. Chiamato dal
    /// sheet su appear e quando i campi che invalidano l'hash dello
    /// spec cambiano. Costo: una lettura JSON piccola, idempotente.
    func refreshCheckpointInfo() {
        guard let dst = outputDir,
              let src = inputDir else {
            checkpointInfo = nil
            return
        }
        let specHash = PruneCheckpoint.computeSpecHash(
            inputDir: src,
            corpus: corpus,
            coverage: coverage)
        guard let existing = PruneCheckpoint.load(from: dst),
              existing.specHash == specHash
        else {
            checkpointInfo = nil
            return
        }
        checkpointInfo = CheckpointInfo(
            phase: existing.phase.rawValue,
            savedAt: existing.savedAt,
            analyzerFiles: existing.analyzer?.processedFiles.count ?? 0,
            analyzerLines: existing.analyzer?.linesScanned ?? 0,
            rewriterShards: existing.rewriter?.completedShards.count ?? 0)
    }

    /// Cancella il checkpoint dal disco. Usato dal bottone
    /// "Reset checkpoint" nella UI.
    func resetCheckpoint() {
        guard let dst = outputDir else { return }
        PruneCheckpoint.delete(from: dst)
        checkpointInfo = nil
    }
}
