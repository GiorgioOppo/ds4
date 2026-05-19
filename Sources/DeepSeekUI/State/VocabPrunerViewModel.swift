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
    /// True dal momento in cui l'utente preme Stop finché il task
    /// background non rilascia. Con la nuova strategia di bail-out
    /// cooperativo (Task.isCancelled ogni riga + yield ogni 50k
    /// righe) la transizione tipica è <100ms, ma resta cooperativa:
    /// se l'encode BPE è in volo su una riga gigante, finisce
    /// quella riga prima di rilasciare. Quando true:
    ///   - il bottone Stop diventa disabled + label "Stopping…"
    ///   - la ProgressView mostra il testo di stop in corso
    ///   - i nuovi `VocabPruneEvent` non aggiornano più
    ///     `progressFraction` così la barra resta visivamente
    ///     "ferma" al punto di stop invece di continuare ad
    ///     avanzare
    @Published var isStopping: Bool = false
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
        isStopping = false
        lastError = nil
        status = VocabPruneStatus()
        lastDecision = nil
        let token = CancellationToken()
        self.cancellation = token

        // Capture references for the closure (sendable hop).
        let specCopy = spec
        self.task = Task { [weak self] in
            // Sia il path success che il path catch finiscono in
            // `finishRun`. Garantisce che `isRunning`/`isStopping`
            // siano sempre resettati anche se in futuro aggiungiamo
            // branch di errore non strutturati. Mantiene la
            // distinzione user-stop vs failure tramite lo snapshot
            // di `isStopping` letto sul MainActor.
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
                    self?.finishRun(error: nil)
                }
            } catch {
                await MainActor.run { [weak self] in
                    self?.finishRun(error: error)
                }
            }
        }
    }

    /// Chiusura della run: chiamata sul MainActor sia dal success
    /// che dal catch del Task in `start()`. Resetta lo stato di
    /// runtime e, se l'errore corrisponde a uno stop user-initiated
    /// (rilevato via `isStopping == true` al momento della chiusura),
    /// promuove lo stato a `stopped` invece che a `error`.
    private func finishRun(error: Error?) {
        let wasStopping = isStopping
        isRunning = false
        isStopping = false
        guard let error = error else {
            // Path success: niente da fare oltre il reset dei flag.
            return
        }
        // Se il task è stato terminato da uno Stop esplicito
        // dell'utente, non mostriamo l'errore come fallimento — è
        // UX previsto. Per gli altri errori (file I/O, parse, etc.)
        // mostriamo il messaggio.
        if wasStopping {
            lastError = nil
            // Reset UI dello stato run per permettere un nuovo
            // Start pulito. Lo `status.stopped` resta true così
            // la sheet può mostrare un banner "Scan stopped —
            // progress saved".
            status = VocabPruneStatus(stopped: true)
            lastDecision = nil
            // Refresh delle info checkpoint così l'utente vede
            // quanti file/shard sono stati salvati dallo stop
            // (info pre-flight per il next Start).
            refreshCheckpointInfo()
        } else {
            let cancelMsg = (error as NSError).userInfo[NSLocalizedDescriptionKey] as? String
            lastError = (error as? LocalizedError)?.errorDescription
                ?? cancelMsg
                ?? error.localizedDescription
        }
    }

    private func handle(event: VocabPruneEvent, spec: VocabPruneSpec) {
        // Quando l'utente ha premuto Stop, congeliamo la UI sul
        // punto raggiunto. Con il bail-out responsivo gli eventi
        // residui sono pochi (<100ms tipicamente), ma se l'encode
        // BPE era in volo su una riga gigante possono arrivare
        // ancora dopo lo stop: li scartiamo così la progress bar
        // non avanza più e l'utente capisce che lo stop è stato
        // accettato.
        if isStopping { return }
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

    /// Interrompe la run corrente senza perdere progressi: il
    /// checkpoint su disco resta intatto, e un successivo `start()`
    /// riparte dal punto raggiunto. Idempotente — doppi click non
    /// fanno danni.
    func stop() {
        guard isRunning, !isStopping else { return }
        isStopping = true
        cancellation?.cancel()
        // Propaga lo stop a Swift Concurrency: il task root e
        // tutti i child task del TaskGroup interno ricevono
        // `Task.isCancelled == true` istantaneamente. Combina col
        // CancellationToken (cooperativo lato API pubblica) per
        // garantire il bail-out più veloce — i loop di scan
        // controllano Task.isCancelled OGNI riga e fanno
        // `await Task.yield()` ogni 50k righe, quindi la
        // transizione a `stopped` avviene tipicamente in <100ms.
        task?.cancel()
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
