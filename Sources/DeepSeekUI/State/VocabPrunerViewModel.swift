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
    /// Numero di thread paralleli per la Fase 1. Default 1.
    /// Range raccomandato in UI: 1..cpuCount.
    @Published var concurrency: Int = 1
    /// Se true (default), riprende dal checkpoint se compatibile.
    @Published var resumeEnabled: Bool = true

    // ---- Runtime state ----
    @Published var status: VocabPruneStatus = VocabPruneStatus()
    @Published var isRunning: Bool = false
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
        lastError = nil
        status = VocabPruneStatus()
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
                }
            } catch {
                await MainActor.run { [weak self] in
                    self?.isRunning = false
                    let cancelMsg = (error as NSError).userInfo[NSLocalizedDescriptionKey] as? String
                    self?.lastError = (error as? LocalizedError)?.errorDescription
                        ?? cancelMsg
                        ?? error.localizedDescription
                }
            }
        }
    }

    private func handle(event: VocabPruneEvent, spec: VocabPruneSpec) {
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
