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

    // ---- Runtime state ----
    @Published var status: VocabPruneStatus = VocabPruneStatus()
    @Published var isRunning: Bool = false
    @Published var lastError: String? = nil

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
            dryRun: dryRun)

        isRunning = true
        lastError = nil
        status = VocabPruneStatus()
        let token = CancellationToken()
        self.cancellation = token

        self.task = Task { [weak self] in
            do {
                try await VocabPruner.run(
                    spec: spec,
                    cancellation: token,
                    onEvent: { [weak self] event in
                        Task { @MainActor in
                            self?.status.apply(event)
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

    func cancel() {
        cancellation?.cancel()
    }
}
