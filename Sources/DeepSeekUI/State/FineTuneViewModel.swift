import Foundation
import SwiftUI
import DeepSeekTraining

/// Drives `FineTuneSheet`. Mirrors `ConvertViewModel` in shape:
/// form state + a current `FineTuneStatus` rolled up from the
/// runner's event stream + the running cancellation token / Task
/// handle.
@MainActor
final class FineTuneViewModel: ObservableObject {
    // ---- Form: paths ----
    @Published var baseModelPath: URL?
    @Published var datasetPath: URL?
    @Published var evalDatasetPath: URL?
    @Published var outputPath: URL?

    // ---- Form: format + precision ----
    @Published var format: DatasetFormat = .jsonlChat
    @Published var precision: TrainingPrecision = .bf16

    // ---- Form: hyperparameters ----
    @Published var learningRate: Double = 5e-5
    @Published var epochs: Int = 3
    @Published var batchSize: Int = 1
    @Published var gradientAccumulationSteps: Int = 8
    @Published var maxSequenceLength: Int = 2048
    @Published var warmupSteps: Int = 100
    @Published var weightDecay: Double = 0.01
    @Published var optimizer: FineTuneOptimizer = .adamW
    @Published var evalSplit: Double = 0.05
    @Published var saveEverySteps: Int = 500
    @Published var seed: UInt64 = 42

    // ---- Runtime ----
    @Published var status: FineTuneStatus = FineTuneStatus()
    @Published var isRunning: Bool = false
    @Published var lastError: String? = nil
    /// Set by the runner when it throws `FineTuneNotImplemented`. The
    /// sheet renders this with an info icon instead of the red error
    /// chrome — it's a "scaffold not wired yet" message, not a
    /// failure.
    @Published var notImplementedNotice: String? = nil

    private var task: Task<Void, Never>? = nil
    private var cancellation: TrainingCancellationToken? = nil

    /// True iff the form has enough information to start. Each path
    /// must exist where required, paths must be distinct, and
    /// hyperparameters must be sane.
    var canStart: Bool {
        guard !isRunning else { return false }
        guard let base = baseModelPath, let ds = datasetPath, let out = outputPath else {
            return false
        }
        let fm = FileManager.default
        guard fm.fileExists(atPath: base.path) else { return false }
        guard fm.fileExists(atPath: ds.path) else { return false }
        if let eval = evalDatasetPath,
           !fm.fileExists(atPath: eval.path) { return false }
        guard base.standardizedFileURL != out.standardizedFileURL else { return false }
        guard learningRate > 0, epochs > 0,
              batchSize > 0, gradientAccumulationSteps > 0,
              maxSequenceLength > 0 else { return false }
        return true
    }

    func start() {
        guard canStart,
              let base = baseModelPath,
              let ds = datasetPath,
              let out = outputPath else { return }

        let spec = FineTuneSpec(
            baseModelPath: base,
            datasetPath: ds,
            evalDatasetPath: evalDatasetPath,
            outputPath: out,
            format: format,
            precision: precision,
            learningRate: learningRate,
            epochs: epochs,
            batchSize: batchSize,
            gradientAccumulationSteps: gradientAccumulationSteps,
            maxSequenceLength: maxSequenceLength,
            warmupSteps: warmupSteps,
            weightDecay: weightDecay,
            optimizer: optimizer,
            evalSplit: evalSplit,
            saveEverySteps: saveEverySteps,
            seed: seed)

        isRunning = true
        lastError = nil
        notImplementedNotice = nil
        status = FineTuneStatus()
        let token = TrainingCancellationToken()
        self.cancellation = token

        self.task = Task { [weak self] in
            do {
                try await FineTuner.run(
                    spec: spec,
                    cancellation: token,
                    onEvent: { [weak self] event in
                        // Hop to MainActor to mutate @Published state.
                        // Capture weakly at the outer Sendable
                        // boundary so the inner Task uses the strong
                        // ref and avoids the Swift 6 concurrent
                        // capture warning.
                        Task { @MainActor in
                            self?.status.apply(event)
                        }
                    })
                await MainActor.run { [weak self] in
                    self?.isRunning = false
                    if self?.status.finishedAt == nil {
                        self?.status.apply(.finished(outputBytes: 0))
                    }
                }
            } catch let cancelled as FineTuneCancelled {
                await MainActor.run { [weak self] in
                    self?.isRunning = false
                    self?.lastError = cancelled.errorDescription
                }
            } catch let notImpl as FineTuneNotImplemented {
                await MainActor.run { [weak self] in
                    self?.isRunning = false
                    self?.notImplementedNotice = notImpl.detail
                }
            } catch {
                await MainActor.run { [weak self] in
                    self?.isRunning = false
                    self?.lastError = (error as? LocalizedError)?.errorDescription
                        ?? error.localizedDescription
                }
            }
            _ = self  // keep reference alive across await
        }
    }

    func cancel() {
        cancellation?.cancel()
    }
}
