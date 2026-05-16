import Foundation

/// Programmatic entry point for full fine-tuning of a local
/// DeepSeek checkpoint. SwiftUI (and any other Swift caller) drives a
/// training run through `run` instead of having to manage a process
/// loop / cancellation handle inline.
///
/// The current backend is a *stub* — see `FineTuneRunner` for the
/// rationale and the validation it performs. Calls that pass
/// pre-flight throw `FineTuneNotImplemented` so the UI can render a
/// clear explanation. The signature is stable: when a real backend
/// lands (native Metal trainer in DeepSeekKit, or an external CLI),
/// the swap is local to `FineTuneRunner.run` and callers don't need
/// to change.
public enum FineTuner {

    public static func run(
        spec: FineTuneSpec,
        cancellation: TrainingCancellationToken = TrainingCancellationToken(),
        onEvent: @escaping @Sendable (FineTuneEvent) -> Void
    ) async throws {
        // Detach so the synchronous validation + future training loop
        // doesn't block the calling actor (mirrors how
        // `Converter.runQuantize` defers to a background task).
        try await Task.detached(priority: .userInitiated) {
            try FineTuneRunner.run(
                spec: spec,
                cancellation: cancellation,
                onEvent: onEvent)
        }.value
    }
}
