import Foundation
import DeepSeekKit

/// Programmatic entry point for the offline checkpoint converter.
/// Wraps the long-form imperative logic that historically lived in
/// `Sources/converter/main.swift` so the UI (and tests, scripts,
/// downstream tools) can drive a conversion without spawning a
/// separate Process.
///
/// The conversion is CPU/IO-heavy and runs on the calling Task's
/// executor; pin it to a `@MainActor` view only if you don't mind
/// blocking the UI for 10-30 minutes. The intended pattern is:
///
///     let token = CancellationToken()
///     try await Task.detached {
///         try await Converter.runQuantize(spec: spec,
///                                          cancellation: token,
///                                          onEvent: { event in
///             // Hop back to MainActor and update @Published state.
///         })
///     }.value
///
/// Implementation is staged in commits — this first commit defines
/// the public API surface and stubs that throw a "not yet
/// implemented" error so the UI integration can be wired up
/// independently. The actual migration of the imperative logic
/// from main.swift lands in the next commits.
public enum Converter {

    /// Run a HF → native conversion (quantize or transcode).
    /// Mirrors the existing `converter` CLI flow.
    public static func runQuantize(
        spec: QuantizeSpec,
        cancellation: CancellationToken = CancellationToken(),
        onEvent: @escaping @Sendable (ConversionEvent) -> Void
    ) async throws {
        // Hand back to the migrated implementation in a subsequent
        // commit. The stub throws so callers don't silently no-op
        // during the multi-commit refactor.
        throw NSError(
            domain: "DeepSeekConverter", code: 100,
            userInfo: [NSLocalizedDescriptionKey:
                "Converter.runQuantize is not yet wired through — use the `converter` CLI binary while the library refactor is in flight."])
        // The CLI binary (Sources/converter/main.swift) remains
        // fully functional; only the library hook is staged.
        // Silence the warning about the unreachable closure parameters.
        _ = (spec, cancellation, onEvent)
    }

    /// Reverse direction: read a previously-quantized checkpoint
    /// (INT8/INT4/INT2 weights + F16 group scales) and emit a
    /// plain BF16 checkpoint with the same naming convention.
    public static func runDequantize(
        spec: DequantizeSpec,
        cancellation: CancellationToken = CancellationToken(),
        onEvent: @escaping @Sendable (ConversionEvent) -> Void
    ) async throws {
        guard spec.target == .bf16 || spec.target == .f16 else {
            throw NSError(
                domain: "DeepSeekConverter", code: 101,
                userInfo: [NSLocalizedDescriptionKey:
                    "Dequantize target must be .bf16 or .f16 (got \(spec.target.rawValue))."])
        }
        throw NSError(
            domain: "DeepSeekConverter", code: 102,
            userInfo: [NSLocalizedDescriptionKey:
                "Converter.runDequantize is not yet implemented — coming in the next commits along with the quantize-from-INT8 path."])
        _ = (spec, cancellation, onEvent)
    }
}
