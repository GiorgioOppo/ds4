import Foundation
import DeepSeekKit

/// Programmatic entry point for the offline checkpoint converter.
/// SwiftUI (and any other Swift caller) drives a conversion through
/// `runQuantize` / `runDequantize` instead of having to manage
/// subprocess plumbing inline.
///
/// Current implementation: `runQuantize` shells out to the
/// `converter` CLI binary (the imperative implementation living in
/// `Sources/converter/main.swift`) and translates its stdout into
/// a stream of `ConversionEvent`s. The CLI is the source of truth
/// for quantization; this wrapper exists so we can swap the impl
/// to a fully-native path later without changing callers.
///
/// `runDequantize` is still a stub — landing in a follow-up commit
/// alongside the native dequant paths in DeepSeekKit.
public enum Converter {

    /// Run a HF → native conversion (quantize or transcode).
    /// Mirrors the existing `converter` CLI flow.
    ///
    /// - Parameter binaryOverridePath: explicit filesystem path to the
    ///   `converter` executable. When `nil`, the runner searches a
    ///   handful of well-known locations (Bundle Resources, sibling
    ///   of the running executable, .build/{debug,release}/converter,
    ///   /usr/local/bin/converter). Throws if no plausible binary is
    ///   found — the caller (UI Settings) can prompt the user to
    ///   point at one explicitly.
    public static func runQuantize(
        spec: QuantizeSpec,
        cancellation: CancellationToken = CancellationToken(),
        binaryOverridePath: String? = nil,
        onEvent: @escaping @Sendable (ConversionEvent) -> Void
    ) async throws {
        guard let binary = ConverterRunner.locateConverterBinary(
            overridePath: binaryOverridePath) else {
            throw NSError(
                domain: "DeepSeekConverter", code: 200,
                userInfo: [NSLocalizedDescriptionKey:
                    "Couldn't find a `converter` binary. Build it once with " +
                    "`swift build -c release` from the repo root, then either " +
                    "set the path in Settings → Convert or copy it to " +
                    "/usr/local/bin/converter."])
        }
        // Run on a detached task so the synchronous Process polling
        // loop doesn't block the calling actor.
        try await Task.detached(priority: .userInitiated) {
            try ConverterRunner.runQuantizeViaSubprocess(
                spec: spec,
                binaryURL: binary,
                cancellation: cancellation,
                onEvent: onEvent)
        }.value
    }

    /// Reverse direction: read a previously-quantized checkpoint
    /// (INT8/INT4/INT2 weights + F16 group scales) and emit a
    /// plain BF16 checkpoint with the same naming convention.
    /// Native implementation lands in a follow-up commit; for now
    /// throws so UI can disable the dequant dropdown gracefully.
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
                "Dequantize is not yet implemented — coming in a follow-up commit. The UI dropdown is disabled until then."])
    }
}
