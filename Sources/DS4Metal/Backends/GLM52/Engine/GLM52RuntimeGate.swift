import Foundation

/// THE GLM 5.2 enablement switch, visible to every target (DS4Engine keys
/// the backend selector and catalog off it; DS4Demo dispatches on it
/// directly because it cannot import DS4Engine).
///
/// STATE: enabled EARLY at the owner's explicit request (2026-07-18) with
/// the engine smoke green on real weights but the full logits-parity gate
/// (`GLM52LogitsParityIntegrationTests`) NOT yet confirmed on hardware.
/// Treat GLM output as experimental until that test is green; if parity
/// fails, flip this back to false while the divergence is fixed.
public enum GLM52RuntimeGate {
    public static let enabled = true
}
