import Foundation

/// THE GLM 5.2 enablement switch, visible to every target (DS4Engine keys
/// the backend selector and catalog off it; DS4Demo dispatches on it
/// directly because it cannot import DS4Engine). It turns true only when
/// the real-GGUF logits parity gate has passed on hardware, so the flip is
/// one reviewable line.
public enum GLM52RuntimeGate {
    public static let enabled = false
}
