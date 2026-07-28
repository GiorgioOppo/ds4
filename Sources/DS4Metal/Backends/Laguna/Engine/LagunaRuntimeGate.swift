import Foundation

/// THE Laguna S 2.1 enablement switch, visible to every target (DS4Engine
/// keys the backend selector and catalog off it; DS4Demo dispatches on it
/// directly because it cannot import DS4Engine).
///
/// STATE: disabled by default. The Laguna frontend (recognition, geometry
/// validation, tokenizer, chat/tool protocol, tensor-schema validation,
/// catalog) is in place and the first-cut resident engine is written, but the
/// end-to-end logits-parity gate against the reference `laguna-s2.1` engine
/// has not run on hardware yet; see `docs/PORTING-GAPS.md`.
///
/// Bring-up escape hatch: `DS4_LAGUNA_RUNTIME=1` enables the runtime for the
/// current process without flipping the default, so the engine can be
/// compiled, smoke-tested and parity-checked on a Mac before the gate is
/// committed open.
public enum LagunaRuntimeGate {
    public static let enabled: Bool = {
        ProcessInfo.processInfo.environment["DS4_LAGUNA_RUNTIME"] == "1"
            || enabledByDefault
    }()

    /// Flip only after the end-to-end logits-parity gate passes on real
    /// weights on hardware.
    static let enabledByDefault = false
}
