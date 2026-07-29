import Foundation

/// THE Laguna S 2.1 enablement switch, visible to every target (DS4Engine
/// keys the backend selector and catalog off it; DS4Demo dispatches on it
/// directly because it cannot import DS4Engine).
///
/// STATE: disabled by default. The frontend and the first-cut resident Metal
/// engine (`LagunaResidentModel`) are in place, but the end-to-end
/// logits-parity gate against the reference C engine has not run on real
/// weights yet; see `docs/PORTING-GAPS.md` (Gap 4).
///
/// `DS4_LAGUNA_RUNTIME=1` opts a single process into the bring-up engine
/// (demo CLI decode plus selector/catalog routing) for local experiments.
/// Treat that output as unvalidated until the parity gate is green. Flip
/// the compiled-in default only after it passes, like the GLM gate.
public enum LagunaRuntimeGate {
    public static let enabled =
        ProcessInfo.processInfo.environment["DS4_LAGUNA_RUNTIME"] == "1"
}
