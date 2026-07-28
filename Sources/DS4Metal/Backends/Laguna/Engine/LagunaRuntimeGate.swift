import Foundation

/// THE Laguna S 2.1 enablement switch, visible to every target (DS4Engine
/// keys the backend selector and catalog off it; DS4Demo dispatches on it
/// directly because it cannot import DS4Engine).
///
/// STATE: disabled. The Laguna frontend (recognition, geometry validation,
/// tokenizer, chat/tool protocol, tensor-schema validation, catalog) is in
/// place, but the Metal decoder from the reference `laguna-s2.1` branch
/// (`metal/laguna.metal` plus the `ds4.c`/`ds4_metal.m` driver paths, and the
/// optional DFlash speculative companion) has not been ported yet. Flip this
/// only after the end-to-end logits-parity gate passes on real weights on
/// hardware; see `docs/PORTING-GAPS.md`.
public enum LagunaRuntimeGate {
    public static let enabled = false
}
