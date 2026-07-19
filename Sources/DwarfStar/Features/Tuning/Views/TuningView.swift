import SwiftUI
import DS4Engine

/// Runtime-tuning panel. The content is provided by the backend UI class
/// (`BackendSettingsUI.tuningPanel()`): DeepSeek exposes the expert
/// slot-cache and the usage-imatrix profile; GLM re-exposes its residency/
/// streaming steppers with a note pointing at the measurement benchmark.
/// Backends without a panel (Qwen, unknown, no model) fall back to the
/// unavailable view. Weight-level fine-tuning is NOT possible on-device:
/// the engine is inference-only and quantized weights are not trainable.
struct TuningView: View {
    @Bindable var store: ChatStore

    @ViewBuilder var body: some View {
        if let panel = BackendSettingsUI.make(store: store, dist: nil).tuningPanel() {
            panel
        } else {
            ContentUnavailableView(
                "Tuning non disponibile",
                systemImage: "slider.horizontal.3",
                description: Text("Il modello selezionato non espone controlli di tuning runtime in questo backend.")
            )
        }
    }
}
