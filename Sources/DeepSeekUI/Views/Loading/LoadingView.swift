import SwiftUI
import DeepSeekKit

/// Bridges `ContentView` to `InferenceService.loadModel`. While loading,
/// renders the `LoadPlan` summary (when available) plus an
/// indeterminate spinner. On error, replaces the spinner with the
/// localized message and three buttons (Free RAM tooltip, Re-shard
/// tooltip, Force Load retry).
struct LoadingView: View {
    let modelDir: URL
    let service: InferenceService
    /// Called when load completes successfully (config carries the
    /// possibly-inferred-from-checkpoint values).
    var onLoaded: (ModelConfig) -> Void
    /// Called when the user explicitly cancels back to the picker.
    var onCancel: () -> Void

    @State private var plan: LoadPlan?
    @State private var error: String?
    @State private var isRetryingWithForce = false
    @AppStorage(AppSettingsKey.loadStrategy) private var loadStrategy: String = "auto"
    @AppStorage(AppSettingsKey.forceLoad) private var forceLoad: Bool = false

    var body: some View {
        VStack(spacing: 24) {
            Text("Loading model")
                .font(.title2)

            if let p = plan {
                PreflightSummaryView(plan: p)
            } else {
                Text("Probing system & shards…")
                    .foregroundStyle(.secondary)
            }

            if let err = error {
                errorPanel(err)
            } else {
                ProgressView()
                    .controlSize(.large)
                Text(plan == nil
                     ? "Reading shard sizes from disk…"
                     : "Mapping weights into memory… first load can take a minute.")
                    .font(.caption)
                    .foregroundStyle(.tertiary)
            }

            Button("Cancel", role: .cancel, action: onCancel)
                .keyboardShortcut(.cancelAction)
        }
        .padding(40)
        .frame(maxWidth: .infinity, maxHeight: .infinity)
        .task(id: "\(modelDir.path)-\(isRetryingWithForce)") {
            await runLoad(force: forceLoad || isRetryingWithForce)
        }
    }

    @ViewBuilder
    private func errorPanel(_ message: String) -> some View {
        VStack(alignment: .leading, spacing: 14) {
            Label("Refused to load", systemImage: "exclamationmark.triangle.fill")
                .foregroundStyle(.orange)
                .font(.headline)
            Text(message)
                .font(.system(.body, design: .monospaced))
                .textSelection(.enabled)
                .fixedSize(horizontal: false, vertical: true)
            HStack {
                Button("Try again") {
                    error = nil
                    isRetryingWithForce.toggle()  // no-op for the load itself, just retriggers .task
                }
                Button("Force Load") {
                    error = nil
                    isRetryingWithForce = true
                }
                .tint(.red)
                Spacer()
                Button("Choose other folder…", action: onCancel)
            }
        }
        .padding(16)
        .background(Color(NSColor.controlBackgroundColor),
                     in: RoundedRectangle(cornerRadius: 12))
        .frame(maxWidth: 520)
    }

    private func runLoad(force: Bool) async {
        do {
            let cfg = try await service.loadModel(
                at: modelDir,
                strategyOverride: loadStrategy == "auto" ? nil : loadStrategy,
                forceLoad: force,
                onPlan: { p in Task { @MainActor in self.plan = p } })
            await MainActor.run {
                AppSettings.setLastModelDir(modelDir.path)
                onLoaded(cfg)
            }
        } catch {
            await MainActor.run {
                self.error = (error as? LocalizedError)?.errorDescription
                    ?? error.localizedDescription
            }
        }
    }
}
