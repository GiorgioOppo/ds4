import Foundation
import SwiftUI
import DeepSeekConverter

/// Drives `ConvertSheet`. Holds the form state, a current
/// `ConversionStatus` rolled up from the library's event stream,
/// and the running cancellation token / Task handle.
@MainActor
final class ConvertViewModel: ObservableObject {
    // ---- Form state ----
    @Published var sourcePath: URL?
    @Published var destPath: URL?
    @Published var nExperts: Int = 256
    @Published var target: ConversionTarget = .int4
    @Published var shardSizeGB: Double = 5.0
    @Published var modelParallel: Int = 1

    // ---- Runtime state ----
    @Published var status: ConversionStatus = ConversionStatus()
    @Published var isRunning: Bool = false
    @Published var lastError: String? = nil

    private var task: Task<Void, Never>? = nil
    private var cancellation: CancellationToken? = nil

    /// True iff the form has enough information to start. Source
    /// directory must exist, dest must be set, and the source and
    /// dest must be distinct (avoid writing into the same dir we
    /// read from).
    var canStart: Bool {
        guard let src = sourcePath, let dst = destPath else { return false }
        guard FileManager.default.fileExists(atPath: src.path) else { return false }
        guard src.standardizedFileURL != dst.standardizedFileURL else { return false }
        guard nExperts > 0 else { return false }
        return !isRunning
    }

    func start(binaryOverridePath: String?) {
        guard canStart, let src = sourcePath, let dst = destPath else { return }
        let spec = QuantizeSpec(
            hfPath: src,
            savePath: dst,
            nExperts: nExperts,
            modelParallel: modelParallel,
            target: target,
            shardSizeGB: shardSizeGB)

        isRunning = true
        lastError = nil
        status = ConversionStatus()
        let token = CancellationToken()
        self.cancellation = token

        self.task = Task { [weak self] in
            do {
                try await Converter.runQuantize(
                    spec: spec,
                    cancellation: token,
                    binaryOverridePath: binaryOverridePath,
                    onEvent: { event in
                        // Hop to MainActor to mutate @Published state.
                        Task { @MainActor [weak self] in
                            self?.status.apply(event)
                        }
                    })
                await MainActor.run {
                    self?.isRunning = false
                    // Ensure the rolled-up state shows completion.
                    if self?.status.finishedAt == nil {
                        self?.status.apply(.finished(outputBytes: 0))
                    }
                }
            } catch {
                await MainActor.run {
                    self?.isRunning = false
                    self?.lastError = (error as? LocalizedError)?.errorDescription
                        ?? error.localizedDescription
                }
            }
        }
    }

    func cancel() {
        cancellation?.cancel()
    }
}
