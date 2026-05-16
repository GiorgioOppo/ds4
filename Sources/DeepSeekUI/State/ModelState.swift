import Foundation
import DeepSeekKit
import SwiftUI

/// Reactive façade over `InferenceService`'s load lifecycle. The
/// service itself isn't an `ObservableObject` (it predates the
/// in-chat picker and was meant to be a singleton handed around
/// by reference), so this small `@MainActor` class publishes the
/// status that SwiftUI views need to render the picker label, the
/// load progress banner, and the composer's enabled state.
///
/// Single source of truth for "what model is the chat talking to
/// right now". The service is still the canonical RAM holder of
/// transformer + tokenizer; ModelState mirrors only the bits the
/// UI cares about + drives loads / unloads in response to picker
/// actions.
@MainActor
final class ModelState: ObservableObject {
    /// Coarse load lifecycle. `.idle` covers both "app just
    /// launched, nothing loaded yet" and "user explicitly
    /// unloaded". `.loading` carries the optional `LoadPlan`
    /// once probing has returned it so the banner can show the
    /// shard summary instead of a bare spinner. `.error` is
    /// terminal until the user retries — the previous .loaded
    /// state is lost (the service unloaded as part of the
    /// failed load).
    enum LoadStatus: Equatable {
        case idle
        case loading(endpoint: ModelEndpoint, plan: LoadPlan?)
        case loaded(endpoint: ModelEndpoint, config: ModelConfig)
        case error(endpoint: ModelEndpoint, message: String)

        static func == (lhs: LoadStatus, rhs: LoadStatus) -> Bool {
            switch (lhs, rhs) {
            case (.idle, .idle): return true
            case (.loading(let a, _), .loading(let b, _)): return a == b
            case (.loaded(let a, _), .loaded(let b, _)):   return a == b
            case (.error(let a, let m), .error(let b, let n)):
                return a == b && m == n
            default: return false
            }
        }
    }

    @Published private(set) var status: LoadStatus = .idle

    let service: InferenceService
    let library: ModelLibrary

    init(service: InferenceService, library: ModelLibrary) {
        self.service = service
        self.library = library
    }

    /// Convenience: is a model fully ready for inference? The
    /// composer reads this to enable/disable Send.
    var isReady: Bool {
        if case .loaded = status { return true }
        return false
    }

    /// Currently-loaded endpoint, if any. Convenience for the
    /// toolbar label.
    var loadedEndpoint: ModelEndpoint? {
        if case .loaded(let ep, _) = status { return ep }
        return nil
    }

    /// Load `endpoint` through the service, updating `status` as
    /// each phase begins. Safe to call when another load is in
    /// flight — the new call effectively cancels by replacing
    /// status (the old `loadModel` Task can still complete and
    /// the service will sit on those weights until the new load
    /// finishes the `unloadModel + loadModel` dance below).
    ///
    /// Remote endpoints are rejected here with a descriptive
    /// error until the backend lands; the UI surfaces that as a
    /// normal `.error` state so the user sees something
    /// actionable.
    func load(_ endpoint: ModelEndpoint, force: Bool = false) async {
        switch endpoint {
        case .localDirectory(let path):
            await loadLocal(path: path, endpoint: endpoint, force: force)
        case .openRouter(let modelID):
            await loadRemoteOpenRouter(modelID: modelID, endpoint: endpoint)
        }
    }

    /// Activate a remote OpenRouter endpoint. No weights to map
    /// and nothing to probe, but we still drop any locally-loaded
    /// model (so the chat unambiguously talks to one backend),
    /// validate that the Keychain holds a key, and ping
    /// `/auth/key` so an invalid token fails here instead of on
    /// the first send. `library.touch` bumps the recents.
    private func loadRemoteOpenRouter(modelID: String,
                                        endpoint: ModelEndpoint) async {
        await service.unloadModel()
        status = .loading(endpoint: endpoint, plan: nil)
        guard let key = KeychainStore.get(
            account: KeychainAccount.openRouterAPIKey), !key.isEmpty
        else {
            status = .error(endpoint: endpoint,
                             message: "Add an OpenRouter API key under Settings → API Keys first.")
            return
        }
        do {
            try await OpenRouterClient().validateKey(key)
            library.touch(endpoint)
            status = .loaded(endpoint: endpoint, config: ModelConfig())
        } catch {
            let msg = (error as? LocalizedError)?.errorDescription
                ?? error.localizedDescription
            status = .error(endpoint: endpoint, message: msg)
        }
    }

    private func loadLocal(path: String,
                            endpoint: ModelEndpoint,
                            force: Bool) async {
        let url = URL(fileURLWithPath: path)
        guard FileManager.default.fileExists(atPath: path) else {
            status = .error(endpoint: endpoint,
                             message: "Model folder no longer exists at: \(path)")
            library.forget(endpoint)
            return
        }
        // If another model is loaded, drop it first so RAM
        // doesn't blow up holding two copies during the swap.
        await service.unloadModel()
        status = .loading(endpoint: endpoint, plan: nil)
        do {
            let rawStrategy = AppSettings.loadStrategy
            let strategy: String? = (rawStrategy == "auto") ? nil : rawStrategy
            let cfg = try await service.loadModel(
                at: url,
                strategyOverride: strategy,
                forceLoad: force || AppSettings.forceLoad,
                onPlan: { [weak self] plan in
                    Task { @MainActor in
                        guard let self else { return }
                        if case .loading(let ep, _) = self.status,
                           ep == endpoint {
                            self.status = .loading(endpoint: ep, plan: plan)
                        }
                    }
                })
            library.touch(endpoint)
            AppSettings.setLastModelDir(path)
            status = .loaded(endpoint: endpoint, config: cfg)
        } catch {
            let msg = (error as? LocalizedError)?.errorDescription
                ?? error.localizedDescription
            status = .error(endpoint: endpoint, message: msg)
        }
    }

    /// Drop the currently-loaded model. Picker reverts to
    /// "No model"; composer disables Send. Conversations stay
    /// open but can't send until the user picks again.
    func unload() async {
        await service.unloadModel()
        status = .idle
    }

    /// Re-try the last failed load with `forceLoad = true`. Used
    /// by the error banner's "Force load" button so the user
    /// doesn't have to manually flip the AppStorage flag.
    func retryWithForce() async {
        guard case .error(let endpoint, _) = status else { return }
        await load(endpoint, force: true)
    }
}

