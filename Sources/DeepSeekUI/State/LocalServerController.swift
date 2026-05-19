import Foundation
import SwiftUI

/// SwiftUI-facing lifecycle wrapper around `LocalServer`. Owns the
/// actor, tracks the `isRunning` flag for binding into the Settings →
/// Server tab, and surfaces the most recent error so the user can see
/// why a bind failed (port already taken, permission denied, etc.).
///
/// Bearer token is read on demand from `KeychainStore` (account
/// `KeychainAccount.serverBearerToken`); the controller never holds
/// the token in `@Published` state to avoid it landing in any
/// SwiftUI debug log accidentally.
@MainActor
final class LocalServerController: ObservableObject {
    @Published private(set) var isRunning: Bool = false
    @Published private(set) var lastError: String? = nil

    private let server = LocalServer()
    private let service: InferenceService

    init(service: InferenceService) {
        self.service = service
    }

    /// Try to start the listener on `port:address`. Idempotent — a
    /// running server is stopped first. Updates `isRunning` /
    /// `lastError` on the main actor when done.
    func start(port: UInt16, address: String) async {
        let token = KeychainStore.get(account: KeychainAccount.serverBearerToken)
        do {
            try await server.start(port: port,
                                    address: address,
                                    bearerToken: token)
            await LocalServerRoutes.register(on: server, service: service)
            self.isRunning = true
            self.lastError = nil
        } catch {
            self.isRunning = false
            self.lastError = (error as? LocalizedError)?.errorDescription
                ?? error.localizedDescription
        }
    }

    func stop() async {
        try? await server.stop()
        self.isRunning = false
    }
}
