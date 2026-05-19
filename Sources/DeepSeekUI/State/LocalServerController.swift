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
///
/// `mcpPool` is held by reference and brokers the tool-call loop
/// inside `/v1/chat/completions`: the controller exposes
/// `composeToolSchemasJSON(allowedNames:)` (used to seed
/// `EncodingDSV4.encodeMessages`) and
/// `invokeQualified(name:argsJSON:)` (used after the model emits
/// `ToolCall`s in a `.done` Message). Both helpers are `@MainActor`
/// so the underlying pool's `@Published` state can keep its
/// single-threaded contract.
@MainActor
final class LocalServerController: ObservableObject {
    @Published private(set) var isRunning: Bool = false
    @Published private(set) var lastError: String? = nil

    private let server = LocalServer()
    private let service: InferenceService
    private let mcpPool: MCPClientPool

    init(service: InferenceService, mcpPool: MCPClientPool) {
        self.service = service
        self.mcpPool = mcpPool
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
            await LocalServerRoutes.register(
                on: server, service: service, controller: self)
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

    // MARK: - Tool helpers (called by LocalServerRoutes on the main actor)

    /// Build the toolSchemasJSON blob passed to
    /// `EncodingDSV4.encodeMessages`. Returns nil when no tools are
    /// registered or the caller explicitly opted out via `allowedNames:
    /// []`. When `allowedNames` is nil every MCP tool in the pool is
    /// exposed; when non-nil only the listed qualified names pass through.
    func composeToolSchemasJSON(allowedNames: Set<String>?) -> String? {
        if let allowed = allowedNames, allowed.isEmpty { return nil }
        var schemas: [[String: Any]] = []
        for tool in mcpPool.allTools() {
            if let allowed = allowedNames,
               !allowed.contains(tool.qualifiedName) { continue }
            schemas.append([
                "name": tool.qualifiedName,
                "description": tool.description,
                "inputSchema": tool.inputSchema,
            ])
        }
        if schemas.isEmpty { return nil }
        guard let data = try? JSONSerialization.data(
            withJSONObject: schemas,
            options: [.prettyPrinted, .sortedKeys])
        else { return nil }
        return String(data: data, encoding: .utf8)
    }

    /// Forward a `ToolCall.name` (already qualified as
    /// `<server>__<tool>`) to the MCP pool. Errors are flattened to
    /// a string by the pool so this never throws.
    func invokeQualified(_ name: String, argsJSON: String) async -> String {
        await mcpPool.invokeQualified(name, argsJSON: argsJSON)
    }
}
