import Foundation
import SwiftUI
import DeepSeekTools

/// `MainActor`-side owner of the native tool runtime. Bundles the
/// `ToolRegistry`, the `PlanStore`, and the bridges that turn a
/// `PermissionRequest` into a SwiftUI sheet.
///
/// Stays separate from `MCPClientPool` (which manages stdio servers)
/// so the two can coexist: the chat layer asks `NativeToolHost` for
/// native-tool schemas and `MCPClientPool` for remote ones, merges
/// them into the system block, and routes each `tools/call` to the
/// right side based on the qualified name.
@MainActor
final class NativeToolHost: ObservableObject {
    let registry: ToolRegistry
    let planStore: PlanStore
    /// Snapshot of the registered schemas. Refreshed from `registry`
    /// after `register(...)` calls so the UI can render a tool list.
    @Published private(set) var schemas: [ToolSchema] = []
    /// Currently pending consent request, surfaced as a sheet.
    @Published var pendingRequest: PendingRequest?
    /// Per-session "always allow" decisions made by the user.
    /// Mirrors the registry's internal cache for inspection in the
    /// permissions tab.
    @Published private(set) var sessionGrants: Set<String> = []

    private let permissionDelegate: GUIPermissionDelegate

    init() {
        let store = PlanStore()
        let registry = ToolRegistry()
        let delegate = GUIPermissionDelegate()
        self.planStore = store
        self.registry = registry
        self.permissionDelegate = delegate
        delegate.host = self
        Task { @MainActor [registry] in
            // Shell escluso esplicitamente: troppo invasivo per
            // l'esposizione di default al chat — il modello tende a
            // raggiungerlo anche quando le tool dedicate (read /
            // edit / grep / glob) fanno il lavoro più sicuro e
            // strutturato. Resterà registrabile manualmente in
            // futuro tramite un toggle.
            await registry.registerAll(
                DefaultTools.standard(planStore: store,
                                       includeShell: false))
            self.schemas = await registry.availableSchemas(mode: .build)
        }
    }

    /// Run a native tool by registry name. The chat flow looks up
    /// the unqualified name with this method; MCP-qualified calls
    /// (`server__tool`) go to the MCP pool instead.
    func dispatch(name: String,
                  input: [String: Any],
                  mode: AgentMode,
                  rootDirectory: URL) async -> ToolOutput {
        let context = ToolContext(
            rootDirectory: rootDirectory,
            mode: mode,
            permission: permissionDelegate,
            environment: nil,
            isCancelled: { false } // wired from chat in InferenceService
        )
        return await registry.dispatch(name: name, input: input, context: context)
    }

    func refreshSchemas(mode: AgentMode) async {
        schemas = await registry.availableSchemas(mode: mode)
    }

    func resetPermissions() {
        Task { await registry.resetSessionCache() }
        sessionGrants.removeAll()
    }

    // MARK: - Permission sheet bridge

    struct PendingRequest: Identifiable {
        let id = UUID()
        let request: PermissionRequest
        let resolve: (PermissionDecision) -> Void
    }

    fileprivate func record(grant key: String) {
        sessionGrants.insert(key)
    }

    fileprivate func present(request: PermissionRequest) async -> PermissionDecision {
        await withCheckedContinuation { (cont: CheckedContinuation<PermissionDecision, Never>) in
            self.pendingRequest = PendingRequest(request: request) { decision in
                cont.resume(returning: decision)
            }
        }
    }
}

/// Bridges `PermissionDelegate` (Sendable) to the `@MainActor` GUI.
/// Stored as a class so the registry can hold a `Sendable` reference
/// while the host wires itself in after init.
final class GUIPermissionDelegate: PermissionDelegate, @unchecked Sendable {
    weak var host: NativeToolHost?

    func decide(request: PermissionRequest) async -> PermissionDecision {
        guard let host else { return .deny }
        let decision = await host.present(request: request)
        if decision == .alwaysAllow {
            await MainActor.run {
                host.record(grant: "\(request.tool):\(request.category.rawValue)")
            }
        }
        return decision
    }
}
