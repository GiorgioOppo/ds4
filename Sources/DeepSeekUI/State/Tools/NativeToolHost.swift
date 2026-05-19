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
        // TODO §8: pick the web-search backend from AppStorage +
        // Keychain. Missing key on the selected provider falls
        // back to DuckDuckGo with a stderr note so the tool stays
        // useful even when configuration's incomplete.
        let searchProvider = Self.resolveWebSearchProvider()
        // TODO §9: opt-in sandbox-exec wrapper on ShellTool.
        let useSandbox = UserDefaults.standard.bool(
            forKey: AppSettingsKey.useShellSandbox)
        // Unix + Xcode toolboxes default ON: bool(forKey:) returns
        // false for an unset key, so we route through `object` to
        // distinguish "unset" (= default true) from "explicitly off"
        // (= user turned the toggle off in Settings).
        let enableUnix = Self.boolDefaultingTrue(
            key: AppSettingsKey.enableUnixTools)
        let enableXcode = Self.boolDefaultingTrue(
            key: AppSettingsKey.enableXcodeTools)
        Task { @MainActor [registry] in
            // Shell escluso esplicitamente: troppo invasivo per
            // l'esposizione di default al chat — il modello tende a
            // raggiungerlo anche quando le tool dedicate (read /
            // edit / grep / glob) fanno il lavoro più sicuro e
            // strutturato. Resterà registrabile manualmente in
            // futuro tramite un toggle.
            // Shell stays disabled by default (UPKmT decision) even
            // though the sandbox flag is wired — flipping the
            // `useShellSandbox` toggle is meaningful only once the
            // shell tool itself gets re-enabled (manual register or
            // future Settings toggle). `shellUsesSandbox` is passed
            // through so DefaultTools threads it to a future
            // ShellTool construction without us having to edit this
            // call site again.
            await registry.registerAll(
                DefaultTools.standard(
                    planStore: store,
                    includeShell: false,
                    includeUnixTools: enableUnix,
                    includeXcodeTools: enableXcode,
                    shellUsesSandbox: useSandbox,
                    webSearchProvider: searchProvider))
            self.schemas = await registry.availableSchemas(mode: .build)
        }
    }

    /// `UserDefaults.bool(forKey:)` returns false for an unset key,
    /// which would silently keep the Unix/Xcode toolboxes off on a
    /// fresh install. This helper distinguishes unset (= return
    /// default) from explicitly set false (= respect the user's
    /// choice).
    private static func boolDefaultingTrue(key: String) -> Bool {
        let defaults = UserDefaults.standard
        if defaults.object(forKey: key) == nil { return true }
        return defaults.bool(forKey: key)
    }

    /// Resolve the user-configured search backend (or nil to keep
    /// the DefaultTools default of `DuckDuckGoLiteProvider`).
    /// Reads `AppSettingsKey.webSearchProvider` from UserDefaults
    /// + the matching `KeychainAccount.*APIKey` from the Keychain.
    /// Emits a one-time stderr note when the user asked for a
    /// key'd provider but didn't store the key.
    private static func resolveWebSearchProvider() -> WebSearchProvider? {
        let raw = UserDefaults.standard.string(
            forKey: AppSettingsKey.webSearchProvider) ?? "duckduckgo"
        switch raw {
        case "tavily":
            guard let key = KeychainStore.get(
                account: KeychainAccount.tavilyAPIKey),
                  !key.isEmpty
            else {
                FileHandle.standardError.write(Data((
                    "[websearch] webSearchProvider=tavily but no Keychain "
                    + "entry under tavilyAPIKey — falling back to DuckDuckGo\n").utf8))
                return nil
            }
            return TavilyProvider(apiKey: key)
        case "brave":
            guard let key = KeychainStore.get(
                account: KeychainAccount.braveSearchAPIKey),
                  !key.isEmpty
            else {
                FileHandle.standardError.write(Data((
                    "[websearch] webSearchProvider=brave but no Keychain "
                    + "entry under braveSearchAPIKey — falling back to DuckDuckGo\n").utf8))
                return nil
            }
            return BraveProvider(apiKey: key)
        case "serper":
            guard let key = KeychainStore.get(
                account: KeychainAccount.serperAPIKey),
                  !key.isEmpty
            else {
                FileHandle.standardError.write(Data((
                    "[websearch] webSearchProvider=serper but no Keychain "
                    + "entry under serperAPIKey — falling back to DuckDuckGo\n").utf8))
                return nil
            }
            return SerperProvider(apiKey: key)
        default:
            return nil  // duckduckgo (DefaultTools.standard's default)
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
