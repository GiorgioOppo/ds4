import SwiftUI
import DeepSeekKit
import DeepSeekTools

/// SwiftUI entry point for the DeepSeek-V4 macOS chat app. Owns the
/// long-lived singletons (`InferenceService`, `DocumentLibrary`,
/// `ProjectLibrary`) and hands them down to both the chat window and
/// the Settings scene.
@main
struct DeepSeekUIApp: App {
    /// Reference type, immutable for the lifetime of the app. Held as
    /// a `let` because `InferenceService` is not an ObservableObject
    /// (its mutating fields are guarded by an internal serial queue),
    /// and SwiftUI's `@StateObject` requires Observable conformance.
    private let service: InferenceService
    @StateObject private var documents: DocumentLibrary
    @StateObject private var projects: ProjectLibrary
    @StateObject private var mcp: MCPServerLibrary
    @StateObject private var mcpPool: MCPClientPool
    @StateObject private var agents: AgentLibrary
    @StateObject private var modelLibrary: ModelLibrary
    @StateObject private var modelState: ModelState
    @StateObject private var openRouterCatalog: OpenRouterCatalog
    // Native tool runtime + adjacent stores. Added when the toolbox
    // landed (read/write/edit/grep/glob/shell/...). Each store has
    // its own JSON file under Application Support and is loaded
    // lazily; the app launches even if these files are absent.
    @StateObject private var nativeTools: NativeToolHost
    @StateObject private var permissions: PermissionStore
    @StateObject private var skills: SkillLibrary
    @StateObject private var themes: ThemeStore
    @StateObject private var keybindings: KeybindingStore
    @StateObject private var slashCommands: SlashCommandLibrary
    @StateObject private var serverController: LocalServerController

    init() {
        // Bridge `AppSettings.lazyExpertLoad` into the DeepSeekKit
        // `StreamingPool` toggle BEFORE InferenceService gets a chance
        // to load a model. The pool reads the flag at every
        // `ensureLayer` call so a later UI change also takes effect
        // mid-session, but doing it here keeps the very first token
        // of the first run on the right path.
        //
        // IMPORTANT: if `DEEPSEEK_LAZY_EXPERT` is set in the
        // environment, it WINS — both `StreamingPool.swift`'s class
        // doc and the kit-side static initializer treat the env var
        // as the override of last resort, so we must not clobber it
        // here. Without this guard, launching with the env var set
        // gets silently overridden by whatever happens to be in
        // `UserDefaults` (false for users who never touched the
        // toggle), and the "DEEPSEEK_LAZY_EXPERT=1 forces it on"
        // contract advertised in the kit comment is broken.
        if ProcessInfo.processInfo
            .environment["DEEPSEEK_LAZY_EXPERT"] == nil {
            StreamingPool.lazyExpertEnabled = AppSettings.lazyExpertLoad
        }

        // Bridge the UI's per-token active-expert override into
        // ModelConfig. No env-var guard needed: ModelConfig.init reads
        // DEEPSEEK_TOPK_EXPERTS before this override, so the env var
        // still wins.
        ModelConfig.activeExpertsOverride = AppSettings.activeExpertsOverride

        let service = InferenceService()
        self.service = service
        self._documents = StateObject(wrappedValue: DocumentLibrary())
        self._projects = StateObject(wrappedValue: ProjectLibrary())
        let mcpLibrary = MCPServerLibrary()
        let pool = MCPClientPool()
        // Wire the pool to the library at construction time so
        // already-saved enabled servers start spawning on launch
        // without waiting for the user to open the Settings tab.
        pool.attach(to: mcpLibrary)
        pool.librarySynced(mcpLibrary)
        self._mcp = StateObject(wrappedValue: mcpLibrary)
        self._mcpPool = StateObject(wrappedValue: pool)
        self._agents = StateObject(wrappedValue: AgentLibrary())
        let lib = ModelLibrary()
        self._modelLibrary = StateObject(wrappedValue: lib)
        self._modelState = StateObject(wrappedValue:
            ModelState(service: service, library: lib))
        self._openRouterCatalog = StateObject(wrappedValue: OpenRouterCatalog())
        self._nativeTools = StateObject(wrappedValue: NativeToolHost())
        self._permissions = StateObject(wrappedValue: PermissionStore())
        self._skills = StateObject(wrappedValue: SkillLibrary())
        self._themes = StateObject(wrappedValue: ThemeStore())
        self._keybindings = StateObject(wrappedValue: KeybindingStore())
        self._slashCommands = StateObject(wrappedValue: SlashCommandLibrary())
        // Local OpenAI-compatible HTTP server (TODO §10.1 / T1). The
        // controller stays idle until the user flips the Settings →
        // Server toggle; nothing binds at launch.
        self._serverController = StateObject(wrappedValue:
            LocalServerController(service: service, mcpPool: pool))
    }

    var body: some Scene {
        WindowGroup {
            ContentView(service: service,
                         documents: documents,
                         projects: projects,
                         mcpPool: mcpPool,
                         agents: agents,
                         modelLibrary: modelLibrary,
                         modelState: modelState,
                         openRouterCatalog: openRouterCatalog,
                         nativeTools: nativeTools)
                .frame(minWidth: 720, minHeight: 480)
                // Theme overrides applied at the WindowGroup root so
                // every descendant — including modal sheets — sees
                // them. `preferredColorScheme(nil)` opts back into
                // the system setting.
                .preferredColorScheme(themes.preferredColorScheme)
                .tint(swiftUIColor(hex: themes.active.accent) ?? .accentColor)
                // Auto-start the local server if the user left
                // Settings → Server enabled on the previous quit.
                // `controller.start` is idempotent so reopening the
                // window after closing it won't double-bind.
                .task { await maybeAutoStartServer() }
        }
        .windowResizability(.contentMinSize)

        SettingsScene(documents: documents,
                       projects: projects,
                       mcp: mcp,
                       mcpPool: mcpPool,
                       agents: agents,
                       service: service,
                       nativeTools: nativeTools,
                       permissions: permissions,
                       skills: skills,
                       themes: themes,
                       keybindings: keybindings,
                       serverController: serverController)
    }

    private func maybeAutoStartServer() async {
        let defaults = UserDefaults.standard
        guard defaults.bool(forKey: AppSettingsKey.serverEnabled),
              !serverController.isRunning
        else { return }
        let rawPort = defaults.integer(forKey: AppSettingsKey.serverPort)
        let port = UInt16(clamping: rawPort == 0 ? 8080 : rawPort)
        let address = defaults.string(forKey: AppSettingsKey.serverBindAddress)
            ?? "127.0.0.1"
        await serverController.start(
            port: port,
            address: address.isEmpty ? "127.0.0.1" : address)
    }
}
