import SwiftUI

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

    init() {
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
    }

    var body: some Scene {
        WindowGroup {
            ContentView(service: service,
                         documents: documents,
                         projects: projects,
                         mcpPool: mcpPool,
                         agents: agents,
                         modelLibrary: modelLibrary,
                         modelState: modelState)
                .frame(minWidth: 720, minHeight: 480)
                .task {
                    // Auto-resume the most recently used model on
                    // launch. Done from a top-level .task instead
                    // of `init` so the load runs after the SwiftUI
                    // window is on screen — that way the loading
                    // banner is visible from the first frame.
                    if case .idle = modelState.status,
                       let last = AppSettings.lastModelDir,
                       FileManager.default.fileExists(atPath: last) {
                        await modelState.load(.localDirectory(path: last))
                    }
                }
        }
        .windowResizability(.contentMinSize)

        SettingsScene(documents: documents,
                       projects: projects,
                       mcp: mcp,
                       mcpPool: mcpPool,
                       agents: agents,
                       service: service)
    }
}
