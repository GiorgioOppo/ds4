import SwiftUI
import AppKit
import DS4Engine
import DS4Core

@main
struct DwarfStarApp: App {
    @State private var settings: AppSettings
    @State private var store: ChatStore
    @State private var mcp: MCPStore

    init() {
        // La GUI opta nel runtime Laguna (bring-up): il gate
        // LagunaRuntimeGate legge DS4_LAGUNA_RUNTIME al primo accesso, e
        // qui siamo prima di qualunque load/inspect. overwrite=0: un
        // DS4_LAGUNA_RUNTIME=0 esplicito nell'ambiente vince ancora.
        setenv("DS4_LAGUNA_RUNTIME", "1", 0)
        // Profilo Laguna misurato sul M1 Pro con ~10 GiB liberi. Va
        // installato PRIMA di ChatStore: quest'ultimo pubblica i knob comuni
        // ottimizzati per DeepSeek, mentre gli alias DS4_LAGUNA_* devono
        // conservare chunk/cache/I/O specifici del backend. Gli override
        // espliciti nell'ambiente vincono perché il preset non sovrascrive.
        LagunaInferenceService.installGUIEnvironmentDefaults()
        // Capture the C engine's stderr so Metal/kernel errors are visible.
        EngineLog.shared.install()
        // Recover before AppSettings/ChatStore stored-property initializers read
        // UserDefaults. A crash during the multi-key auto-tune commit can then
        // never expose a half-winner/half-baseline configuration to this launch.
        let recovery = MachineAutoTuneTransactionStore.recoverAtLaunch()
        if let message = recovery.logMessage {
            DS4Log.info("autotune", message)
        }
        let settings = AppSettings()
        _settings = State(initialValue: settings)
        _store = State(initialValue: ChatStore(settings: settings))
        // Connect the configured MCP servers at launch so their tools are
        // available on the first chat, not only after opening the MCP panel.
        _mcp = State(initialValue: MCPStore())
        // Unbundled executables (`swift run DwarfStar`) are not activated by
        // Launch Services: the window appears but never becomes KEY, so the
        // keyboard never reaches any TextField (clicks still work). Activate
        // explicitly once the run loop is up — harmless in a bundled .app.
        DispatchQueue.main.async {
            NSApplication.shared.setActivationPolicy(.regular)
            NSApplication.shared.activate(ignoringOtherApps: true)
        }
    }

    var body: some Scene {
        WindowGroup("DwarfStar") {
            RootView(store: store, settings: settings, mcp: mcp)
                .frame(minWidth: 860, minHeight: 600)
        }
        .windowResizability(.contentMinSize)
    }
}
