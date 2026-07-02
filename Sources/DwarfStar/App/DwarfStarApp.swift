import SwiftUI
import AppKit

@main
struct DwarfStarApp: App {
    @State private var settings: AppSettings
    @State private var store: ChatStore

    init() {
        // Capture the C engine's stderr so Metal/kernel errors are visible.
        EngineLog.shared.install()
        let settings = AppSettings()
        _settings = State(initialValue: settings)
        _store = State(initialValue: ChatStore(settings: settings))
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
            RootView(store: store, settings: settings)
                .frame(minWidth: 860, minHeight: 600)
        }
        .windowResizability(.contentMinSize)
    }
}
