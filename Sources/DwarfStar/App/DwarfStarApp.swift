import SwiftUI

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
    }

    var body: some Scene {
        WindowGroup("DwarfStar") {
            RootView(store: store, settings: settings)
                .frame(minWidth: 860, minHeight: 600)
        }
        .windowResizability(.contentMinSize)
    }
}
