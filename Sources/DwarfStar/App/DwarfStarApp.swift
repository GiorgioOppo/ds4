import SwiftUI

@main
struct DwarfStarApp: App {
    @State private var store = ChatStore()

    init() {
        // Capture the C engine's stderr so Metal/kernel errors are visible.
        EngineLog.shared.install()
    }

    var body: some Scene {
        WindowGroup("DwarfStar") {
            RootView(store: store)
                .frame(minWidth: 860, minHeight: 600)
        }
        .windowResizability(.contentMinSize)
    }
}
