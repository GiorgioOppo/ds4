import SwiftUI

/// SwiftUI entry point for the DeepSeek-V4 macOS chat app. Skeleton:
/// just brings up a window with the model picker. The rest of the
/// scaffolding (InferenceService, chat, persistence, settings) is
/// layered on in subsequent commits per the plan.
@main
struct DeepSeekUIApp: App {
    var body: some Scene {
        WindowGroup {
            ContentView()
                .frame(minWidth: 720, minHeight: 480)
        }
        .windowResizability(.contentMinSize)

        SettingsScene()
    }
}
