import SwiftUI

/// SwiftUI entry point for the DeepSeek-V4 macOS chat app. Owns the
/// long-lived singletons (`InferenceService`, `DocumentLibrary`) and
/// hands them down both to the chat window and to the Settings scene
/// so the Documents tab can reach the active tokenizer / model dir.
@main
struct DeepSeekUIApp: App {
    /// Reference type, immutable for the lifetime of the app. Held as
    /// a `let` because `InferenceService` is not an ObservableObject
    /// (its mutating fields are guarded by an internal serial queue),
    /// and SwiftUI's `@StateObject` requires Observable conformance.
    private let service: InferenceService
    @StateObject private var library: DocumentLibrary

    init() {
        self.service = InferenceService()
        self._library = StateObject(wrappedValue: DocumentLibrary())
    }

    var body: some Scene {
        WindowGroup {
            ContentView(service: service)
                .frame(minWidth: 720, minHeight: 480)
        }
        .windowResizability(.contentMinSize)

        SettingsScene(library: library, service: service)
    }
}
