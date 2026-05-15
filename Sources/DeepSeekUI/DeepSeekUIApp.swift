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

    init() {
        self.service = InferenceService()
        self._documents = StateObject(wrappedValue: DocumentLibrary())
        self._projects = StateObject(wrappedValue: ProjectLibrary())
    }

    var body: some Scene {
        WindowGroup {
            ContentView(service: service,
                         documents: documents,
                         projects: projects)
                .frame(minWidth: 720, minHeight: 480)
        }
        .windowResizability(.contentMinSize)

        SettingsScene(documents: documents,
                       projects: projects,
                       service: service)
    }
}
