import SwiftUI
import DeepSeekKit

/// macOS Settings scene (cmd+,). Five tabs covering generation,
/// loading, the entire ModelConfig surface, the global documents
/// library, and storage.
struct SettingsScene: Scene {
    @ObservedObject var library: DocumentLibrary
    let service: InferenceService

    var body: some Scene {
        Settings {
            TabView {
                GenerationSettingsTab()
                    .tabItem { Label("Generation", systemImage: "slider.horizontal.3") }
                LoadingSettingsTab()
                    .tabItem { Label("Loading", systemImage: "tray.and.arrow.down") }
                ModelConfigSettingsTab()
                    .tabItem { Label("Model Config", systemImage: "gearshape.2") }
                DocumentsView(library: library, service: service)
                    .tabItem { Label("Documents", systemImage: "doc.text") }
                StorageSettingsTab()
                    .tabItem { Label("Storage", systemImage: "externaldrive") }
            }
            .frame(width: 620, height: 540)
        }
    }
}
