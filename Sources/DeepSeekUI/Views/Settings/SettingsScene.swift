import SwiftUI
import DeepSeekKit

/// macOS Settings scene (cmd+,). Four tabs covering generation,
/// loading, the entire ModelConfig surface, and storage.
struct SettingsScene: Scene {
    var body: some Scene {
        Settings {
            TabView {
                GenerationSettingsTab()
                    .tabItem { Label("Generation", systemImage: "slider.horizontal.3") }
                LoadingSettingsTab()
                    .tabItem { Label("Loading", systemImage: "tray.and.arrow.down") }
                ModelConfigSettingsTab()
                    .tabItem { Label("Model Config", systemImage: "gearshape.2") }
                StorageSettingsTab()
                    .tabItem { Label("Storage", systemImage: "externaldrive") }
            }
            .frame(width: 620, height: 540)
        }
    }
}
