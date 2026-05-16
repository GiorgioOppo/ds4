import SwiftUI
import DeepSeekKit

/// macOS Settings scene (cmd+,). Tabs covering generation, loading,
/// the entire ModelConfig surface, the global documents library, the
/// projects library, the MCP server registry, native tools,
/// permissions, skills, theme, keybindings, and storage.
struct SettingsScene: Scene {
    @ObservedObject var documents: DocumentLibrary
    @ObservedObject var projects: ProjectLibrary
    @ObservedObject var mcp: MCPServerLibrary
    @ObservedObject var mcpPool: MCPClientPool
    @ObservedObject var agents: AgentLibrary
    let service: InferenceService
    @ObservedObject var nativeTools: NativeToolHost
    @ObservedObject var permissions: PermissionStore
    @ObservedObject var skills: SkillLibrary
    @ObservedObject var themes: ThemeStore
    @ObservedObject var keybindings: KeybindingStore

    var body: some Scene {
        Settings {
            TabView {
                GenerationSettingsTab()
                    .tabItem { Label("Generation", systemImage: "slider.horizontal.3") }
                LoadingSettingsTab()
                    .tabItem { Label("Loading", systemImage: "tray.and.arrow.down") }
                ModelConfigSettingsTab()
                    .tabItem { Label("Model Config", systemImage: "gearshape.2") }
                AgentsView(library: agents, mcpPool: mcpPool)
                    .tabItem { Label("Agents", systemImage: "person.2") }
                ToolsSettingsTab(host: nativeTools)
                    .tabItem { Label("Tools", systemImage: "wrench.and.screwdriver") }
                PermissionsSettingsTab(host: nativeTools, store: permissions)
                    .tabItem { Label("Permissions", systemImage: "checkmark.shield") }
                SkillsSettingsTab(library: skills)
                    .tabItem { Label("Skills", systemImage: "sparkles") }
                ThemeSettingsTab(store: themes)
                    .tabItem { Label("Theme", systemImage: "paintbrush") }
                KeybindingsSettingsTab(store: keybindings)
                    .tabItem { Label("Keys", systemImage: "keyboard") }
                DocumentsView(library: documents, service: service)
                    .tabItem { Label("Documents", systemImage: "doc.text") }
                ProjectsView(library: projects,
                              documents: documents,
                              service: service)
                    .tabItem { Label("Projects", systemImage: "folder") }
                MCPServersView(library: mcp, pool: mcpPool)
                    .tabItem { Label("MCP", systemImage: "server.rack") }
                APIKeysSettingsTab()
                    .tabItem { Label("API Keys", systemImage: "key") }
                StorageSettingsTab()
                    .tabItem { Label("Storage", systemImage: "externaldrive") }
            }
            .frame(width: 760, height: 580)
        }
    }
}
