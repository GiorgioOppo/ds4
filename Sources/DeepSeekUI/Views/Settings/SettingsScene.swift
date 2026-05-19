import SwiftUI
import DeepSeekKit

/// macOS Settings scene (cmd+,). A sidebar-driven layout covering
/// generation, loading, the entire ModelConfig surface, the global
/// documents library, the projects library, the MCP server registry,
/// native tools, permissions, skills, theme, keybindings, and storage.
///
/// We use a NavigationSplitView instead of a TabView because the latter
/// silently collapses overflowing tabs into a ">>" menu on macOS where
/// the items render as disabled and cannot be selected — a long-standing
/// SwiftUI bug that bit us once the tab count crossed ~10.
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
    @ObservedObject var serverController: LocalServerController

    var body: some Scene {
        Settings {
            SettingsRoot(documents: documents,
                         projects: projects,
                         mcp: mcp,
                         mcpPool: mcpPool,
                         agents: agents,
                         service: service,
                         nativeTools: nativeTools,
                         permissions: permissions,
                         skills: skills,
                         themes: themes,
                         keybindings: keybindings,
                         serverController: serverController)
        }
    }
}

private enum SettingsCategory: String, CaseIterable, Identifiable, Hashable {
    case generation, loading, modelConfig, quantization
    case agents, tools, permissions, skills
    case theme, keybindings
    case documents, projects, mcp, apiKeys, server, storage

    var id: String { rawValue }

    var label: String {
        switch self {
        case .generation:  return "Generation"
        case .loading:     return "Loading"
        case .modelConfig: return "Model Config"
        case .quantization:return "Quantization"
        case .agents:      return "Agents"
        case .tools:       return "Tools"
        case .permissions: return "Permissions"
        case .skills:      return "Skills"
        case .theme:       return "Theme"
        case .keybindings: return "Keys"
        case .documents:   return "Documents"
        case .projects:    return "Projects"
        case .mcp:         return "MCP"
        case .apiKeys:     return "API Keys"
        case .server:      return "Server"
        case .storage:     return "Storage"
        }
    }

    var systemImage: String {
        switch self {
        case .generation:  return "slider.horizontal.3"
        case .loading:     return "tray.and.arrow.down"
        case .modelConfig: return "gearshape.2"
        case .quantization:return "rectangle.compress.vertical"
        case .agents:      return "person.2"
        case .tools:       return "wrench.and.screwdriver"
        case .permissions: return "checkmark.shield"
        case .skills:      return "sparkles"
        case .theme:       return "paintbrush"
        case .keybindings: return "keyboard"
        case .documents:   return "doc.text"
        case .projects:    return "folder"
        case .mcp:         return "server.rack"
        case .apiKeys:     return "key"
        case .server:      return "network"
        case .storage:     return "externaldrive"
        }
    }
}

private struct SettingsRoot: View {
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
    @ObservedObject var serverController: LocalServerController

    @State private var selection: SettingsCategory = .generation

    var body: some View {
        NavigationSplitView {
            List(SettingsCategory.allCases, selection: $selection) { cat in
                Label(cat.label, systemImage: cat.systemImage)
                    .tag(cat)
            }
            .navigationSplitViewColumnWidth(min: 180, ideal: 200, max: 240)
        } detail: {
            detail
                .navigationTitle(selection.label)
                .frame(minWidth: 560, minHeight: 520)
        }
        .frame(minWidth: 820, minHeight: 580)
    }

    @ViewBuilder
    private var detail: some View {
        switch selection {
        case .generation:   GenerationSettingsTab()
        case .loading:      LoadingSettingsTab()
        case .modelConfig:  ModelConfigSettingsTab()
        case .quantization: QuantizationSettingsTab()
        case .agents:       AgentsView(library: agents, mcpPool: mcpPool)
        case .tools:        ToolsSettingsTab(host: nativeTools)
        case .permissions:  PermissionsSettingsTab(host: nativeTools, store: permissions)
        case .skills:       SkillsSettingsTab(library: skills)
        case .theme:        ThemeSettingsTab(store: themes)
        case .keybindings:  KeybindingsSettingsTab(store: keybindings)
        case .documents:    DocumentsView(library: documents, service: service)
        case .projects:     ProjectsView(library: projects,
                                         documents: documents,
                                         service: service)
        case .mcp:          MCPServersView(library: mcp, pool: mcpPool)
        case .apiKeys:      APIKeysSettingsTab()
        case .server:       ServerSettingsTab(controller: serverController)
        case .storage:      StorageSettingsTab()
        }
    }
}
