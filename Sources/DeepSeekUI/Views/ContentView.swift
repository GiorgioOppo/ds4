import SwiftUI
import DeepSeekKit
#if canImport(AppKit)
import AppKit
#endif

/// Top-level view. Always renders the chat container — the model
/// load step that used to gate this view is now an in-chat
/// affordance (toolbar picker + status banner) so the user can
/// browse history, edit agents/projects, and queue draft messages
/// before a model is even loaded.
struct ContentView: View {
    let service: InferenceService
    @ObservedObject var documents: DocumentLibrary
    @ObservedObject var projects: ProjectLibrary
    @ObservedObject var mcpPool: MCPClientPool
    @ObservedObject var agents: AgentLibrary
    @ObservedObject var modelLibrary: ModelLibrary
    @ObservedObject var modelState: ModelState
    @ObservedObject var openRouterCatalog: OpenRouterCatalog
    @ObservedObject var nativeTools: NativeToolHost

    var body: some View {
        ChatContainer(
            store: ChatStore(service: service,
                              documents: documents,
                              projects: projects,
                              mcpPool: mcpPool,
                              agents: agents,
                              modelState: modelState,
                              nativeTools: nativeTools),
            projects: projects,
            agents: agents,
            modelLibrary: modelLibrary,
            modelState: modelState,
            openRouterCatalog: openRouterCatalog)
    }
}

/// Hosts the `ChatStore` for the app's lifetime + lays out the
/// NavigationSplitView (sidebar + detail). Toolbar items: model
/// picker, agent picker, project picker, convert sheet trigger.
struct ChatContainer: View {
    @StateObject var store: ChatStore
    @ObservedObject var projects: ProjectLibrary
    @ObservedObject var agents: AgentLibrary
    @ObservedObject var modelLibrary: ModelLibrary
    @ObservedObject var modelState: ModelState
    @ObservedObject var openRouterCatalog: OpenRouterCatalog

    @State private var showConvert: Bool = false
    @State private var showFineTune: Bool = false
    @State private var showVocabPruner: Bool = false
    @State private var showPlayground: Bool = false

    var body: some View {
        NavigationSplitView {
            ConversationListView(store: store)
                .frame(minWidth: 200)
        } detail: {
            ChatView(store: store, modelState: modelState)
                .toolbar {
                    ToolbarItem(placement: .navigation) {
                        ModelPicker(modelState: modelState,
                                     library: modelLibrary,
                                     catalog: openRouterCatalog)
                    }
                    ToolbarItem(placement: .navigation) {
                        agentPicker
                    }
                    ToolbarItem(placement: .navigation) {
                        projectPicker
                    }
                    ToolbarItem {
                        Button {
                            showConvert = true
                        } label: {
                            Label("Convert model…", systemImage: "wand.and.stars")
                        }
                    }
                    ToolbarItem {
                        Button {
                            showFineTune = true
                        } label: {
                            Label("Fine-tune model…", systemImage: "graduationcap")
                        }
                    }
                    ToolbarItem {
                        Button {
                            showVocabPruner = true
                        } label: {
                            Label("Prune vocab…", systemImage: "scissors")
                        }
                    }
                    ToolbarItem {
                        Button {
                            showPlayground = true
                        } label: {
                            Label("OOP playground", systemImage: "puzzlepiece.extension")
                        }
                    }
                }
        }
        .sheet(isPresented: $showConvert) {
            ConvertSheet()
        }
        .sheet(isPresented: $showFineTune) {
            FineTuneSheet()
        }
        .sheet(isPresented: $showVocabPruner) {
            VocabPrunerSheet()
        }
        .sheet(isPresented: $showPlayground) {
            PlaygroundSheet()
        }
    }

    @ViewBuilder
    private var agentPicker: some View {
        if let c = store.selectedConversation {
            let attached = c.agentID.flatMap { agents.agent(id: $0) }
            Menu {
                Button {
                    store.setAgent(nil, for: c.id)
                } label: {
                    if attached == nil {
                        Label("None", systemImage: "checkmark")
                    } else {
                        Text("None")
                    }
                }
                if !agents.agents.isEmpty { Divider() }
                ForEach(agents.agents) { a in
                    Button {
                        store.setAgent(a.id, for: c.id)
                    } label: {
                        if attached?.id == a.id {
                            Label(a.name, systemImage: "checkmark")
                        } else {
                            Text(a.name)
                        }
                    }
                }
            } label: {
                Label(attached?.name ?? "No agent",
                       systemImage: attached?.iconName ?? "person.crop.circle")
            }
            .help(attached == nil
                   ? "Attach an agent from Settings → Agents"
                   : "Agent: \(attached!.name)")
        } else {
            EmptyView()
        }
    }

    @ViewBuilder
    private var projectPicker: some View {
        if let c = store.selectedConversation {
            let attached = c.projectID.flatMap { projects.project(id: $0) }
            Menu {
                Button {
                    store.setProject(nil, for: c.id)
                } label: {
                    if attached == nil {
                        Label("None", systemImage: "checkmark")
                    } else {
                        Text("None")
                    }
                }
                if !projects.projects.isEmpty {
                    Divider()
                }
                ForEach(projects.projects) { p in
                    Button {
                        store.setProject(p.id, for: c.id)
                    } label: {
                        if attached?.id == p.id {
                            Label(p.name, systemImage: "checkmark")
                        } else {
                            Text(p.name)
                        }
                    }
                }
            } label: {
                Label(attached?.name ?? "No project",
                       systemImage: attached == nil ? "folder" : "folder.fill")
            }
            .help(attached == nil
                   ? "Attach a project from Settings → Projects"
                   : "Project attached: \(attached!.name)")
        } else {
            EmptyView()
        }
    }
}

/// Toolbar menu that drives `ModelState` from inside the chat.
/// Shows the currently-loaded model (or "Loading…" / "No model")
/// as its label, lists recents from `ModelLibrary` for one-click
/// switching, and exposes the NSOpenPanel-backed Browse action +
/// an Unload affordance when a model is loaded.
private struct ModelPicker: View {
    @ObservedObject var modelState: ModelState
    @ObservedObject var library: ModelLibrary
    @ObservedObject var catalog: OpenRouterCatalog

    @State private var showAddOpenRouter: Bool = false
    @State private var showAddAnthropic: Bool = false

    var body: some View {
        Menu {
            Section("Current") {
                switch modelState.status {
                case .idle:
                    Label("No model loaded", systemImage: "circle")
                case .loading(let ep, _):
                    Label("Loading \(ep.displayName)…", systemImage: "arrow.down.circle")
                case .loaded(let ep, _):
                    Label(ep.displayName, systemImage: "checkmark.circle")
                case .error(let ep, _):
                    Label("Failed: \(ep.displayName)",
                           systemImage: "exclamationmark.octagon")
                }
            }
            let recents = library.recents()
            if !recents.isEmpty {
                Section("Recent") {
                    ForEach(recents) { entry in
                        Button {
                            Task { await modelState.load(entry.endpoint) }
                        } label: {
                            HStack {
                                Image(systemName: entry.endpoint.iconName)
                                Text(entry.name)
                            }
                        }
                    }
                    Divider()
                    Menu("Forget…") {
                        ForEach(recents) { entry in
                            Button(entry.name) {
                                library.forget(entry.endpoint)
                            }
                        }
                    }
                }
            }
            Section {
                Button {
                    browse()
                } label: {
                    Label("Choose model folder…",
                           systemImage: "folder.badge.plus")
                }
                Button {
                    showAddOpenRouter = true
                } label: {
                    Label("Add OpenRouter model…", systemImage: "cloud")
                }
                Button {
                    showAddAnthropic = true
                } label: {
                    Label("Add Anthropic model…", systemImage: "cloud.fill")
                }
                if modelState.isReady {
                    Button(role: .destructive) {
                        Task { await modelState.unload() }
                    } label: {
                        Label("Unload current model",
                               systemImage: "eject")
                    }
                }
                if case .error = modelState.status {
                    Button {
                        Task { await modelState.retryWithForce() }
                    } label: {
                        Label("Retry with Force Load",
                               systemImage: "arrow.clockwise.heavy")
                    }
                }
            }
        } label: {
            label
        }
        .help(helpText)
        .sheet(isPresented: $showAddOpenRouter) {
            AddOpenRouterModelSheet(catalog: catalog,
                                     modelState: modelState)
        }
        .sheet(isPresented: $showAddAnthropic) {
            AddAnthropicModelSheet(modelState: modelState)
        }
    }

    @ViewBuilder
    private var label: some View {
        switch modelState.status {
        case .idle:
            Label("No model", systemImage: "cpu")
        case .loading(let ep, _):
            Label("Loading \(ep.displayName)",
                   systemImage: "arrow.down.circle")
        case .loaded(let ep, _):
            Label(ep.displayName, systemImage: ep.iconName)
        case .error(let ep, _):
            Label("Failed: \(ep.displayName)",
                   systemImage: "exclamationmark.octagon.fill")
        }
    }

    private var helpText: String {
        switch modelState.status {
        case .idle:
            return "No model loaded — choose one to start chatting"
        case .loading(let ep, _):
            return "Loading \(ep.subtitle)…"
        case .loaded(let ep, _):
            return ep.subtitle
        case .error(_, let msg):
            return "Load failed: \(msg)"
        }
    }

    private func browse() {
        #if canImport(AppKit)
        let panel = NSOpenPanel()
        panel.canChooseFiles = false
        panel.canChooseDirectories = true
        panel.allowsMultipleSelection = false
        panel.prompt = "Choose"
        panel.message = "Select the converted DeepSeek model directory."
        if panel.runModal() == .OK, let url = panel.url {
            Task { await modelState.load(.localDirectory(path: url.path)) }
        }
        #endif
    }
}
