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

    var body: some View {
        ChatContainer(
            store: ChatStore(service: service,
                              documents: documents,
                              projects: projects,
                              mcpPool: mcpPool,
                              agents: agents,
                              modelState: modelState),
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
                                     store: store,
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
    @ObservedObject var store: ChatStore
    @ObservedObject var library: ModelLibrary
    @ObservedObject var catalog: OpenRouterCatalog

    @State private var showAddOpenRouter: Bool = false
    @State private var showLocalModelSettings: Bool = false

    /// Endpoint mostrato in toolbar = quello della chat selezionata
    /// (post-refactor multi-endpoint). Permette di vedere a colpo
    /// d'occhio "questa chat parla a X" anche se in background
    /// un'altra chat sta usando un endpoint diverso. Fallback su
    /// `modelState.loadedEndpoint` per chat senza endpoint
    /// proprio (pre-migration) o quando nessuna chat è selezionata.
    private var currentEndpoint: ModelEndpoint? {
        if let id = store.selectedID {
            return store.endpoint(of: id)
        }
        return modelState.loadedEndpoint
    }

    var body: some View {
        Menu {
            Section("This chat") {
                if let ep = currentEndpoint {
                    Label(ep.displayName, systemImage: ep.iconName)
                } else {
                    Label("No model bound", systemImage: "circle")
                }
                // Stato del local model loader (rilevante solo se la
                // chat usa un local endpoint o se l'utente sta per
                // selezionarne uno).
                switch modelState.status {
                case .idle:
                    EmptyView()
                case .loading(let ep, _):
                    Label("Loading \(ep.displayName)…", systemImage: "arrow.down.circle")
                case .loaded:
                    EmptyView()
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
                            bind(entry.endpoint)
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
                    showLocalModelSettings = true
                } label: {
                    Label("Customize local model settings…",
                           systemImage: "slider.horizontal.3")
                }
                if modelState.isReady,
                   case .localDirectory = modelState.loadedEndpoint {
                    Button(role: .destructive) {
                        Task { await modelState.unload() }
                    } label: {
                        Label("Unload local model from RAM",
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
                                     modelState: modelState,
                                     store: store)
        }
        .sheet(isPresented: $showLocalModelSettings) {
            LocalModelSettingsSheet(modelState: modelState)
        }
    }

    @ViewBuilder
    private var label: some View {
        // La label di toolbar segue la chat selezionata, non il
        // global state — così se chat A è locale e chat B è remota,
        // la toolbar cambia quando l'utente sposta la selezione.
        if case .loading(let ep, _) = modelState.status {
            Label("Loading \(ep.displayName)",
                   systemImage: "arrow.down.circle")
        } else if case .error(let ep, _) = modelState.status,
                  currentEndpoint == ep {
            Label("Failed: \(ep.displayName)",
                   systemImage: "exclamationmark.octagon.fill")
        } else if let ep = currentEndpoint {
            Label(ep.displayName, systemImage: ep.iconName)
        } else {
            Label("No model", systemImage: "cpu")
        }
    }

    private var helpText: String {
        if case .loading(let ep, _) = modelState.status {
            return "Loading \(ep.subtitle)…"
        }
        if case .error(_, let msg) = modelState.status {
            return "Load failed: \(msg)"
        }
        if let ep = currentEndpoint {
            return ep.subtitle
        }
        return "No model bound to this chat — pick one to start"
    }

    /// Cuore del refactor multi-endpoint: l'utente sceglie un
    /// endpoint dal picker e questa funzione lo lega alla CHAT
    /// CORRENTE (non globalmente). Per local: carichiamo il modello
    /// nel service se non è già quello — il modello locale è
    /// singleton in RAM, condiviso fra tutte le chat con endpoint
    /// local. Per remote: validiamo la chiave OpenRouter via
    /// `modelState.load` ma NON tocchiamo il local loaded. Le chat
    /// in volo restano sul loro endpoint.
    private func bind(_ endpoint: ModelEndpoint) {
        if let id = store.selectedID {
            store.setEndpoint(endpoint, for: id)
        }
        switch endpoint {
        case .localDirectory(let path):
            // Skip load se il service ha già *questo* local model in
            // RAM (la verifica corretta è sul `currentModelDir`, non
            // sullo `loadedEndpoint` di ModelState che post-refactor
            // riflette l'ultimo endpoint *scelto*, non quello fisico).
            // Evita un unload+reload spurio se l'utente seleziona di
            // nuovo lo stesso local — operazione che ucciderebbe
            // ogni altra chat local in volo.
            if store.service.currentModelDir()?.path != path {
                Task { await modelState.load(endpoint) }
            }
        case .openRouter:
            // Remote: il load valida la chiave, non scarica il local.
            // Sicuro chiamarlo a ogni bind (idempotente lato server).
            Task { await modelState.load(endpoint) }
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
            bind(.localDirectory(path: url.path))
        }
        #endif
    }
}
