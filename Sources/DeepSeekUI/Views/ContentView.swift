import SwiftUI
import DeepSeekKit

enum AppPhase {
    case picking
    case loading(URL)
    case ready(URL, ModelConfig)
}

/// Top-level view. Receives the shared `InferenceService` and
/// `ProjectLibrary` from the App scene and routes between picker →
/// loading → ready. The project library is threaded down so the chat
/// surface can show / pick the active project.
struct ContentView: View {
    let service: InferenceService
    @ObservedObject var projects: ProjectLibrary
    @State private var phase: AppPhase = .picking

    var body: some View {
        Group {
            switch phase {
            case .picking:
                ModelPickerView { url in
                    phase = .loading(url)
                }
                .onAppear {
                    // Auto-resume the previously loaded model if reachable.
                    if let last = AppSettings.lastModelDir,
                       FileManager.default.fileExists(atPath: last) {
                        phase = .loading(URL(fileURLWithPath: last))
                    }
                }

            case .loading(let url):
                LoadingView(modelDir: url,
                            service: service,
                            onLoaded: { cfg in phase = .ready(url, cfg) },
                            onCancel:  { phase = .picking })

            case .ready(let url, _):
                ChatContainer(
                    store: ChatStore(modelDirPath: url.path,
                                      service: service),
                    projects: projects,
                    onUnload: { phase = .picking })
            }
        }
    }
}

/// Hosts the `ChatStore` for the lifetime of the .ready phase and
/// lays out the NavigationSplitView (sidebar + detail). Toolbar
/// items: convert model, unload, and a project picker that lets the
/// active conversation reference a `Project` from the library.
private struct ChatContainer: View {
    @StateObject var store: ChatStore
    @ObservedObject var projects: ProjectLibrary
    var onUnload: () -> Void

    @State private var showConvert: Bool = false

    var body: some View {
        NavigationSplitView {
            ConversationListView(store: store)
                .frame(minWidth: 200)
        } detail: {
            ChatView(store: store)
                .toolbar {
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
                    ToolbarItem(placement: .destructiveAction) {
                        Button {
                            onUnload()
                        } label: {
                            Label("Unload model", systemImage: "eject")
                        }
                    }
                }
        }
        .sheet(isPresented: $showConvert) {
            ConvertSheet()
        }
    }

    @ViewBuilder
    private var projectPicker: some View {
        // Disabled when there is no active conversation; the menu is
        // a no-op without a target. The label reads the currently-
        // attached project (or "No project") so the user can tell at
        // a glance which context the chat is wired to.
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
