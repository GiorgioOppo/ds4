import SwiftUI

enum AppSection: String, CaseIterable, Identifiable {
    case chat = "Chat"
    case settings = "Impostazioni"
    case agents = "Agenti"
    case project = "Progetto"
    case tuning = "Tuning"
    case server = "Server"
    case distributed = "Worker"
    case benchmark = "Benchmark"
    case diagnostics = "Diagnostica"

    var id: String { rawValue }
    var icon: String {
        switch self {
        case .chat: return "bubble.left.and.bubble.right"
        case .settings: return "gearshape"
        case .agents: return "person.2"
        case .project: return "folder"
        case .tuning: return "slider.horizontal.3"
        case .server: return "server.rack"
        case .distributed: return "cpu"
        case .benchmark: return "gauge.with.dots.needle.67percent"
        case .diagnostics: return "stethoscope"
        }
    }
}

/// App shell: a sidebar selects the panel. The model + engine mode are set once
/// in Impostazioni (AppSettings) and inherited by every panel's controller.
struct RootView: View {
    @Bindable var store: ChatStore
    let settings: AppSettings
    @State private var distributed: DistributedController
    @State private var server: ServerController
    @State private var bench: BenchController
    @State private var diagnostics: DiagnosticsController
    @State private var selection: AppSection? = .chat

    init(store: ChatStore, settings: AppSettings) {
        self.store = store
        self.settings = settings
        _distributed = State(initialValue: DistributedController(settings: settings))
        _server = State(initialValue: ServerController(settings: settings))
        _bench = State(initialValue: BenchController(settings: settings))
        _diagnostics = State(initialValue: DiagnosticsController(settings: settings))
    }

    var body: some View {
        NavigationSplitView {
            List(AppSection.allCases, selection: $selection) { section in
                Label(section.rawValue, systemImage: section.icon).tag(section)
            }
            .navigationSplitViewColumnWidth(min: 150, ideal: 170, max: 220)
            .navigationTitle("DwarfStar")
        } detail: {
            switch selection ?? .chat {
            case .chat:
                ChatTabView(store: store, dist: distributed, settings: settings,
                            openSettings: { selection = .settings })
            case .settings:
                SettingsView(settings: settings, store: store, dist: distributed)
            case .agents:
                AgentsView(store: store)
            case .project:
                ProjectView(store: store)
            case .tuning:
                TuningView(store: store)
            case .server:
                ServerView(controller: server, modelLoadedInProcess: store.isReady)
            case .distributed:
                WorkerView(controller: distributed)
            case .benchmark:
                BenchView(controller: bench)
            case .diagnostics:
                DiagnosticsView(controller: diagnostics)
            }
        }
    }
}
