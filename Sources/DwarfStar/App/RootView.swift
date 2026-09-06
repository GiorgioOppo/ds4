import SwiftUI

enum AppSection: String, CaseIterable, Identifiable {
    case chat = "Chat"
    case settings = "Settings"
    case agents = "Agents"
    case mcp = "MCP"
    case project = "Project"
    case tuning = "Tuning"
    case server = "Server"
    case distributed = "Worker"
    case benchmark = "Benchmark"
    case miniToolBench = "Mini Tool Bench"
    case conversion = "Conversione"
    case diagnostics = "Diagnostics"

    var id: String { rawValue }

    var title: String {
        switch self {
        case .chat: return "Chat"
        case .settings: return "Modelli e impostazioni"
        case .agents: return "Agenti"
        case .mcp: return "Connessioni MCP"
        case .project: return "Progetti"
        case .tuning: return "Ottimizzazione"
        case .server: return "Server API"
        case .distributed: return "Worker distribuito"
        case .benchmark: return "Benchmark"
        case .miniToolBench: return "Mini Tool Bench"
        case .conversion: return "Conversione modelli"
        case .diagnostics: return "Diagnostica"
        }
    }

    static let workspace: [AppSection] = [.chat, .project, .agents]
    static let configuration: [AppSection] = [.settings, .mcp]
    static let advanced: [AppSection] = [.tuning, .server, .distributed, .benchmark,
                                         .miniToolBench, .conversion, .diagnostics]
    var icon: String {
        switch self {
        case .chat: return "bubble.left.and.bubble.right"
        case .settings: return "gearshape"
        case .agents: return "person.2"
        case .mcp: return "puzzlepiece.extension"
        case .project: return "folder"
        case .tuning: return "slider.horizontal.3"
        case .server: return "server.rack"
        case .distributed: return "cpu"
        case .benchmark: return "gauge.with.dots.needle.67percent"
        case .miniToolBench: return "terminal"
        case .conversion: return "arrow.triangle.2.circlepath"
        case .diagnostics: return "stethoscope"
        }
    }
}

/// App shell: a sidebar selects the panel. The model + engine mode are set once
/// in Impostazioni (AppSettings) and inherited by every panel's controller.
struct RootView: View {
    @Bindable var store: ChatStore
    let settings: AppSettings
    let mcp: MCPStore
    @State private var distributed: DistributedController
    @State private var server: ServerController
    @State private var bench: BenchController
    @State private var miniToolBench: MiniToolBenchController
    @State private var diagnostics: DiagnosticsController
    @State private var conversion = SafetensorsConversionController()
    @State private var selection: AppSection? = .chat

    init(store: ChatStore, settings: AppSettings, mcp: MCPStore) {
        self.store = store
        self.settings = settings
        self.mcp = mcp
        let distributed = DistributedController(settings: settings)
        let server = ServerController(settings: settings, store: store)
        _distributed = State(initialValue: distributed)
        _server = State(initialValue: server)
        _miniToolBench = State(initialValue: MiniToolBenchController(store: store, server: server))
        _bench = State(initialValue: BenchController(settings: settings, dist: distributed, store: store))
        _diagnostics = State(initialValue: DiagnosticsController(settings: settings))
    }

    var body: some View {
        NavigationSplitView {
            List(selection: $selection) {
                Section("Spazio di lavoro") {
                    navigationRows(AppSection.workspace)
                }
                Section("Configurazione") {
                    navigationRows(AppSection.configuration)
                }
                Section("Strumenti avanzati") {
                    navigationRows(AppSection.advanced)
                }
            }
            .listStyle(.sidebar)
            .navigationSplitViewColumnWidth(min: 190, ideal: 220, max: 260)
            .navigationTitle("DwarfStar")
            .safeAreaInset(edge: .bottom, spacing: 0) {
                Button { selection = .settings } label: {
                    HStack(spacing: 8) {
                        Image(systemName: engineStatus.icon)
                            .foregroundStyle(engineStatus.color)
                        VStack(alignment: .leading, spacing: 2) {
                            Text(engineStatus.title)
                                .font(.caption.weight(.medium))
                            Text(settings.mode == .local ? "Esecuzione locale" : "Esecuzione distribuita")
                                .font(.caption2)
                                .foregroundStyle(.secondary)
                        }
                        Spacer(minLength: 0)
                        Image(systemName: "chevron.right").font(.caption2)
                            .foregroundStyle(.secondary)
                    }
                    .padding(12)
                    .contentShape(Rectangle())
                }
                .buttonStyle(.plain)
                .background(.bar)
                .help("Apri la configurazione del modello e del motore")
            }
        } detail: {
            VStack(spacing: 0) {
                if store.benchRunning {
                    HStack(spacing: 10) {
                        ProgressView()
                            .controlSize(.small)
                        Text(store.benchStatus ?? "Operazione esclusiva sul motore in corso…")
                            .font(.caption)
                            .lineLimit(1)
                        Spacer()
                        Button("Apri impostazioni") { selection = .settings }
                            .font(.caption)
                        Button("Interrompi", role: .destructive) { store.cancelAutoTune() }
                            .font(.caption)
                    }
                    .padding(.horizontal, 12)
                    .padding(.vertical, 7)
                    .background(.orange.opacity(0.13))
                    Divider()
                }

                Group {
                    switch selection ?? .chat {
                    case .chat:
                        ChatTabView(store: store, dist: distributed, settings: settings,
                                    openSettings: { selection = .settings })
                    case .settings:
                        SettingsView(settings: settings, store: store, dist: distributed)
                    case .agents:
                        AgentsView(store: store)
                    case .mcp:
                        MCPServersView(store: mcp)
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
                    case .miniToolBench:
                        MiniToolBenchView(controller: miniToolBench,
                                          openServer: { selection = .server })
                    case .conversion:
                        SafetensorsConversionView(controller: conversion)
                    case .diagnostics:
                        DiagnosticsView(controller: diagnostics)
                    }
                }
                // Navigation remains available, but no other screen may retain
                // or mutate the single engine while the tuner swaps services.
                .allowsHitTesting(!store.benchRunning || selection == .settings || selection == .miniToolBench)
            }
        }
    }

    private func navigationRows(_ sections: [AppSection]) -> some View {
        ForEach(sections) { section in
            Label(section.title, systemImage: section.icon)
                .lineLimit(1)
                .help(section.title)
                .tag(section)
        }
    }

    private var engineStatus: (title: String, icon: String, color: Color) {
        if settings.mode == .distributed {
            return ("Motore distribuito", "network", .secondary)
        }
        switch store.phase {
        case .ready:
            return ("Modello pronto", "checkmark.circle.fill", .green)
        case .loading:
            return ("Caricamento in corso…", "hourglass", .accentColor)
        case .failed:
            return ("Controlla il modello", "exclamationmark.triangle", .orange)
        case .needsModel:
            return ("Configura il modello", "shippingbox", .secondary)
        }
    }
}
