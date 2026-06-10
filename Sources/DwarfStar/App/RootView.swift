import SwiftUI

enum AppSection: String, CaseIterable, Identifiable {
    case chat = "Chat"
    case tuning = "Tuning"
    case server = "Server"
    case benchmark = "Benchmark"
    case diagnostics = "Diagnostica"

    var id: String { rawValue }
    var icon: String {
        switch self {
        case .chat: return "bubble.left.and.bubble.right"
        case .tuning: return "slider.horizontal.3"
        case .server: return "server.rack"
        case .benchmark: return "gauge.with.dots.needle.67percent"
        case .diagnostics: return "stethoscope"
        }
    }
}

/// App shell: a sidebar selects between chat, server/agent, benchmark and
/// diagnostics panels.
struct RootView: View {
    @Bindable var store: ChatStore
    @State private var server = ServerController()
    @State private var bench = BenchController()
    @State private var diagnostics = DiagnosticsController()
    @State private var selection: AppSection? = .chat

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
                ContentView(store: store)
            case .tuning:
                TuningView(store: store)
            case .server:
                ServerView(controller: server, modelLoadedInProcess: store.isReady)
            case .benchmark:
                BenchView(controller: bench)
            case .diagnostics:
                DiagnosticsView(controller: diagnostics)
            }
        }
    }
}
