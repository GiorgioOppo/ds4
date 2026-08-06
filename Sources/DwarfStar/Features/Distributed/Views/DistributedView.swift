import SwiftUI
import DS4Engine

/// This Mac as a distributed WORKER: owns a layer slice and listens for the
/// coordinator (the Distribuito sidebar tab). The coordinator lives in the Chat
/// tab → Distribuito mode.
struct WorkerView: View {
    @Bindable var controller: DistributedController

    var body: some View {
        VStack(spacing: 0) {
            Form {
                Section {
                    Label("This Mac as a worker: it listens for the coordinator, which sends EVERYTHING — the GGUF, the sidecar, the settings, and the layer slice. The first connect transfers the files (it can take a while on a big model); afterwards SHA-256 manifests verify them instantly and nothing is re-sent.",
                          systemImage: "rectangle.3.group")
                        .font(.callout).foregroundStyle(.secondary)
                }
                Section("Local GGUF (optional, avoids the first transfer)") {
                    LabeledContent("GGUF", value: (controller.modelPath as NSString).lastPathComponent)
                    Text("If a local file matches the coordinator's SHA-256 (this GGUF from Settings or a same-named sibling), it is used directly and nothing is transferred. Received files land in Application Support/DwarfStar/dist-models.")
                        .font(.caption).foregroundStyle(.secondary)
                }

                Section("Worker - \(controller.modelLayersLabel) layer") {
                    TextField("Port", value: $controller.port, format: .number.grouping(.never))
                    Text("Layer slice, context, and cache budget are chosen by the coordinator and shown in the log when the assignment arrives.")
                        .font(.caption).foregroundStyle(.secondary)
                }
                .disabled(controller.workerRunning)

                Section {
                    HStack(spacing: 12) {
                        if controller.workerRunning {
                            Button(role: .destructive) { controller.stopWorker() } label: {
                                Label("Stop Worker", systemImage: "stop.fill")
                            }
                            Label(controller.workerSummary, systemImage: "dot.radiowaves.left.and.right")
                                .foregroundStyle(.green).font(.callout)
                        } else {
                            Button { controller.startWorker() } label: {
                                Label("Start Worker", systemImage: "play.fill")
                            }
                        }
                    }
                }
            }
            .formStyle(.grouped)

            DistLogView(text: controller.workerLog, height: 140)
        }
        .task(id: controller.modelPath) { await controller.refreshModelGeometry() }
    }
}

/// The coordinator: a chat that runs across the worker cluster, shown inside the
/// Chat tab when "Distribuito" is selected. Before connecting: a full-screen
/// setup form (like the local model-load screen). Once connected: the SAME
/// layout as the local chat — header (model name + route info, Thinking, Nuova
/// chat, Disconnetti), full-screen transcript, composer with live status.
struct CoordinatorChatView: View {
    @Bindable var controller: DistributedController
    @State private var projects: [ProjectLibrary.SavedProject] = []
    @State private var activeProjectName: String?

    /// The route setup lives in Impostazioni; this view is the chat itself.
    /// `openSettings` lets the not-connected placeholder jump there.
    var openSettings: () -> Void = {}

    var body: some View {
        if controller.connected {
            VStack(spacing: 0) {
                header
                Divider()
                transcript
                Divider()
                composer
            }
        } else {
            ContentUnavailableView {
                Label("Cluster Not Connected", systemImage: "rectangle.3.group")
            } description: {
                Text("Configure the model, workers, and route in Settings, then press Connect.")
            } actions: {
                Button("Open Settings") { openSettings() }
            }
        }
    }

    // MARK: Connected — local-chat look

    private var modelName: String { (controller.modelPath as NSString).lastPathComponent }

    private var header: some View {
        HStack(spacing: 12) {
            VStack(alignment: .leading, spacing: 2) {
                Text(modelName)
                    .font(.headline)
                Text("distributed · \(controller.modelLayersLabel) layers on \(controller.parsePeers().count) workers · ctx \(controller.contextSize) · \(controller.forwardEnabled ? "forwarding" : "relay")")
                    .font(.caption)
                    .foregroundStyle(.secondary)
            }
            Spacer()
            projectMenu
            Picker("Agent", selection: Binding(get: { controller.selectedAgentId },
                                                set: { controller.selectAgent($0) })) {
                ForEach(controller.agents) { agent in
                    Label(agent.name, systemImage: agent.icon).tag(agent.id)
                }
            }
            .pickerStyle(.menu)
            .fixedSize()
            .help("Distributed chat role: starts a new chat with the agent's system prompt and tools. Tools run on this Mac, the coordinator.")
            Toggle("Thinking", isOn: $controller.think)
                .toggleStyle(.switch)
            Button {
                controller.newChat()
            } label: {
                Label("New Chat", systemImage: "square.and.pencil")
            }
            Button(role: .destructive) {
                controller.disconnectCoordinator()
            } label: {
                Label("Disconnect", systemImage: "xmark.circle")
            }
        }
        .padding(.horizontal)
        .padding(.vertical, 10)
    }

    /// Same project menu as the local chat: the active project feeds the
    /// project_* tools, which run on this (coordinator) Mac.
    private var projectMenu: some View {
        Menu {
            if projects.isEmpty {
                Text("No saved projects")
            } else {
                ForEach(projects) { p in
                    Button {
                        ProjectLibrary.activate(p)
                        refreshProject()
                    } label: {
                        if p.name == activeProjectName { Label(p.name, systemImage: "checkmark") }
                        else { Text(p.name) }
                    }
                }
            }
            Divider()
            Button {
                if let p = ProjectLibrary.pickAndAdd() {
                    ProjectLibrary.activate(p)
                    refreshProject()
                }
            } label: {
                Label("Add New Project…", systemImage: "folder.badge.plus")
            }
        } label: {
            Label(activeProjectName ?? "Project", systemImage: "folder")
        }
        .fixedSize()
        .help("Passed to project-capable agents and used by project_* tools on the coordinator.")
        .onAppear { refreshProject() }
    }

    private func refreshProject() {
        ProjectLibrary.syncClonedRepos()   // repos cloned via github_clone appear too
        projects = ProjectLibrary.all()
        activeProjectName = ProjectCache.shared.info()?.name
    }

    private var transcript: some View {
        ScrollViewReader { proxy in
            ScrollView {
                LazyVStack(alignment: .leading, spacing: 14) {
                    ForEach(controller.messages) { MessageRow(message: $0).id($0.id) }
                }
                .padding()
            }
            .onChange(of: controller.messages.last.map { $0.text.count + $0.reasoning.count }) {
                if let last = controller.messages.last { withAnimation { proxy.scrollTo(last.id, anchor: .bottom) } }
            }
        }
    }

    private var composer: some View {
        VStack(alignment: .leading, spacing: 4) {
            if controller.isGenerating && !controller.status.isEmpty {
                HStack(spacing: 6) {
                    ProgressView().controlSize(.mini)
                    Text(controller.status)
                        .font(.caption.monospacedDigit())
                        .foregroundStyle(.secondary)
                }
            }
            HStack(alignment: .bottom, spacing: 8) {
                TextField("Write a message...", text: $controller.chatInput, axis: .vertical)
                    .textFieldStyle(.roundedBorder)
                    .lineLimit(1...6)
                    .onSubmit { controller.sendChat() }
                if controller.isGenerating {
                    Button(role: .destructive) { controller.stopGeneration() } label: {
                        Image(systemName: "stop.fill")
                    }
                } else {
                    Button { controller.sendChat() } label: {
                        Image(systemName: "arrow.up.circle.fill")
                    }
                    .disabled(controller.chatInput.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty)
                }
            }
        }
        .padding(10)
    }

}

/// Shared monospaced log strip (hidden when empty).
struct DistLogView: View {
    let text: String
    var height: CGFloat = 140
    var body: some View {
        if !text.isEmpty {
            Divider()
            ScrollView {
                Text(text)
                    .font(.system(.caption, design: .monospaced))
                    .frame(maxWidth: .infinity, alignment: .leading)
                    .textSelection(.enabled).padding(8)
            }
            .frame(height: height)
            .background(Color.black.opacity(0.05))
        }
    }
}
