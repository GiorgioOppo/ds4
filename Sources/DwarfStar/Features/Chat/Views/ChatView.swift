import SwiftUI
import DS4Engine
import DS4Core

struct ChatView: View {
    @Bindable var store: ChatStore
    @State private var showTools = false
    @State private var showChats = false
    @State private var projects: [ProjectLibrary.SavedProject] = []
    @State private var activeProjectName: String?
    @State private var lastAutoScroll = Date.distantPast

    var body: some View {
        VStack(spacing: 0) {
            header
            Divider()
            transcript
            Divider()
            composer
        }
        .sheet(isPresented: $showTools) { ToolPickerView(store: store) }
        .sheet(isPresented: $store.awaitingManualResults) {
            ManualToolResultsView(store: store)
                .interactiveDismissDisabled()
        }
    }

    private var header: some View {
        HStack(spacing: 12) {
            VStack(alignment: .leading, spacing: 2) {
                Text(store.info?.displayName
                     ?? store.inspectedModelDescriptor?.displayName
                     ?? "Nessun modello caricato")
                    .font(.headline)
                if let info = store.info {
                    Text("\(info.architecture.rawValue) · \(info.layers) layer · \(info.quantizationSummary) · ctx \(info.contextSize) · KV ~\(kvSize(info.kvCacheBytes))")
                        .font(.caption)
                        .foregroundStyle(.secondary)
                } else if let descriptor = store.inspectedModelDescriptor {
                    Text("\(descriptor.architecture.rawValue) · backend \(descriptor.backendAvailability.rawValue)")
                        .font(.caption)
                        .foregroundStyle(.secondary)
                }
            }
            Spacer()
            projectMenu
            Picker("Agent", selection: Binding(get: { store.selectedAgentId },
                                                set: { store.selectAgent($0) })) {
                ForEach(store.agents) { agent in
                    Label(agent.name, systemImage: agent.icon).tag(agent.id)
                }
            }
            .pickerStyle(.menu)
            .fixedSize()
            .help("Change role: starts a new chat with the agent's system prompt and tools; the expert cache warms from that agent's usage profile.")
            if store.modelCapabilities.contains(.tools) {
                Button {
                    showTools = true
                } label: {
                    Label(toolButtonTitle, systemImage: "wrench.and.screwdriver")
                }
            }
            temperatureMenu
            if store.supportsReasoning {
                Toggle("Thinking", isOn: $store.think)
                    .toggleStyle(.switch)
            }
            Button {
                showChats = true
            } label: {
                Label("Chat", systemImage: "bubble.left.and.bubble.right")
            }
            .popover(isPresented: $showChats, arrowEdge: .bottom) {
                ChatListView(store: store)
            }
            .help("Apri, rinomina o elimina le conversazioni salvate")
            Button {
                store.newChat()
            } label: {
                Label("New Chat", systemImage: "square.and.pencil")
            }
        }
        .padding(.horizontal)
        .padding(.vertical, 10)
    }

    /// Temperature control: lower = più focalizzato e meno deriva (utile sui
    /// modelli molto quantizzati); più alto = più creativo/variabile.
    private var temperatureMenu: some View {
        Menu {
            VStack(alignment: .leading) {
                Text("Temperature: \(store.temperature, format: .number.precision(.fractionLength(2)))")
                    .font(.caption).foregroundStyle(.secondary)
                Slider(value: $store.temperature, in: 0...1.5, step: 0.05)
                    .frame(width: 220)
                Text("Low = more focused, less drift. High = more creative. 0 = greedy, deterministic like the demo.")
                    .font(.caption2).foregroundStyle(.secondary)
                HStack {
                    Button("Greedy (0)") { store.temperature = 0 }
                    Button("Precise (0.3)") { store.temperature = 0.3 }
                    Button("Default (0.6)") { store.temperature = 0.6 }
                }
                .buttonStyle(.borderless).font(.caption)

                Divider()
                Text("Repetition penalty: \(store.repetitionPenalty, format: .number.precision(.fractionLength(2)))")
                    .font(.caption).foregroundStyle(.secondary)
                Slider(value: $store.repetitionPenalty, in: 1.0...1.5, step: 0.05)
                    .frame(width: 220)
                Text("Raise it (1.15-1.3) if the model starts repeating after many tokens.")
                    .font(.caption2).foregroundStyle(.secondary)
            }
            .padding(8)
        } label: {
            Label(store.temperature.formatted(.number.precision(.fractionLength(1))),
                  systemImage: "thermometer.medium")
        }
        .menuStyle(.borderlessButton)
        .fixedSize()
        .help("Sampling temperature: lower it (0.3-0.4) if the model drifts or repeats.")
    }

    private var toolButtonTitle: String {
        guard store.toolsEnabled else { return "Tool" }
        return "Tool (\(store.enabledToolNames.count))"
    }

    /// Import/switch the active project right from the chat: the agent's
    /// project_* tools read the active one; the chat memory is untouched.
    private var projectMenu: some View {
        Menu {
            if projects.isEmpty {
                Text("No saved projects")
            }
            ForEach(projects) { p in
                Button {
                    if ProjectLibrary.activate(p) != nil { refreshProject() }
                } label: {
                    if p.id == ProjectLibrary.activeId {
                        Label(p.name, systemImage: "checkmark")
                    } else {
                        Text(p.name)
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
                Label("Import Folder...", systemImage: "folder.badge.plus")
            }
        } label: {
            Label(activeProjectName ?? "Project", systemImage: "folder")
        }
        .fixedSize()
        .help("Active project for the agent's project_* tools. Importing does not touch chat memory.")
        .onAppear { refreshProject() }
    }

    private func refreshProject() {
        ProjectLibrary.syncClonedRepos()   // repos cloned via github_clone appear too
        projects = ProjectLibrary.all()
        activeProjectName = ProjectCache.shared.info()?.name
    }

    private func kvSize(_ bytes: UInt64) -> String {
        ByteCountFormatter.string(fromByteCount: Int64(bytes), countStyle: .memory)
    }

    private var transcript: some View {
        ScrollViewReader { proxy in
            ScrollView {
                LazyVStack(alignment: .leading, spacing: 14) {
                    ForEach(store.messages) { message in
                        MessageRow(message: message)
                            .id(message.id)
                    }
                }
                .padding()
            }
            .onChange(of: store.messages.last.map { $0.reasoning.count + $0.text.count + $0.toolStreamText.count }) {
                if let last = store.messages.last {
                    let now = Date()
                    guard !store.isGenerating || now.timeIntervalSince(lastAutoScroll) >= 0.20 else {
                        return
                    }
                    lastAutoScroll = now

                    // Live deltas must not accumulate overlapping SwiftUI
                    // animations. Apart from wasting main-thread time, an
                    // animation can retain the ScrollViewProxy while this view
                    // is being removed after a sidebar navigation.
                    if store.isGenerating {
                        proxy.scrollTo(last.id, anchor: .bottom)
                    } else {
                        withAnimation { proxy.scrollTo(last.id, anchor: .bottom) }
                    }
                }
            }
        }
    }

    private var composer: some View {
        VStack(alignment: .leading, spacing: 6) {
        if store.isGenerating && !store.status.isEmpty {
            HStack(spacing: 6) {
                ProgressView().controlSize(.mini)
                Text(store.status)
                    .font(.caption.monospacedDigit())
                    .foregroundStyle(.secondary)
            }
        }
        if !store.attachments.isEmpty {
            ScrollView(.horizontal, showsIndicators: false) {
                HStack(spacing: 6) {
                    ForEach(store.attachments) { att in
                        AttachmentChip(name: att.name, bytes: att.bytes) {
                            store.removeAttachment(att.id)
                        }
                    }
                }
            }
        }
        if let note = store.attachmentNote {
            Label(note, systemImage: "exclamationmark.triangle")
                .font(.caption2).foregroundStyle(.orange)
        }
        if let est = store.attachmentTokenEstimate, est > store.contextSize - 256 {
            Label("Attachments are ~\(est) tokens and may exceed the context (\(store.contextSize)). Reduce files or increase context in Settings.",
                  systemImage: "exclamationmark.triangle")
                .font(.caption2).foregroundStyle(.orange)
        }
        if store.contextUsed > 0, store.contextUsed * 100 >= store.contextSize * 85 {
            Label("Context nearly full: \(store.contextUsed)/\(store.contextSize) tokens. Responses may soon be truncated: start a new chat or increase context in Settings.",
                  systemImage: "exclamationmark.triangle.fill")
                .font(.caption2).foregroundStyle(.orange)
        }
        HStack(alignment: .bottom, spacing: 8) {
            Button { store.pickAndAttachFiles() } label: {
                Image(systemName: "paperclip")
            }
            .buttonStyle(.borderless)
            .help("Import text files into the conversation")
            .disabled(store.isGenerating)
            TextField("Write a message...", text: $store.input, axis: .vertical)
                .textFieldStyle(.roundedBorder)
                .lineLimit(1...6)
                .onSubmit { store.send() }
            if store.isGenerating {
                Button(role: .destructive) { store.stop() } label: {
                    Image(systemName: "stop.fill")
                }
            } else {
                Button { store.send() } label: {
                    Image(systemName: "arrow.up.circle.fill")
                }
                .disabled(store.input.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty
                          && store.attachments.isEmpty)
            }
        }
        }
        .padding(.horizontal)
        .padding(.vertical, 10)
    }
}
