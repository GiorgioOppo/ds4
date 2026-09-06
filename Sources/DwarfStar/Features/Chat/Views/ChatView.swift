import SwiftUI
import DS4Engine
import DS4Core

struct ChatView: View {
    @Bindable var store: ChatStore
    @State private var showTools = false
    @State private var showChats = false
    @State private var showResponseSettings = false
    @State private var projects: [ProjectLibrary.SavedProject] = []
    @State private var activeProjectName: String?
    @State private var lastAutoScroll = Date.distantPast
    @State private var followsResponse = true
    @State private var previousTranscriptTop: CGFloat?
    @State private var isDropTargeted = false
    @FocusState private var composerFocused: Bool

    private let transcriptEnd = "conversation-end"

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
        .onAppear { composerFocused = true }
        .onChange(of: store.activeSessionId) { composerFocused = true }
    }

    private var header: some View {
        VStack(spacing: 12) {
            HStack(spacing: 12) {
                VStack(alignment: .leading, spacing: 4) {
                    Text(store.info?.displayName
                         ?? store.inspectedModelDescriptor?.displayName
                         ?? "Nessun modello caricato")
                        .font(.headline)
                        .lineLimit(1)
                    Text(modelDetails)
                        .font(.caption)
                        .foregroundStyle(.secondary)
                        .lineLimit(1)
                        .help(modelDetails)
                }
                Spacer(minLength: 4)
                Button {
                    showChats = true
                } label: {
                    Label("Cronologia", systemImage: "clock.arrow.circlepath")
                }
                .labelStyle(.iconOnly)
                .controlSize(.large)
                .popover(isPresented: $showChats, arrowEdge: .bottom) {
                    ChatListView(store: store)
                }
                .help("Cerca, apri e gestisci le conversazioni salvate")
                Button {
                    store.newChat()
                    composerFocused = true
                } label: {
                    Label("Nuova chat", systemImage: "square.and.pencil")
                }
                .keyboardShortcut("n", modifiers: .command)
                .help("Inizia una nuova conversazione (⌘N). La chat attuale viene salvata.")
            }
            HStack(spacing: 10) {
                projectMenu
                    .frame(maxWidth: 170)
                Picker("Agente", selection: Binding(get: { store.selectedAgentId },
                                                    set: { store.selectAgent($0) })) {
                    ForEach(store.agents) { agent in
                        Label(agent.name, systemImage: agent.icon).tag(agent.id)
                    }
                }
                .labelsHidden()
                .pickerStyle(.menu)
                .frame(maxWidth: 180)
                .disabled(store.isGenerating)
                .help("Cambia agente e inizia una nuova chat con il suo ruolo e i suoi strumenti.")
                Spacer(minLength: 0)
                if store.modelCapabilities.contains(.tools) {
                    Button {
                        showTools = true
                    } label: {
                        Label(toolButtonTitle, systemImage: "wrench.and.screwdriver")
                    }
                    .help("Scegli gli strumenti disponibili per l’agente")
                }
                Button {
                    showResponseSettings = true
                } label: {
                    Label("Risposta", systemImage: "slider.horizontal.3")
                }
                .popover(isPresented: $showResponseSettings) {
                    ChatResponseSettings(store: store)
                }
                .help("Regola creatività, ripetizioni e ragionamento")
            }
            .controlSize(.small)
        }
        .padding(.horizontal)
        .padding(.vertical, 12)
        .background(.bar)

    }

    private var modelDetails: String {
        if let info = store.info {
            return "\(info.architecture.rawValue) · \(info.layers) layer · \(info.quantizationSummary) · contesto \(info.contextSize) · \(kvLabel(info))"
        }
        return store.inspectedModelDescriptor.map {
            "\($0.architecture.rawValue) · backend \($0.backendAvailability.rawValue)"
        } ?? "Modello locale"
    }

    private var toolButtonTitle: String {
        guard store.toolsEnabled else { return "Strumenti" }
        return "Strumenti (\(store.enabledToolNames.count))"
    }

    /// Add/switch the active project right from the chat: the agent's
    /// project_* tools read the active one; the chat memory is untouched.
    private var projectMenu: some View {
        Menu {
            if projects.isEmpty {
                Text("Nessun progetto salvato")
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
                Label("Aggiungi un progetto…", systemImage: "folder.badge.plus")
            }
        } label: {
            Label(activeProjectName ?? "Progetto", systemImage: "folder")
                .lineLimit(1)
        }
        .disabled(store.isGenerating)
        .help("Scegli il progetto da usare al prossimo messaggio. La conversazione attuale viene mantenuta.")
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

    private func kvLabel(_ info: ModelInfo) -> String {
        if info.architecture
            == LagunaBackendDefinition.supportedArchitecture {
            return "KV lazy iniziale ~\(kvSize(info.kvCacheBytes))"
        }
        return "KV ~\(kvSize(info.kvCacheBytes))"
    }

    private var transcript: some View {
        GeometryReader { viewport in
            ScrollViewReader { proxy in
                ScrollView {
                    LazyVStack(alignment: .leading, spacing: 20) {
                        if store.messages.isEmpty {
                            ChatEmptyState(agentName: store.selectedAgent.name) { prompt in
                                store.input = prompt
                                composerFocused = true
                            }
                            .frame(maxWidth: .infinity)
                        }
                        ForEach(store.messages) { message in
                            MessageRow(message: message)
                                .id(message.id)
                        }
                        Color.clear.frame(height: 1).id(transcriptEnd)
                    }
                    .frame(maxWidth: 880)
                    .padding(24)
                    .frame(maxWidth: .infinity)
                    .background {
                        GeometryReader { content in
                            let frame = content.frame(in: .named("chat-transcript"))
                            Color.clear.preference(key: TranscriptBoundsKey.self,
                                                   value: TranscriptBounds(top: frame.minY, bottom: frame.maxY))
                        }
                    }
                }
                .coordinateSpace(name: "chat-transcript")
                .onPreferenceChange(TranscriptBoundsKey.self) { bounds in
                    // Growing text changes the bottom edge; reading older messages
                    // moves the top edge down. Only the latter pauses following.
                    if let previous = previousTranscriptTop,
                       bounds.top > previous + 2,
                       bounds.bottom > viewport.size.height + 40 {
                        followsResponse = false
                    }
                    if bounds.bottom <= viewport.size.height + 24 {
                        followsResponse = true
                    }
                    previousTranscriptTop = bounds.top
                }
                .overlay(alignment: .bottom) {
                    if !followsResponse && !store.messages.isEmpty {
                        Button {
                            followsResponse = true
                            proxy.scrollTo(transcriptEnd, anchor: .bottom)
                        } label: {
                            Label("Vai all’ultimo messaggio", systemImage: "arrow.down")
                        }
                        .buttonStyle(.borderedProminent)
                        .clipShape(Capsule())
                        .padding(.bottom, 12)
                    }
                }
                .onAppear {
                    if !store.messages.isEmpty { proxy.scrollTo(transcriptEnd, anchor: .bottom) }
                }
                .onChange(of: store.activeSessionId) {
                    followsResponse = true
                    previousTranscriptTop = nil
                    if !store.messages.isEmpty { proxy.scrollTo(transcriptEnd, anchor: .bottom) }
                }
                .onChange(of: store.messages.count) {
                    if followsResponse && !store.messages.isEmpty {
                        proxy.scrollTo(transcriptEnd, anchor: .bottom)
                    }
                }
                .onChange(of: store.messages.last.map { $0.reasoning.count + $0.text.count + $0.toolStreamText.count }) {
                    guard followsResponse, !store.messages.isEmpty else { return }
                    let now = Date()
                    guard !store.isGenerating || now.timeIntervalSince(lastAutoScroll) >= 0.20 else {
                        return
                    }
                    lastAutoScroll = now
                    // No overlapping animations during streaming or navigation.
                    proxy.scrollTo(transcriptEnd, anchor: .bottom)
                }
                .onChange(of: store.isGenerating) {
                    if followsResponse && !store.messages.isEmpty {
                        proxy.scrollTo(transcriptEnd, anchor: .bottom)
                    }
                }
            }
        }
        .background(Color.secondary.opacity(0.025))
    }

    private var composer: some View {
        VStack(alignment: .leading, spacing: 10) {
            if store.isGenerating {
                HStack(spacing: 6) {
                    ProgressView().controlSize(.mini)
                    Text(store.status.isEmpty ? "Preparazione della risposta…" : store.status)
                        .font(.caption.monospacedDigit())
                        .foregroundStyle(.secondary)
                        .lineLimit(2)
                }
            }
            if !store.attachments.isEmpty {
                ScrollView(.horizontal, showsIndicators: false) {
                    HStack(spacing: 6) {
                        ForEach(store.attachments) { att in
                            AttachmentChip(name: att.name, bytes: att.bytes, imageData: att.imageData) {
                                store.removeAttachment(att.id)
                            }
                        }
                    }
                }
            }
            if let note = store.attachmentNote {
                Label(note, systemImage: "exclamationmark.triangle")
                    .font(.caption).foregroundStyle(.orange)
            }
            if let est = store.attachmentTokenEstimate, est > store.contextSize - 256 {
                Label("Gli allegati occupano circa \(est) token e possono superare il contesto (\(store.contextSize)). Riduci i file o aumenta il contesto nelle impostazioni.",
                      systemImage: "exclamationmark.triangle")
                    .font(.caption).foregroundStyle(.orange)
            }
            if store.contextUsed > 0, store.contextUsed * 100 >= store.contextSize * 85 {
                Label("Contesto quasi pieno: \(store.contextUsed)/\(store.contextSize) token. Per evitare risposte troncate, inizia una nuova chat o aumenta il contesto nelle impostazioni.",
                      systemImage: "exclamationmark.triangle.fill")
                    .font(.caption).foregroundStyle(.orange)
            }
            HStack(alignment: .bottom, spacing: 10) {
                Button { store.pickAndAttachFiles() } label: {
                    Label("Allega file", systemImage: "paperclip")
                }
                .labelStyle(.iconOnly)
                .controlSize(.large)
                .help(store.canAttachImages ? "Allega immagini o file di testo alla conversazione" : "Allega file di testo alla conversazione")
                .disabled(store.isGenerating)
                TextField("Scrivi un messaggio…", text: $store.input, axis: .vertical)
                    .textFieldStyle(.plain)
                    .font(.body)
                    .lineLimit(2...8)
                    .focused($composerFocused)
                    .accessibilityLabel("Messaggio")
                    .onSubmit(sendMessage)
                    .padding(.vertical, 6)
                if store.isGenerating {
                    Button(role: .destructive) { store.stop() } label: {
                        Label("Interrompi", systemImage: "stop.fill")
                    }
                    .keyboardShortcut(.escape, modifiers: [])
                    .help("Interrompi la risposta (Esc)")
                } else {
                    Button(action: sendMessage) {
                        Label("Invia", systemImage: "arrow.up")
                    }
                    .buttonStyle(.borderedProminent)
                    .keyboardShortcut(.return, modifiers: .command)
                    .disabled(store.input.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty
                              && store.attachments.isEmpty)
                    .help("Invia il messaggio (⌘Invio)")
                }
            }
            .padding(10)
            .background(.background, in: RoundedRectangle(cornerRadius: 14))
            .overlay {
                RoundedRectangle(cornerRadius: 14)
                    .strokeBorder(composerFocused ? Color.accentColor.opacity(0.6) : Color.primary.opacity(0.15))
            }
            HStack {
                Text(store.isGenerating ? "Puoi preparare il prossimo messaggio." : "⌘Invio per inviare")
                if store.canAttachImages {
                    Label("Vision attivo", systemImage: "eye")
                        .foregroundStyle(.tint)
                        .help(store.visionConfigurationNote)
                }
                Spacer()
                if store.contextUsed > 0 {
                    Text("Contesto: \(store.contextUsed.formatted()) / \(store.contextSize.formatted())")
                        .monospacedDigit()
                        .help("Token utilizzati nella finestra di contesto del modello")
                }
            }
            .font(.caption2)
            .foregroundStyle(.secondary)
        }
        .frame(maxWidth: 880)
        .padding(.horizontal)
        .padding(.vertical, 12)
        .frame(maxWidth: .infinity)
        .background(.bar)
        .overlay {
            if isDropTargeted {
                RoundedRectangle(cornerRadius: 12)
                    .strokeBorder(Color.accentColor, style: StrokeStyle(lineWidth: 2, dash: [6]))
                    .padding(5)
                    .allowsHitTesting(false)
            }
        }
        .dropDestination(for: URL.self) { urls, _ in
            let files = urls.filter(\.isFileURL)
            guard !files.isEmpty else { return false }
            store.importFiles(files)
            return true
        } isTargeted: { isDropTargeted = $0 }
    }

    private func sendMessage() {
        guard !store.isGenerating else { return }
        followsResponse = true
        store.send()
        composerFocused = true
    }
}

private struct TranscriptBounds: Equatable {
    var top: CGFloat = 0
    var bottom: CGFloat = 0
}

private struct TranscriptBoundsKey: PreferenceKey {
    static let defaultValue = TranscriptBounds()

    static func reduce(value: inout TranscriptBounds, nextValue: () -> TranscriptBounds) {
        value = nextValue()
    }
}
