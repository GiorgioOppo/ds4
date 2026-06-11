import SwiftUI
import DS4Engine
import DS4Core

struct ChatView: View {
    @Bindable var store: ChatStore
    @State private var showTools = false
    @State private var projects: [ProjectLibrary.SavedProject] = []
    @State private var activeProjectName: String?

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
                Text(store.info?.name ?? "DeepSeek V4")
                    .font(.headline)
                if let info = store.info {
                    Text("\(info.layers) layer · \(info.routedQuantBits)-bit · ctx \(info.contextSize) · KV ~\(kvSize(info.kvCacheBytes))")
                        .font(.caption)
                        .foregroundStyle(.secondary)
                }
            }
            Spacer()
            projectMenu
            Picker("Agente", selection: Binding(get: { store.selectedAgentId },
                                                set: { store.selectAgent($0) })) {
                ForEach(store.agents) { agent in
                    Label(agent.name, systemImage: agent.icon).tag(agent.id)
                }
            }
            .pickerStyle(.menu)
            .fixedSize()
            .help("Cambia ruolo: nuova chat con il system prompt e i tool dell'agente; la cache esperti si ri-scalda col SUO profilo d'uso.")
            Button {
                showTools = true
            } label: {
                Label(toolButtonTitle, systemImage: "wrench.and.screwdriver")
            }
            Toggle("Thinking", isOn: $store.think)
                .toggleStyle(.switch)
            Button {
                store.newChat()
            } label: {
                Label("Nuova chat", systemImage: "square.and.pencil")
            }
        }
        .padding(.horizontal)
        .padding(.vertical, 10)
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
                Label("Importa cartella…", systemImage: "folder.badge.plus")
            }
        } label: {
            Label(activeProjectName ?? "Progetto", systemImage: "folder")
        }
        .fixedSize()
        .help("Progetto attivo per i tool project_* dell'agente. L'import non tocca la memoria della chat.")
        .onAppear { refreshProject() }
    }

    private func refreshProject() {
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
                    withAnimation { proxy.scrollTo(last.id, anchor: .bottom) }
                }
            }
        }
    }

    private var composer: some View {
        VStack(alignment: .leading, spacing: 4) {
        if store.isGenerating && !store.status.isEmpty {
            HStack(spacing: 6) {
                ProgressView().controlSize(.mini)
                Text(store.status)
                    .font(.caption.monospacedDigit())
                    .foregroundStyle(.secondary)
            }
        }
        HStack(alignment: .bottom, spacing: 8) {
            TextField("Scrivi un messaggio…", text: $store.input, axis: .vertical)
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
                .disabled(store.input.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty)
            }
        }
        }
        .padding(.horizontal)
        .padding(.vertical, 10)
    }
}

struct MessageRow: View {
    let message: UIMessage

    var body: some View {
        if message.role == .tool {
            ToolResultRow(text: message.text)
        } else {
            HStack {
                if message.role == .user { Spacer(minLength: 40) }
                VStack(alignment: message.role == .user ? .trailing : .leading, spacing: 6) {
                    if !message.reasoning.isEmpty {
                        ReasoningView(text: message.reasoning)
                    }
                    if !message.text.isEmpty {
                        Text(message.text)
                            .textSelection(.enabled)
                            .padding(10)
                            .background(bubbleColor)
                            .clipShape(RoundedRectangle(cornerRadius: 12))
                    } else if message.role == .assistant && message.reasoning.isEmpty
                                && message.toolCalls.isEmpty && message.toolStreamText.isEmpty {
                        ProgressView().controlSize(.small)
                    }
                    if !message.toolStreamText.isEmpty {
                        ToolStreamView(text: message.toolStreamText)
                    }
                    ForEach(message.toolCalls) { call in
                        ToolCallView(call: call)
                    }
                }
                if message.role == .assistant { Spacer(minLength: 40) }
            }
        }
    }

    private var bubbleColor: Color {
        message.role == .user ? Color.accentColor.opacity(0.18) : Color.secondary.opacity(0.12)
    }
}

/// The raw tool-call markup as it streams, shown live during generation. Once the
/// block closes it is replaced by the formatted ToolCallView card.
struct ToolStreamView: View {
    let text: String

    var body: some View {
        VStack(alignment: .leading, spacing: 2) {
            Label("Generazione chiamata tool…", systemImage: "wrench.and.screwdriver")
                .font(.caption.bold())
                .foregroundStyle(.orange.opacity(0.7))
            Text(text)
                .font(.system(.caption2, design: .monospaced))
                .textSelection(.enabled)
                .foregroundStyle(.secondary)
        }
        .padding(8)
        .frame(maxWidth: .infinity, alignment: .leading)
        .background(Color.orange.opacity(0.05))
        .overlay(RoundedRectangle(cornerRadius: 10).strokeBorder(Color.orange.opacity(0.25),
                                                                 style: StrokeStyle(lineWidth: 1, dash: [4])))
        .clipShape(RoundedRectangle(cornerRadius: 10))
    }
}

/// A tool the model decided to call.
struct ToolCallView: View {
    let call: ToolCall

    var body: some View {
        VStack(alignment: .leading, spacing: 2) {
            Label("Chiamata tool: \(call.name)", systemImage: "wrench.and.screwdriver.fill")
                .font(.caption.bold())
                .foregroundStyle(.orange)
            Text(call.argumentsJSON)
                .font(.system(.caption2, design: .monospaced))
                .textSelection(.enabled)
                .foregroundStyle(.secondary)
        }
        .padding(8)
        .frame(maxWidth: .infinity, alignment: .leading)
        .background(Color.orange.opacity(0.08))
        .clipShape(RoundedRectangle(cornerRadius: 10))
    }
}

/// The result of a tool, fed back to the model.
struct ToolResultRow: View {
    let text: String
    var body: some View {
        Label(text, systemImage: "arrow.uturn.left")
            .font(.system(.caption, design: .monospaced))
            .foregroundStyle(.secondary)
            .padding(8)
            .frame(maxWidth: .infinity, alignment: .leading)
            .background(Color.green.opacity(0.08))
            .clipShape(RoundedRectangle(cornerRadius: 10))
    }
}

/// Collapsible chain-of-thought block.
struct ReasoningView: View {
    let text: String
    @State private var expanded = false

    var body: some View {
        DisclosureGroup(isExpanded: $expanded) {
            Text(text)
                .font(.callout)
                .foregroundStyle(.secondary)
                .textSelection(.enabled)
                .frame(maxWidth: .infinity, alignment: .leading)
        } label: {
            Label("Ragionamento", systemImage: "brain")
                .font(.caption)
                .foregroundStyle(.secondary)
        }
        .padding(8)
        .background(Color.secondary.opacity(0.06))
        .clipShape(RoundedRectangle(cornerRadius: 10))
    }
}

/// Sheet to enable tools and choose which built-ins are exposed to the model.
struct ToolPickerView: View {
    @Bindable var store: ChatStore
    @Environment(\.dismiss) private var dismiss

    var body: some View {
        VStack(alignment: .leading, spacing: 12) {
            Text("Tool").font(.title2).bold()
            Toggle("Abilita i tool (function calling)", isOn: $store.toolsEnabled)
                .onChange(of: store.toolsEnabled) { store.syncTools() }
            Text("Quando abilitati, i tool selezionati vengono dichiarati al modello. I tool integrati vengono eseguiti automaticamente; per altri tool potrai inserire il risultato a mano.")
                .font(.caption).foregroundStyle(.secondary)

            Divider()
            Text("Tool integrati").font(.headline)
            ForEach(store.availableTools) { tool in
                Toggle(isOn: Binding(
                    get: { store.enabledToolNames.contains(tool.name) },
                    set: { on in
                        if on { store.enabledToolNames.insert(tool.name) }
                        else { store.enabledToolNames.remove(tool.name) }
                        store.syncTools()
                    })) {
                    VStack(alignment: .leading) {
                        Text(tool.name).font(.body.monospaced())
                        Text(tool.description).font(.caption).foregroundStyle(.secondary)
                    }
                }
                .disabled(!store.toolsEnabled)
            }

            Divider()
            Toggle("Dichiarazione compatta (solo nome+parametri)", isOn: $store.compactTools)
                .onChange(of: store.compactTools) { store.syncTools() }
                .disabled(!store.toolsEnabled)
            Text("Meno token di prefill: invece dello schema completo manda solo `nome(parametri)` + una riga di formato. Più economico ma si discosta dal testo di addestramento.")
                .font(.caption).foregroundStyle(.secondary)

            Spacer()
            HStack {
                Spacer()
                Button("Chiudi") { dismiss() }.keyboardShortcut(.defaultAction)
            }
        }
        .padding()
        .frame(minWidth: 460, minHeight: 360)
    }
}

/// Sheet to enter results for tool calls that aren't built-in.
struct ManualToolResultsView: View {
    @Bindable var store: ChatStore
    @State private var contents: [String: String] = [:]

    var body: some View {
        VStack(alignment: .leading, spacing: 12) {
            Text("Risultati dei tool").font(.title2).bold()
            Text("Il modello ha chiamato dei tool non integrati. Inserisci il risultato (idealmente JSON) per ciascuno e invia per continuare.")
                .font(.caption).foregroundStyle(.secondary)

            ForEach(store.pendingManualCalls) { call in
                VStack(alignment: .leading, spacing: 4) {
                    Label("\(call.name)  \(call.argumentsJSON)", systemImage: "wrench.and.screwdriver.fill")
                        .font(.caption.monospaced())
                    TextEditor(text: Binding(get: { contents[call.id] ?? "" },
                                             set: { contents[call.id] = $0 }))
                        .font(.system(.caption, design: .monospaced))
                        .frame(height: 70)
                        .overlay(RoundedRectangle(cornerRadius: 6).stroke(Color.secondary.opacity(0.3)))
                }
            }

            HStack {
                Button("Annulla") { store.cancelManualResults() }
                Spacer()
                Button("Invia risultati") { store.submitManualResults(contents) }
                    .keyboardShortcut(.defaultAction)
            }
        }
        .padding()
        .frame(minWidth: 480, minHeight: 320)
    }
}
