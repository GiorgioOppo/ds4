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
            temperatureMenu
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

    /// Temperature control: lower = più focalizzato e meno deriva (utile sui
    /// modelli molto quantizzati); più alto = più creativo/variabile.
    private var temperatureMenu: some View {
        Menu {
            VStack(alignment: .leading) {
                Text("Temperatura: \(store.temperature, format: .number.precision(.fractionLength(2)))")
                    .font(.caption).foregroundStyle(.secondary)
                Slider(value: $store.temperature, in: 0...1.5, step: 0.05)
                    .frame(width: 220)
                Text("Bassa = più focalizzato, meno deriva. Alta = più creativo.")
                    .font(.caption2).foregroundStyle(.secondary)
                HStack {
                    Button("Preciso (0.3)") { store.temperature = 0.3 }
                    Button("Default (0.6)") { store.temperature = 0.6 }
                }
                .buttonStyle(.borderless).font(.caption)

                Divider()
                Text("Penalità ripetizione: \(store.repetitionPenalty, format: .number.precision(.fractionLength(2)))")
                    .font(.caption).foregroundStyle(.secondary)
                Slider(value: $store.repetitionPenalty, in: 1.0...1.5, step: 0.05)
                    .frame(width: 220)
                Text("Alza (1.15–1.3) se il modello entra in loop di ripetizione dopo molti token.")
                    .font(.caption2).foregroundStyle(.secondary)
            }
            .padding(8)
        } label: {
            Label(store.temperature.formatted(.number.precision(.fractionLength(1))),
                  systemImage: "thermometer.medium")
        }
        .menuStyle(.borderlessButton)
        .fixedSize()
        .help("Temperatura di campionamento: abbassala (0.3–0.4) se il modello sbanda o ripete.")
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
            Label("Allegati ~\(est) token: rischiano di superare il contesto (\(store.contextSize)). Riduci i file o aumenta il contesto in Impostazioni.",
                  systemImage: "exclamationmark.triangle")
                .font(.caption2).foregroundStyle(.orange)
        }
        if store.contextUsed > 0, store.contextUsed * 100 >= store.contextSize * 85 {
            Label("Contesto quasi pieno: \(store.contextUsed)/\(store.contextSize) token. A breve la risposta verrà troncata: inizia una nuova chat o aumenta il contesto (Impostazioni).",
                  systemImage: "exclamationmark.triangle.fill")
                .font(.caption2).foregroundStyle(.orange)
        }
        HStack(alignment: .bottom, spacing: 8) {
            Button { store.pickAndAttachFiles() } label: {
                Image(systemName: "paperclip")
            }
            .buttonStyle(.borderless)
            .help("Importa file di testo nella conversazione")
            .disabled(store.isGenerating)
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
                .disabled(store.input.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty
                          && store.attachments.isEmpty)
            }
        }
        }
        .padding(.horizontal)
        .padding(.vertical, 10)
    }
}

/// A staged text-file attachment shown above the composer, with a remove button.
struct AttachmentChip: View {
    let name: String
    let bytes: Int
    let onRemove: () -> Void

    var body: some View {
        HStack(spacing: 4) {
            Image(systemName: "doc.text").font(.caption2)
            Text(name).font(.caption).lineLimit(1)
            Text(ByteCountFormatter.string(fromByteCount: Int64(bytes), countStyle: .file))
                .font(.caption2).foregroundStyle(.secondary)
            Button(action: onRemove) {
                Image(systemName: "xmark.circle.fill")
            }
            .buttonStyle(.borderless)
            .foregroundStyle(.secondary)
            .help("Rimuovi allegato")
        }
        .padding(.horizontal, 8).padding(.vertical, 4)
        .background(Color.secondary.opacity(0.12))
        .clipShape(Capsule())
    }
}

/// Filename badges shown under a user message that imported text files.
struct AttachmentBadges: View {
    let names: [String]

    var body: some View {
        ScrollView(.horizontal, showsIndicators: false) {
            HStack(spacing: 6) {
                ForEach(names, id: \.self) { name in
                    Label(name, systemImage: "doc.text")
                        .font(.caption2).lineLimit(1)
                        .padding(.horizontal, 6).padding(.vertical, 3)
                        .background(Color.secondary.opacity(0.12))
                        .clipShape(Capsule())
                }
            }
        }
    }
}

struct MessageRow: View {
    let message: UIMessage

    var body: some View {
        if message.role == .tool {
            if let run = message.subAgent {
                SubAgentView(run: run)
            } else {
                ToolResultRow(text: message.text)
            }
        } else {
            HStack {
                if message.role == .user { Spacer(minLength: 40) }
                VStack(alignment: message.role == .user ? .trailing : .leading, spacing: 6) {
                    if !message.reasoning.isEmpty {
                        ReasoningView(text: message.reasoning)
                    }
                    if !message.text.isEmpty {
                        Group {
                            if message.role == .assistant {
                                MarkdownView(text: message.text)
                            } else {
                                Text(message.text).textSelection(.enabled)
                            }
                        }
                        .padding(10)
                        .background(bubbleColor)
                        .clipShape(RoundedRectangle(cornerRadius: 12))
                    } else if message.role == .assistant && message.reasoning.isEmpty
                                && message.toolCalls.isEmpty && message.toolStreamText.isEmpty {
                        ProgressView().controlSize(.small)
                    }
                    if !message.attachments.isEmpty {
                        AttachmentBadges(names: message.attachments)
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

// MARK: - Markdown rendering

/// Lightweight Markdown renderer for assistant messages: fenced code blocks,
/// headings, bullet/ordered lists, blockquotes, and paragraphs with inline
/// markdown (bold/italic/`code`/links). Re-parses on each update — cheap at chat
/// length and streaming-friendly. Avoids a dependency; AttributedString handles
/// the inline syntax, this splits the block structure SwiftUI's Text won't.
struct MarkdownView: View {
    let text: String

    var body: some View {
        VStack(alignment: .leading, spacing: 8) {
            let blocks = Self.parse(text)
            ForEach(Array(blocks.enumerated()), id: \.offset) { _, block in
                render(block)
            }
        }
        .frame(maxWidth: .infinity, alignment: .leading)
    }

    @ViewBuilder
    private func render(_ block: Block) -> some View {
        switch block {
        case .code(let code):
            ScrollView(.horizontal, showsIndicators: false) {
                Text(code)
                    .font(.system(.callout, design: .monospaced))
                    .textSelection(.enabled)
            }
            .padding(8)
            .frame(maxWidth: .infinity, alignment: .leading)
            .background(Color.black.opacity(0.06))
            .clipShape(RoundedRectangle(cornerRadius: 8))
        case .heading(let level, let s):
            Text(Self.inline(s))
                .font(level <= 1 ? .title3.bold() : (level == 2 ? .headline : .subheadline.bold()))
                .textSelection(.enabled)
        case .list(let items, let ordered):
            VStack(alignment: .leading, spacing: 3) {
                ForEach(Array(items.enumerated()), id: \.offset) { i, item in
                    HStack(alignment: .firstTextBaseline, spacing: 6) {
                        Text(ordered ? "\(i + 1)." : "•")
                            .font(.callout.monospacedDigit())
                            .foregroundStyle(.secondary)
                        Text(Self.inline(item)).textSelection(.enabled)
                    }
                }
            }
        case .quote(let s):
            Text(Self.inline(s))
                .foregroundStyle(.secondary)
                .padding(.leading, 10)
                .overlay(alignment: .leading) {
                    Rectangle().frame(width: 3).foregroundStyle(.secondary.opacity(0.4))
                }
        case .paragraph(let s):
            Text(Self.inline(s)).textSelection(.enabled)
        }
    }

    /// Inline markdown (bold/italic/code/links), whitespace preserved, never throws.
    static func inline(_ s: String) -> AttributedString {
        (try? AttributedString(markdown: s, options: .init(
            interpretedSyntax: .inlineOnlyPreservingWhitespace,
            failurePolicy: .returnPartiallyParsedIfPossible))) ?? AttributedString(s)
    }

    enum Block { case code(String), heading(Int, String), list([String], ordered: Bool)
                 case quote(String), paragraph(String) }

    /// Split text into block-level elements (the part SwiftUI's Text won't do).
    static func parse(_ text: String) -> [Block] {
        var blocks: [Block] = []
        let lines = text.components(separatedBy: "\n")
        var para: [String] = []
        func flush() {
            if !para.isEmpty { blocks.append(.paragraph(para.joined(separator: "\n"))); para = [] }
        }
        var i = 0
        while i < lines.count {
            let line = lines[i]
            let t = line.trimmingCharacters(in: .whitespaces)

            if t.hasPrefix("```") {                         // fenced code block
                flush()
                var code: [String] = []
                i += 1
                while i < lines.count, !lines[i].trimmingCharacters(in: .whitespaces).hasPrefix("```") {
                    code.append(lines[i]); i += 1
                }
                i += 1                                       // skip closing fence
                blocks.append(.code(code.joined(separator: "\n")))
                continue
            }
            if let h = headingLevel(t) {                     // # heading
                flush()
                let content = String(t.drop(while: { $0 == "#" })).trimmingCharacters(in: .whitespaces)
                blocks.append(.heading(h, content)); i += 1; continue
            }
            if isBullet(t) || isOrdered(t) {                 // list (grouped)
                flush()
                let ordered = isOrdered(t)
                var items: [String] = []
                while i < lines.count {
                    let tt = lines[i].trimmingCharacters(in: .whitespaces)
                    if ordered, isOrdered(tt) { items.append(stripOrdered(tt)) }
                    else if !ordered, isBullet(tt) { items.append(String(tt.dropFirst(2))) }
                    else { break }
                    i += 1
                }
                blocks.append(.list(items, ordered: ordered)); continue
            }
            if t.hasPrefix("> ") {                           // blockquote
                flush(); blocks.append(.quote(String(t.dropFirst(2)))); i += 1; continue
            }
            if t.isEmpty { flush() } else { para.append(line) }
            i += 1
        }
        flush()
        return blocks
    }

    private static func headingLevel(_ t: String) -> Int? {
        var n = 0
        for c in t { if c == "#" { n += 1 } else { break } }
        return (n >= 1 && n <= 6 && t.dropFirst(n).first == " ") ? n : nil
    }
    private static func isBullet(_ t: String) -> Bool {
        t.hasPrefix("- ") || t.hasPrefix("* ") || t.hasPrefix("+ ")
    }
    private static func isOrdered(_ t: String) -> Bool {
        guard let dot = t.firstIndex(of: ".") else { return false }
        let num = t[t.startIndex..<dot]
        return !num.isEmpty && num.allSatisfy(\.isNumber) && t[dot...].hasPrefix(". ")
    }
    private static func stripOrdered(_ t: String) -> String {
        guard let dot = t.firstIndex(of: ".") else { return t }
        return String(t[t.index(after: dot)...]).trimmingCharacters(in: .whitespaces)
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

/// A completed isolated sub-agent run: target + answer, with a collapsible trace
/// of the internal steps (which never entered the main conversation's context).
struct SubAgentView: View {
    let run: InferenceService.SubAgentRun
    @State private var expanded = false

    var body: some View {
        VStack(alignment: .leading, spacing: 6) {
            Label("Sub-agent · \(run.target)", systemImage: "person.3.sequence")
                .font(.caption.bold()).foregroundStyle(.purple)
            if !run.question.isEmpty {
                Text(run.question).font(.caption2).foregroundStyle(.secondary).lineLimit(2)
            }
            Text(run.answer).font(.callout).textSelection(.enabled)
            if !run.steps.isEmpty {
                DisclosureGroup(isExpanded: $expanded) {
                    VStack(alignment: .leading, spacing: 2) {
                        ForEach(Array(run.steps.enumerated()), id: \.offset) { _, step in
                            Text(step)
                                .font(.system(.caption2, design: .monospaced))
                                .foregroundStyle(.secondary)
                                .textSelection(.enabled)
                                .frame(maxWidth: .infinity, alignment: .leading)
                        }
                    }
                } label: {
                    Label("Passi interni (\(run.steps.count))", systemImage: "list.bullet.indent")
                        .font(.caption2).foregroundStyle(.secondary)
                }
            }
        }
        .padding(10)
        .frame(maxWidth: .infinity, alignment: .leading)
        .background(Color.purple.opacity(0.07))
        .overlay(RoundedRectangle(cornerRadius: 10).strokeBorder(Color.purple.opacity(0.25)))
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
