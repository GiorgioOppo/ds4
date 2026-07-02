import SwiftUI
import AppKit
import UniformTypeIdentifiers
import DS4Engine

/// Curated SF Symbols selectable as agent icons.
enum AgentIcons {
    static let all = [
        "person", "person.fill", "person.2", "brain",
        "chevron.left.forwardslash.chevron.right", "terminal", "function", "pencil",
        "book", "graduationcap", "magnifyingglass", "globe",
        "lightbulb", "hammer", "wrench.and.screwdriver", "chart.bar",
        "briefcase", "stethoscope", "music.note", "gamecontroller",
    ]
}

/// Agent (role) management: view and EDIT each agent's defining prompt, icon and
/// the tools it exposes; create custom agents, restore the defaults, switch the
/// active one, and export/import the whole set as JSON. Edits apply on the next
/// new chat / agent switch.
struct AgentsView: View {
    @Bindable var store: ChatStore
    @State private var ioMessage = ""

    var body: some View {
        Form {
            Section {
                Text("An agent is a role: a system prompt that defines behavior, the tools it may call, and a dedicated expert-usage profile. The cache warms with the experts used by that role. Changes apply on the next new chat or agent switch.")
                    .font(.caption).foregroundStyle(.secondary)
            }

            ForEach($store.agents) { $agent in
                Section {
                    HStack {
                        Label(agent.name, systemImage: agent.icon).font(.headline)
                        if agent.id == store.selectedAgentId {
                            Text("active")
                                .font(.caption)
                                .padding(.horizontal, 6).padding(.vertical, 2)
                                .background(Color.accentColor.opacity(0.2))
                                .clipShape(Capsule())
                        }
                        Spacer()
                        Button("Use") { store.selectAgent(agent.id) }
                            .disabled(agent.id == store.selectedAgentId)
                    }

                    TextField("Name", text: $agent.name)

                    DisclosureGroup("Icon") {
                        LazyVGrid(columns: [GridItem(.adaptive(minimum: 36))], spacing: 8) {
                            ForEach(AgentIcons.all, id: \.self) { symbol in
                                Button { agent.icon = symbol } label: {
                                    Image(systemName: symbol)
                                        .frame(width: 30, height: 30)
                                        .background(agent.icon == symbol
                                                    ? Color.accentColor.opacity(0.25) : .clear)
                                        .clipShape(RoundedRectangle(cornerRadius: 6))
                                }
                                .buttonStyle(.plain)
                                .help(symbol)
                            }
                        }
                        .padding(.vertical, 4)
                    }

                    VStack(alignment: .leading, spacing: 4) {
                        Text("System prompt (role definition)")
                            .font(.caption).foregroundStyle(.secondary)
                        TextEditor(text: $agent.systemPrompt)
                            .font(.body)
                            .frame(minHeight: 64, maxHeight: 150)
                            .overlay(RoundedRectangle(cornerRadius: 6)
                                .stroke(Color.secondary.opacity(0.3)))
                    }

                    DisclosureGroup("Agent tools (\(agent.toolNames.count))") {
                        ForEach(store.availableTools) { tool in
                            Toggle(isOn: Binding(
                                get: { agent.toolNames.contains(tool.name) },
                                set: { on in
                                    if on {
                                        if !agent.toolNames.contains(tool.name) { agent.toolNames.append(tool.name) }
                                    } else {
                                        agent.toolNames.removeAll { $0 == tool.name }
                                    }
                                })) {
                                VStack(alignment: .leading) {
                                    Text(tool.name).font(.body.monospaced())
                                    Text(tool.description).font(.caption).foregroundStyle(.secondary)
                                }
                            }
                        }
                    }

                    if !store.isDefaultAgent(agent.id) {
                        Button(role: .destructive) { store.deleteAgent(agent.id) } label: {
                            Label("Delete Agent", systemImage: "trash")
                        }
                    }
                }
            }

            Section {
                Button { store.addAgent() } label: {
                    Label("New Agent", systemImage: "plus")
                }
                Button { store.restoreDefaultAgents() } label: {
                    Label("Restore Defaults", systemImage: "arrow.counterclockwise")
                }
                HStack {
                    Button { exportAgents() } label: {
                        Label("Export...", systemImage: "square.and.arrow.up")
                    }
                    Button { importAgents() } label: {
                        Label("Import...", systemImage: "square.and.arrow.down")
                    }
                }
                if !ioMessage.isEmpty {
                    Text(ioMessage).font(.caption).foregroundStyle(.secondary)
                }
                if !store.isReady {
                    Text("No model loaded: the selection is remembered and applied when a model loads.")
                        .font(.caption).foregroundStyle(.tertiary)
                }
            }
        }
        .formStyle(.grouped)
        .onChange(of: store.agents) { store.saveAgents() }
    }

    /// Save the whole agent set as JSON (user-picked location, sandbox-safe).
    private func exportAgents() {
        let panel = NSSavePanel()
        panel.nameFieldStringValue = "dwarfstar-agents.json"
        panel.allowedContentTypes = [.json]
        panel.title = "Export Agents"
        guard panel.runModal() == .OK, let url = panel.url,
              let data = store.exportAgentsData() else { return }
        do {
            try data.write(to: url)
            ioMessage = "Exported \(store.agents.count) agents to \(url.lastPathComponent)."
        } catch {
            ioMessage = "Export failed: \(error.localizedDescription)"
        }
    }

    /// Merge agents from a JSON file (matching ids updated, new ones appended).
    private func importAgents() {
        let panel = NSOpenPanel()
        panel.allowedContentTypes = [.json]
        panel.allowsMultipleSelection = false
        panel.title = "Import Agents"
        guard panel.runModal() == .OK, let url = panel.url,
              let data = try? Data(contentsOf: url) else { return }
        let n = store.importAgents(from: data)
        ioMessage = n > 0 ? "Imported/updated \(n) agents." : "Invalid file: no agents imported."
    }
}
