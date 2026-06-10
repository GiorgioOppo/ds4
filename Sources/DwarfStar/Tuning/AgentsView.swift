import SwiftUI
import DS4Engine

/// Agent (role) management: view and EDIT each agent's defining prompt and the
/// tools it exposes; create custom agents, restore the defaults, and switch the
/// active one. Edits apply on the next new chat / agent switch.
struct AgentsView: View {
    @Bindable var store: ChatStore

    var body: some View {
        Form {
            Section {
                Text("Un agente è un ruolo: un system prompt che definisce il comportamento, i tool che può chiamare, e un profilo d'uso esperti dedicato (la cache si scalda con gli esperti di QUEL ruolo). Le modifiche si applicano alla prossima nuova chat o al cambio di agente.")
                    .font(.caption).foregroundStyle(.secondary)
            }

            ForEach($store.agents) { $agent in
                Section {
                    HStack {
                        Label(agent.name, systemImage: agent.icon).font(.headline)
                        if agent.id == store.selectedAgentId {
                            Text("attivo")
                                .font(.caption)
                                .padding(.horizontal, 6).padding(.vertical, 2)
                                .background(Color.accentColor.opacity(0.2))
                                .clipShape(Capsule())
                        }
                        Spacer()
                        Button("Usa") { store.selectAgent(agent.id) }
                            .disabled(agent.id == store.selectedAgentId)
                    }

                    TextField("Nome", text: $agent.name)

                    VStack(alignment: .leading, spacing: 4) {
                        Text("System prompt (definizione del ruolo)")
                            .font(.caption).foregroundStyle(.secondary)
                        TextEditor(text: $agent.systemPrompt)
                            .font(.body)
                            .frame(minHeight: 64, maxHeight: 150)
                            .overlay(RoundedRectangle(cornerRadius: 6)
                                .stroke(Color.secondary.opacity(0.3)))
                    }

                    DisclosureGroup("Tool dell'agente (\(agent.toolNames.count))") {
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
                            Label("Elimina agente", systemImage: "trash")
                        }
                    }
                }
            }

            Section {
                Button { store.addAgent() } label: {
                    Label("Nuovo agente", systemImage: "plus")
                }
                Button { store.restoreDefaultAgents() } label: {
                    Label("Ripristina predefiniti", systemImage: "arrow.counterclockwise")
                }
                if !store.isReady {
                    Text("Nessun modello caricato: la selezione viene ricordata e applicata al caricamento.")
                        .font(.caption).foregroundStyle(.tertiary)
                }
            }
        }
        .formStyle(.grouped)
        .onChange(of: store.agents) { store.saveAgents() }
    }
}
