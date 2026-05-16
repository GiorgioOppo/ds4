import SwiftUI
#if canImport(AppKit)
import AppKit
#endif

/// Settings tab: registry of agent presets. Master-detail layout
/// matching the Projects / MCP tabs. The MCP pool is observed so
/// the per-agent tool allowlist editor can present the live tool
/// catalogue (step A2 wires `Conversation.agentID` into the chat
/// flow; this tab only manages config).
struct AgentsView: View {
    @ObservedObject var library: AgentLibrary
    @ObservedObject var mcpPool: MCPClientPool

    @State private var selectedID: UUID?
    @State private var editing: AgentConfig?

    var body: some View {
        HStack(spacing: 0) {
            sidebar
                .frame(width: 220)
                .background(Color(NSColor.controlBackgroundColor))
            Divider()
            detail
                .frame(maxWidth: .infinity, maxHeight: .infinity)
        }
        .sheet(item: $editing) { agent in
            AgentEditSheet(initial: agent, mcpPool: mcpPool) { updated in
                if library.agents.contains(where: { $0.id == updated.id }) {
                    library.update(updated)
                } else {
                    library.add(updated)
                }
                if selectedID == nil { selectedID = updated.id }
            }
        }
    }

    // MARK: - sidebar

    private var sidebar: some View {
        VStack(spacing: 0) {
            HStack {
                Text("Agents").font(.headline)
                Spacer()
                Button {
                    editing = AgentConfig(name: "Untitled agent")
                } label: {
                    Label("New agent", systemImage: "plus")
                }
                .controlSize(.small)
            }
            .padding(.horizontal, 10)
            .padding(.vertical, 8)
            Divider()
            List(selection: $selectedID) {
                ForEach(library.agents) { a in
                    HStack(spacing: 8) {
                        Image(systemName: a.iconName)
                            .foregroundStyle(AgentTint.color(for: a.tint))
                            .frame(width: 18)
                        VStack(alignment: .leading, spacing: 1) {
                            Text(a.name).lineLimit(1)
                            if !a.summary.isEmpty {
                                Text(a.summary)
                                    .font(.caption2)
                                    .foregroundStyle(.tertiary)
                                    .lineLimit(1)
                            }
                        }
                    }
                    .tag(a.id)
                    .contextMenu {
                        Button("Edit…") { editing = a }
                        Divider()
                        Button("Delete", role: .destructive) {
                            library.delete(a.id)
                            if selectedID == a.id { selectedID = nil }
                        }
                    }
                }
            }
            .listStyle(.sidebar)
        }
    }

    // MARK: - detail

    @ViewBuilder
    private var detail: some View {
        if let id = selectedID,
           let a = library.agents.first(where: { $0.id == id }) {
            ScrollView {
                VStack(alignment: .leading, spacing: 14) {
                    header(a)
                    Divider()
                    promptSection(a)
                    Divider()
                    samplingSection(a)
                    Divider()
                    toolsSection(a)
                }
                .padding(16)
            }
        } else {
            VStack(spacing: 8) {
                Image(systemName: "person.crop.circle.badge.plus")
                    .font(.system(size: 36))
                    .foregroundStyle(.tertiary)
                Text(library.agents.isEmpty
                      ? "Define an agent to pin a system prompt + tool subset to a chat."
                      : "Select an agent to inspect its configuration.")
                    .foregroundStyle(.secondary)
                    .multilineTextAlignment(.center)
                if library.agents.isEmpty {
                    Button("New agent…") {
                        editing = AgentConfig(name: "Untitled agent")
                    }
                    .controlSize(.large)
                }
            }
            .frame(maxWidth: .infinity, maxHeight: .infinity)
            .padding()
        }
    }

    private func header(_ a: AgentConfig) -> some View {
        HStack(alignment: .top, spacing: 12) {
            Image(systemName: a.iconName)
                .font(.system(size: 28))
                .foregroundStyle(AgentTint.color(for: a.tint))
                .frame(width: 40)
            VStack(alignment: .leading, spacing: 2) {
                Text(a.name).font(.title3.bold())
                if !a.summary.isEmpty {
                    Text(a.summary)
                        .font(.callout)
                        .foregroundStyle(.secondary)
                }
                Text("Created \(a.createdAt.formatted(date: .abbreviated, time: .shortened))")
                    .font(.caption)
                    .foregroundStyle(.tertiary)
            }
            Spacer()
            Button("Edit…") { editing = a }
        }
    }

    private func promptSection(_ a: AgentConfig) -> some View {
        VStack(alignment: .leading, spacing: 6) {
            Text("System prompt").font(.headline)
            if a.systemPrompt.isEmpty {
                Text("No system prompt — agent behaves as a vanilla assistant.")
                    .font(.caption)
                    .foregroundStyle(.tertiary)
            } else {
                Text(a.systemPrompt)
                    .font(.callout.monospaced())
                    .textSelection(.enabled)
                    .fixedSize(horizontal: false, vertical: true)
                    .padding(8)
                    .frame(maxWidth: .infinity, alignment: .leading)
                    .background(Color(NSColor.textBackgroundColor),
                                 in: RoundedRectangle(cornerRadius: 6))
            }
        }
    }

    private func samplingSection(_ a: AgentConfig) -> some View {
        VStack(alignment: .leading, spacing: 6) {
            Text("Sampling defaults").font(.headline)
            HStack(alignment: .top, spacing: 24) {
                stat("Mode",        modeLabel(a.defaultMode))
                stat("Temperature", String(format: "%.2f", a.temperature))
                stat("Top-P",       String(format: "%.2f", a.topP))
                stat("Top-K",       a.topK == 0 ? "off" : "\(a.topK)")
                stat("Rep. penalty", String(format: "%.2f", a.repetitionPenalty))
                stat("Max tokens",  "\(a.maxTokens)")
            }
        }
    }

    private func stat(_ label: String, _ value: String) -> some View {
        VStack(alignment: .leading, spacing: 1) {
            Text(label)
                .font(.caption2)
                .foregroundStyle(.tertiary)
            Text(value)
                .font(.callout.monospacedDigit())
        }
    }

    @ViewBuilder
    private func toolsSection(_ a: AgentConfig) -> some View {
        VStack(alignment: .leading, spacing: 6) {
            Text("Tools").font(.headline)
            let allowed = a.allowedToolNames
            let available = mcpPool.allTools()
            if allowed == nil {
                Label("All connected MCP tools allowed.",
                       systemImage: "checkmark.seal.fill")
                    .font(.caption)
                    .foregroundStyle(.green)
            } else if allowed?.isEmpty == true {
                Label("No tools — agent runs prose-only.",
                       systemImage: "xmark.seal.fill")
                    .font(.caption)
                    .foregroundStyle(.secondary)
            } else if let allowed {
                VStack(alignment: .leading, spacing: 2) {
                    ForEach(Array(allowed).sorted(), id: \.self) { name in
                        HStack(spacing: 6) {
                            Image(systemName: "checkmark.circle")
                                .foregroundStyle(Color.green)
                                .font(.caption)
                            Text(name)
                                .font(.callout.monospaced())
                        }
                    }
                    let liveNames = Set(available.map(\.qualifiedName))
                    let stale = allowed.subtracting(liveNames)
                    if !stale.isEmpty {
                        Text("⚠︎ \(stale.count) name\(stale.count == 1 ? "" : "s") not currently connected.")
                            .font(.caption2)
                            .foregroundStyle(.orange)
                    }
                }
            }
        }
    }

    private func modeLabel(_ raw: String) -> String {
        switch raw {
        case "max":  return "Max"
        case "high": return "High"
        default:     return "Chat"
        }
    }
}
