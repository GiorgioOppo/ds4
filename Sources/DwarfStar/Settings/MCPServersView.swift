import SwiftUI
import AppKit
import UniformTypeIdentifiers
import DS4Engine
import DS4Core

// MCP panel: configure the MCP servers the app connects to as a client. The
// persisted configs live here (UserDefaults, `mcpServers`-compatible JSON via
// import/export); the live connections belong to MCPManager.shared, which the
// chat/agents/distributed tool paths read directly.

/// Owns the persisted server configs and mirrors MCPManager's live statuses
/// for SwiftUI. Created once at app launch so enabled servers connect
/// immediately, not only when the panel is first opened.
@MainActor
@Observable
final class MCPStore {
    private static let defaultsKey = "DS4MCPServers"

    var configs: [MCPServerConfig] = [] {
        didSet {
            persist()
            MCPManager.shared.setConfigs(configs)   // its change handler refreshes `statuses`
        }
    }
    /// Live snapshot mirrored from MCPManager (pushed by its change handler).
    var statuses: [MCPServerStatus] = []

    init() {
        if let data = UserDefaults.standard.data(forKey: Self.defaultsKey),
           let saved = try? JSONDecoder().decode([MCPServerConfig].self, from: data) {
            configs = saved
        }
        // Connection state changes arrive from arbitrary threads; mirror them
        // into this observable so the panel re-renders without polling.
        MCPManager.shared.addChangeHandler { [weak self] in
            Task { @MainActor in self?.refresh() }
        }
        MCPManager.shared.setConfigs(configs)   // didSet doesn't fire in init
        refresh()
    }

    private func persist() {
        if let data = try? JSONEncoder().encode(configs) {
            UserDefaults.standard.set(data, forKey: Self.defaultsKey)
        }
    }

    /// Pull the manager's current snapshot into the observable mirror.
    func refresh() {
        statuses = MCPManager.shared.statuses()
    }

    // MARK: Mutations

    func add(_ config: MCPServerConfig) -> String? {
        guard !configs.contains(where: { $0.id == config.id }) else {
            return "A server named '\(config.name)' (id \(config.id)) already exists."
        }
        configs.append(config)
        return nil
    }

    func remove(id: String) { configs.removeAll { $0.id == id } }

    func setEnabled(id: String, _ enabled: Bool) {
        guard let i = configs.firstIndex(where: { $0.id == id }) else { return }
        configs[i].enabled = enabled
    }

    func reconnect(id: String) {
        MCPManager.shared.reconnect(id: id)
    }

    /// Merge servers from `mcpServers` JSON (Claude Desktop / Cursor format).
    /// Existing ids are replaced. Returns (imported count, error).
    func importJSON(_ json: String) -> (Int, String?) {
        do {
            let imported = try MCPServerConfig.importJSON(json)
            var merged = configs
            for server in imported {
                if let i = merged.firstIndex(where: { $0.id == server.id }) { merged[i] = server }
                else { merged.append(server) }
            }
            configs = merged
            return (imported.count, nil)
        } catch {
            return (0, "\(error)")
        }
    }

    var exportedJSON: String { MCPServerConfig.exportJSON(configs) }
}

/// The MCP sidebar panel: server list with live status and tools, add form,
/// and JSON import/export.
struct MCPServersView: View {
    @Bindable var store: MCPStore
    @State private var showAdd = false
    @State private var ioMessage = ""

    var body: some View {
        Form {
            Section {
                Text("MCP (Model Context Protocol) servers extend the model with external tools: at connect time each server declares its tools, which then appear next to the built-ins in the chat Tool picker and in each agent's tool list (named mcp_<server>_<tool>). stdio servers are launched by the app as child processes; HTTP servers are reached at their URL. In a sandboxed (App Store) build, child processes inherit the sandbox — servers that need broader access should run externally and be reached over HTTP.")
                    .font(.caption).foregroundStyle(.secondary)
            }

            if store.statuses.isEmpty {
                Section {
                    Text("No MCP servers configured.").foregroundStyle(.secondary)
                }
            }

            ForEach(store.statuses) { server in
                Section {
                    MCPServerRow(store: store, server: server)
                }
            }

            Section {
                Button { showAdd = true } label: {
                    Label("Add Server…", systemImage: "plus")
                }
                HStack {
                    Button { exportJSON() } label: {
                        Label("Export JSON…", systemImage: "square.and.arrow.up")
                    }
                    Button { importJSON() } label: {
                        Label("Import JSON…", systemImage: "square.and.arrow.down")
                    }
                }
                if !ioMessage.isEmpty {
                    Text(ioMessage).font(.caption).foregroundStyle(.secondary)
                }
                Text("Import/Export uses the standard {\"mcpServers\": …} format shared with Claude Desktop, Cursor, and VS Code.")
                    .font(.caption).foregroundStyle(.secondary)
            }
        }
        .formStyle(.grouped)
        .sheet(isPresented: $showAdd) {
            MCPAddServerView(store: store)
        }
    }

    private func exportJSON() {
        let panel = NSSavePanel()
        panel.allowedContentTypes = [.json]
        panel.nameFieldStringValue = "mcp-servers.json"
        guard panel.runModal() == .OK, let url = panel.url else { return }
        do {
            try store.exportedJSON.data(using: .utf8)?.write(to: url)
            ioMessage = "Exported \(store.configs.count) server(s)."
        } catch {
            ioMessage = "Export failed: \(error.localizedDescription)"
        }
    }

    private func importJSON() {
        let panel = NSOpenPanel()
        panel.allowedContentTypes = [.json]
        panel.allowsMultipleSelection = false
        guard panel.runModal() == .OK, let url = panel.url,
              let json = try? String(contentsOf: url, encoding: .utf8) else { return }
        let (count, error) = store.importJSON(json)
        ioMessage = error.map { "Import failed: \($0)" } ?? "Imported \(count) server(s)."
    }
}

/// One configured server: status line, enable toggle, actions, tool list.
private struct MCPServerRow: View {
    @Bindable var store: MCPStore
    let server: MCPServerStatus

    var body: some View {
        HStack {
            statusDot
            VStack(alignment: .leading) {
                Text(server.name).font(.headline)
                Text(server.transportSummary)
                    .font(.caption.monospaced()).foregroundStyle(.secondary)
                    .lineLimit(1).truncationMode(.middle)
            }
            Spacer()
            Toggle("", isOn: Binding(
                get: { server.enabled },
                set: { store.setEnabled(id: server.id, $0) }))
                .toggleStyle(.switch)
                .labelsHidden()
                .help(server.enabled ? "Disable (disconnect)" : "Enable (connect)")
        }

        HStack(spacing: 12) {
            statusText
            Spacer()
            if server.enabled {
                Button("Reconnect") { store.reconnect(id: server.id) }
                    .font(.caption)
            }
            Button(role: .destructive) { store.remove(id: server.id) } label: {
                Image(systemName: "trash")
            }
            .help("Remove server")
        }

        if case .connected = server.state, !server.tools.isEmpty {
            // `tools` are the namespaced specs the model actually sees, so the
            // listed names are exact even after a cross-server name collision.
            DisclosureGroup("Tools (\(server.tools.count))") {
                ForEach(server.tools) { tool in
                    VStack(alignment: .leading) {
                        Text(tool.name).font(.body.monospaced())
                        if !tool.description.isEmpty {
                            Text(tool.description)
                                .font(.caption).foregroundStyle(.secondary).lineLimit(3)
                        }
                    }
                    .padding(.vertical, 2)
                }
            }
        }
    }

    private var statusDot: some View {
        Circle().frame(width: 9, height: 9).foregroundStyle(dotColor)
    }

    private var dotColor: Color {
        switch server.state {
        case .connected: return .green
        case .connecting: return .yellow
        case .failed: return .red
        case .disabled: return .gray
        }
    }

    @ViewBuilder
    private var statusText: some View {
        switch server.state {
        case .connected:
            Text("Connected" + (server.serverInfo.isEmpty ? "" : " · \(server.serverInfo)")
                 + " · \(server.tools.count) tool(s)")
                .font(.caption).foregroundStyle(.secondary)
        case .connecting:
            Text("Connecting…").font(.caption).foregroundStyle(.secondary)
        case .failed(let reason):
            Text(reason).font(.caption).foregroundStyle(.red)
                .lineLimit(3).textSelection(.enabled)
        case .disabled:
            Text("Disabled").font(.caption).foregroundStyle(.secondary)
        }
    }
}

/// Sheet to add a server manually (stdio command or HTTP URL).
private struct MCPAddServerView: View {
    @Bindable var store: MCPStore
    @Environment(\.dismiss) private var dismiss

    private enum Kind: String, CaseIterable, Identifiable {
        case stdio = "stdio (local command)"
        case http = "HTTP (remote URL)"
        var id: String { rawValue }
    }

    @State private var name = ""
    @State private var kind: Kind = .stdio
    @State private var command = ""
    @State private var arguments = ""
    @State private var environment = ""
    @State private var url = ""
    @State private var headers = ""
    @State private var error = ""

    var body: some View {
        VStack(alignment: .leading, spacing: 12) {
            Text("Add MCP Server").font(.title2).bold()

            TextField("Name (e.g. filesystem, github)", text: $name)
            Picker("Transport", selection: $kind) {
                ForEach(Kind.allCases) { Text($0.rawValue).tag($0) }
            }
            .pickerStyle(.segmented)

            switch kind {
            case .stdio:
                TextField("Command (e.g. npx, uvx, /path/to/server)", text: $command)
                    .font(.body.monospaced())
                TextField("Arguments (space-separated; quote paths with spaces: \"/Users/Me/My Docs\")",
                          text: $arguments)
                    .font(.body.monospaced())
                TextField("Environment (KEY=value, one per line)", text: $environment, axis: .vertical)
                    .font(.body.monospaced())
                    .lineLimit(2...4)
                Text("The command is resolved via PATH (including /usr/local/bin and /opt/homebrew/bin) and spawned as a child process speaking JSON-RPC on stdin/stdout.")
                    .font(.caption).foregroundStyle(.secondary)
            case .http:
                TextField("URL (e.g. https://example.com/mcp)", text: $url)
                    .font(.body.monospaced())
                TextField("Headers (Name: value, one per line)", text: $headers, axis: .vertical)
                    .font(.body.monospaced())
                    .lineLimit(2...4)
                Text("Streamable-HTTP transport: JSON-RPC is POSTed to the URL. Use the headers for an Authorization token if the server requires one.")
                    .font(.caption).foregroundStyle(.secondary)
            }

            if !error.isEmpty {
                Text(error).font(.caption).foregroundStyle(.red)
            }

            HStack {
                Spacer()
                Button("Cancel") { dismiss() }.keyboardShortcut(.cancelAction)
                Button("Add") { add() }.keyboardShortcut(.defaultAction)
                    .disabled(name.trimmingCharacters(in: .whitespaces).isEmpty)
            }
        }
        .padding()
        .frame(minWidth: 520)
    }

    private func add() {
        let trimmedName = name.trimmingCharacters(in: .whitespaces)
        let transport: MCPTransportConfig
        switch kind {
        case .stdio:
            let cmd = command.trimmingCharacters(in: .whitespaces)
            guard !cmd.isEmpty else { error = "Command is required."; return }
            transport = .stdio(command: cmd, arguments: Self.tokenizeArguments(arguments),
                               environment: Self.parsePairs(environment, separator: "="))
        case .http:
            let u = url.trimmingCharacters(in: .whitespaces)
            guard u.lowercased().hasPrefix("http://") || u.lowercased().hasPrefix("https://") else {
                error = "URL must start with http:// or https://."; return
            }
            transport = .http(url: u, headers: Self.parsePairs(headers, separator: ":"))
        }
        if let problem = store.add(MCPServerConfig(name: trimmedName, transport: transport)) {
            error = problem
            return
        }
        dismiss()
    }

    /// Split the arguments field shell-style: spaces separate, "…" and '…'
    /// group (there is no shell in between, so quotes must be handled here —
    /// a path like "/Users/John Smith/Docs" is ONE argument).
    static func tokenizeArguments(_ text: String) -> [String] {
        var out: [String] = []
        var current = ""
        var quote: Character?
        var hasToken = false
        for ch in text {
            if let q = quote {
                if ch == q { quote = nil } else { current.append(ch) }
            } else if ch == "\"" || ch == "'" {
                quote = ch; hasToken = true
            } else if ch == " " || ch == "\t" || ch.isNewline {
                if hasToken { out.append(current); current = ""; hasToken = false }
            } else {
                current.append(ch); hasToken = true
            }
        }
        if hasToken { out.append(current) }
        return out
    }

    /// Parse "KEY=value" / "Name: value" lines into a dictionary.
    private static func parsePairs(_ text: String, separator: Character) -> [String: String] {
        var out: [String: String] = [:]
        for line in text.split(whereSeparator: \.isNewline) {
            guard let sep = line.firstIndex(of: separator) else { continue }
            let key = line[..<sep].trimmingCharacters(in: .whitespaces)
            let value = line[line.index(after: sep)...].trimmingCharacters(in: .whitespaces)
            if !key.isEmpty { out[key] = value }
        }
        return out
    }
}
