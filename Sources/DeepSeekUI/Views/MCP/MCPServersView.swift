import SwiftUI
#if canImport(AppKit)
import AppKit
#endif

/// Settings tab: registry of MCP (Model Context Protocol) servers.
/// Manages metadata only — process spawning + JSON-RPC traffic lands
/// in Step M2 (`MCPClientPool`). The "Status" column currently shows
/// "Idle" for every row until that client lands.
struct MCPServersView: View {
    @ObservedObject var library: MCPServerLibrary
    @ObservedObject var pool: MCPClientPool

    @State private var selectedID: UUID?
    @State private var editing: MCPServerConfig?
    @State private var showImport: Bool = false
    @State private var importError: String?

    var body: some View {
        HStack(spacing: 0) {
            sidebar
                .frame(width: 220)
                .background(Color(NSColor.controlBackgroundColor))
            Divider()
            detail
                .frame(maxWidth: .infinity, maxHeight: .infinity)
        }
        .sheet(item: $editing) { server in
            MCPServerEditSheet(initial: server) { updated in
                if library.servers.contains(where: { $0.id == updated.id }) {
                    library.update(updated)
                } else {
                    library.add(updated)
                }
                pool.librarySynced(library)
                if selectedID == nil { selectedID = updated.id }
            }
        }
        .alert("Import failed",
               isPresented: Binding(get: { importError != nil },
                                     set: { _ in importError = nil })) {
            Button("OK") { importError = nil }
        } message: {
            Text(importError ?? "")
        }
    }

    // MARK: - sidebar

    private var sidebar: some View {
        VStack(spacing: 0) {
            HStack {
                Text("MCP servers").font(.headline)
                Spacer()
                Menu {
                    Button {
                        editing = MCPServerConfig(
                            name: "new-server",
                            command: "npx")
                    } label: {
                        Label("New server…", systemImage: "plus")
                    }
                    Button {
                        importFromFile()
                    } label: {
                        Label("Import from Claude Desktop config…",
                               systemImage: "tray.and.arrow.down")
                    }
                } label: {
                    Image(systemName: "plus")
                }
                .menuStyle(.borderlessButton)
                .frame(width: 28)
            }
            .padding(.horizontal, 10)
            .padding(.vertical, 8)
            Divider()
            List(selection: $selectedID) {
                ForEach(library.servers) { s in
                    HStack(spacing: 8) {
                        Image(systemName: s.enabled
                               ? "circle.fill"
                               : "circle.dotted")
                            .foregroundStyle(s.enabled
                                              ? Color.green
                                              : Color.secondary)
                            .font(.caption2)
                        VStack(alignment: .leading, spacing: 1) {
                            Text(s.name).lineLimit(1)
                            Text(s.command)
                                .font(.caption2.monospaced())
                                .foregroundStyle(.tertiary)
                                .lineLimit(1)
                                .truncationMode(.middle)
                        }
                    }
                    .tag(s.id)
                    .contextMenu {
                        Button(s.enabled ? "Disable" : "Enable") {
                            library.setEnabled(s.id, !s.enabled)
                            pool.librarySynced(library)
                        }
                        Button("Edit…") {
                            editing = s
                        }
                        Divider()
                        Button("Delete", role: .destructive) {
                            library.delete(s.id)
                            pool.librarySynced(library)
                            if selectedID == s.id { selectedID = nil }
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
           let s = library.servers.first(where: { $0.id == id }) {
            VStack(alignment: .leading, spacing: 12) {
                header(s)
                Divider()
                spawnSection(s)
                Divider()
                envSection(s)
                Spacer()
                statusFooter(s)
            }
            .padding(16)
        } else {
            VStack(spacing: 8) {
                Image(systemName: "server.rack")
                    .font(.system(size: 36))
                    .foregroundStyle(.tertiary)
                Text(library.servers.isEmpty
                      ? "Register an MCP server to grant the model external tools."
                      : "Select a server to inspect its configuration.")
                    .foregroundStyle(.secondary)
                    .multilineTextAlignment(.center)
                if library.servers.isEmpty {
                    Button("Add server…") {
                        editing = MCPServerConfig(
                            name: "new-server",
                            command: "npx")
                    }
                    .controlSize(.large)
                }
            }
            .frame(maxWidth: .infinity, maxHeight: .infinity)
            .padding()
        }
    }

    private func header(_ s: MCPServerConfig) -> some View {
        HStack(alignment: .firstTextBaseline) {
            VStack(alignment: .leading, spacing: 2) {
                Text(s.name).font(.title3.bold())
                Text("Added \(s.createdAt.formatted(date: .abbreviated, time: .shortened))")
                    .font(.caption)
                    .foregroundStyle(.tertiary)
            }
            Spacer()
            Toggle("Enabled", isOn: Binding(
                get: { s.enabled },
                set: {
                    library.setEnabled(s.id, $0)
                    pool.librarySynced(library)
                }))
                .toggleStyle(.switch)
            Button("Edit…") { editing = s }
        }
    }

    private func spawnSection(_ s: MCPServerConfig) -> some View {
        VStack(alignment: .leading, spacing: 6) {
            Text("Spawn command").font(.headline)
            HStack(alignment: .top, spacing: 8) {
                Text(s.command)
                    .font(.body.monospaced())
                if !s.args.isEmpty {
                    Text(s.args.joined(separator: " "))
                        .font(.body.monospaced())
                        .foregroundStyle(.secondary)
                        .lineLimit(2)
                        .truncationMode(.middle)
                } else {
                    Text("(no arguments)")
                        .font(.body)
                        .foregroundStyle(.tertiary)
                }
            }
            .padding(8)
            .frame(maxWidth: .infinity, alignment: .leading)
            .background(Color(NSColor.textBackgroundColor))
            .clipShape(RoundedRectangle(cornerRadius: 6))
        }
    }

    @ViewBuilder
    private func envSection(_ s: MCPServerConfig) -> some View {
        VStack(alignment: .leading, spacing: 6) {
            Text("Environment").font(.headline)
            if s.env.isEmpty {
                Text("No extra variables.")
                    .font(.caption)
                    .foregroundStyle(.tertiary)
            } else {
                VStack(alignment: .leading, spacing: 2) {
                    ForEach(s.env.keys.sorted(), id: \.self) { k in
                        let v = s.env[k] ?? ""
                        HStack(alignment: .firstTextBaseline) {
                            Text(k)
                                .font(.callout.monospaced())
                            Text("=")
                                .foregroundStyle(.tertiary)
                            Text(masked(v))
                                .font(.callout.monospaced())
                                .foregroundStyle(.secondary)
                                .lineLimit(1)
                                .truncationMode(.middle)
                        }
                    }
                }
                .padding(8)
                .frame(maxWidth: .infinity, alignment: .leading)
                .background(Color(NSColor.textBackgroundColor))
                .clipShape(RoundedRectangle(cornerRadius: 6))
            }
        }
    }

    @ViewBuilder
    private func statusFooter(_ s: MCPServerConfig) -> some View {
        if let client = pool.client(forServer: s.id) {
            StatusRow(observing: client)
        } else if !s.enabled {
            HStack(spacing: 6) {
                Image(systemName: "circle.slash")
                    .foregroundStyle(Color.secondary)
                Text("Disabled — flip the switch above to spawn the server")
                    .font(.caption)
                    .foregroundStyle(.tertiary)
            }
        } else {
            HStack(spacing: 6) {
                Image(systemName: "circle")
                    .foregroundStyle(Color.secondary)
                Text("Not connected yet")
                    .font(.caption)
                    .foregroundStyle(.tertiary)
            }
        }
    }

    /// Live status line that observes one MCPClient and re-renders
    /// when its @Published `status` / `tools` change.
    private struct StatusRow: View {
        @ObservedObject var observing: MCPClient

        var body: some View {
            VStack(alignment: .leading, spacing: 6) {
                HStack(spacing: 8) {
                    iconAndLabel
                    Spacer()
                    Button("Reconnect") {
                        Task {
                            observing.disconnect()
                            await observing.connect()
                        }
                    }
                    .controlSize(.small)
                }
                if case .connected = observing.status, !observing.tools.isEmpty {
                    toolList
                }
            }
        }

        @ViewBuilder
        private var iconAndLabel: some View {
            switch observing.status {
            case .idle:
                HStack(spacing: 6) {
                    Image(systemName: "circle")
                        .foregroundStyle(Color.secondary)
                    Text("Idle").foregroundStyle(.secondary).font(.caption)
                }
            case .connecting:
                HStack(spacing: 6) {
                    ProgressView().controlSize(.mini)
                    Text("Connecting…").font(.caption)
                        .foregroundStyle(.secondary)
                }
            case .connected(let n):
                HStack(spacing: 6) {
                    Image(systemName: "checkmark.circle.fill")
                        .foregroundStyle(Color.green)
                    Text("Connected · \(n) tool\(n == 1 ? "" : "s")")
                        .font(.caption)
                }
            case .error(let msg):
                HStack(spacing: 6) {
                    Image(systemName: "exclamationmark.octagon.fill")
                        .foregroundStyle(Color.red)
                    Text(msg)
                        .font(.caption)
                        .foregroundStyle(.red)
                        .lineLimit(2)
                        .truncationMode(.middle)
                }
            }
        }

        @ViewBuilder
        private var toolList: some View {
            DisclosureGroup("Tools (\(observing.tools.count))") {
                VStack(alignment: .leading, spacing: 4) {
                    ForEach(observing.tools, id: \.self) { t in
                        VStack(alignment: .leading, spacing: 1) {
                            Text(t.toolName)
                                .font(.callout.monospaced())
                            if !t.description.isEmpty {
                                Text(t.description)
                                    .font(.caption2)
                                    .foregroundStyle(.secondary)
                                    .lineLimit(2)
                            }
                        }
                        .padding(.vertical, 2)
                    }
                }
                .padding(.leading, 8)
            }
            .font(.caption)
        }
    }

    // MARK: - import

    private func importFromFile() {
        #if canImport(AppKit)
        let panel = NSOpenPanel()
        panel.allowsMultipleSelection = false
        panel.canChooseDirectories = false
        panel.canChooseFiles = true
        panel.title = "Choose a Claude Desktop config JSON"
        guard panel.runModal() == .OK, let url = panel.url else { return }
        do {
            let data = try Data(contentsOf: url)
            let n = try library.importClaudeDesktopJSON(data)
            if n == 0 {
                importError = "No new servers found in that file. (Existing names are skipped.)"
            } else {
                pool.librarySynced(library)
            }
        } catch {
            importError = "Import failed: \(error.localizedDescription)"
        }
        #endif
    }

    /// Hide environment values that look like secrets (any non-empty
    /// value containing "KEY", "TOKEN", "SECRET", "PASS" in the
    /// variable name's letters, or longer than 20 chars). The full
    /// value is still on disk in mcp.json — this is just to keep the
    /// settings UI screenshot-safe.
    private func masked(_ value: String) -> String {
        if value.isEmpty { return "" }
        if value.count <= 6 { return String(repeating: "•", count: value.count) }
        let head = value.prefix(3)
        let tail = value.suffix(3)
        return "\(head)…\(tail)"
    }
}
