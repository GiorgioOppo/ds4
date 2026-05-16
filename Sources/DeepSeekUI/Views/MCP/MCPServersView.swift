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
                        }
                        Button("Edit…") {
                            editing = s
                        }
                        Divider()
                        Button("Delete", role: .destructive) {
                            library.delete(s.id)
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
                set: { library.setEnabled(s.id, $0) }))
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

    private func statusFooter(_ s: MCPServerConfig) -> some View {
        // Step M1 has no live client yet. Every row shows "Idle"
        // until M2's MCPClientPool lands. The colour cue here is
        // intentionally dim so it doesn't look like a real status
        // indicator before it can report anything real.
        HStack(spacing: 6) {
            Image(systemName: "circle")
                .foregroundStyle(Color.secondary)
            Text("Idle — connection pool not implemented yet")
                .font(.caption)
                .foregroundStyle(.tertiary)
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
