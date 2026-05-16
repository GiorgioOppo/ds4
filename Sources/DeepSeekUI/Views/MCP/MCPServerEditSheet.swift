import SwiftUI

/// Modal sheet that edits one `MCPServerConfig`. Backed by a local
/// editable copy so Cancel really discards changes; commit replaces
/// the entry in the library via the caller's `onSave` closure.
///
/// Args + env are surfaced as multi-line text fields (one per line)
/// because that maps cleanly to what the user pastes from a Claude
/// Desktop JSON — array commas would be a nuisance. Parser is
/// permissive: empty lines are skipped, `KEY=VAL` for env, raw
/// strings for args.
struct MCPServerEditSheet: View {
    let initial: MCPServerConfig
    let onSave: (MCPServerConfig) -> Void

    @Environment(\.dismiss) private var dismiss

    @State private var name: String = ""
    @State private var command: String = ""
    @State private var argsText: String = ""
    @State private var envText: String = ""
    @State private var enabled: Bool = true

    var body: some View {
        VStack(alignment: .leading, spacing: 12) {
            Text(initial.name == "new-server"
                  ? "New MCP server"
                  : "Edit \(initial.name)")
                .font(.title3.bold())
            Form {
                TextField("Name", text: $name)
                TextField("Command", text: $command,
                           prompt: Text("npx | uvx | python | …"))
                VStack(alignment: .leading, spacing: 4) {
                    Text("Arguments (one per line)")
                        .font(.caption)
                        .foregroundStyle(.secondary)
                    TextEditor(text: $argsText)
                        .font(.body.monospaced())
                        .frame(minHeight: 80, maxHeight: 120)
                        .overlay(RoundedRectangle(cornerRadius: 4)
                            .stroke(Color.secondary.opacity(0.3)))
                }
                VStack(alignment: .leading, spacing: 4) {
                    Text("Environment (KEY=VALUE, one per line)")
                        .font(.caption)
                        .foregroundStyle(.secondary)
                    TextEditor(text: $envText)
                        .font(.body.monospaced())
                        .frame(minHeight: 60, maxHeight: 120)
                        .overlay(RoundedRectangle(cornerRadius: 4)
                            .stroke(Color.secondary.opacity(0.3)))
                }
                Toggle("Enabled", isOn: $enabled)
            }
            .formStyle(.grouped)
            Spacer()
            HStack {
                Spacer()
                Button("Cancel") { dismiss() }
                Button("Save") { commit() }
                    .keyboardShortcut(.return, modifiers: .command)
                    .disabled(!canSave)
                    .buttonStyle(.borderedProminent)
            }
        }
        .padding(20)
        .frame(width: 520, height: 520)
        .onAppear { hydrate() }
    }

    private var canSave: Bool {
        !name.trimmingCharacters(in: .whitespaces).isEmpty
            && !command.trimmingCharacters(in: .whitespaces).isEmpty
    }

    private func hydrate() {
        name = initial.name
        command = initial.command
        argsText = initial.args.joined(separator: "\n")
        envText = initial.env
            .sorted { $0.key < $1.key }
            .map { "\($0.key)=\($0.value)" }
            .joined(separator: "\n")
        enabled = initial.enabled
    }

    private func commit() {
        var updated = initial
        updated.name = name.trimmingCharacters(in: .whitespaces)
        updated.command = command.trimmingCharacters(in: .whitespaces)
        updated.args = argsText
            .split(separator: "\n", omittingEmptySubsequences: true)
            .map { String($0).trimmingCharacters(in: .whitespaces) }
            .filter { !$0.isEmpty }
        var env: [String: String] = [:]
        for line in envText.split(separator: "\n", omittingEmptySubsequences: true) {
            let s = String(line).trimmingCharacters(in: .whitespaces)
            guard let eq = s.firstIndex(of: "=") else { continue }
            let k = String(s[..<eq]).trimmingCharacters(in: .whitespaces)
            let v = String(s[s.index(after: eq)...]).trimmingCharacters(in: .whitespaces)
            guard !k.isEmpty else { continue }
            env[k] = v
        }
        updated.env = env
        updated.enabled = enabled
        onSave(updated)
        dismiss()
    }
}
