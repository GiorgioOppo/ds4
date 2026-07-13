import SwiftUI
import DS4Engine
import DS4Core

/// Sheet to enable tools and choose which built-ins are exposed to the model.
struct ToolPickerView: View {
    @Bindable var store: ChatStore
    @Environment(\.dismiss) private var dismiss

    var body: some View {
        VStack(alignment: .leading, spacing: 12) {
            Text("Tool").font(.title2).bold()
            Toggle("Enable tools (function calling)", isOn: $store.toolsEnabled)
                .onChange(of: store.toolsEnabled) { store.syncTools() }
            Text("When enabled, selected tools are declared to the model. Built-in tools run automatically; for other tools you can enter the result manually.")
                .font(.caption).foregroundStyle(.secondary)

            Divider()
            ScrollView {
                VStack(alignment: .leading, spacing: 12) {
                    Text("Built-in Tools").font(.headline)
                    toolToggles(store.builtinTools)
                    if !store.mcpTools.isEmpty {
                        Divider()
                        Text("MCP Tools").font(.headline)
                        Text("Exposed by the connected MCP servers (configured in the MCP panel); they run on the external server.")
                            .font(.caption).foregroundStyle(.secondary)
                        toolToggles(store.mcpTools)
                    }
                }
                .frame(maxWidth: .infinity, alignment: .leading)
            }

            Divider()
            Toggle("Compact declaration (name + parameters only)", isOn: $store.compactTools)
                .onChange(of: store.compactTools) { store.syncTools() }
                .disabled(!store.toolsEnabled)
            Text("Fewer prefill tokens: sends only `name(parameters)` plus one format line instead of the full schema. Cheaper, but less faithful to the training text.")
                .font(.caption).foregroundStyle(.secondary)

            Spacer()
            HStack {
                Spacer()
                Button("Close") { dismiss() }.keyboardShortcut(.defaultAction)
            }
        }
        .padding()
        .frame(minWidth: 460, minHeight: 360, maxHeight: 640)
    }

    /// One enable/disable toggle per tool spec, wired to `enabledToolNames`.
    @ViewBuilder
    private func toolToggles(_ tools: [ToolSpec]) -> some View {
        ForEach(tools) { tool in
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
                        .lineLimit(3)
                }
            }
            .disabled(!store.toolsEnabled)
        }
    }
}

/// Sheet to enter results for tool calls that aren't built-in.
struct ManualToolResultsView: View {
    @Bindable var store: ChatStore
    @State private var contents: [String: String] = [:]

    var body: some View {
        VStack(alignment: .leading, spacing: 12) {
            Text("Tool Results").font(.title2).bold()
            Text("The model called tools that are not built in. Enter a result, ideally JSON, for each one and submit to continue.")
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
                Button("Cancel") { store.cancelManualResults() }
                Spacer()
                Button("Submit Results") { store.submitManualResults(contents) }
                    .keyboardShortcut(.defaultAction)
            }
        }
        .padding()
        .frame(minWidth: 480, minHeight: 320)
    }
}

