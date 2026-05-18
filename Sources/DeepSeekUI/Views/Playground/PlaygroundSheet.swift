import SwiftUI
import DeepSeekTools

/// Sheet "playground" che dimostra il pattern OOP introdotto in
/// `Sources/DeepSeekTools/Abstractions/`:
///   - una `Chat` (DemoEchoChat) sopra un `AgentBase` (DemoEchoAgent),
///   - un `PluginRegistry` con un `DemoLoggerPlugin` osservatore,
///   - envelope `Question` / `Answer` che attraversano entrambi.
///
/// Non è una chat di produzione — l'agent è un echo finto. Serve
/// come riferimento visuale del pattern per chi vuole capire
/// come si compongono i pezzi.
struct PlaygroundSheet: View {
    @StateObject private var vm = PlaygroundViewModel()
    @Environment(\.dismiss) private var dismiss

    var body: some View {
        VStack(spacing: 0) {
            header
            Divider()
            HStack(spacing: 0) {
                chatPane
                    .frame(maxWidth: .infinity, maxHeight: .infinity)
                Divider()
                pluginPane
                    .frame(width: 280)
                    .background(Color(NSColor.controlBackgroundColor))
            }
            Divider()
            input
                .padding(12)
        }
        .frame(minWidth: 760, minHeight: 540)
    }

    // ---- Header ----

    @ViewBuilder
    private var header: some View {
        HStack {
            Image(systemName: "puzzlepiece.extension")
                .font(.title2)
                .foregroundStyle(.tint)
            VStack(alignment: .leading) {
                Text("OOP Playground").font(.headline)
                Text("Demo del pattern Chat → Agent → Plugin. " +
                     "Vedi Sources/DeepSeekTools/Abstractions/.")
                    .font(.caption)
                    .foregroundStyle(.secondary)
            }
            Spacer()
            Button {
                vm.clearAll()
            } label: {
                Label("Clear", systemImage: "trash")
            }
            .controlSize(.small)
            .disabled(vm.chatSnapshot.turns.isEmpty
                       && vm.pluginEvents.isEmpty)
            Button {
                dismiss()
            } label: {
                Image(systemName: "xmark.circle.fill")
                    .foregroundStyle(.secondary)
            }
            .buttonStyle(.borderless)
            .keyboardShortcut(.cancelAction)
        }
        .padding(12)
    }

    // ---- Chat history ----

    @ViewBuilder
    private var chatPane: some View {
        VStack(alignment: .leading, spacing: 0) {
            sectionHeader("Chat history",
                          subtitle: "Q&A pairs handled by DemoEchoChat.ask(...)")
            Divider()
            if vm.chatSnapshot.turns.isEmpty {
                emptyState(icon: "bubble.left.and.bubble.right",
                            text: "Scrivi un prompt sotto e premi Send.")
            } else {
                ScrollView {
                    VStack(alignment: .leading, spacing: 12) {
                        ForEach(Array(vm.chatSnapshot.turns.enumerated()),
                                 id: \.offset) { idx, turn in
                            turnView(turn: turn, index: idx)
                        }
                    }
                    .padding(12)
                }
            }
        }
    }

    @ViewBuilder
    private func turnView(turn: ChatTurn, index: Int) -> some View {
        VStack(alignment: .leading, spacing: 6) {
            Label {
                Text(turn.question.content)
                    .textSelection(.enabled)
            } icon: {
                Image(systemName: "person.circle.fill")
                    .foregroundStyle(.blue)
            }
            .padding(.leading, 4)

            if let a = turn.answer {
                Label {
                    Text(a.content)
                        .textSelection(.enabled)
                        .font(.system(.body, design: .monospaced))
                } icon: {
                    Image(systemName: "cpu")
                        .foregroundStyle(.purple)
                }
                .padding(.leading, 4)
            } else {
                Label {
                    Text("(generating…)")
                        .foregroundStyle(.tertiary)
                } icon: {
                    ProgressView().controlSize(.small)
                }
                .padding(.leading, 4)
            }

            Divider()
        }
    }

    // ---- Plugin event log ----

    @ViewBuilder
    private var pluginPane: some View {
        VStack(alignment: .leading, spacing: 0) {
            sectionHeader("Plugin events",
                          subtitle: "DemoLoggerPlugin.observe(envelope:)")
            Divider()
            if vm.pluginEvents.isEmpty {
                emptyState(icon: "bell.slash",
                            text: "No envelope yet.")
            } else {
                ScrollView {
                    VStack(alignment: .leading, spacing: 4) {
                        ForEach(vm.pluginEvents) { entry in
                            HStack(spacing: 6) {
                                Text("#\(entry.index)")
                                    .font(.caption2.monospacedDigit())
                                    .foregroundStyle(.tertiary)
                                Text(entry.kind)
                                    .font(.system(.caption, design: .monospaced))
                                    .padding(.horizontal, 4)
                                    .padding(.vertical, 1)
                                    .background(badgeColor(for: entry.kind).opacity(0.15))
                                    .foregroundStyle(badgeColor(for: entry.kind))
                                    .cornerRadius(3)
                                Spacer()
                            }
                        }
                    }
                    .padding(8)
                }
            }
        }
    }

    private func badgeColor(for kind: String) -> Color {
        switch kind {
        case "chat.question": return .blue
        case "chat.answer":   return .purple
        case "agent.token":   return .orange
        case "agent.done":    return .green
        case "model.delta":   return .pink
        default:              return .gray
        }
    }

    // ---- Input bar ----

    @ViewBuilder
    private var input: some View {
        VStack(alignment: .leading, spacing: 8) {
            if let err = vm.lastError {
                Label(err, systemImage: "exclamationmark.triangle.fill")
                    .foregroundStyle(.red)
                    .font(.caption)
            }
            HStack(spacing: 8) {
                TextField("Prompt (premi Invio per Send)", text: $vm.inputText)
                    .textFieldStyle(.roundedBorder)
                    .onSubmit { vm.send() }
                    .disabled(vm.isRunning)
                Button {
                    vm.send()
                } label: {
                    if vm.isRunning {
                        ProgressView().controlSize(.small)
                            .frame(width: 60)
                    } else {
                        Label("Send", systemImage: "paperplane.fill")
                            .frame(width: 60)
                    }
                }
                .keyboardShortcut(.defaultAction)
                .disabled(!vm.canSend)
            }
            Text("DemoEchoAgent prepende `echo:` al testo. Plugin viewer " +
                 "registra ogni envelope pubblicato dalla chat.")
                .font(.caption2)
                .foregroundStyle(.tertiary)
        }
    }

    // ---- Helpers ----

    @ViewBuilder
    private func sectionHeader(_ title: String, subtitle: String) -> some View {
        VStack(alignment: .leading, spacing: 2) {
            Text(title).font(.subheadline.bold())
            Text(subtitle).font(.caption2).foregroundStyle(.tertiary)
        }
        .padding(.horizontal, 12)
        .padding(.vertical, 8)
    }

    @ViewBuilder
    private func emptyState(icon: String, text: String) -> some View {
        VStack(spacing: 6) {
            Image(systemName: icon)
                .font(.system(size: 28))
                .foregroundStyle(.tertiary)
            Text(text)
                .font(.callout)
                .foregroundStyle(.secondary)
                .multilineTextAlignment(.center)
        }
        .frame(maxWidth: .infinity, maxHeight: .infinity)
    }
}
