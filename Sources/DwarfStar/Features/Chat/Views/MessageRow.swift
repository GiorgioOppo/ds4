import SwiftUI
import AppKit
import DS4Engine
import DS4Core

struct MessageRow: View {
    let message: UIMessage
    @State private var responseCopied = false

    var body: some View {
        if message.role == .tool {
            if let run = message.subAgent {
                SubAgentView(run: run, running: message.subAgentRunning)
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
                        .contextMenu {
                            if message.role == .assistant {
                                Button("Copia risposta", systemImage: "doc.on.doc") {
                                    copyResponse()
                                }
                            }
                        }
                        if message.role == .assistant {
                            Button(action: copyResponse) {
                                Label(responseCopied ? "Copiata" : "Copia risposta",
                                      systemImage: responseCopied ? "checkmark" : "doc.on.doc")
                                    .contentTransition(.symbolEffect(.replace))
                            }
                            .buttonStyle(.plain)
                            .font(.caption)
                            .foregroundStyle(responseCopied ? Color.green : Color.secondary)
                            .help("Copia negli appunti la risposta completa")
                            .accessibilityLabel(responseCopied ? "Risposta copiata" : "Copia risposta")
                        }
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

    private func copyResponse() {
        NSPasteboard.general.clearContents()
        NSPasteboard.general.setString(message.text, forType: .string)
        responseCopied = true

        Task { @MainActor in
            try? await Task.sleep(for: .seconds(1.5))
            responseCopied = false
        }
    }
}
