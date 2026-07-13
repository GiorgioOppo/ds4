import SwiftUI
import DS4Engine
import DS4Core

struct MessageRow: View {
    let message: UIMessage

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
}

