import SwiftUI
import DS4Engine

struct ChatView: View {
    @Bindable var store: ChatStore

    var body: some View {
        VStack(spacing: 0) {
            header
            Divider()
            transcript
            Divider()
            composer
        }
    }

    private var header: some View {
        HStack(spacing: 12) {
            VStack(alignment: .leading, spacing: 2) {
                Text(store.info?.name ?? "DeepSeek V4")
                    .font(.headline)
                if let info = store.info {
                    Text("\(info.layers) layer · \(info.routedQuantBits)-bit · ctx \(info.contextSize) · KV ~\(kvSize(info.kvCacheBytes))")
                        .font(.caption)
                        .foregroundStyle(.secondary)
                }
            }
            Spacer()
            Toggle("Thinking", isOn: $store.think)
                .toggleStyle(.switch)
            Button {
                store.newChat()
            } label: {
                Label("Nuova chat", systemImage: "square.and.pencil")
            }
        }
        .padding(.horizontal)
        .padding(.vertical, 10)
    }

    private func kvSize(_ bytes: UInt64) -> String {
        ByteCountFormatter.string(fromByteCount: Int64(bytes), countStyle: .memory)
    }

    private var transcript: some View {
        ScrollViewReader { proxy in
            ScrollView {
                LazyVStack(alignment: .leading, spacing: 14) {
                    ForEach(store.messages) { message in
                        MessageRow(message: message)
                            .id(message.id)
                    }
                }
                .padding()
            }
            .onChange(of: store.messages.last?.text) {
                if let last = store.messages.last {
                    withAnimation { proxy.scrollTo(last.id, anchor: .bottom) }
                }
            }
        }
    }

    private var composer: some View {
        VStack(alignment: .leading, spacing: 4) {
        if store.isGenerating && !store.status.isEmpty {
            HStack(spacing: 6) {
                ProgressView().controlSize(.mini)
                Text(store.status)
                    .font(.caption.monospacedDigit())
                    .foregroundStyle(.secondary)
            }
        }
        HStack(alignment: .bottom, spacing: 8) {
            TextField("Scrivi un messaggio…", text: $store.input, axis: .vertical)
                .textFieldStyle(.roundedBorder)
                .lineLimit(1...6)
                .onSubmit { store.send() }
            if store.isGenerating {
                Button(role: .destructive) { store.stop() } label: {
                    Image(systemName: "stop.fill")
                }
            } else {
                Button { store.send() } label: {
                    Image(systemName: "arrow.up.circle.fill")
                }
                .disabled(store.input.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty)
            }
        }
        }
        .padding(.horizontal)
        .padding(.vertical, 10)
    }
}

struct MessageRow: View {
    let message: UIMessage

    var body: some View {
        HStack {
            if message.role == .user { Spacer(minLength: 40) }
            VStack(alignment: message.role == .user ? .trailing : .leading, spacing: 6) {
                if !message.reasoning.isEmpty {
                    ReasoningView(text: message.reasoning)
                }
                if !message.text.isEmpty {
                    Text(message.text)
                        .textSelection(.enabled)
                        .padding(10)
                        .background(bubbleColor)
                        .clipShape(RoundedRectangle(cornerRadius: 12))
                } else if message.role == .assistant && message.reasoning.isEmpty {
                    ProgressView().controlSize(.small)
                }
            }
            if message.role == .assistant { Spacer(minLength: 40) }
        }
    }

    private var bubbleColor: Color {
        message.role == .user ? Color.accentColor.opacity(0.18) : Color.secondary.opacity(0.12)
    }
}

/// Collapsible chain-of-thought block.
struct ReasoningView: View {
    let text: String
    @State private var expanded = false

    var body: some View {
        DisclosureGroup(isExpanded: $expanded) {
            Text(text)
                .font(.callout)
                .foregroundStyle(.secondary)
                .textSelection(.enabled)
                .frame(maxWidth: .infinity, alignment: .leading)
        } label: {
            Label("Ragionamento", systemImage: "brain")
                .font(.caption)
                .foregroundStyle(.secondary)
        }
        .padding(8)
        .background(Color.secondary.opacity(0.06))
        .clipShape(RoundedRectangle(cornerRadius: 10))
    }
}
