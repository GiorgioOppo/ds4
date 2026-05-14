import SwiftUI

/// Sidebar list of conversations. Selection drives `ChatStore.selectedID`.
/// Right-click on a row → Delete; toolbar `+` creates a new chat.
struct ConversationListView: View {
    @ObservedObject var store: ChatStore

    var body: some View {
        List(selection: Binding(
            get: { store.selectedID },
            set: { store.selectedID = $0 }
        )) {
            ForEach(store.conversations) { c in
                row(c)
                    .tag(c.id)
                    .contextMenu {
                        Button("Delete", role: .destructive) {
                            store.delete(c.id)
                        }
                    }
            }
        }
        .navigationTitle("Chats")
        .toolbar {
            ToolbarItem {
                Button {
                    store.newChat()
                } label: {
                    Label("New Chat", systemImage: "plus.bubble")
                }
                .keyboardShortcut("n", modifiers: .command)
            }
        }
    }

    @ViewBuilder
    private func row(_ c: Conversation) -> some View {
        VStack(alignment: .leading, spacing: 2) {
            Text(c.title)
                .lineLimit(1)
            HStack(spacing: 6) {
                Text(c.createdAt, format: .dateTime.month().day().hour().minute())
                    .font(.caption2)
                    .foregroundStyle(.tertiary)
                switch store.phase(of: c.id) {
                case .streaming, .prefilling:
                    ProgressView().controlSize(.mini)
                default: EmptyView()
                }
            }
        }
        .padding(.vertical, 2)
    }
}
