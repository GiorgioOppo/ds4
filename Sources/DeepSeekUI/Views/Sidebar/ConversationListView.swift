import SwiftUI

/// Sidebar list of conversations. Selection drives `ChatStore.selectedID`.
/// Right-click on a row → Delete; toolbar `+` creates a new chat.
struct ConversationListView: View {
    @ObservedObject var store: ChatStore

    var body: some View {
        // Bind directly to the @Published projected value rather than
        // a custom Binding(get:set:). A get/set binding's setter is
        // sometimes invoked during a view update — that write into
        // `store.selectedID` (a @Published) triggers a publish, which
        // SwiftUI flags as "Publishing changes from within view
        // updates is not allowed, this will cause undefined behavior."
        List(selection: $store.selectedID) {
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
