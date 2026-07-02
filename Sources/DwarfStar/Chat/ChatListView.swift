import SwiftUI

/// Popover list of persisted chats: switch between them, rename, delete, or start
/// a new one. Backed by `ChatStore.sessions` (newest first).
struct ChatListView: View {
    @Bindable var store: ChatStore
    @Environment(\.dismiss) private var dismiss
    @State private var renamingId: String?
    @State private var renameText = ""

    var body: some View {
        VStack(alignment: .leading, spacing: 0) {
            HStack {
                Text("Chat").font(.headline)
                Spacer()
                Button {
                    store.newChat()
                    dismiss()
                } label: {
                    Label("Nuova", systemImage: "square.and.pencil")
                }
                .buttonStyle(.borderless)
                .help("Inizia una nuova conversazione")
            }
            .padding(10)
            Divider()

            if store.sessions.isEmpty {
                Text("Nessuna chat salvata")
                    .foregroundStyle(.secondary)
                    .frame(maxWidth: .infinity, maxHeight: .infinity)
            } else {
                ScrollView {
                    LazyVStack(spacing: 0) {
                        ForEach(store.sessions) { session in
                            row(session)
                            Divider()
                        }
                    }
                }
            }
        }
        .frame(width: 320, height: 420)
        .alert("Rinomina chat", isPresented: Binding(
            get: { renamingId != nil },
            set: { if !$0 { renamingId = nil } })) {
            TextField("Titolo", text: $renameText)
            Button("Annulla", role: .cancel) { renamingId = nil }
            Button("Salva") {
                if let id = renamingId { store.renameSession(id, to: renameText) }
                renamingId = nil
            }
        }
    }

    private func row(_ session: ChatSession) -> some View {
        let isActive = session.id == store.activeSessionId
        return Button {
            store.switchSession(session.id)
            dismiss()
        } label: {
            HStack(spacing: 8) {
                Image(systemName: isActive ? "checkmark.circle.fill" : "bubble.left")
                    .foregroundStyle(isActive ? Color.accentColor : .secondary)
                VStack(alignment: .leading, spacing: 2) {
                    Text(session.title.isEmpty ? ChatSession.untitled : session.title)
                        .lineLimit(1)
                    Text("\(session.messages.count) messaggi · \(session.updatedAt.formatted(date: .abbreviated, time: .shortened))")
                        .font(.caption)
                        .foregroundStyle(.secondary)
                }
                Spacer()
            }
            .contentShape(Rectangle())
            .padding(.horizontal, 10)
            .padding(.vertical, 8)
        }
        .buttonStyle(.plain)
        .background(isActive ? Color.accentColor.opacity(0.08) : Color.clear)
        .contextMenu {
            Button {
                renameText = session.title
                renamingId = session.id
            } label: { Label("Rinomina", systemImage: "pencil") }
            Button(role: .destructive) {
                store.deleteSession(session.id)
            } label: { Label("Elimina", systemImage: "trash") }
        }
    }
}
