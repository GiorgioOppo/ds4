import SwiftUI

/// Popover list of persisted chats: switch between them, rename, delete, or start
/// a new one. Backed by `ChatStore.sessions` (newest first).
struct ChatListView: View {
    @Bindable var store: ChatStore
    @Environment(\.dismiss) private var dismiss
    @State private var renamingId: String?
    @State private var renameText = ""
    @State private var searchText = ""
    @State private var deletingSession: ChatSession?

    private var filteredSessions: [ChatSession] {
        let query = searchText.trimmingCharacters(in: .whitespacesAndNewlines)
        return query.isEmpty ? store.sessions : store.sessions.filter {
            $0.title.localizedStandardContains(query)
        }
    }

    var body: some View {
        VStack(alignment: .leading, spacing: 0) {
            HStack {
                Text("Conversazioni").font(.headline)
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
            .padding(14)
            HStack(spacing: 6) {
                Image(systemName: "magnifyingglass").foregroundStyle(.secondary)
                TextField("Cerca per titolo", text: $searchText)
                    .textFieldStyle(.plain)
                    .accessibilityLabel("Cerca conversazioni per titolo")
                if !searchText.isEmpty {
                    Button { searchText = "" } label: {
                        Label("Cancella ricerca", systemImage: "xmark.circle.fill")
                    }
                    .labelStyle(.iconOnly)
                    .buttonStyle(.plain)
                    .foregroundStyle(.secondary)
                }
            }
            .padding(8)
            .background(Color.secondary.opacity(0.1), in: RoundedRectangle(cornerRadius: 8))
            .padding(.horizontal, 14)
            .padding(.bottom, 12)
            Divider()

            if store.sessions.isEmpty {
                ContentUnavailableView("Nessuna conversazione", systemImage: "bubble.left.and.bubble.right",
                                       description: Text("Le chat vengono salvate qui automaticamente."))
            } else if filteredSessions.isEmpty {
                ContentUnavailableView.search(text: searchText)
            } else {
                ScrollView {
                    LazyVStack(spacing: 0) {
                        ForEach(filteredSessions) { session in
                            row(session)
                            Divider()
                        }
                    }
                }
            }
        }
        .frame(width: 380, height: 460)
        .alert("Rinomina conversazione", isPresented: Binding(
            get: { renamingId != nil },
            set: { if !$0 { renamingId = nil } })) {
            TextField("Titolo", text: $renameText)
            Button("Annulla", role: .cancel) { renamingId = nil }
            Button("Salva") {
                if let id = renamingId { store.renameSession(id, to: renameText) }
                renamingId = nil
            }
        }
        .alert("Eliminare la conversazione?", isPresented: Binding(
            get: { deletingSession != nil },
            set: { if !$0 { deletingSession = nil } })) {
            Button("Annulla", role: .cancel) { deletingSession = nil }
            Button("Elimina", role: .destructive) {
                if let session = deletingSession { store.deleteSession(session.id) }
                deletingSession = nil
            }
        } message: {
            Text("“\(deletingSession?.title ?? ChatSession.untitled)” verrà eliminata dal Mac. Questa operazione non può essere annullata.")
        }
    }

    private func row(_ session: ChatSession) -> some View {
        let isActive = session.id == store.activeSessionId
        return HStack(spacing: 8) {
            Button {
                store.switchSession(session.id)
                dismiss()
            } label: {
                HStack(spacing: 8) {
                    Image(systemName: isActive ? "checkmark.circle.fill" : "bubble.left")
                        .foregroundStyle(isActive ? Color.accentColor : .secondary)
                    VStack(alignment: .leading, spacing: 4) {
                        Text(session.title.isEmpty ? ChatSession.untitled : session.title)
                            .lineLimit(1)
                        Text("\(session.messages.count) messaggi · \(session.updatedAt.formatted(date: .abbreviated, time: .shortened))")
                            .font(.caption)
                            .foregroundStyle(.secondary)
                    }
                    Spacer(minLength: 0)
                }
                .contentShape(Rectangle())
                .padding(.vertical, 10)
            }
            .buttonStyle(.plain)
            .accessibilityLabel("\(session.title)\(isActive ? ", conversazione attuale" : "")")
            Menu {
                sessionActions(session)
            } label: {
                Label("Azioni conversazione", systemImage: "ellipsis")
            }
            .labelStyle(.iconOnly)
            .menuIndicator(.hidden)
            .menuStyle(.borderlessButton)
            .fixedSize()
            .help("Rinomina o elimina questa conversazione")
        }
        .padding(.horizontal, 14)
        .background(isActive ? Color.accentColor.opacity(0.08) : Color.clear)
        .contextMenu { sessionActions(session) }
    }

    @ViewBuilder
    private func sessionActions(_ session: ChatSession) -> some View {
        Button {
            renameText = session.title
            renamingId = session.id
        } label: { Label("Rinomina", systemImage: "pencil") }
        Button(role: .destructive) {
            deletingSession = session
        } label: { Label("Elimina…", systemImage: "trash") }
    }
}
