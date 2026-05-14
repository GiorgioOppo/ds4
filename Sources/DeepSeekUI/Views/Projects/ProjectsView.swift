import SwiftUI

/// Settings tab: master-detail over the projects library. The
/// left-hand list shows every project; the right pane shows the
/// active project's source paths, index status, and discovered
/// documents.
struct ProjectsView: View {
    @ObservedObject var library: ProjectLibrary
    @ObservedObject var documents: DocumentLibrary
    let service: InferenceService

    @State private var selectedID: UUID?
    @State private var showCreate: Bool = false

    var body: some View {
        HStack(spacing: 0) {
            sidebar
                .frame(width: 200)
                .background(Color(NSColor.controlBackgroundColor))
            Divider()
            detail
                .frame(maxWidth: .infinity, maxHeight: .infinity)
        }
        .sheet(isPresented: $showCreate) {
            CreateProjectSheet(library: library) { newID in
                selectedID = newID
            }
        }
    }

    // MARK: - sidebar

    private var sidebar: some View {
        VStack(spacing: 0) {
            List(selection: $selectedID) {
                ForEach(library.projects) { p in
                    HStack {
                        Image(systemName: "folder")
                            .foregroundStyle(Color.accentColor)
                        VStack(alignment: .leading, spacing: 2) {
                            Text(p.name).lineLimit(1)
                            Text(subtitle(for: p))
                                .font(.caption2)
                                .foregroundStyle(.tertiary)
                        }
                    }
                    .tag(p.id)
                    .contextMenu {
                        Button("Delete", role: .destructive) {
                            library.delete(p.id, documents: documents)
                            if selectedID == p.id { selectedID = nil }
                        }
                    }
                }
            }
            Divider()
            HStack {
                Button {
                    showCreate = true
                } label: {
                    Image(systemName: "plus")
                }
                .buttonStyle(.borderless)
                .help("New project")
                Spacer()
            }
            .padding(8)
        }
    }

    private func subtitle(for p: Project) -> String {
        let docs = documents.documents(for: p.id).count
        return docs == 0 ? "not indexed" : "\(docs) files"
    }

    // MARK: - detail

    @ViewBuilder
    private var detail: some View {
        if let id = selectedID, let p = library.project(id: id) {
            ProjectDetailView(project: p,
                               library: library,
                               documents: documents,
                               service: service)
                .id(p.id)
        } else {
            VStack(spacing: 8) {
                Image(systemName: "folder.badge.plus")
                    .font(.system(size: 36))
                    .foregroundStyle(.tertiary)
                Text(library.projects.isEmpty
                      ? "Create a project to pre-index a codebase."
                      : "Select a project to manage its sources.")
                    .foregroundStyle(.secondary)
                if library.projects.isEmpty {
                    Button("New project…") { showCreate = true }
                        .controlSize(.large)
                }
            }
            .frame(maxWidth: .infinity, maxHeight: .infinity)
        }
    }
}
