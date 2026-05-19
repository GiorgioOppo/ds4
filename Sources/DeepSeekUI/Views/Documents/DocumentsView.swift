import SwiftUI

/// Settings tab: lists the global document library and exposes the
/// import-document sheet. The actual chat-side attach (Step 2) and
/// KV-cache precompute (Step 3) are not part of this build yet, so
/// for now this is the only place a user can manage documents.
struct DocumentsView: View {
    @ObservedObject var library: DocumentLibrary
    let service: InferenceService

    @State private var showImport: Bool = false
    @State private var selection: UUID?

    var body: some View {
        VStack(alignment: .leading, spacing: 12) {
            header
            list
        }
        .padding(20)
        .sheet(isPresented: $showImport) {
            ImportDocumentSheet(library: library, service: service)
        }
    }

    private var header: some View {
        HStack {
            VStack(alignment: .leading, spacing: 4) {
                Text("Documents")
                    .font(.title3.bold())
                Text("Tokenize a text file once and reuse it across chats. The file is stored as a precomputed token sequence under Application Support.")
                    .font(.caption)
                    .foregroundStyle(.secondary)
            }
            Spacer()
            Button {
                showImport = true
            } label: {
                Label("Import…", systemImage: "plus")
            }
            .disabled(!service.isModelLoaded())
            .help(!service.isModelLoaded()
                   ? "Load a model first"
                   : "Import a new text document")
        }
    }

    @ViewBuilder
    private var list: some View {
        let standalone = library.standaloneDocuments
        if standalone.isEmpty {
            VStack(spacing: 6) {
                Image(systemName: "doc.text.below.ecg")
                    .font(.system(size: 32))
                    .foregroundStyle(.tertiary)
                Text("No documents yet.")
                    .foregroundStyle(.secondary)
                if !service.isModelLoaded() {
                    Text("Load a model from the picker, then come back to import a document.")
                        .font(.caption)
                        .foregroundStyle(.tertiary)
                        .multilineTextAlignment(.center)
                } else {
                    Text("Single-file imports show up here. For codebases, see the Projects tab.")
                        .font(.caption)
                        .foregroundStyle(.tertiary)
                        .multilineTextAlignment(.center)
                }
            }
            .frame(maxWidth: .infinity, maxHeight: .infinity)
        } else {
            List(selection: $selection) {
                ForEach(standalone) { d in
                    row(d)
                        .tag(d.id)
                        .contextMenu {
                            Button("Delete", role: .destructive) {
                                library.delete(d.id)
                            }
                        }
                }
            }
            .frame(minHeight: 240)
        }
    }

    private func row(_ d: VectorizedDocument) -> some View {
        HStack(alignment: .top, spacing: 12) {
            Image(systemName: d.hasPrecomputedCache
                   ? "doc.text.fill"
                   : "doc.text")
                .font(.title2)
                .foregroundStyle(d.hasPrecomputedCache
                                  ? Color.accentColor
                                  : Color.secondary)
            VStack(alignment: .leading, spacing: 2) {
                Text(d.name)
                    .font(.body)
                Text(detailString(d))
                    .font(.caption.monospacedDigit())
                    .foregroundStyle(.secondary)
            }
            Spacer()
        }
        .padding(.vertical, 4)
    }

    private func detailString(_ d: VectorizedDocument) -> String {
        let tokens = "\(d.tokenCount) tok"
        let bytes  = byteString(d.byteCount)
        let when   = d.createdAt.formatted(date: .numeric, time: .shortened)
        let kvNote = d.hasPrecomputedCache ? " · cache ready" : " · tokens only"
        return "\(tokens) · \(bytes)\(kvNote) · \(when)"
    }

    private func byteString(_ n: Int) -> String {
        if n < 1024 { return "\(n) B" }
        let kb = Double(n) / 1024.0
        if kb < 1024 { return String(format: "%.1f KB", kb) }
        let mb = kb / 1024.0
        return String(format: "%.1f MB", mb)
    }
}
