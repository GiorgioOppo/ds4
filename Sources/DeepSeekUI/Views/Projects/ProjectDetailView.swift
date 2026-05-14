import SwiftUI
#if canImport(AppKit)
import AppKit
#endif

/// Right pane of the Projects tab. Shows the project's name (editable),
/// its source paths (add/remove), the index status, and the per-file
/// list of documents produced by the last index run.
struct ProjectDetailView: View {
    let project: Project
    @ObservedObject var library: ProjectLibrary
    @ObservedObject var documents: DocumentLibrary
    let service: InferenceService

    @State private var name: String = ""
    @State private var nameDebouncer: Task<Void, Never>? = nil
    @State private var indexing: Bool = false
    @State private var progressText: String = ""
    @State private var indexedCount: Int = 0
    @State private var totalCount: Int = 0
    @State private var errorMessage: String?

    var body: some View {
        VStack(alignment: .leading, spacing: 12) {
            header
            Divider()
            sources
            Divider()
            documentsSection
        }
        .padding(16)
        .onAppear { name = project.name }
        .onChange(of: project.id) { _, _ in
            name = project.name
            indexing = false
            progressText = ""
            indexedCount = 0
            totalCount = 0
            errorMessage = nil
        }
    }

    // MARK: - header (name + index status)

    private var header: some View {
        VStack(alignment: .leading, spacing: 8) {
            TextField("Project name", text: $name)
                .textFieldStyle(.roundedBorder)
                .font(.title3)
                .onChange(of: name) { _, newValue in
                    scheduleNameSave(newValue)
                }
            HStack(spacing: 12) {
                statusLabel
                Spacer()
                Button(action: indexNow) {
                    Label(indexing ? "Indexing…" : "Index now",
                           systemImage: "arrow.triangle.2.circlepath")
                }
                .disabled(indexing
                          || project.sourcePaths.isEmpty
                          || service.currentTokenizer() == nil)
                .help(indexButtonHelp)
            }
            if indexing {
                ProgressView(value: Double(indexedCount),
                              total: Double(max(totalCount, 1))) {
                    Text(progressText)
                        .font(.caption)
                        .foregroundStyle(.secondary)
                }
            }
            if let err = errorMessage {
                Label(err, systemImage: "exclamationmark.octagon.fill")
                    .foregroundStyle(.red)
                    .font(.caption)
            }
        }
    }

    @ViewBuilder
    private var statusLabel: some View {
        if let when = project.lastIndexedAt {
            Text("Last indexed \(when.formatted(date: .abbreviated, time: .shortened))")
                .font(.caption)
                .foregroundStyle(.tertiary)
        } else {
            Text("Never indexed")
                .font(.caption)
                .foregroundStyle(.tertiary)
        }
    }

    private var indexButtonHelp: String {
        if service.currentTokenizer() == nil { return "Load a model first" }
        if project.sourcePaths.isEmpty { return "Add at least one source path" }
        return "Re-scan every source and tokenize the discovered files"
    }

    /// Placeholder rendered when a project has zero source paths.
    /// Promotes the Add actions to large, obvious call-to-action
    /// buttons so the user isn't left staring at a blank panel.
    private var emptySourcesHint: some View {
        VStack(spacing: 10) {
            Image(systemName: "folder.badge.questionmark")
                .font(.system(size: 28))
                .foregroundStyle(.tertiary)
            Text("No sources yet")
                .font(.callout)
                .foregroundStyle(.secondary)
            HStack(spacing: 8) {
                Button {
                    addPath(directory: false)
                } label: {
                    Label("Add files…", systemImage: "doc.badge.plus")
                }
                .buttonStyle(.borderedProminent)
                .controlSize(.regular)
                Button {
                    addPath(directory: true)
                } label: {
                    Label("Add folder…", systemImage: "folder.badge.plus")
                }
                .controlSize(.regular)
            }
            Text("Pick code files or whole directories. The next index pass tokenizes every text file found under them.")
                .font(.caption)
                .foregroundStyle(.tertiary)
                .multilineTextAlignment(.center)
        }
        .frame(maxWidth: .infinity)
        .padding(.vertical, 12)
    }

    // MARK: - sources

    private var sources: some View {
        VStack(alignment: .leading, spacing: 6) {
            HStack {
                Text("Sources").font(.headline)
                Spacer()
                Button {
                    addPath(directory: false)
                } label: {
                    Label("Add file", systemImage: "doc.badge.plus")
                }
                .controlSize(.small)
                Button {
                    addPath(directory: true)
                } label: {
                    Label("Add folder", systemImage: "folder.badge.plus")
                }
                .controlSize(.small)
            }
            if project.sourcePaths.isEmpty {
                emptySourcesHint
            } else {
                List {
                    ForEach(project.sourcePaths, id: \.self) { path in
                        HStack {
                            Image(systemName: isDirectory(path)
                                   ? "folder"
                                   : "doc")
                                .foregroundStyle(.secondary)
                            Text(path)
                                .font(.callout.monospaced())
                                .lineLimit(1)
                                .truncationMode(.middle)
                            Spacer()
                            Button {
                                removePath(path)
                            } label: {
                                Image(systemName: "minus.circle")
                            }
                            .buttonStyle(.borderless)
                        }
                    }
                }
                .frame(minHeight: 80, maxHeight: 140)
            }
        }
    }

    // MARK: - indexed documents

    private var documentsSection: some View {
        VStack(alignment: .leading, spacing: 6) {
            HStack {
                Text("Indexed files").font(.headline)
                Spacer()
                Text("\(documents.documents(for: project.id).count) files")
                    .font(.caption.monospacedDigit())
                    .foregroundStyle(.tertiary)
            }
            let docs = documents.documents(for: project.id)
            if docs.isEmpty {
                Text("Run Index now to scan the sources and produce token sequences.")
                    .font(.caption)
                    .foregroundStyle(.tertiary)
            } else {
                List {
                    ForEach(docs) { d in
                        HStack {
                            Image(systemName: d.hasPrecomputedCache
                                   ? "doc.text.fill" : "doc.text")
                                .foregroundStyle(d.hasPrecomputedCache
                                                  ? Color.accentColor
                                                  : Color.secondary)
                            VStack(alignment: .leading, spacing: 1) {
                                Text(d.displayPath ?? d.name)
                                    .font(.callout)
                                    .lineLimit(1)
                                    .truncationMode(.middle)
                                Text("\(d.tokenCount) tok · \(byteString(d.byteCount))")
                                    .font(.caption2.monospacedDigit())
                                    .foregroundStyle(.tertiary)
                            }
                            Spacer()
                        }
                    }
                }
                .frame(minHeight: 120)
            }
        }
    }

    // MARK: - actions

    private func scheduleNameSave(_ newValue: String) {
        nameDebouncer?.cancel()
        nameDebouncer = Task { [project, newValue] in
            try? await Task.sleep(nanoseconds: 400_000_000)
            if Task.isCancelled { return }
            var updated = project
            updated.name = newValue
            library.update(updated)
        }
    }

    private func addPath(directory: Bool) {
        #if canImport(AppKit)
        let panel = NSOpenPanel()
        panel.allowsMultipleSelection = true
        panel.canChooseDirectories = directory
        panel.canChooseFiles = !directory
        panel.title = directory ? "Add source folders" : "Add source files"
        guard panel.runModal() == .OK else { return }
        var updated = project
        for url in panel.urls {
            let p = url.path
            if !updated.sourcePaths.contains(p) {
                updated.sourcePaths.append(p)
            }
        }
        library.update(updated)
        #endif
    }

    private func removePath(_ path: String) {
        var updated = project
        updated.sourcePaths.removeAll { $0 == path }
        library.update(updated)
    }

    private func indexNow() {
        guard !indexing else { return }
        let projectSnapshot = project
        let fingerprint = ModelFingerprint.of(
            modelDirPath: service.currentModelDir()?.path ?? "")

        // Wipe any prior documents for this project — re-index is
        // always full for now.
        documents.purge(projectID: projectSnapshot.id)

        let candidates = ProjectIndexer.scan(projectSnapshot)
        totalCount = candidates.count
        indexedCount = 0
        progressText = candidates.isEmpty
            ? "Nothing to index."
            : "Scanning…"
        indexing = true
        errorMessage = nil

        Task { [documents] in
            for cand in candidates {
                progressText = "Tokenizing \(cand.displayPath)"
                guard let text = readText(at: cand.url) else {
                    indexedCount += 1
                    continue
                }
                if let tokens = await service.tokenize(text) {
                    do {
                        _ = try documents.add(
                            name: cand.url.lastPathComponent,
                            sourceFilename: cand.url.lastPathComponent,
                            byteCount: cand.byteCount,
                            tokens: tokens,
                            modelFingerprint: fingerprint,
                            projectID: projectSnapshot.id,
                            displayPath: cand.displayPath)
                    } catch {
                        errorMessage = "Save failed for \(cand.displayPath): \(error.localizedDescription)"
                    }
                }
                indexedCount += 1
            }
            // Stamp the project as indexed.
            var updated = projectSnapshot
            updated.lastIndexedAt = .now
            updated.modelFingerprint = fingerprint
            library.update(updated)
            indexing = false
            progressText = "Indexed \(indexedCount) of \(totalCount) files."
        }
    }

    // MARK: - helpers

    private func isDirectory(_ path: String) -> Bool {
        var isDir: ObjCBool = false
        _ = FileManager.default.fileExists(atPath: path, isDirectory: &isDir)
        return isDir.boolValue
    }

    private func readText(at url: URL) -> String? {
        guard let data = try? Data(contentsOf: url) else { return nil }
        return String(data: data, encoding: .utf8)
            ?? String(data: data, encoding: .isoLatin1)
    }

    private func byteString(_ n: Int) -> String {
        if n < 1024 { return "\(n) B" }
        let kb = Double(n) / 1024.0
        if kb < 1024 { return String(format: "%.1f KB", kb) }
        let mb = kb / 1024.0
        return String(format: "%.1f MB", mb)
    }
}
