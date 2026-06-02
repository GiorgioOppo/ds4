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
    @State private var prefilling: Bool = false
    @State private var progressText: String = ""
    @State private var indexedCount: Int = 0
    @State private var totalCount: Int = 0
    @State private var errorMessage: String?

    var body: some View {
        VStack(alignment: .leading, spacing: 12) {
            header
            Divider()
            sources
            if project.pendingSymlinkRoots?.isEmpty == false {
                Divider()
                pendingSymlinkSection
            }
            Divider()
            contextModeSection
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
                          || !service.isModelLoaded())
                .help(indexButtonHelp)
                Button(action: prefillAll) {
                    Label(prefilling ? "Prefilling…" : "Prefill files",
                           systemImage: "bolt.horizontal.circle")
                }
                .disabled(indexing || prefilling
                          || documents.documents(for: project.id).isEmpty
                          || !service.isModelLoaded())
                .help("Run a prefill forward over each indexed file and "
                      + "save its KV cache (.vec) to disk. Files larger "
                      + "than the model's context window are skipped.")
            }
            if indexing || prefilling {
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
        if !service.isModelLoaded() { return "Load a model first" }
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

    // MARK: - external symlink targets ("Grant access" affordance)

    /// Surfaces directories discovered during the last farm rebuild
    /// where one or more source-side symlinks land outside every
    /// granted `sourcePaths` entry. Without an explicit grant for
    /// those directories, the macOS sandbox refuses to open() through
    /// the link and the model sees a confusing "permission denied"
    /// instead of the file. One NSOpenPanel pick per directory
    /// unlocks every sibling link landing there.
    @ViewBuilder
    private var pendingSymlinkSection: some View {
        VStack(alignment: .leading, spacing: 6) {
            HStack {
                Text("Linked external folders").font(.headline)
                Spacer()
                Text("\(project.pendingSymlinkRoots?.count ?? 0)")
                    .font(.caption)
                    .foregroundStyle(.secondary)
                    .padding(.horizontal, 6)
                    .padding(.vertical, 1)
                    .background(Color.orange.opacity(0.18),
                                 in: Capsule())
            }
            Text("Some files inside your sources are symlinks pointing at the directories below. Grant access to make them readable; dismiss to ignore until the next rebuild.")
                .font(.caption)
                .foregroundStyle(.secondary)
            if let pending = project.pendingSymlinkRoots {
                List {
                    ForEach(pending, id: \.self) { path in
                        HStack {
                            Image(systemName: "link.badge.plus")
                                .foregroundStyle(.orange)
                            Text(path)
                                .font(.callout.monospaced())
                                .lineLimit(1)
                                .truncationMode(.middle)
                            Spacer()
                            // On macOS, a SwiftUI `Button` inside a
                            // `List` row needs an explicit `.buttonStyle`
                            // — without one the tap is captured by the
                            // row's selection gesture and the action
                            // never fires. `.bordered` matches the
                            // visual weight the user expects from a
                            // primary action, while the dismiss X
                            // stays `.borderless` so it doesn't
                            // compete for attention.
                            Button("Grant access") {
                                grantAccess(to: path)
                            }
                            .buttonStyle(.bordered)
                            .controlSize(.small)
                            Button {
                                library.dismissSymlinkRoot(
                                    path: path, for: project.id)
                            } label: {
                                Image(systemName: "xmark.circle")
                            }
                            .buttonStyle(.borderless)
                            .help("Dismiss this entry from the list")
                        }
                    }
                }
                .frame(minHeight: 60, maxHeight: 140)
            }
        }
    }

    /// NSOpenPanel pre-pointed at `suggestedPath` so the user can
    /// either confirm the suggestion or navigate to a containing
    /// directory and grant the broader scope in one click. The
    /// resulting URL becomes a `sourcePaths` entry on the project
    /// and the rebuild's allowed-roots check picks up every symlink
    /// landing under it on the next pass.
    private func grantAccess(to suggestedPath: String) {
        #if canImport(AppKit)
        let panel = NSOpenPanel()
        panel.canChooseDirectories = true
        panel.canChooseFiles = false
        panel.allowsMultipleSelection = false
        panel.directoryURL = URL(fileURLWithPath: suggestedPath)
        panel.message = "DeepSeek can't open files linked from this folder. Grant access here so the symlinks resolve."
        panel.prompt = "Grant"
        guard panel.runModal() == .OK,
              let url = panel.urls.first,
              let bookmark = ProjectSourceAccess.makeBookmark(for: url)
        else { return }
        library.grantSymlinkRoot(path: url.path,
                                  bookmark: bookmark,
                                  for: project.id)
        #endif
    }

    // MARK: - context mode (paths-only vs indexed-content)

    /// Sezione di configurazione del `ProjectContextMode` per-progetto.
    /// Override del default globale (`AppSettings.projectContextMode`).
    /// Vedi `docs/PROJECTS.md` / `Sources/DeepSeekUI/State/ProjectInventory.swift`.
    @ViewBuilder
    private var contextModeSection: some View {
        VStack(alignment: .leading, spacing: 8) {
            Text("Context mode").font(.headline)
            Text("Come il progetto viene presentato al modello sulla prima " +
                 "turn di una chat: paths-only (esplorazione via tool) o " +
                 "indexed-content (legacy, token injection).")
                .font(.caption)
                .foregroundStyle(.secondary)

            HStack(spacing: 12) {
                Picker("Mode", selection: contextModeBinding) {
                    Text("Use global default")
                        .tag(Optional<ProjectContextMode>.none)
                    ForEach(ProjectContextMode.allCases, id: \.self) { mode in
                        Text(mode.displayName)
                            .tag(Optional<ProjectContextMode>.some(mode))
                    }
                }
                .pickerStyle(.menu)
                .frame(maxWidth: 320)
                Text("(effective: \(project.effectiveContextMode.displayName))")
                    .font(.caption)
                    .foregroundStyle(.tertiary)
                Spacer()
            }

            // Mostra il cap solo quando il path-only è effettivamente
            // attivo (override esplicito o default globale).
            if project.effectiveContextMode == .pathsOnly {
                HStack(spacing: 8) {
                    Text("Max files in inventory:")
                        .frame(width: 180, alignment: .leading)
                    Stepper(value: maxFilesBinding,
                             in: 50...10_000, step: 50) {
                        Text("\(project.effectiveMaxInventoryFiles)")
                            .font(.system(.body, design: .monospaced))
                            .frame(width: 80, alignment: .trailing)
                    }
                    if project.maxInventoryFiles != nil {
                        Button("Reset to global") {
                            var p = project
                            p.maxInventoryFiles = nil
                            library.update(p)
                        }
                        .controlSize(.small)
                    }
                    Spacer()
                }
                .font(.callout)
            }

            // Spiega cosa fa la modalità effettiva.
            Text(project.effectiveContextMode.summary)
                .font(.caption2)
                .foregroundStyle(.tertiary)
                .padding(.top, 4)
        }
    }

    /// Binding bidirezionale che salva nel `library` ad ogni modifica.
    private var contextModeBinding: Binding<ProjectContextMode?> {
        Binding(
            get: { project.contextMode },
            set: { newValue in
                var p = project
                p.contextMode = newValue
                library.update(p)
            })
    }

    private var maxFilesBinding: Binding<Int> {
        Binding(
            get: { project.effectiveMaxInventoryFiles },
            set: { newValue in
                var p = project
                p.maxInventoryFiles = newValue
                library.update(p)
            })
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
        // Bookmarks are kept positional with `sourcePaths`: grow the
        // bookmark array to match the existing path list before
        // appending so old entries stay aligned. Re-picking an existing
        // path overwrites its bookmark in place — that's how the user
        // recovers sandbox access after a stale grant.
        var bookmarks = updated.sourceBookmarks ?? []
        while bookmarks.count < updated.sourcePaths.count {
            bookmarks.append(Data())
        }
        for url in panel.urls {
            let p = url.path
            let fresh = ProjectSourceAccess.makeBookmark(for: url) ?? Data()
            if let existing = updated.sourcePaths.firstIndex(of: p) {
                bookmarks[existing] = fresh
            } else {
                updated.sourcePaths.append(p)
                bookmarks.append(fresh)
            }
        }
        updated.sourceBookmarks = bookmarks
        library.update(updated)
        #endif
    }

    private func removePath(_ path: String) {
        var updated = project
        // Drop the bookmark at the matching index so the parallel
        // array stays aligned with `sourcePaths`.
        if let idx = updated.sourcePaths.firstIndex(of: path) {
            updated.sourcePaths.remove(at: idx)
            if var bookmarks = updated.sourceBookmarks,
               idx < bookmarks.count {
                bookmarks.remove(at: idx)
                updated.sourceBookmarks = bookmarks
            }
        }
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

    /// Step 3: run a prefill forward over every indexed file in the
    /// project and persist each file's KV snapshot to its `.vec`.
    /// Compute + save only; chats don't consume the `.vec` yet.
    private func prefillAll() {
        guard !prefilling, !indexing else { return }
        let docs = documents.documents(for: project.id)
        guard !docs.isEmpty else { return }
        totalCount = docs.count
        indexedCount = 0
        progressText = "Prefilling…"
        prefilling = true
        errorMessage = nil

        Task { [documents] in
            var skipped = 0
            for d in docs {
                progressText = "Prefilling \(d.displayPath ?? d.name)"
                do {
                    let toks = try documents.tokens(of: d.id)
                    let ok = await service.prefillDocument(id: d.id,
                                                            tokens: toks)
                    documents.setPrecomputedCache(ok, for: d.id)
                    if !ok { skipped += 1 }
                } catch {
                    skipped += 1
                    documents.setPrecomputedCache(false, for: d.id)
                    errorMessage = "Prefill failed for "
                        + "\(d.displayPath ?? d.name): "
                        + "\(error.localizedDescription)"
                }
                indexedCount += 1
            }
            prefilling = false
            progressText = skipped == 0
                ? "Prefilled \(indexedCount) file(s)."
                : "Prefilled \(indexedCount - skipped) of \(indexedCount)"
                    + " — \(skipped) skipped (too large for the context"
                    + " window, or save failed)."
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
