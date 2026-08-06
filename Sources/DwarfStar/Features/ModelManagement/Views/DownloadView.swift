import AppKit
import SwiftUI
import DS4Engine

/// Catalog-only model manager. The Engine owns which artifacts are supported;
/// this sheet only renders installation state and drives native downloads.
struct DownloadView: View {
    @Bindable var store: ChatStore
    @State private var runner = DownloadRunner()
    @Environment(\.dismiss) private var dismiss

    var body: some View {
        VStack(alignment: .leading, spacing: 14) {
            header

            ScrollView {
                LazyVStack(spacing: 0) {
                    ForEach(ModelCatalogRegistry.downloadEntries) { entry in
                        catalogRow(entry)
                        if entry.id != ModelCatalogRegistry.downloadEntries.last?.id {
                            Divider()
                        }
                    }
                }
            }
            .background(Color(nsColor: .controlBackgroundColor))
            .clipShape(RoundedRectangle(cornerRadius: 10))
            .overlay {
                RoundedRectangle(cornerRadius: 10)
                    .stroke(Color(nsColor: .separatorColor).opacity(0.65), lineWidth: 1)
            }

            footer
        }
        .padding(18)
        .frame(minWidth: 680, idealWidth: 760, minHeight: 560, idealHeight: 680)
        .onAppear(perform: configureRunner)
        .onChange(of: store.modelPath) { _, _ in configureRunner() }
        .onDisappear { runner.cancel() }
    }

    private var header: some View {
        VStack(alignment: .leading, spacing: 8) {
            HStack(alignment: .firstTextBaseline) {
                VStack(alignment: .leading, spacing: 3) {
                    Text("Modelli GGUF")
                        .font(.title2.bold())
                    Text("Scarica o riprendi i modelli catalogati senza comandi esterni. I file completi già presenti vengono riutilizzati.")
                        .font(.callout)
                        .foregroundStyle(.secondary)
                }
                Spacer()
                Button {
                    runner.checkForUpdates()
                } label: {
                    Label("Controlla aggiornamenti", systemImage: "arrow.clockwise")
                }
                .disabled(runner.isRunning)
            }

            HStack(spacing: 14) {
                Label(
                    runner.availableBytes > 0
                        ? "Spazio libero: \(formatBytes(runner.availableBytes))"
                        : "Spazio libero non disponibile",
                    systemImage: "internaldrive"
                )
                if let source = HFTokenStore.activeSourceDescription() {
                    Label("Token: \(source)", systemImage: "key.fill")
                } else {
                    Label("Download pubblico senza token", systemImage: "key.slash")
                }
            }
            .font(.caption)
            .foregroundStyle(.secondary)
        }
    }

    @ViewBuilder
    private func catalogRow(_ entry: ModelCatalogEntry) -> some View {
        let installation = runner.installation(for: entry)
        let active = runner.isActive(entry) ? runner.active : nil
        let selected = selectedPath(for: entry, installation: installation)
        let isSelected = selected.map { pathsAreEqual($0, store.modelPath) } ?? false

        VStack(alignment: .leading, spacing: 10) {
            HStack(alignment: .top, spacing: 12) {
                VStack(alignment: .leading, spacing: 4) {
                    Text(entry.displayName)
                        .font(.headline)
                    Text(entry.summary)
                        .font(.caption)
                        .foregroundStyle(.secondary)
                        .fixedSize(horizontal: false, vertical: true)
                }
                Spacer(minLength: 16)
                VStack(alignment: .trailing, spacing: 5) {
                    Label(installation.state.title, systemImage: installation.state.symbol)
                        .foregroundStyle(installStateColor(installation.state))
                    Text(entry.expectedSizeBytes.map(formatBytes)
                         ?? "circa \(entry.approximateSizeGB) GB")
                        .foregroundStyle(.secondary)
                }
                .font(.caption)
            }

            HStack(spacing: 12) {
                if entry.artifacts.allSatisfy({ $0.role == .optionalComponent }) {
                    Label("Accessorio · non sostituisce il modello selezionato",
                          systemImage: "puzzlepiece.extension")
                } else if entry.isSplitFragmentPackage {
                    Label("GGUF diviso · \(entry.artifacts.count) parti consecutive",
                          systemImage: "rectangle.split.3x1")
                } else if entry.artifacts.count > 1 {
                    Label("Pacchetto split distribuito · \(entry.artifacts.count) shard",
                          systemImage: "rectangle.split.2x1")
                } else if entry.isSelectable {
                    Label("Modello completo · selezionabile", systemImage: "checkmark.seal")
                } else {
                    Label("Modello completo · solo download", systemImage: "shippingbox")
                }

                runtimeLabel(entry.runtimeAvailability)
            }
            .font(.caption)

            if installation.state == .partial {
                let local = installation.localBytes + installation.partialBytes
                Label(
                    "\(installation.installedArtifacts)/\(installation.artifactCount) file completi · \(formatBytes(local)) locali. "
                        + (installation.partialBytes > 0
                           ? "Il prossimo tentativo riprende il .part."
                           : "Verranno scaricati soltanto i file mancanti."),
                    systemImage: "arrow.clockwise"
                )
                .font(.caption)
                .foregroundStyle(.orange)
            } else if installation.state == .installed {
                Text("\(installation.artifactCount) file · \(formatBytes(installation.localBytes)) su disco")
                    .font(.caption)
                    .foregroundStyle(.secondary)
            } else if installation.state == .invalidLocalFile {
                ForEach(installation.invalidArtifacts.keys.sorted(), id: \.self) { targetID in
                    if let invalid = installation.invalidArtifacts[targetID] {
                        Label(
                            "\(URL(fileURLWithPath: invalid.path).lastPathComponent): \(invalid.reason). "
                                + "Sposta, rinomina o rimuovi il file; DwarfStar non lo sovrascrive automaticamente.",
                            systemImage: "exclamationmark.octagon.fill"
                        )
                        .font(.caption)
                        .foregroundStyle(.red)
                        .textSelection(.enabled)
                    }
                }
            }

            updateLabel(for: entry)

            let packageBytes = entry.expectedSizeBytes
                ?? Int64(entry.approximateSizeGB) * 1_000_000_000
            let estimatedRequired = ModelDownloader.requiredFreeSpace(
                totalBytes: packageBytes,
                existingPartialBytes: installation.localBytes + installation.partialBytes
            )
            if installation.state != .installed,
               installation.state != .invalidLocalFile,
               runner.availableBytes > 0,
               estimatedRequired > runner.availableBytes {
                Label(
                    "Spazio probabilmente insufficiente: servono circa \(formatBytes(estimatedRequired)) inclusa la riserva, disponibili \(formatBytes(runner.availableBytes)).",
                    systemImage: "exclamationmark.triangle.fill"
                )
                .font(.caption)
                .foregroundStyle(.red)
            }

            if let active {
                VStack(alignment: .leading, spacing: 5) {
                    ProgressView(value: min(max(active.overallFraction, 0), 1))
                    HStack {
                        Text("\(active.phase) · \(active.byteSummary)")
                        Spacer()
                        Text("file \(active.artifactIndex)/\(active.artifactCount)")
                    }
                    .font(.caption.monospacedDigit())
                    .foregroundStyle(.secondary)
                    Text(active.artifactName)
                        .font(.caption2.monospaced())
                        .foregroundStyle(.tertiary)
                        .lineLimit(1)
                }
            }

            if let error = runner.error(for: entry) {
                Label(error, systemImage: "exclamationmark.triangle.fill")
                    .font(.caption)
                    .foregroundStyle(.red)
                    .textSelection(.enabled)
            } else if let notice = runner.notice(for: entry) {
                Label(notice, systemImage: "info.circle")
                    .font(.caption)
                    .foregroundStyle(.secondary)
            }

            HStack {
                Spacer()
                rowAction(entry, installation: installation, isSelected: isSelected)
            }
        }
        .padding(13)
    }

    @ViewBuilder
    private func runtimeLabel(_ availability: ModelRuntimeAvailability) -> some View {
        switch availability {
        case .runnable:
            Label("Runtime disponibile", systemImage: "bolt.fill")
                .foregroundStyle(.green)
        case .downloadOnly(let reason):
            Label(reason, systemImage: "exclamationmark.triangle")
                .foregroundStyle(.orange)
        }
    }

    @ViewBuilder
    private func rowAction(_ entry: ModelCatalogEntry,
                           installation: CatalogInstallation,
                           isSelected: Bool) -> some View {
        if runner.isActive(entry) {
            Button(role: .destructive) {
                runner.cancel()
            } label: {
                Label("Annulla", systemImage: "xmark.circle")
            }
        } else if installation.state == .installed, entry.isSelectable,
                  let path = selectedPath(for: entry, installation: installation) {
            HStack {
                if runner.updateState(for: entry) == .available {
                    Button { runner.installUpdate(entry, onSelectableModel: selectCatalogPath) } label: {
                        Label("Aggiorna modello", systemImage: "arrow.down.circle.fill")
                    }
                }
                Button { selectCatalogPath(path) } label: {
                    Label(isSelected ? "Selezionato" : "Seleziona",
                          systemImage: isSelected ? "checkmark.circle.fill" : "circle")
                }
                .disabled(isSelected || runner.isRunning)
            }
        } else if installation.state == .installed {
            if runner.updateState(for: entry) == .available {
                Button { runner.installUpdate(entry, onSelectableModel: selectCatalogPath) } label: {
                    Label("Aggiorna pacchetto", systemImage: "arrow.down.circle.fill")
                }
                .disabled(runner.isRunning)
            } else {
                Label("Download completo · non selezionabile",
                      systemImage: "checkmark.circle")
                    .font(.caption)
                    .foregroundStyle(.secondary)
            }
        } else if installation.state == .invalidLocalFile,
                  let invalid = installation.invalidArtifacts.values.first {
            Button {
                NSWorkspace.shared.activateFileViewerSelecting([
                    URL(fileURLWithPath: invalid.path),
                ])
            } label: {
                Label("Mostra file non valido", systemImage: "folder.badge.questionmark")
            }
            .disabled(runner.isRunning)
        } else {
            Button {
                runner.acquire(entry, onSelectableModel: selectCatalogPath)
            } label: {
                let retry = runner.error(for: entry) != nil
                Label(retry ? "Riprova" : installation.state == .partial ? "Riprendi" : "Scarica",
                      systemImage: retry ? "arrow.clockwise" : "arrow.down.circle")
            }
            .disabled(runner.isRunning)
        }
    }

    @ViewBuilder
    private func updateLabel(for entry: ModelCatalogEntry) -> some View {
        switch runner.updateState(for: entry) {
        case .unknown: EmptyView()
        case .checking:
            Label("Controllo versione remota…", systemImage: "arrow.clockwise")
                .font(.caption).foregroundStyle(.secondary)
        case .current:
            Label("Versione più recente", systemImage: "checkmark.shield")
                .font(.caption).foregroundStyle(.green)
        case .available:
            Label("È disponibile una versione aggiornata del modello.",
                  systemImage: "arrow.down.circle.fill")
                .font(.caption).foregroundStyle(.orange)
        case .failed(let message):
            Label("Controllo aggiornamenti non riuscito: \(message)",
                  systemImage: "exclamationmark.triangle")
                .font(.caption).foregroundStyle(.orange)
        }
    }

    private var footer: some View {
        HStack(alignment: .center, spacing: 10) {
            VStack(alignment: .leading, spacing: 2) {
                Text("Cartella download")
                    .font(.caption)
                    .foregroundStyle(.secondary)
                Text(AppEnvironment.modelDownloadDirectory.path)
                    .font(.caption2.monospaced())
                    .textSelection(.enabled)
                    .lineLimit(2)
            }
            Button("Mostra nel Finder") {
                NSWorkspace.shared.activateFileViewerSelecting([AppEnvironment.modelDownloadDirectory])
            }
            .font(.caption)
            Spacer()
            Button("Chiudi") { dismiss() }
                .keyboardShortcut(.cancelAction)
                .disabled(runner.isRunning)
                .help(runner.isRunning ? "Annulla il download prima di chiudere." : "Chiudi")
        }
    }

    private func configureRunner() {
        var directories = [
            AppEnvironment.modelDownloadDirectory,
            URL(fileURLWithPath: store.scriptDir, isDirectory: true),
            URL(fileURLWithPath: store.scriptDir, isDirectory: true)
                .appendingPathComponent("gguf", isDirectory: true),
        ]
        if !store.modelPath.isEmpty {
            directories.append(URL(fileURLWithPath: store.modelPath)
                .deletingLastPathComponent())
        }
        runner.configure(searchDirectories: directories,
                         destination: AppEnvironment.modelDownloadDirectory)
    }

    @discardableResult
    private func selectCatalogPath(_ path: String) -> Bool {
        store.selectCatalogModel(path: path)
    }

    private func selectedPath(for entry: ModelCatalogEntry,
                              installation: CatalogInstallation) -> String? {
        guard entry.isSelectable, entry.artifacts.count == 1,
              let target = entry.artifacts.first else { return nil }
        return installation.pathsByTargetID[target.id]
    }

    private func pathsAreEqual(_ lhs: String, _ rhs: String) -> Bool {
        URL(fileURLWithPath: lhs).resolvingSymlinksInPath().standardizedFileURL.path
            == URL(fileURLWithPath: rhs).resolvingSymlinksInPath().standardizedFileURL.path
    }

    private func installStateColor(_ state: CatalogInstallState) -> Color {
        switch state {
        case .notDownloaded: .secondary
        case .partial: .orange
        case .invalidLocalFile: .red
        case .installed: .green
        }
    }

    private func formatBytes(_ bytes: Int64) -> String {
        ByteCountFormatter.string(fromByteCount: bytes, countStyle: .file)
    }
}
