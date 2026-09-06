import AppKit
import Foundation
import Observation
import UniformTypeIdentifiers
import DS4Engine

enum MiniToolBenchResultFilter: String, CaseIterable, Identifiable {
    case all = "Tutti"
    case passed = "Superati"
    case failed = "Non superati"
    case incomplete = "Incompleti"
    var id: String { rawValue }
}

/// Runs the Linux benchmark through Podman using the shared API server, and
/// imports its verifier results. The server owns the one engine lease.
@MainActor @Observable
final class MiniToolBenchController {
    var configuration: MiniToolBenchConfiguration {
        didSet { saveConfiguration() }
    }
    private(set) var report: MiniToolBenchReport?
    private(set) var reportRevision = 0
    private(set) var isImporting = false
    var resultSearch = ""
    var resultFilter = MiniToolBenchResultFilter.all
    var errorMessage: String?
    private(set) var statusMessage: String?
    private(set) var isManagedRunning = false
    private(set) var isManagedStopping = false
    private(set) var managedStatus = "Pronto per avviare il test"
    private(set) var managedLog = ""
    private(set) var managedError: String?

    @ObservationIgnored private let defaults: UserDefaults
    @ObservationIgnored private let store: ChatStore
    @ObservationIgnored private let server: ServerController
    @ObservationIgnored private let runner = MiniToolBenchRunner()
    @ObservationIgnored private var managedTask: Task<Void, Never>?
    @ObservationIgnored private var stopTask: Task<Void, Never>?
    @ObservationIgnored private var cancellationRequested = false
    @ObservationIgnored private var startedServerForRun = false
    @ObservationIgnored private var managedSecret = ""
    @ObservationIgnored private var importedDirectory: MiniToolBenchScopedDirectory?
    @ObservationIgnored private var importTask: Task<Void, Never>?
    private static let configurationKey = "DwarfStar.MiniToolBench.configuration.v1"

    init(store: ChatStore, server: ServerController, defaults: UserDefaults = .standard) {
        self.store = store
        self.server = server
        self.defaults = defaults
        if let data = defaults.data(forKey: Self.configurationKey),
           let configuration = try? JSONDecoder().decode(MiniToolBenchConfiguration.self, from: data) {
            self.configuration = configuration
        } else {
            self.configuration = MiniToolBenchConfiguration()
        }
    }

    var validationErrors: [String] { MiniToolBenchCommandBuilder.validationErrors(configuration) }
    var command: String { (try? MiniToolBenchCommandBuilder.command(configuration)) ?? "" }
    var canExportCommand: Bool { validationErrors.isEmpty && !command.isEmpty }
    var modelIsReady: Bool { store.isReady }
    var servedModelName: String { server.modelId }
    var managedEndpoint: String { "http://host.containers.internal:\(server.port)/v1" }
    private var managedConfiguration: MiniToolBenchConfiguration {
        var config = configuration
        config.endpoint = managedEndpoint
        config.pythonExecutable = "python3"
        config.runnerDirectory = "~/terminal-bench-mini"
        return config
    }
    var managedValidationErrors: [String] {
        var errors = MiniToolBenchCommandBuilder.validationErrors(managedConfiguration)
        if !(1...65_535).contains(server.port) { errors.append("La porta del Server API deve essere compresa tra 1 e 65535.") }
        return errors
    }
    var canStartManagedRun: Bool {
        modelIsReady && !isManagedRunning && !isManagedStopping && !isImporting && managedValidationErrors.isEmpty
    }

    func startManagedRun() {
        guard !isManagedRunning, !isManagedStopping, !isImporting else { return }
        managedError = nil
        guard store.isReady else {
            managedError = "Carica il modello nelle impostazioni prima di avviare il test."
            return
        }
        guard managedValidationErrors.isEmpty else {
            managedError = managedValidationErrors.joined(separator: "\n")
            return
        }
        let config = managedConfiguration
        isManagedRunning = true
        isManagedStopping = false
        cancellationRequested = false
        startedServerForRun = false
        managedLog = ""
        statusMessage = nil
        managedStatus = "Preparazione del Server API…"
        managedTask = Task { [weak self] in
            guard let self else { return }
            defer {
                self.isManagedRunning = false
                if self.stopTask == nil { self.isManagedStopping = false }
                self.managedTask = nil
                self.managedSecret = ""
            }
            do {
                try await self.ensureServerStarted()
                try self.checkManagedCancellation()
                let key = self.server.apiKey.trimmingCharacters(in: .whitespacesAndNewlines)
                self.managedSecret = key
                self.appendManagedLog("Server API pronto. Modello condiviso: \(self.server.modelId)")
                self.appendManagedLog("Endpoint per Podman: \(config.endpoint)")
                self.managedStatus = "Preparazione dell’ambiente Linux…"
                let directory = try await self.runner.run(configuration: config,
                    apiKey: key.isEmpty ? nil : key) { [weak self] event in
                        self?.receiveManagedEvent(event)
                    }
                try self.checkManagedCancellation()
                if let directory {
                    self.managedStatus = "Test terminato. Importazione dei risultati…"
                    self.appendManagedLog("Risultati disponibili: \(directory.path)")
                    self.importReport(from: directory, managed: true)
                } else {
                    self.managedStatus = "Esecuzione terminata senza un report da importare."
                    self.appendManagedLog(self.managedStatus)
                }
            } catch is CancellationError {
                self.managedStatus = self.cancelledStatus
                self.appendManagedLog(self.managedStatus)
            } catch {
                if self.cancellationRequested {
                    self.managedStatus = self.cancelledStatus
                } else {
                    self.managedStatus = "Il test non è stato completato."
                    self.managedError = self.redacted(error.localizedDescription)
                }
                self.appendManagedLog(self.redacted(error.localizedDescription))
            }
        }
    }

    func stopManagedRun() {
        guard isManagedRunning, !isManagedStopping else { return }
        cancellationRequested = true
        isManagedStopping = true
        managedStatus = "Interruzione del test in corso…"
        let taskToCancel = managedTask
        let cancellingStartup = !runner.isRunning
        let stopOwnedStartup = cancellingStartup && startedServerForRun && server.isLoading
        stopTask = Task { [weak self] in
            guard let self else { return }
            defer {
                self.stopTask = nil
                if !self.isManagedRunning { self.isManagedStopping = false }
            }
            do {
                if cancellingStartup {
                    taskToCancel?.cancel()
                    if stopOwnedStartup { self.server.stop() }
                } else {
                    // Keep the local reader alive until the remote runner has
                    // stopped its process group and drained its final output.
                    try await self.runner.cancel()
                }
            } catch {
                self.cancellationRequested = false
                self.isManagedStopping = false
                self.managedError = "Interruzione non riuscita: \(self.redacted(error.localizedDescription))"
                self.appendManagedLog(self.managedError ?? "Interruzione non riuscita")
            }
        }
    }

    func copyManagedLog() { copy(managedLog, message: "Log del test copiato.") }

    private func ensureServerStarted() async throws {
        try checkManagedCancellation()
        guard !server.isStopping else {
            throw MiniToolBenchManagedError.message("Il Server API si sta arrestando. Attendi il completamento e riprova.")
        }
        if server.isRunning { return }
        appendManagedLog("Avvio del Server API sul modello già caricato…")
        if !server.isLoading {
            startedServerForRun = true
            server.start()
        }
        let deadline = ContinuousClock.now.advanced(by: .seconds(180))
        while server.isLoading && !server.isRunning {
            try checkManagedCancellation()
            guard ContinuousClock.now < deadline else {
                throw MiniToolBenchManagedError.message("Il Server API non è pronto dopo 3 minuti. Controlla il pannello Server API; l’avvio può proseguire in background.")
            }
            try await Task.sleep(for: .milliseconds(200))
        }
        try checkManagedCancellation()
        guard server.isRunning else {
            let detail = server.log.trimmingCharacters(in: .whitespacesAndNewlines)
            throw MiniToolBenchManagedError.message("Impossibile avviare il Server API. \(detail.isEmpty ? "Controlla il modello e le operazioni attive sul motore." : detail)")
        }
    }

    private func checkManagedCancellation() throws {
        try Task.checkCancellation()
        if cancellationRequested { throw CancellationError() }
    }

    private var cancelledStatus: String {
        server.isRunning ? "Test interrotto. Il Server API resta disponibile." : "Test interrotto."
    }

    private func receiveManagedEvent(_ event: MiniToolBenchRunEvent) {
        if let message = event.message, !message.isEmpty {
            appendManagedLog(message)
            if event.event != "log" && !isManagedStopping {
                managedStatus = redacted(message)
            }
        }
    }

    private func redacted(_ text: String) -> String {
        managedSecret.isEmpty ? text : text.replacingOccurrences(of: managedSecret, with: "[chiave API]")
    }

    private func appendManagedLog(_ text: String) {
        managedLog += redacted(text) + (text.hasSuffix("\n") ? "" : "\n")
        if managedLog.count > 100_000 { managedLog = String(managedLog.suffix(100_000)) }
    }

    var filteredTasks: [MiniToolBenchTaskResult] {
        guard let report else { return [] }
        let query = resultSearch.trimmingCharacters(in: .whitespacesAndNewlines)
        return report.tasks.filter { task in
            let matchesSearch = query.isEmpty || task.task.localizedCaseInsensitiveContains(query)
            let matchesFilter: Bool
            switch resultFilter {
            case .all: matchesFilter = true
            case .passed: matchesFilter = task.passed
            case .failed: matchesFilter = task.completed && !task.passed
            case .incomplete: matchesFilter = !task.completed
            }
            return matchesSearch && matchesFilter
        }.sorted { $0.task.localizedStandardCompare($1.task) == .orderedAscending }
    }

    func copySetup() { copy(MiniToolBenchCommandBuilder.setupCommand, message: "Comandi di preparazione copiati.") }

    func copyCommand() {
        guard canExportCommand else { return }
        copy(command, message: "Comandi doctor e run copiati. Eseguili sul runner Linux.")
    }

    func exportScript() {
        guard canExportCommand else { return }
        let panel = NSSavePanel()
        panel.title = "Esporta script per il runner Linux"
        panel.nameFieldStringValue = "mini-tool-bench.sh"
        panel.allowedContentTypes = [UTType(filenameExtension: "sh") ?? .plainText]
        panel.message = "Lo script esegue doctor e poi run nella cartella Linux configurata. Preparala prima con i comandi di installazione."
        guard panel.runModal() == .OK, let url = panel.url else { return }
        let script = """
        #!/usr/bin/env bash
        set -euo pipefail
        # Mini Tool Bench — Terminal-Bench-Local
        # Runner Linux, repository revision: \(MiniToolBenchCommandBuilder.pinnedRevision)
        # Prepare the runner with the setup commands shown in DwarfStar first.

        \(command)

        """
        do {
            try script.write(to: url, atomically: true, encoding: .utf8)
            statusMessage = "Script esportato: \(url.lastPathComponent). Eseguilo con bash sul runner Linux."
        } catch { errorMessage = "Impossibile esportare lo script: \(error.localizedDescription)" }
    }

    func chooseReportDirectory() {
        guard !isImporting, !isManagedRunning else { return }
        let panel = NSOpenPanel()
        panel.title = "Importa risultati Mini Tool Bench"
        panel.prompt = "Importa risultati"
        panel.canChooseDirectories = true
        panel.canChooseFiles = false
        panel.allowsMultipleSelection = false
        panel.message = "Scegli la cartella che contiene summary.json, i file results-*.json e gli eventuali transcript copiati dal runner Linux."
        guard panel.runModal() == .OK, let directory = panel.url else { return }
        importReport(from: directory)
    }

    func importReport(from directory: URL, managed: Bool = false) {
        importTask?.cancel()
        isImporting = true
        errorMessage = nil
        statusMessage = nil
        let access = MiniToolBenchScopedDirectory(url: directory)
        importTask = Task { [weak self] in
            do {
                let loaded = try await Task.detached(priority: .userInitiated) {
                    try Task.checkCancellation()
                    return try MiniToolBenchReport.load(from: directory)
                }.value
                try Task.checkCancellation()
                guard let self else { return }
                self.report = loaded
                self.reportRevision += 1
                self.importedDirectory = access
                self.resultSearch = ""
                self.resultFilter = .all
                self.isImporting = false
                self.statusMessage = "Importati \(loaded.tasks.count) risultati verificati."
                if managed {
                    self.managedStatus = "Esecuzione terminata. Risultati importati."
                    self.appendManagedLog(self.statusMessage ?? "Risultati importati")
                }
            } catch is CancellationError {
                // A subsequent selection owns the current import state.
            } catch {
                guard let self, !Task.isCancelled else { return }
                self.isImporting = false
                let message = "Impossibile importare i risultati: \(error.localizedDescription)"
                if managed {
                    self.managedStatus = "Esecuzione terminata; controlla il report dei risultati."
                    self.managedError = message
                    self.appendManagedLog(message)
                } else { self.errorMessage = message }
            }
        }
    }

    func openTranscript(for task: MiniToolBenchTaskResult) {
        guard let report, let url = report.transcriptURL(for: task) else { return }
        if !NSWorkspace.shared.open(url) {
            errorMessage = "Nessuna applicazione disponibile per aprire \(url.lastPathComponent)."
        }
    }

    func revealReport() {
        guard let report else { return }
        NSWorkspace.shared.activateFileViewerSelecting([report.directory.appendingPathComponent("summary.json")])
    }

    private func copy(_ text: String, message: String) {
        NSPasteboard.general.clearContents()
        if NSPasteboard.general.setString(text, forType: .string) { statusMessage = message }
        else { errorMessage = "Impossibile copiare negli appunti." }
    }

    private func saveConfiguration() {
        if let data = try? JSONEncoder().encode(configuration) {
            defaults.set(data, forKey: Self.configurationKey)
        }
        statusMessage = nil
    }
}

/// Keep access to siblings of summary.json for as long as their report is
/// shown. Replacing the imported report releases the previous directory.
private final class MiniToolBenchScopedDirectory {
    let url: URL
    private let didStart: Bool
    init(url: URL) {
        self.url = url
        didStart = url.startAccessingSecurityScopedResource()
    }
    deinit { if didStart { url.stopAccessingSecurityScopedResource() } }
}

private enum MiniToolBenchManagedError: LocalizedError {
    case message(String)
    var errorDescription: String? {
        switch self { case .message(let text): return text }
    }
}
