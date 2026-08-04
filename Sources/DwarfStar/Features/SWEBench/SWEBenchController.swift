import Foundation
import AppKit
import DS4Engine

struct SWEBenchPrediction: Codable, Sendable {
    let instanceID: String
    let modelName: String
    let modelPatch: String?

    enum CodingKeys: String, CodingKey {
        case instanceID = "instance_id"
        case modelName = "model_name_or_path"
        case modelPatch = "model_patch"
    }
}

private struct SWELocalTaskDocument: Decodable {
    let instanceID: String
    let goldPatch: String?
    let repo: String?
    let baseCommit: String?
    let problemStatement: String?
    enum CodingKeys: String, CodingKey {
        case instanceID = "instance_id"
        case goldPatch = "patch"
        case repo, baseCommit = "base_commit", problemStatement = "problem_statement"
    }
}

struct SWEGoldCatalogTask: Identifiable, Sendable {
    let id: String
    let repo: String
    let version: String
    let taskJSON: Data
}

struct SWEModelExchange: Identifiable, Codable, Sendable {
    let id: UUID
    let createdAt: Date
    let instanceID: String
    let prompt: String
    let thinking: String
    let response: String
    let extractedPatch: String

    enum CodingKeys: String, CodingKey {
        case id, createdAt = "created_at", instanceID = "instance_id"
        case prompt, thinking, response, extractedPatch = "extracted_patch"
    }
}

enum SWELocalPredictionMode: String, CaseIterable, Identifiable {
    case model = "Generata dal modello"
    case gold = "Patch Gold del task"
    case jsonl = "Prediction JSONL"
    var id: String { rawValue }
}

struct SWELocalGradeReport: Decodable, Sendable {
    let instanceID: String
    let patchApplied: Bool
    let resolved: Bool
    let failToPassSuccess: [String]
    let failToPassFailure: [String]
    let passToPassSuccess: [String]
    let passToPassFailure: [String]
    let exitCode: Int32
    let timedOut: Bool
    let verificationOnly: Bool?

    enum CodingKeys: String, CodingKey {
        case instanceID = "instance_id", patchApplied = "patch_applied", resolved
        case failToPassSuccess = "fail_to_pass_success", failToPassFailure = "fail_to_pass_failure"
        case passToPassSuccess = "pass_to_pass_success", passToPassFailure = "pass_to_pass_failure"
        case exitCode = "exit_code", timedOut = "timed_out"
        case verificationOnly = "verification_only"
    }
}

enum SWEBenchValidationError: LocalizedError {
    case unreadable(String), invalidLine(Int, String), duplicate(String), empty

    var errorDescription: String? {
        switch self {
        case .unreadable(let message): return "Impossibile leggere le prediction: \(message)"
        case .invalidLine(let line, let message): return "JSONL non valido alla riga \(line): \(message)"
        case .duplicate(let id): return "instance_id duplicato: \(id)"
        case .empty: return "Il file non contiene prediction SWE-bench."
        }
    }
}

enum SWEBenchPredictions {
    static func load(path: String) throws -> [SWEBenchPrediction] {
        let contents: String
        do { contents = try String(contentsOfFile: path, encoding: .utf8) }
        catch { throw SWEBenchValidationError.unreadable(error.localizedDescription) }
        var values: [SWEBenchPrediction] = [], ids = Set<String>()
        for (offset, raw) in contents.components(separatedBy: .newlines).enumerated() {
            let line = raw.trimmingCharacters(in: .whitespacesAndNewlines)
            guard !line.isEmpty else { continue }
            do {
                let value = try JSONDecoder().decode(SWEBenchPrediction.self, from: Data(line.utf8))
                guard !value.instanceID.isEmpty, !value.modelName.isEmpty else {
                    throw SWEBenchValidationError.invalidLine(offset + 1, "instance_id e model_name_or_path sono obbligatori")
                }
                guard ids.insert(value.instanceID).inserted else {
                    throw SWEBenchValidationError.duplicate(value.instanceID)
                }
                values.append(value)
            } catch let error as SWEBenchValidationError { throw error }
            catch { throw SWEBenchValidationError.invalidLine(offset + 1, error.localizedDescription) }
        }
        guard !values.isEmpty else { throw SWEBenchValidationError.empty }
        return values
    }
}

@MainActor
@Observable
final class SWEBenchController {
    let store: ChatStore
    var predictionMode: SWELocalPredictionMode = .model
    var predictionsPath = ""
    var outputDirectory = ""
    var runID = "dwarfstar"
    var timeoutSeconds = 1_800
    var localRunnerPath = ""
    var localTaskPath = ""
    var localWorkspace = ""
    var allowUnsafeHostExecution = false
    var patchOnlyMode = true
    var selectedTaskInstanceID = ""
    var selectedTaskHasGoldPatch = false
    var goldCatalog: [SWEGoldCatalogTask] = []
    var goldCatalogFilter = ""
    var selectedGoldTaskID = ""
    var goldCatalogLoading = false
    var goldCatalogStatus = ""
    var modelGenerationStatus = ""
    var isGeneratingPatch = false
    var currentModelPrompt = ""
    var currentModelThinking = ""
    var currentModelResponse = ""
    var modelExchanges: [SWEModelExchange] = []
    var lastModelTranscriptPath = ""
    var isBatchRunning = false
    var batchCompleted = 0
    var batchTotal = 0
    var batchStatus = ""

    var predictionCount = 0
    var nonEmptyPatchCount = 0
    var isRunning = false
    var log = ""
    var localReport: SWELocalGradeReport?
    var lastError: String?

    private var scopedURLs: [URL] = []
    private var externalMonitor: Task<Void, Never>?
    private var externalLogURL: URL?
    private var externalStatusURL: URL?
    private var externalLogBytes: UInt64 = 0
    private var pendingLocalReportURL: URL?
    private var modelGenerationTask: Task<Void, Never>?
    private var batchTask: Task<Void, Never>?

    init(store: ChatStore) {
        self.store = store
        outputDirectory = FileManager.default.urls(for: .applicationSupportDirectory,
                                                    in: .userDomainMask).first?
            .appendingPathComponent("DwarfStar/SWEBench", isDirectory: true).path ?? NSTemporaryDirectory()
        localWorkspace = URL(fileURLWithPath: outputDirectory, isDirectory: true)
            .appendingPathComponent("local-workspaces", isDirectory: true).path
        let bundled = Bundle.main.resourceURL?.appendingPathComponent("DS4SWEBenchLocal").path ?? ""
        let sibling = Bundle.main.bundleURL.deletingLastPathComponent().appendingPathComponent("ds4-swe-local").path
        let cwd = URL(fileURLWithPath: FileManager.default.currentDirectoryPath)
        let candidates = [bundled, sibling,
                          cwd.appendingPathComponent(".build/debug/ds4-swe-local").path,
                          cwd.appendingPathComponent(".build/release/ds4-swe-local").path]
            .filter { FileManager.default.isExecutableFile(atPath: $0) }
        localRunnerPath = candidates.max { lhs, rhs in
            let left = (try? FileManager.default.attributesOfItem(atPath: lhs)[.modificationDate] as? Date) ?? .distantPast
            let right = (try? FileManager.default.attributesOfItem(atPath: rhs)[.modificationDate] as? Date) ?? .distantPast
            return left < right
        } ?? ""
    }

    func generateModelPrediction() {
        guard !isGeneratingPatch else { return }
        guard let backend = store.isReady ? store.chatBackend : nil else {
            lastError = "Carica prima un modello locale nelle Impostazioni."
            return
        }
        guard let data = try? Data(contentsOf: URL(fileURLWithPath: localTaskPath)),
              let task = try? JSONDecoder().decode(SWELocalTaskDocument.self, from: data),
              let statement = task.problemStatement?.trimmingCharacters(in: .whitespacesAndNewlines),
              !statement.isEmpty else {
            lastError = "Il task non contiene problem_statement."
            return
        }
        let gate = EngineActivityGate.shared
        guard let lease = gate.acquire(.sweBench) else {
            lastError = "Motore occupato da \(gate.activeOwner?.displayName ?? "un'altra operazione")."
            return
        }
        isGeneratingPatch = true
        modelGenerationStatus = "Invio del prompt SWE-bench al modello…"
        currentModelThinking = ""
        currentModelResponse = ""
        lastError = nil
        let prompt = Self.modelPrompt(task: task, statement: statement)
        currentModelPrompt = prompt
        modelGenerationTask = Task { [weak self] in
            do {
                let stream = await backend.complete(
                    turns: [.system("Sei un software engineer. Rispondi esclusivamente con una unified diff git applicabile, senza spiegazioni."),
                            .user(prompt)],
                    tools: [], thinkMode: .high,
                    sampling: SamplingParams(temperature: 0.2, topK: 20, topP: 0.9, minP: 0.02),
                    maxTokens: 4_096)
                var response = ""
                var thinking = ""
                for try await event in stream {
                    switch event {
                    case .text(let text):
                        response += text
                        self?.currentModelResponse = response
                    case .reasoning(let text):
                        thinking += text
                        self?.currentModelThinking = thinking
                    case .progress(let value): self?.modelGenerationStatus = value
                    case .toolStream, .toolCall: break
                    }
                }
                guard let self else { _ = gate.release(lease); return }
                let patch = Self.extractUnifiedDiff(from: response)
                try FileManager.default.createDirectory(atPath: self.outputDirectory, withIntermediateDirectories: true)
                let exchange = SWEModelExchange(id: UUID(), createdAt: Date(),
                                                instanceID: task.instanceID, prompt: prompt,
                                                thinking: thinking, response: response,
                                                extractedPatch: patch)
                let transcriptURL = URL(fileURLWithPath: self.outputDirectory, isDirectory: true)
                    .appendingPathComponent("ds4-swe-model-\(task.instanceID)-\(exchange.id.uuidString).json")
                let transcriptEncoder = JSONEncoder()
                transcriptEncoder.outputFormatting = [.prettyPrinted, .sortedKeys]
                transcriptEncoder.dateEncodingStrategy = .iso8601
                try transcriptEncoder.encode(exchange).write(to: transcriptURL, options: .atomic)
                self.modelExchanges.insert(exchange, at: 0)
                self.lastModelTranscriptPath = transcriptURL.path
                guard !patch.isEmpty else {
                    throw SWEBenchValidationError.invalidLine(1, "il modello non ha prodotto una unified diff; transcript salvato")
                }
                let prediction = SWEBenchPrediction(instanceID: task.instanceID,
                                                    modelName: "dwarfstar-local", modelPatch: patch)
                let url = URL(fileURLWithPath: self.outputDirectory, isDirectory: true)
                    .appendingPathComponent("ds4-swe-local-\(task.instanceID)-model.jsonl")
                var encoded = try JSONEncoder().encode(prediction); encoded.append(0x0A)
                try encoded.write(to: url, options: .atomic)
                self.predictionsPath = url.path
                self.validatePredictions()
                self.modelGenerationStatus = "Prompt eseguito: model_patch salvata (\(patch.utf8.count) byte)."
                self.isGeneratingPatch = false
                _ = gate.release(lease)
            } catch {
                self?.lastError = "Generazione patch fallita: \(error.localizedDescription)"
                self?.modelGenerationStatus = ""
                self?.isGeneratingPatch = false
                _ = gate.release(lease)
            }
        }
    }

    func generateAllModelPredictions() {
        guard !isBatchRunning, !goldCatalog.isEmpty else { return }
        guard store.isReady, store.chatBackend != nil else {
            lastError = "Carica prima un modello locale nelle Impostazioni."
            return
        }
        isBatchRunning = true
        batchCompleted = 0
        batchTotal = goldCatalog.count
        batchStatus = "Preparazione batch SWE-bench Lite…"
        lastError = nil
        let taskIDs = goldCatalog.map(\.id)
        batchTask = Task { [weak self] in
            guard let self else { return }
            for id in taskIDs {
                if Task.isCancelled { break }
                self.batchStatus = "\(self.batchCompleted + 1)/\(self.batchTotal) · \(id)"
                self.selectedGoldTaskID = id
                self.selectGoldTask(id)
                self.predictionMode = .model
                self.generateModelPrediction()
                while self.isGeneratingPatch && !Task.isCancelled {
                    try? await Task.sleep(for: .milliseconds(250))
                }
                if Task.isCancelled { break }
                self.batchCompleted += 1
            }
            let cancelled = Task.isCancelled
            self.isBatchRunning = false
            self.batchStatus = cancelled
                ? "Batch interrotto dopo \(self.batchCompleted)/\(self.batchTotal) task."
                : "Batch completato: \(self.batchCompleted)/\(self.batchTotal) task."
            self.batchTask = nil
        }
    }

    func stopModelBatch() {
        batchTask?.cancel()
        batchTask = nil
        modelGenerationTask?.cancel()
        batchStatus = "Interruzione batch in corso…"
    }

    private static func modelPrompt(task: SWELocalTaskDocument, statement: String) -> String {
        """
        Repository: \(task.repo ?? "sconosciuto")
        Commit di base: \(task.baseCommit ?? "sconosciuto")

        Problema SWE-bench:
        \(statement)

        Produci la correzione come unified diff git. Non includere markdown o spiegazioni.
        """
    }

    private static func extractUnifiedDiff(from response: String) -> String {
        var value = response.trimmingCharacters(in: .whitespacesAndNewlines)
        if value.hasPrefix("```") {
            let lines = value.components(separatedBy: .newlines)
            if lines.count >= 3 { value = lines.dropFirst().dropLast().joined(separator: "\n") }
        }
        guard value.contains("diff --git ") || (value.contains("--- ") && value.contains("+++ ")) else { return "" }
        return value + (value.hasSuffix("\n") ? "" : "\n")
    }

    func validatePredictions() {
        do {
            let values = try SWEBenchPredictions.load(path: predictionsPath)
            predictionCount = values.count
            nonEmptyPatchCount = values.filter { !($0.modelPatch ?? "").isEmpty }.count
            lastError = nil
        } catch {
            predictionCount = 0; nonEmptyPatchCount = 0
            lastError = error.localizedDescription
        }
    }

    func selectPredictions(_ url: URL) { retain(url); predictionsPath = url.path; validatePredictions() }
    func selectOutputDirectory(_ url: URL) { retain(url); outputDirectory = url.path }
    func selectLocalTask(_ url: URL) {
        retain(url); localTaskPath = url.path
        inspectSelectedTask()
    }
    func selectLocalRunner(_ url: URL) { retain(url); localRunnerPath = url.path }

    private func retain(_ url: URL) {
        let value = url.standardizedFileURL
        guard !scopedURLs.contains(value) else { return }
        if value.startAccessingSecurityScopedResource() { scopedURLs.append(value) }
    }

    private func inspectSelectedTask() {
        selectedTaskInstanceID = ""; selectedTaskHasGoldPatch = false
        guard let data = try? Data(contentsOf: URL(fileURLWithPath: localTaskPath)),
              let task = try? JSONDecoder().decode(SWELocalTaskDocument.self, from: data) else {
            lastError = "Task JSON non valido o privo di instance_id."
            return
        }
        selectedTaskInstanceID = task.instanceID
        selectedTaskHasGoldPatch = !(task.goldPatch ?? "").isEmpty
        lastError = nil
    }

    var filteredGoldCatalog: [SWEGoldCatalogTask] {
        let query = goldCatalogFilter.trimmingCharacters(in: .whitespacesAndNewlines).lowercased()
        guard !query.isEmpty else { return goldCatalog }
        return goldCatalog.filter { $0.id.lowercased().contains(query) || $0.repo.lowercased().contains(query) }
    }

    func downloadSWEGoldCatalog() {
        guard !goldCatalogLoading else { return }
        goldCatalogLoading = true; goldCatalogStatus = "Scaricamento Gold SWE-bench Lite…"; lastError = nil
        Task { [weak self] in
            do {
                var result: [SWEGoldCatalogTask] = []
                for offset in stride(from: 0, to: 300, by: 100) {
                    var url = URLComponents(string: "https://datasets-server.huggingface.co/rows")!
                    url.queryItems = [URLQueryItem(name: "dataset", value: "SWE-bench/SWE-bench_Lite"),
                                      URLQueryItem(name: "config", value: "default"),
                                      URLQueryItem(name: "split", value: "test"),
                                      URLQueryItem(name: "offset", value: String(offset)),
                                      URLQueryItem(name: "length", value: "100")]
                    let (data, response) = try await URLSession.shared.data(from: url.url!)
                    guard (response as? HTTPURLResponse)?.statusCode == 200,
                          let root = try JSONSerialization.jsonObject(with: data) as? [String: Any],
                          let rows = root["rows"] as? [[String: Any]] else { throw URLError(.badServerResponse) }
                    for wrapper in rows {
                        guard let row = wrapper["row"] as? [String: Any],
                              let id = row["instance_id"] as? String,
                              let repo = row["repo"] as? String else { continue }
                        result.append(SWEGoldCatalogTask(
                            id: id, repo: repo, version: row["version"] as? String ?? "",
                            taskJSON: try JSONSerialization.data(withJSONObject: row, options: [.sortedKeys])))
                    }
                }
                guard let self else { return }
                self.goldCatalog = result.sorted { $0.id < $1.id }
                self.goldCatalogLoading = false
                self.goldCatalogStatus = "\(result.count) patch Gold disponibili"
            } catch {
                guard let self else { return }
                self.goldCatalogLoading = false; self.goldCatalogStatus = ""
                self.lastError = "Download Gold non riuscito: \(error.localizedDescription)"
            }
        }
    }

    func selectGoldTask(_ id: String) {
        guard let task = goldCatalog.first(where: { $0.id == id }) else { return }
        do {
            try FileManager.default.createDirectory(atPath: outputDirectory, withIntermediateDirectories: true)
            let url = URL(fileURLWithPath: outputDirectory, isDirectory: true)
                .appendingPathComponent("swe-gold-task-\(task.id).json")
            try task.taskJSON.write(to: url, options: .atomic)
            localTaskPath = url.path
            predictionsPath = ""
            predictionCount = 0
            nonEmptyPatchCount = 0
            modelGenerationStatus = ""
            inspectSelectedTask()
        } catch { lastError = "Impossibile salvare il task Gold: \(error.localizedDescription)" }
    }

    private func materializeGoldPrediction(runID: String) throws -> String {
        let taskData = try Data(contentsOf: URL(fileURLWithPath: localTaskPath))
        let task = try JSONDecoder().decode(SWELocalTaskDocument.self, from: taskData)
        guard let patch = task.goldPatch, !patch.isEmpty else {
            throw SWEBenchValidationError.invalidLine(1, "il task non contiene la patch Gold")
        }
        let prediction = SWEBenchPrediction(instanceID: task.instanceID,
                                            modelName: "gold", modelPatch: patch)
        let url = URL(fileURLWithPath: outputDirectory, isDirectory: true)
            .appendingPathComponent("ds4-swe-local-\(runID)-gold.jsonl")
        var data = try JSONEncoder().encode(prediction); data.append(0x0A)
        try data.write(to: url, options: .atomic)
        return url.path
    }

    func run() {
        guard !isRunning else { return }
        guard !localRunnerPath.isEmpty, FileManager.default.isExecutableFile(atPath: localRunnerPath) else {
            lastError = "Runner ds4-swe-local non trovato. Compilalo con `swift build --product ds4-swe-local` o seleziona l’eseguibile."
            return
        }
        guard !localTaskPath.isEmpty else {
            lastError = "Seleziona un task Swift-bench JSON."
            return
        }
        guard patchOnlyMode || allowUnsafeHostExecution else {
            lastError = "Conferma esplicitamente l’esecuzione di codice non attendibile sul Mac."
            return
        }
        let id = runID.trimmingCharacters(in: .whitespacesAndNewlines)
        guard !id.isEmpty, id.range(of: #"^[A-Za-z0-9._-]+$"#, options: .regularExpression) != nil else {
            lastError = "Run ID non valido: usa lettere, numeri, punto, trattino o underscore."
            return
        }

        let taskSnapshot: String, predictionSnapshot: String
        do {
            try FileManager.default.createDirectory(atPath: outputDirectory, withIntermediateDirectories: true)
            try FileManager.default.createDirectory(atPath: localWorkspace, withIntermediateDirectories: true)
            if predictionMode == .gold {
                predictionsPath = try materializeGoldPrediction(runID: id)
            }
            validatePredictions()
            guard lastError == nil, predictionCount == 1 else {
                throw SWEBenchValidationError.invalidLine(1, "il runner locale richiede esattamente una prediction")
            }
            let output = URL(fileURLWithPath: outputDirectory, isDirectory: true)
            let taskURL = output.appendingPathComponent("ds4-swe-local-\(id)-task.json")
            let predictionURL = output.appendingPathComponent("ds4-swe-local-\(id)-prediction.json")
            try Data(contentsOf: URL(fileURLWithPath: localTaskPath)).write(to: taskURL, options: .atomic)
            try Data(contentsOf: URL(fileURLWithPath: predictionsPath)).write(to: predictionURL, options: .atomic)
            taskSnapshot = taskURL.path; predictionSnapshot = predictionURL.path
        } catch {
            lastError = "Impossibile preparare il workspace locale: \(error.localizedDescription)"
            return
        }
        let reportURL = URL(fileURLWithPath: outputDirectory, isDirectory: true)
            .appendingPathComponent("ds4-swe-local-\(id).json")
        pendingLocalReportURL = reportURL
        var arguments = ["--task", taskSnapshot, "--prediction", predictionSnapshot,
                         "--workspace", localWorkspace, "--report", reportURL.path,
                         "--timeout", String(timeoutSeconds)]
        arguments.append(patchOnlyMode ? "--patch-only" : "--allow-unsafe-host-execution")
        runInTerminal(arguments: arguments, runID: id)
    }

    private func runInTerminal(arguments: [String], runID: String) {
        let output = URL(fileURLWithPath: outputDirectory, isDirectory: true)
        let logURL = output.appendingPathComponent("dwarfstar-swebench-local-\(runID).log")
        let statusURL = output.appendingPathComponent("dwarfstar-swebench-local-\(runID).status")
        try? FileManager.default.removeItem(at: statusURL)
        let command = ([localRunnerPath] + arguments).map(Self.shellQuote).joined(separator: " ")
        let script = """
        cd \(Self.shellQuote(outputDirectory))
        : > \(Self.shellQuote(logURL.path))
        \(command) > >(tee -a \(Self.shellQuote(logURL.path))) 2>&1
        ds4_status=$?
        print -r -- $ds4_status > \(Self.shellQuote(statusURL.path))
        exit $ds4_status
        """
        NSPasteboard.general.clearContents()
        NSPasteboard.general.setString("/bin/zsh -lc " + Self.shellQuote(script), forType: .string)
        localReport = nil; lastError = nil; isRunning = true
        externalLogURL = logURL; externalStatusURL = statusURL; externalLogBytes = 0
        log = "Comando Swift locale copiato. In Terminal premi ⌘V e Invio.\n"
        let configuration = NSWorkspace.OpenConfiguration(); configuration.activates = true
        NSWorkspace.shared.openApplication(
            at: URL(fileURLWithPath: "/System/Applications/Utilities/Terminal.app"),
            configuration: configuration
        ) { [weak self] _, error in
            guard let error else { return }
            Task { @MainActor [weak self] in
                self?.externalMonitor?.cancel(); self?.externalMonitor = nil
                self?.isRunning = false
                self?.lastError = "Impossibile aprire Terminal: \(error.localizedDescription)"
            }
        }
        monitorExternalRun()
    }

    private nonisolated static func shellQuote(_ value: String) -> String {
        "'" + value.replacingOccurrences(of: "'", with: "'\\''") + "'"
    }

    private func monitorExternalRun() {
        externalMonitor?.cancel()
        externalMonitor = Task { [weak self] in
            while !Task.isCancelled {
                try? await Task.sleep(for: .seconds(1))
                guard let self else { return }
                self.consumeExternalLog()
                if let url = self.externalStatusURL,
                   let raw = try? String(contentsOf: url, encoding: .utf8),
                   let code = Int32(raw.trimmingCharacters(in: .whitespacesAndNewlines)) {
                    self.finish(exitCode: code); return
                }
            }
        }
    }

    private func consumeExternalLog() {
        guard let url = externalLogURL, let handle = try? FileHandle(forReadingFrom: url) else { return }
        defer { try? handle.close() }
        do {
            try handle.seek(toOffset: externalLogBytes)
            let data = try handle.readToEnd() ?? Data()
            externalLogBytes += UInt64(data.count)
            if let text = String(data: data, encoding: .utf8) { log += text }
        } catch { }
    }

    func stop() {
        guard externalMonitor != nil else { return }
        lastError = "Il runner gira in Terminal: interrompilo con Ctrl-C nella finestra Terminal."
    }

    private func finish(exitCode: Int32) {
        externalMonitor?.cancel(); externalMonitor = nil
        consumeExternalLog(); isRunning = false
        if let url = pendingLocalReportURL,
           let data = try? Data(contentsOf: url),
           let value = try? JSONDecoder().decode(SWELocalGradeReport.self, from: data) {
            localReport = value; pendingLocalReportURL = nil
        } else {
            lastError = "Runner terminato senza produrre un report valido (codice \(exitCode))."
        }
    }
}
