import Foundation

enum MiniToolBenchSuite: String, Codable, CaseIterable, Identifiable, Sendable {
    case core19
    case legacyMini20 = "legacy-mini20"
    var id: String { rawValue }
    var title: String { self == .core19 ? "Core-19 · 19 task" : "Mini-20 storico · 20 task" }
}

enum MiniToolBenchTier: String, Codable, CaseIterable, Identifiable, Sendable {
    case full, smoke
    var id: String { rawValue }
    var title: String { self == .full ? "Suite completa" : "Smoke · 1 task" }
}

struct MiniToolBenchConfiguration: Codable, Equatable, Sendable {
    var runnerDirectory = "~/terminal-bench-mini"
    var pythonExecutable = "python3"
    var endpoint = "http://127.0.0.1:8080/v1"
    var platform = "apple-silicon"
    var modelName = ""
    var engine = "DwarfStar"
    var backend = "metal"
    var quant = ""
    var inferenceProfile = ""
    var modelID = ""
    var contextLength = ""
    var suite: MiniToolBenchSuite = .core19
    var tier: MiniToolBenchTier = .full
    var attempts = 2

    init() {}

    enum CodingKeys: String, CodingKey {
        case runnerDirectory, pythonExecutable, endpoint, platform, modelName, engine, backend
        case quant, inferenceProfile, modelID, contextLength, suite, tier, attempts
    }

    init(from decoder: Decoder) throws {
        self.init()
        let values = try decoder.container(keyedBy: CodingKeys.self)
        runnerDirectory = try values.decodeIfPresent(String.self, forKey: .runnerDirectory) ?? runnerDirectory
        pythonExecutable = try values.decodeIfPresent(String.self, forKey: .pythonExecutable) ?? pythonExecutable
        endpoint = try values.decodeIfPresent(String.self, forKey: .endpoint) ?? endpoint
        platform = try values.decodeIfPresent(String.self, forKey: .platform) ?? platform
        modelName = try values.decodeIfPresent(String.self, forKey: .modelName) ?? modelName
        engine = try values.decodeIfPresent(String.self, forKey: .engine) ?? engine
        backend = try values.decodeIfPresent(String.self, forKey: .backend) ?? backend
        quant = try values.decodeIfPresent(String.self, forKey: .quant) ?? quant
        inferenceProfile = try values.decodeIfPresent(String.self, forKey: .inferenceProfile) ?? inferenceProfile
        modelID = try values.decodeIfPresent(String.self, forKey: .modelID) ?? modelID
        contextLength = try values.decodeIfPresent(String.self, forKey: .contextLength) ?? contextLength
        suite = try values.decodeIfPresent(MiniToolBenchSuite.self, forKey: .suite) ?? suite
        tier = try values.decodeIfPresent(MiniToolBenchTier.self, forKey: .tier) ?? tier
        attempts = try values.decodeIfPresent(Int.self, forKey: .attempts) ?? attempts
    }
}

enum MiniToolBenchError: LocalizedError {
    case invalid(String)
    var errorDescription: String? {
        switch self { case .invalid(let message): return message }
    }
}

/// Only builds reviewable shell text. The official Linux runner owns the agent,
/// containers, verification and result export; the app never simulates a score.
enum MiniToolBenchCommandBuilder {
    static let repositoryURL = URL(string: "https://github.com/kyuz0/terminal-bench-mini")!
    static let pinnedRevision = "4a84b3dad49750a2db9f2e96d23a9bd8dafe7b66"

    static var setupCommand: String {
        """
        cd "$HOME" &&
        git clone https://github.com/kyuz0/terminal-bench-mini.git terminal-bench-mini &&
        git -C terminal-bench-mini checkout --detach \(pinnedRevision)
        """
    }

    static func validationErrors(_ c: MiniToolBenchConfiguration) -> [String] {
        var errors: [String] = []
        let required = [("Directory del runner", c.runnerDirectory), ("Interprete Python", c.pythonExecutable), ("Piattaforma", c.platform),
                        ("Nome del modello", c.modelName), ("Motore", c.engine), ("Backend", c.backend)]
        for (name, value) in required where value.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty {
            errors.append("\(name): campo obbligatorio.")
        }
        let values = [c.runnerDirectory, c.pythonExecutable, c.endpoint, c.platform, c.modelName, c.engine, c.backend,
                      c.quant, c.inferenceProfile, c.modelID, c.contextLength]
        if values.contains(where: { $0.unicodeScalars.contains(where: { CharacterSet.controlCharacters.contains($0) }) }) {
            errors.append("I campi non possono contenere caratteri di controllo o righe multiple.")
        }
        let endpoint = c.endpoint.trimmingCharacters(in: .whitespacesAndNewlines)
        if let url = URLComponents(string: endpoint), ["http", "https"].contains(url.scheme?.lowercased() ?? ""),
           let host = url.host, !host.isEmpty, url.user == nil, url.password == nil,
           url.query == nil, url.fragment == nil,
           url.path.trimmingCharacters(in: CharacterSet(charactersIn: "/")) == "v1" {
            if host == "0.0.0.0" || host == "::" {
                errors.append("L’endpoint deve essere un indirizzo raggiungibile, non un indirizzo di ascolto.")
            }
        } else {
            errors.append("Inserisci un endpoint http(s) completo con /v1, senza credenziali, query o frammenti.")
        }
        let context = c.contextLength.trimmingCharacters(in: .whitespacesAndNewlines)
        if !context.isEmpty, Int(context).map({ $0 > 0 }) != true {
            errors.append("La capacità del contesto deve essere un intero positivo oppure restare vuota per il rilevamento automatico.")
        }
        if !(1...2).contains(c.attempts) { errors.append("Scegli uno o due tentativi per task.") }
        if c.modelName.lowercased().contains(".gguf") {
            errors.append("Usa il nome della famiglia/revisione del modello; indica quantizzazione e ID servito nei campi separati.")
        }
        return errors
    }

    static func shellQuote(_ value: String) -> String {
        "'" + value.replacingOccurrences(of: "'", with: "'\\''") + "'"
    }

    private static func directoryExpression(_ value: String) -> String {
        if value == "~" { return "\"$HOME\"" }
        if value.hasPrefix("~/") { return "\"$HOME\"/" + shellQuote(String(value.dropFirst(2))) }
        // A leading dash remains a path, never a cd option.
        return shellQuote(value.hasPrefix("-") ? "./" + value : value)
    }

    static func command(_ c: MiniToolBenchConfiguration) throws -> String {
        let errors = validationErrors(c)
        guard errors.isEmpty else { throw MiniToolBenchError.invalid(errors.joined(separator: "\n")) }
        func clean(_ s: String) -> String { s.trimmingCharacters(in: .whitespacesAndNewlines) }
        // Equals form preserves option-looking values as single option values.
        func option(_ key: String, _ value: String) -> String { "--\(key)=\(clean(value))" }
        var common = [option("suite", c.suite.rawValue), option("tier", c.tier.rawValue),
                      option("endpoint", c.endpoint)]
        if !clean(c.modelID).isEmpty { common.append(option("model", c.modelID)) }
        if !clean(c.contextLength).isEmpty { common.append(option("context-length", c.contextLength)) }
        var run = common + [option("platform", c.platform), option("model-name", c.modelName),
                            option("engine", c.engine), option("backend", c.backend),
                            "--attempts=\(c.attempts)", "--concurrency=1"]
        if !clean(c.quant).isEmpty { run.append(option("quant", c.quant)) }
        if !clean(c.inferenceProfile).isEmpty { run.append(option("inference-profile", c.inferenceProfile)) }
        func line(_ action: String, _ args: [String]) -> String {
            ([clean(c.pythonExecutable), "terminal_bench.py", action] + args).map(shellQuote).joined(separator: " ")
        }
        let runtimeCheck = "import sys; sys.exit('Serve Python 3.11 o successivo; interprete corrente: ' + sys.version.split()[0]) if sys.version_info < (3, 11) else None; sys.exit('Esegui questo script in una shell Linux, ad esempio nella VM di Podman, non nella shell macOS.') if sys.platform != 'linux' else None"
        let check = [clean(c.pythonExecutable), "-c", runtimeCheck].map(shellQuote).joined(separator: " ")
        return """
        cd \(directoryExpression(clean(c.runnerDirectory))) &&
        test \"$(git rev-parse HEAD)\" = \(shellQuote(pinnedRevision)) &&
        \(check) &&
        \(line("doctor", common)) &&
        \(line("run", run))
        """
    }
}

struct MiniToolBenchIdentity: Decodable, Sendable {
    let id: String?
    let name: String?
    let version: String?
    let manifestHash: String?
    enum CodingKeys: String, CodingKey { case id, name, version, manifestHash = "manifest_hash" }
}

struct MiniToolBenchTokenUsage: Decodable, Sendable {
    let input: Int?
    let output: Int?
    let cached: Int?
}

struct MiniToolBenchSummary: Decodable, Sendable {
    let schemaVersion: Int
    let totalTasks: Int
    let passedTasks: Int
    let passRate: Double
    let totalDurationMS: Double?
    let tokens: MiniToolBenchTokenUsage?
    let model: MiniToolBenchIdentity
    let platform: MiniToolBenchIdentity
    let suite: MiniToolBenchIdentity?
    let engine: String?
    let backend: String?
    let quant: String?
    let inferenceProfile: String?
    let profileHash: String?
    let results: [String]
    enum CodingKeys: String, CodingKey {
        case schemaVersion = "schema_version", totalTasks = "total_tasks", passedTasks = "passed_tasks"
        case passRate = "pass_rate", totalDurationMS = "total_duration_ms", tokens, model, platform, suite
        case engine, backend, quant, inferenceProfile = "inference_profile", profileHash = "profile_hash", results
    }
}

struct MiniToolBenchAttempt: Decodable, Identifiable, Sendable {
    var id: Int { attempt }
    let attempt: Int
    let passed: Bool
    let durationMS: Double?
    let transcript: String?
    enum CodingKeys: String, CodingKey { case attempt, passed, durationMS = "duration_ms", transcript }
}

struct MiniToolBenchTaskResult: Decodable, Identifiable, Sendable {
    var id: String { task }
    let task: String
    let passed: Bool
    let completed: Bool
    let attempts: [MiniToolBenchAttempt]?
    let durationMS: Double?
    let tokens: MiniToolBenchTokenUsage?
    let transcript: String?
    let model: MiniToolBenchIdentity?
    let suite: MiniToolBenchIdentity?
    let profileHash: String?
    enum CodingKeys: String, CodingKey {
        case task, passed, completed, attempts, durationMS = "duration_ms", tokens, transcript, model, suite
        case profileHash = "profile_hash"
    }
}

struct MiniToolBenchReport: Sendable {
    let summary: MiniToolBenchSummary
    let tasks: [MiniToolBenchTaskResult]
    let warnings: [String]
    let attemptBudget: Int?
    let directory: URL

    /// A selected directory grants access to its summary and sibling results.
    /// A standalone summary is useful too, but missing details are explicit.
    static func load(from selection: URL) throws -> MiniToolBenchReport {
        let isDirectory = (try selection.resourceValues(forKeys: [.isDirectoryKey])).isDirectory == true
        let file = isDirectory ? selection.appendingPathComponent("summary.json") : selection
        let directory = file.deletingLastPathComponent().resolvingSymlinksInPath()
        let decoder = JSONDecoder()
        let summary = try decoder.decode(MiniToolBenchSummary.self, from: readJSON(file))
        guard (1...4).contains(summary.schemaVersion) else {
            throw MiniToolBenchError.invalid("Versione del report non supportata: \(summary.schemaVersion).")
        }
        guard summary.totalTasks >= 0, summary.totalTasks <= 1_000,
              summary.passedTasks >= 0, summary.passedTasks <= summary.totalTasks,
              summary.passRate.isFinite, (0...1).contains(summary.passRate),
              summary.results.count == summary.totalTasks,
              Set(summary.results).count == summary.results.count else {
            throw MiniToolBenchError.invalid("Il riepilogo contiene conteggi o riferimenti ai task incoerenti.")
        }
        let expectedRate = summary.totalTasks == 0 ? 0 : Double(summary.passedTasks) / Double(summary.totalTasks)
        guard abs(summary.passRate - expectedRate) < 0.0001 else {
            throw MiniToolBenchError.invalid("La percentuale non corrisponde al conteggio dei task superati.")
        }
        var tasks: [MiniToolBenchTaskResult] = [], missing = 0
        var ids = Set<String>()
        for name in summary.results {
            let url = try artifactURL(name, directory: directory)
            guard FileManager.default.fileExists(atPath: url.path) else { missing += 1; continue }
            let task = try decoder.decode(MiniToolBenchTaskResult.self, from: readJSON(url))
            guard !task.task.isEmpty, ids.insert(task.task).inserted else {
                throw MiniToolBenchError.invalid("Il report contiene task duplicati o privi di identificativo.")
            }
            if let expected = summary.model.id, let actual = task.model?.id, expected != actual {
                throw MiniToolBenchError.invalid("Il task \(task.task) appartiene a un altro modello.")
            }
            if let expected = summary.suite?.manifestHash, task.suite?.manifestHash != expected {
                throw MiniToolBenchError.invalid("Il task \(task.task) appartiene a un’altra versione della suite.")
            }
            if let expected = summary.profileHash, task.profileHash != expected {
                throw MiniToolBenchError.invalid("Il task \(task.task) appartiene a un altro profilo di valutazione.")
            }
            tasks.append(task)
        }
        if missing == 0, tasks.filter(\.passed).count != summary.passedTasks {
            throw MiniToolBenchError.invalid("Gli esiti dei task non corrispondono al riepilogo.")
        }
        var warnings = ["Il riepilogo aggrega i task esportati: non certifica da solo il completamento di una suite o di una singola esecuzione."]
        if missing > 0 { warnings.append("Mancano \(missing) file dei task. Importa la cartella completa per consultarne i dettagli.") }
        struct RunMeta: Decodable {
            let maxAttempts: Int?
            let profileHash: String?
            enum CodingKeys: String, CodingKey { case maxAttempts = "max_attempts", profileHash = "profile_hash" }
        }
        let metaURL = try artifactURL("run-meta.json", directory: directory)
        var attemptBudget: Int?
        if FileManager.default.fileExists(atPath: metaURL.path) {
            do {
                let meta = try decoder.decode(RunMeta.self, from: readJSON(metaURL))
                if meta.profileHash == summary.profileHash, let count = meta.maxAttempts, count > 0 {
                    attemptBudget = count
                } else { warnings.append("Il budget dei tentativi non è disponibile per questo profilo.") }
            } catch { warnings.append("run-meta.json non leggibile: il budget dei tentativi non viene dedotto.") }
        }
        return MiniToolBenchReport(summary: summary, tasks: tasks.sorted { $0.task < $1.task },
                                   warnings: warnings, attemptBudget: attemptBudget, directory: directory)
    }

    func transcriptURL(for task: MiniToolBenchTaskResult) -> URL? {
        guard let name = task.transcript, let url = try? Self.artifactURL(name, directory: directory),
              url.pathExtension.lowercased() == "json",
              (try? url.resourceValues(forKeys: [.isRegularFileKey]).isRegularFile) == true else { return nil }
        return url
    }

    private static func readJSON(_ url: URL) throws -> Data {
        let values = try url.resourceValues(forKeys: [.isRegularFileKey, .fileSizeKey])
        guard values.isRegularFile == true, let size = values.fileSize, size <= 16 * 1_024 * 1_024 else {
            throw MiniToolBenchError.invalid("Il report deve essere un file JSON di massimo 16 MiB.")
        }
        return try Data(contentsOf: url)
    }

    private static func artifactURL(_ name: String, directory: URL) throws -> URL {
        // Official normalized artifacts are siblings, not arbitrary file URLs.
        guard !name.isEmpty, name != ".", name != "..", !name.contains("/"), !name.contains("\\"),
              !name.unicodeScalars.contains(where: { CharacterSet.controlCharacters.contains($0) }) else {
            throw MiniToolBenchError.invalid("Riferimento a un file esterno alla cartella dei risultati.")
        }
        let url = directory.appendingPathComponent(name).resolvingSymlinksInPath()
        guard url.deletingLastPathComponent() == directory else {
            throw MiniToolBenchError.invalid("Un collegamento nei risultati punta fuori dalla cartella selezionata.")
        }
        return url
    }
}
