import Foundation
import DS4Core

// Tool registry for the GUI: built-in, auto-executable demo tools plus the
// plumbing to run a model-emitted ToolCall. Tools the registry doesn't know are
// left to the UI to answer manually.

/// A tool the app can execute itself. `run` receives the model's raw JSON
/// arguments and returns a JSON (or plain text) result string.
public struct BuiltinTool: Sendable {
    public let spec: ToolSpec
    public let run: @Sendable (_ argumentsJSON: String) -> String
    public init(spec: ToolSpec, run: @escaping @Sendable (_ argumentsJSON: String) -> String) {
        self.spec = spec; self.run = run
    }
}

/// The result of a tool, ready to feed back into the conversation.
public struct ToolOutput: Sendable, Equatable {
    public var callId: String
    public var name: String
    public var content: String
    public init(callId: String, name: String, content: String) {
        self.callId = callId; self.name = name; self.content = content
    }
}

public enum ToolRegistry {
    /// The built-in tools. project_write/project_edit and file_write/file_add/
    /// file_modify have side effects (they modify files INSIDE the active project
    /// root only); everything else is pure.
    public static let builtins: [BuiltinTool] = [clock, calculator, add, subtract, multiply,
                                                 projectList, projectRead, projectSearch,
                                                 projectWrite, projectEdit,
                                                 fileRead, fileWrite, fileAdd, fileModify, git,
                                                 agentsList, subagentSearch, subagentRun]

    /// Tools that require an imported project (a root to operate in). The sub-agent
    /// resolver drops these when no project is loaded.
    public static let projectScoped: Set<String> = [
        "project_list", "project_read", "project_search", "project_edit", "project_write",
        "file_read", "file_write", "file_add", "file_modify", "git",
    ]

    public static func builtin(named name: String) -> BuiltinTool? {
        builtins.first { $0.spec.name == name }
    }

    /// Specs for the named subset (used to declare tools to the model).
    public static func specs(enabled names: Set<String>) -> [ToolSpec] {
        builtins.filter { names.contains($0.spec.name) }.map(\.spec)
    }

    /// Tools a sub-agent may be granted: every built-in EXCEPT the orchestration
    /// tools (no nested sub-agents; `agents_list` is for the orchestrator, not for
    /// doing work). The main agent passes a minimal subset of these to
    /// `subagent_run`; names outside this set are ignored.
    public static var subAgentGrantable: Set<String> {
        Set(builtins.map(\.spec.name)).subtracting(["subagent_run", "subagent_search", "agents_list"])
    }

    /// Run a model-emitted call against the built-ins; nil if it's not a built-in
    /// (the UI must then supply the result manually).
    public static func execute(_ call: ToolCall) -> ToolOutput? {
        guard let tool = builtin(named: call.name) else { return nil }
        return ToolOutput(callId: call.id, name: call.name, content: tool.run(call.argumentsJSON))
    }

    // MARK: - Built-ins

    /// Current date/time in ISO-8601 (no parameters).
    static let clock = BuiltinTool(
        spec: ToolSpec(name: "now",
                       description: "Return the current local date and time in ISO-8601 format.",
                       parametersJSON: #"{"type":"object","properties":{}}"#),
        run: { _ in
            let f = ISO8601DateFormatter()
            return #"{"datetime":"\#(f.string(from: Date()))"}"#
        })

    /// Evaluate a basic arithmetic expression (+ - * / parentheses).
    static let calculator = BuiltinTool(
        spec: ToolSpec(name: "calculator",
                       description: "Evaluate a basic arithmetic expression with + - * / and parentheses.",
                       parametersJSON: #"{"type":"object","properties":{"expression":{"type":"string"}},"required":["expression"]}"#),
        run: { argsJSON in
            guard let data = argsJSON.data(using: .utf8),
                  let obj = try? JSONSerialization.jsonObject(with: data) as? [String: Any],
                  let expr = obj["expression"] as? String else {
                return #"{"error":"missing 'expression' argument"}"#
            }
            return evaluateArithmetic(expr)
        })

    // MARK: Project-exploration tools (read-only over the imported ProjectCache;
    // results enter the chat ONLY when the model calls them, so the project
    // import never alters the conversation memory).

    static let projectList = BuiltinTool(
        spec: ToolSpec(name: "project_list",
                       description: "List files/folders of the imported project. Optional 'path' (relative) lists a subfolder; omit it for the root.",
                       parametersJSON: #"{"type":"object","properties":{"path":{"type":"string"}}}"#),
        run: { argsJSON in
            let path = stringArg(argsJSON, "path") ?? ""
            return ProjectCache.shared.listTool(path: path)
        })

    static let projectRead = BuiltinTool(
        spec: ToolSpec(name: "project_read",
                       description: "Read a project file (about 120 lines per call, with line numbers). 'path' relative; optional 'from_line' to continue.",
                       parametersJSON: #"{"type":"object","properties":{"path":{"type":"string"},"from_line":{"type":"number"}},"required":["path"]}"#),
        run: { argsJSON in
            guard let path = stringArg(argsJSON, "path") else { return "Argomento 'path' mancante." }
            let from = intArg(argsJSON, "from_line") ?? 1
            return ProjectCache.shared.readTool(path: path, fromLine: from)
        })

    static let projectSearch = BuiltinTool(
        spec: ToolSpec(name: "project_search",
                       description: "Search a text (case-insensitive) across the imported project; returns file:line matches.",
                       parametersJSON: #"{"type":"object","properties":{"query":{"type":"string"}},"required":["query"]}"#),
        run: { argsJSON in
            guard let q = stringArg(argsJSON, "query") else { return "Argomento 'query' mancante." }
            return ProjectCache.shared.searchTool(query: q)
        })

    static let projectWrite = BuiltinTool(
        spec: ToolSpec(name: "project_write",
                       description: "Create or overwrite a TEXT file inside the imported project. Use project_edit for small changes to existing files.",
                       parametersJSON: #"{"type":"object","properties":{"path":{"type":"string","description":"relative path"},"content":{"type":"string","description":"full file content"}},"required":["path","content"]}"#),
        run: { argsJSON in
            guard let p = stringArg(argsJSON, "path") else { return "Argomento 'path' mancante." }
            guard let c = stringArg(argsJSON, "content") else { return "Argomento 'content' mancante." }
            return ProjectCache.shared.writeTool(path: p, content: c)
        })

    static let projectEdit = BuiltinTool(
        spec: ToolSpec(name: "project_edit",
                       description: "Replace ONE exact occurrence of 'find' with 'replace' in a project file. 'find' must match exactly (incl. indentation) and be unique in the file — include surrounding lines to disambiguate.",
                       parametersJSON: #"{"type":"object","properties":{"path":{"type":"string"},"find":{"type":"string"},"replace":{"type":"string"}},"required":["path","find","replace"]}"#),
        run: { argsJSON in
            guard let p = stringArg(argsJSON, "path") else { return "Argomento 'path' mancante." }
            guard let f = stringArg(argsJSON, "find") else { return "Argomento 'find' mancante." }
            let r = stringArg(argsJSON, "replace") ?? ""
            return ProjectCache.shared.editTool(path: p, find: f, replace: r)
        })

    /// Read any file inside the project root (raw, not limited to the index),
    /// optionally a line range [from_line, to_line].
    static let fileRead = BuiltinTool(
        spec: ToolSpec(name: "file_read",
                       description: "Leggi un file QUALSIASI dentro la radice del progetto importato (anche non indicizzato). Senza from_line/to_line restituisce l'intero file (cap 96 KB); con from_line/to_line (1-based, inclusi) restituisce SOLO quelle righe, numerate.",
                       parametersJSON: #"{"type":"object","properties":{"path":{"type":"string","description":"percorso relativo alla radice"},"from_line":{"type":"number","description":"prima riga, 1-based (opzionale)"},"to_line":{"type":"number","description":"ultima riga inclusa, 1-based (opzionale)"}},"required":["path"]}"#),
        run: { argsJSON in
            guard let p = stringArg(argsJSON, "path") else { return "Argomento 'path' mancante." }
            return ProjectCache.shared.readFileTool(path: p,
                                                    fromLine: intArg(argsJSON, "from_line"),
                                                    toLine: intArg(argsJSON, "to_line"))
        })

    /// Create/overwrite the WHOLE file inside the project root.
    static let fileWrite = BuiltinTool(
        spec: ToolSpec(name: "file_write",
                       description: "Crea o sovrascrivi l'INTERO file dentro la radice del progetto importato (qualunque estensione; crea le cartelle). Per AGGIUNGERE righe usa file_add, per MODIFICARE righe esistenti usa file_modify.",
                       parametersJSON: #"{"type":"object","properties":{"path":{"type":"string","description":"percorso relativo alla radice"},"content":{"type":"string","description":"contenuto completo del file"}},"required":["path","content"]}"#),
        run: { argsJSON in
            guard let p = stringArg(argsJSON, "path") else { return "Argomento 'path' mancante." }
            guard let c = stringArg(argsJSON, "content") else { return "Argomento 'content' mancante." }
            return ProjectCache.shared.writeFileTool(path: p, content: c)
        })

    /// ADD lines (insert) without overwriting.
    static let fileAdd = BuiltinTool(
        spec: ToolSpec(name: "file_add",
                       description: "AGGIUNGI righe a un file (senza sovrascrivere): inserisce 'content' PRIMA della riga 'at_line' (1-based); senza 'at_line' accoda in fondo. Crea il file se non esiste.",
                       parametersJSON: #"{"type":"object","properties":{"path":{"type":"string","description":"percorso relativo alla radice"},"content":{"type":"string","description":"righe da inserire"},"at_line":{"type":"number","description":"inserisci prima di questa riga, 1-based (opzionale: in coda)"}},"required":["path","content"]}"#),
        run: { argsJSON in
            guard let p = stringArg(argsJSON, "path") else { return "Argomento 'path' mancante." }
            guard let c = stringArg(argsJSON, "content") else { return "Argomento 'content' mancante." }
            return ProjectCache.shared.addLinesTool(path: p, content: c, atLine: intArg(argsJSON, "at_line"))
        })

    /// MODIFY (replace) a line range.
    static let fileModify = BuiltinTool(
        spec: ToolSpec(name: "file_modify",
                       description: "MODIFICA un file sostituendo le righe [from_line, to_line] (1-based, incluse) con 'content' (to_line omesso = una sola riga; 'content' vuoto = cancella quelle righe). Il file deve esistere. Per sostituzioni su testo esatto preferisci project_edit.",
                       parametersJSON: #"{"type":"object","properties":{"path":{"type":"string","description":"percorso relativo alla radice"},"content":{"type":"string","description":"righe sostitutive (vuoto = cancella)"},"from_line":{"type":"number","description":"prima riga da sostituire, 1-based"},"to_line":{"type":"number","description":"ultima riga inclusa, 1-based (opzionale = from_line)"}},"required":["path","content","from_line"]}"#),
        run: { argsJSON in
            guard let p = stringArg(argsJSON, "path") else { return "Argomento 'path' mancante." }
            guard let c = stringArg(argsJSON, "content") else { return "Argomento 'content' mancante." }
            guard let f = intArg(argsJSON, "from_line") else { return "Argomento 'from_line' mancante." }
            return ProjectCache.shared.modifyLinesTool(path: p, content: c, fromLine: f, toLine: intArg(argsJSON, "to_line"))
        })

    static let git = BuiltinTool(
        spec: ToolSpec(name: "git",
                       description: "Run a LOCAL git subcommand in the imported project. Allowed: status, diff, log, show, branch, blame, grep, add, commit, stash, tag, rev-parse, ls-files. No push/pull/network. Example: {\"args\":\"diff --stat\"} or {\"args\":\"commit -am \\\"fix: ...\\\"\"}.",
                       parametersJSON: #"{"type":"object","properties":{"args":{"type":"string","description":"git subcommand and arguments"}},"required":["args"]}"#),
        run: { argsJSON in
            guard let a = stringArg(argsJSON, "args") else { return "Argomento 'args' mancante." }
            return GitTool.run(argsLine: a)
        })

    // MARK: Sub-agent tools (delegate a focused task to an isolated context)

    /// List the available agents (roles) and the tools each one has — so the
    /// orchestrator can pick the right minimal tool set to grant a sub-agent.
    static let agentsList = BuiltinTool(
        spec: ToolSpec(name: "agents_list",
                       description: "Elenca gli agenti (ruoli) disponibili e i tool che ciascuno ha a disposizione (id · nome · tool). Usalo per scegliere quali tool concedere a un sub-agent (parametro 'tools' di subagent_run) in base al ruolo adatto al compito.",
                       parametersJSON: #"{"type":"object","properties":{}}"#),
        run: { _ in AgentRegistry.shared.describe() })

    /// Find loadable sub-agent targets: project files whose name/content match.
    static let subagentSearch = BuiltinTool(
        spec: ToolSpec(name: "subagent_search",
                       description: "Cerca i target caricabili come sub-agent: file del progetto che corrispondono (per contenuto). Restituisce 'file:riga' da cui ricavare il percorso da passare a subagent_run.",
                       parametersJSON: #"{"type":"object","properties":{"query":{"type":"string"}},"required":["query"]}"#),
        run: { argsJSON in
            guard let q = stringArg(argsJSON, "query") else { return "Argomento 'query' mancante." }
            return ProjectCache.shared.searchTool(query: q)
        })

    /// Delegate a focused task to an isolated sub-agent. EXECUTED BY THE ENGINE
    /// (InferenceService.runSubAgent), which intercepts this call so the sub-agent
    /// runs in a separate context; this sentinel only applies if a non-engine path
    /// (HTTP server / distributed) emits the call, where sub-agents are unsupported.
    static let subagentRun = BuiltinTool(
        spec: ToolSpec(name: "subagent_run",
                       description: "Esegui un sub-agent ISOLATO su un TARGET (percorso file del progetto, oppure \"project\" per l'intero progetto) con una DOMANDA. Il sub-agent ha il contenuto già in contesto e restituisce SOLO la risposta. Con 'agent' (id da agents_list) il sub-agent assume quel RUOLO (system prompt + i suoi tool). In alternativa passa in 'tools' l'insieme MINIMO di tool. Precedenza: tools > tool del ruolo > sola lettura. Tool disponibili: project_list, project_read, project_search, project_edit, project_write, git.",
                       parametersJSON: #"{"type":"object","properties":{"target":{"type":"string","description":"percorso file relativo, oppure \"project\""},"question":{"type":"string","description":"compito o domanda per il sub-agent"},"agent":{"type":"string","description":"id di un agente (da agents_list): il sub-agent ne assume ruolo e tool. Opzionale."},"tools":{"type":"array","items":{"type":"string"},"description":"override opzionale: insieme MINIMO di tool concessi. Se assente usa i tool del ruolo 'agent', altrimenti sola lettura."}},"required":["target","question"]}"#),
        run: { _ in #"{"note":"subagent_run è gestito dall'engine (non disponibile in questo contesto)"}"# })

    static func stringArg(_ json: String, _ key: String) -> String? {
        guard let data = json.data(using: .utf8),
              let obj = try? JSONSerialization.jsonObject(with: data) as? [String: Any] else { return nil }
        return obj[key] as? String
    }

    static func intArg(_ json: String, _ key: String) -> Int? {
        guard let data = json.data(using: .utf8),
              let obj = try? JSONSerialization.jsonObject(with: data) as? [String: Any] else { return nil }
        if let n = obj[key] as? NSNumber { return n.intValue }
        if let s = obj[key] as? String { return Int(s) }
        return nil
    }

    // MARK: Two-operand arithmetic tools (a, b)

    /// Sum of two numbers.
    static let add = binaryTool(name: "add", verb: "Add", symbol: "+") { $0 + $1 }
    /// Difference of two numbers (a − b).
    static let subtract = binaryTool(name: "subtract", verb: "Subtract", symbol: "−") { $0 - $1 }
    /// Product of two numbers.
    static let multiply = binaryTool(name: "multiply", verb: "Multiply", symbol: "×") { $0 * $1 }

    /// Build a tool taking two numeric arguments `a` and `b` and returning `op(a,b)`.
    static func binaryTool(name: String, verb: String, symbol: String,
                           _ op: @escaping @Sendable (Double, Double) -> Double) -> BuiltinTool {
        let schema = #"{"type":"object","properties":{"a":{"type":"number"},"b":{"type":"number"}},"required":["a","b"]}"#
        return BuiltinTool(
            spec: ToolSpec(name: name, description: "\(verb) two numbers: a \(symbol) b.", parametersJSON: schema),
            run: { argsJSON in
                guard let (a, b) = parseTwoNumbers(argsJSON) else {
                    return #"{"error":"expected numeric arguments 'a' and 'b'"}"#
                }
                return formatNumberResult(op(a, b))
            })
    }

    /// Parse `a` and `b` from a JSON arguments object. Accepts JSON numbers or
    /// numeric strings (some models quote their arguments).
    static func parseTwoNumbers(_ argsJSON: String) -> (Double, Double)? {
        guard let data = argsJSON.data(using: .utf8),
              let obj = try? JSONSerialization.jsonObject(with: data) as? [String: Any],
              let a = number(obj["a"]), let b = number(obj["b"]) else { return nil }
        return (a, b)
    }

    private static func number(_ any: Any?) -> Double? {
        if let n = any as? NSNumber { return n.doubleValue }
        if let s = any as? String { return Double(s.trimmingCharacters(in: .whitespaces)) }
        return nil
    }

    /// Render a numeric result as JSON, printing whole numbers without a ".0".
    static func formatNumberResult(_ value: Double) -> String {
        guard value.isFinite else { return #"{"error":"non-finite result"}"# }
        let s = (value.rounded() == value && abs(value) < 1e15) ? String(Int64(value)) : String(value)
        return #"{"result":\#(s)}"#
    }

    /// Safe arithmetic evaluation via a small recursive-descent parser (no
    /// NSExpression, so a malformed input returns an error instead of throwing an
    /// uncatchable ObjC exception). Supports + - * / , unary minus, parentheses.
    static func evaluateArithmetic(_ expr: String) -> String {
        guard let value = ArithmeticEvaluator.evaluate(expr) else {
            return #"{"error":"could not evaluate expression"}"#
        }
        return formatNumberResult(value)
    }
}

/// Minimal, crash-free arithmetic evaluator: `expr := term (('+'|'-') term)*`,
/// `term := factor (('*'|'/') factor)*`, `factor := number | '(' expr ')' | '-' factor`.
/// Returns nil on any malformed input. Pure and side-effect free.
enum ArithmeticEvaluator {
    static func evaluate(_ s: String) -> Double? {
        var p = Parser(Array(s))
        guard let v = p.parseExpression(), p.atEnd else { return nil }
        return v
    }

    private struct Parser {
        let c: [Character]
        var i = 0
        init(_ c: [Character]) { self.c = c }

        var atEnd: Bool { mutating get { skipSpaces(); return i >= c.count } }
        mutating func skipSpaces() { while i < c.count, c[i] == " " || c[i] == "\t" { i += 1 } }
        mutating func peek() -> Character? { skipSpaces(); return i < c.count ? c[i] : nil }

        mutating func parseExpression() -> Double? {
            guard var acc = parseTerm() else { return nil }
            while let op = peek(), op == "+" || op == "-" {
                i += 1
                guard let rhs = parseTerm() else { return nil }
                acc = (op == "+") ? acc + rhs : acc - rhs
            }
            return acc
        }

        mutating func parseTerm() -> Double? {
            guard var acc = parseFactor() else { return nil }
            while let op = peek(), op == "*" || op == "/" {
                i += 1
                guard let rhs = parseFactor() else { return nil }
                if op == "/" { if rhs == 0 { return nil }; acc /= rhs } else { acc *= rhs }
            }
            return acc
        }

        mutating func parseFactor() -> Double? {
            guard let ch = peek() else { return nil }
            if ch == "-" { i += 1; guard let f = parseFactor() else { return nil }; return -f }
            if ch == "+" { i += 1; return parseFactor() }
            if ch == "(" {
                i += 1
                guard let v = parseExpression(), peek() == ")" else { return nil }
                i += 1
                return v
            }
            return parseNumber()
        }

        mutating func parseNumber() -> Double? {
            skipSpaces()
            let start = i
            while i < c.count, c[i].isNumber || c[i] == "." { i += 1 }
            guard i > start else { return nil }
            return Double(String(c[start..<i]))
        }
    }
}
