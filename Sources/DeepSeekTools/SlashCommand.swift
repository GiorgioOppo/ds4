import Foundation

/// User-typed slash commands intercepted by the composer before the
/// message reaches the model. Built-ins are defined here; custom
/// commands are loaded from `~/Library/Application Support/.../commands/*.json`
/// (or a project-local `.deepseek/commands/`) by the host.
///
/// Each command has a `name` (the verb the user types after `/`), a
/// `summary` for the palette, and an `action` payload that the host
/// interprets. We keep `action` as a string + dictionary rather than
/// a closure so commands stay Codable / serialisable.
public struct SlashCommand: Codable, Sendable, Identifiable, Hashable {
    public var id: String { name }
    public var name: String
    public var summary: String
    public var action: String
    public var arguments: [String: String]

    public init(name: String,
                summary: String,
                action: String,
                arguments: [String: String] = [:]) {
        self.name = name
        self.summary = summary
        self.action = action
        self.arguments = arguments
    }
}

/// Parsed result of a `/word arg arg` line. `name` is everything
/// between the leading slash and the first space; `rest` is the raw
/// remainder (no shell-quoting — commands that need structured args
/// should parse `rest` themselves).
public struct ParsedSlashCommand: Sendable, Hashable {
    public let name: String
    public let rest: String

    public init(name: String, rest: String) {
        self.name = name
        self.rest = rest
    }

    /// Returns `nil` for messages that don't start with `/`. We treat
    /// a single `/` followed by whitespace as a non-command too —
    /// the user probably typed prose starting with `/`.
    public static func parse(_ message: String) -> ParsedSlashCommand? {
        let trimmed = message.trimmingCharacters(in: .whitespacesAndNewlines)
        guard trimmed.hasPrefix("/") else { return nil }
        let body = String(trimmed.dropFirst())
        guard let first = body.first, !first.isWhitespace else { return nil }
        if let space = body.firstIndex(of: " ") {
            return ParsedSlashCommand(
                name: String(body[..<space]).lowercased(),
                rest: String(body[body.index(after: space)...])
                    .trimmingCharacters(in: .whitespaces))
        }
        return ParsedSlashCommand(name: body.lowercased(), rest: "")
    }
}

/// Built-in slash commands. The host wires each `action` to its own
/// imperative effect (clear conversation, swap mode, etc.). Adding
/// a new built-in here means also wiring its action in the host.
public enum BuiltInSlashCommands {
    public static let all: [SlashCommand] = [
        .init(name: "help", summary: "Mostra i comandi slash disponibili.", action: "help"),
        .init(name: "clear", summary: "Pulisce la conversazione corrente.", action: "clear"),
        .init(name: "new", summary: "Avvia una nuova conversazione.", action: "new"),
        .init(name: "mode",
              summary: "Cambia modalità agente: 'plan' o 'build'.",
              action: "mode"),
        .init(name: "model",
              summary: "Mostra o cambia il modello attivo.",
              action: "model"),
        .init(name: "agent",
              summary: "Collega un agente per nome (senza arg → lista).",
              action: "agent"),
        .init(name: "skill",
              summary: "Attiva una skill per il prossimo turno.",
              action: "skill"),
        .init(name: "tools",
              summary: "Elenca gli strumenti disponibili all'agente corrente.",
              action: "tools"),
        .init(name: "permissions",
              summary: "Mostra / azzera la cache dei permessi della sessione.",
              action: "permissions"),
        .init(name: "theme",
              summary: "Cambia tema dell'app.",
              action: "theme"),
        .init(name: "compact",
              summary: "Chiede al modello di compattare il contesto della conversazione.",
              action: "compact"),
    ]
}
