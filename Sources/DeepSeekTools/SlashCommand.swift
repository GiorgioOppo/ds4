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
        .init(name: "help", summary: "Show available slash commands.", action: "help"),
        .init(name: "clear", summary: "Clear the current conversation.", action: "clear"),
        .init(name: "new", summary: "Start a new conversation.", action: "new"),
        .init(name: "mode",
              summary: "Switch agent mode: 'plan' or 'build'.",
              action: "mode"),
        .init(name: "model",
              summary: "Show or change the active model.",
              action: "model"),
        .init(name: "agent",
              summary: "Attach an agent by name (no arg → list).",
              action: "agent"),
        .init(name: "skill",
              summary: "Activate a skill for the next turn.",
              action: "skill"),
        .init(name: "tools",
              summary: "List the tools available to the current agent.",
              action: "tools"),
        .init(name: "permissions",
              summary: "Show / reset the per-session permission cache.",
              action: "permissions"),
        .init(name: "theme",
              summary: "Switch app theme.",
              action: "theme"),
        .init(name: "compact",
              summary: "Ask the model to compact the conversation context.",
              action: "compact"),
    ]
}
