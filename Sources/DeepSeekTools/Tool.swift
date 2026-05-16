import Foundation

/// Result of one tool invocation. The model only sees `output` and
/// the `isError` flag; the `metadata` field is for the UI (e.g. so
/// `edit` can show a diff disclosure or `shell` can render stderr in
/// red without re-parsing the text).
public struct ToolOutput: Sendable {
    public let output: String
    public let isError: Bool
    /// Stable, JSON-encodable side channel for the UI. Avoid
    /// `[String: Any]` to keep this Sendable across actor hops.
    public let metadata: [String: String]

    public init(output: String,
                isError: Bool = false,
                metadata: [String: String] = [:]) {
        self.output = output
        self.isError = isError
        self.metadata = metadata
    }

    public static func error(_ err: ToolError) -> ToolOutput {
        ToolOutput(output: err.wireMessage, isError: true)
    }
}

/// One callable tool. Implementations must be `Sendable` because the
/// registry can be queried from any actor (the inference loop hops
/// between the GPU executor and the main actor).
public protocol Tool: Sendable {
    /// Static description handed to the model. Same instance is
    /// served to every chat.
    var schema: ToolSchema { get }

    /// Execute the tool. `input` is the JSON object the model emitted
    /// as the function's arguments — already decoded into a Foundation
    /// graph. Implementations should validate aggressively and return
    /// a `ToolError`-wrapped failure for any structural problem
    /// rather than throwing fatally.
    func run(input: [String: Any], context: ToolContext) async throws -> ToolOutput

    /// One-line human-readable summary the permission UI shows.
    /// Default impl falls back to the schema name; override when a
    /// tool can describe what it's about to do more concretely
    /// (e.g. `edit foo.swift`).
    func permissionSummary(input: [String: Any]) -> String
}

extension Tool {
    public func permissionSummary(input: [String: Any]) -> String {
        schema.name
    }
}

// MARK: - Input helpers

/// Small typed accessors over the untyped argument bag. Keeps each
/// tool's `run` body terse — `try input.string("path")` instead of a
/// six-line cast pyramid.
extension Dictionary where Key == String, Value == Any {
    public func string(_ key: String) throws -> String {
        guard let raw = self[key] else {
            throw ToolError.invalidInput("missing field '\(key)'")
        }
        guard let s = raw as? String else {
            throw ToolError.invalidInput("'\(key)' must be a string")
        }
        return s
    }

    public func optionalString(_ key: String) -> String? {
        self[key] as? String
    }

    public func integer(_ key: String) throws -> Int {
        guard let raw = self[key] else {
            throw ToolError.invalidInput("missing field '\(key)'")
        }
        if let i = raw as? Int { return i }
        if let n = raw as? NSNumber { return n.intValue }
        throw ToolError.invalidInput("'\(key)' must be an integer")
    }

    public func optionalInteger(_ key: String) -> Int? {
        if let i = self[key] as? Int { return i }
        if let n = self[key] as? NSNumber { return n.intValue }
        return nil
    }

    public func optionalBool(_ key: String) -> Bool? {
        if let b = self[key] as? Bool { return b }
        if let n = self[key] as? NSNumber { return n.boolValue }
        return nil
    }

    public func optionalStringArray(_ key: String) -> [String]? {
        self[key] as? [String]
    }
}
