import Foundation

/// JSON-schema fragment describing one tool's input. The shape
/// matches the `inputSchema` field that MCP and OpenAI's `tools`
/// array expect, so the same schema can be forwarded verbatim to a
/// remote provider or embedded into a local DSML tools block.
///
/// We use an untyped `[String: Any]` rather than a Codable graph so
/// callers can pass through provider-specific extensions (e.g.
/// `enum`, `format`, `examples`) without having to widen this enum
/// every time. The structural validity is the responsibility of the
/// tool author.
public struct ToolSchema: Sendable {
    public let name: String
    public let description: String
    public let category: ToolCategory
    public let inputSchema: [String: Any]

    public init(name: String,
                description: String,
                category: ToolCategory,
                inputSchema: [String: Any]) {
        self.name = name
        self.description = description
        self.category = category
        self.inputSchema = inputSchema
    }
}

/// Builder helpers so each tool's `static let schema` line stays
/// readable. We don't try to model the whole of JSON Schema here —
/// just the few shapes every tool reuses.
public enum SchemaBuilder {
    public static func object(properties: [String: [String: Any]],
                              required: [String] = []) -> [String: Any] {
        var schema: [String: Any] = [
            "type": "object",
            "properties": properties,
        ]
        if !required.isEmpty { schema["required"] = required }
        return schema
    }

    public static func string(description: String,
                              enumValues: [String]? = nil) -> [String: Any] {
        var s: [String: Any] = ["type": "string", "description": description]
        if let enumValues { s["enum"] = enumValues }
        return s
    }

    public static func integer(description: String,
                               minimum: Int? = nil) -> [String: Any] {
        var s: [String: Any] = ["type": "integer", "description": description]
        if let minimum { s["minimum"] = minimum }
        return s
    }

    public static func boolean(description: String,
                               defaultValue: Bool? = nil) -> [String: Any] {
        var s: [String: Any] = ["type": "boolean", "description": description]
        if let defaultValue { s["default"] = defaultValue }
        return s
    }

    public static func array(itemsType: String,
                             description: String) -> [String: Any] {
        return [
            "type": "array",
            "description": description,
            "items": ["type": itemsType],
        ]
    }
}
