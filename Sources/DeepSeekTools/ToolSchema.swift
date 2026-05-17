import Foundation

/// JSON-schema fragment describing one tool's input. The shape
/// matches the `inputSchema` field that MCP and OpenAI's `tools`
/// array expect, so the same schema can be forwarded verbatim to a
/// remote provider or embedded into a local DSML tools block.
///
/// La forma è tipizzata su `JSONValue` (Sendable + Codable). Resta
/// strutturalmente estendibile: i provider possono aggiungere
/// extension (`enum`, `format`, `examples`, `x-*`) come nested
/// `.object(...)` / `.array(...)` senza dover toccare questo
/// modulo. La validità strutturale è responsabilità dell'autore
/// del tool.
///
/// In API: `inputSchema` viene sempre passato come `.object(...)`.
/// L'accessor `inputSchemaObject` fornisce la mappa interna per
/// callers che si aspettano la forma dict.
public struct ToolSchema: Sendable {
    public let name: String
    public let description: String
    public let category: ToolCategory
    public let inputSchema: JSONValue

    public init(name: String,
                description: String,
                category: ToolCategory,
                inputSchema: JSONValue) {
        self.name = name
        self.description = description
        self.category = category
        self.inputSchema = inputSchema
    }

    /// Convenience: l'inputSchema visto come `[String: JSONValue]`
    /// quando è un oggetto, `nil` altrimenti. Usato dai serializer
    /// che si aspettano la forma dict-of-fields.
    public var inputSchemaObject: [String: JSONValue]? {
        inputSchema.asObject
    }
}

/// Builder helpers so each tool's `static let schema` line stays
/// readable. We don't try to model the whole of JSON Schema here —
/// just the few shapes every tool reuses.
public enum SchemaBuilder {
    public static func object(properties: [String: JSONValue],
                              required: [String] = []) -> JSONValue {
        var fields: [String: JSONValue] = [
            "type": .string("object"),
            "properties": .object(properties),
        ]
        if !required.isEmpty {
            fields["required"] = .array(required.map { .string($0) })
        }
        return .object(fields)
    }

    public static func string(description: String,
                              enumValues: [String]? = nil) -> JSONValue {
        var fields: [String: JSONValue] = [
            "type": .string("string"),
            "description": .string(description),
        ]
        if let enumValues {
            fields["enum"] = .array(enumValues.map { .string($0) })
        }
        return .object(fields)
    }

    public static func integer(description: String,
                               minimum: Int? = nil) -> JSONValue {
        var fields: [String: JSONValue] = [
            "type": .string("integer"),
            "description": .string(description),
        ]
        if let minimum {
            fields["minimum"] = .int(minimum)
        }
        return .object(fields)
    }

    public static func boolean(description: String,
                               defaultValue: Bool? = nil) -> JSONValue {
        var fields: [String: JSONValue] = [
            "type": .string("boolean"),
            "description": .string(description),
        ]
        if let defaultValue {
            fields["default"] = .bool(defaultValue)
        }
        return .object(fields)
    }

    public static func array(itemsType: String,
                             description: String) -> JSONValue {
        return .object([
            "type": .string("array"),
            "description": .string(description),
            "items": .object(["type": .string(itemsType)]),
        ])
    }
}
