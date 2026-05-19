import Foundation

/// JSON Schema → `SchemaMask` compiler (TODO §10.3 / T3). Reads a
/// JSON Schema dictionary (as produced by `JSONSerialization`,
/// matching what OpenAI's `response_format: {type:"json_schema",
/// json_schema:{schema: …}}` carries) and returns a `SchemaMask`
/// that constrains the sampler to outputs valid under that schema.
///
/// Coverage today:
///   - `{"enum": ["A", "B", …]}` — literal string union. Closes
///     the user-facing "give me exactly one of these tokens" case
///     that most non-trivial structured-output workflows actually
///     need.
///
/// Out of scope (deferred to T3 follow-up commits):
///   - `type: "object"` with `properties` / `required` /
///     `additionalProperties`. Needs a recursive
///     value-state pushdown automaton.
///   - `type: "array"` with `items`. Same.
///   - `oneOf` / `anyOf`. Union of N alternatives — extendable
///     from the enum scaffold by tracking which alternatives are
///     still viable as bytes get consumed.
///   - `type: "string"` with `pattern` (regex). Requires a
///     character-class subset of the regex engine.
///
/// When the input schema isn't yet supported we throw
/// `SchemaError.unsupported` so callers see the gap rather than
/// silently getting an unconstrained sample.
public enum SchemaCompiler {
    /// Build a mask from a JSON-Schema dictionary. `tokenizer` and
    /// `vocabSize` are passed through to the resulting `SchemaMask`
    /// for its per-id string cache.
    public static func compile(
        schema: [String: Any],
        tokenizer: any Tokenizer,
        vocabSize: Int) throws -> SchemaMask
    {
        // Enum of literal strings.
        if let enumValues = schema["enum"] as? [Any] {
            let strings = enumValues.compactMap { $0 as? String }
            guard strings.count == enumValues.count else {
                throw SchemaError.unsupported(
                    "enum with non-string members (T3 covers strings only)")
            }
            guard !strings.isEmpty else {
                throw SchemaError.unsupported("empty enum")
            }
            return SchemaMask(tokenizer: tokenizer,
                               allowed: strings,
                               vocabSize: vocabSize)
        }

        // Future: type / properties / items / oneOf / anyOf /
        // pattern. Each maps to a different automaton; punt with a
        // clear error so callers know we haven't silently dropped
        // their constraint on the floor.
        throw SchemaError.unsupported(
            "JSON Schema feature not yet supported by SchemaCompiler "
            + "(this version handles `enum` only — see TODO §10.3 for "
            + "object / array / pattern follow-ups)")
    }

    /// Convenience: parse a JSON Schema from a raw `Data` (the
    /// shape passed in by `response_format` or the `--json-schema`
    /// CLI flag) and compile it.
    public static func compile(
        jsonData: Data,
        tokenizer: any Tokenizer,
        vocabSize: Int) throws -> SchemaMask
    {
        guard let obj = try JSONSerialization.jsonObject(with: jsonData)
                as? [String: Any]
        else {
            throw SchemaError.malformed("JSON Schema must be a JSON object")
        }
        return try compile(schema: obj,
                            tokenizer: tokenizer,
                            vocabSize: vocabSize)
    }
}

public enum SchemaError: LocalizedError {
    case unsupported(String)
    case malformed(String)

    public var errorDescription: String? {
        switch self {
        case .unsupported(let s): return "Schema unsupported: \(s)"
        case .malformed(let s):   return "Schema malformed: \(s)"
        }
    }
}
