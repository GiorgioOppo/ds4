import Foundation

/// JSON Schema → `SchemaMask` compiler (TODO §10.3 / T3). Reads a
/// JSON Schema dictionary (as produced by `JSONSerialization`,
/// matching what OpenAI's `response_format: {type:"json_schema",
/// json_schema:{schema: …}}` carries) and returns a `SchemaMask`
/// that constrains the sampler to outputs valid under that schema.
///
/// Coverage today (all reduce to a finite union of literal strings,
/// which is the only model `SchemaMask` enforces — see that file
/// for the constraint semantics):
///   - `{"enum": [...]}` — literal string union.
///   - `{"const": "X"}` — single literal.
///   - `{"oneOf": [...]}` / `{"anyOf": [...]}` — union over the
///     compiled inner schemas. Recursive: `oneOf` of `enum`s
///     flattens, `oneOf` of `const`s flattens, mixed works too.
///
/// Out of scope (deferred to T3 follow-up commits):
///   - `type: "object"` with `properties` — needs a JSON
///     pushdown automaton (open-brace, key, colon, value, comma,
///     close-brace) where each value can recursively be another
///     schema.
///   - `type: "array"` with `items` — same machinery.
///   - `type: "string"` with `pattern` — regex subset.
///
/// When the input schema isn't reducible to a literal union we
/// throw `SchemaError.unsupported` so callers see the gap rather
/// than silently getting an unconstrained sample.
public enum SchemaCompiler {
    /// Build a mask from a JSON-Schema dictionary. `tokenizer` and
    /// `vocabSize` are passed through to the resulting `SchemaMask`
    /// for its per-id string cache.
    public static func compile(
        schema: [String: Any],
        tokenizer: any Tokenizer,
        vocabSize: Int) throws -> SchemaMask
    {
        let allowed = try collectAllowedStrings(from: schema)
        guard !allowed.isEmpty else {
            throw SchemaError.unsupported("schema admits no string values")
        }
        return SchemaMask(tokenizer: tokenizer,
                           allowed: allowed,
                           vocabSize: vocabSize)
    }

    /// Walk the schema collecting the literal-string union it
    /// describes. Throws if the schema contains a construct
    /// (`type:"object"`, `pattern`, …) we can't reduce to a finite
    /// set today. De-duplicates so `oneOf: [{enum:["A"]},
    /// {const:"A"}]` doesn't double-count.
    static func collectAllowedStrings(
        from schema: [String: Any]) throws -> [String]
    {
        var out: [String] = []
        var seen: Set<String> = []
        try appendAllowedStrings(from: schema, out: &out, seen: &seen)
        return out
    }

    private static func appendAllowedStrings(
        from schema: [String: Any],
        out: inout [String],
        seen: inout Set<String>) throws
    {
        // enum: list of literal strings.
        if let enumValues = schema["enum"] as? [Any] {
            let strings = enumValues.compactMap { $0 as? String }
            guard strings.count == enumValues.count else {
                throw SchemaError.unsupported(
                    "enum with non-string members "
                    + "(this version handles string unions only)")
            }
            for s in strings where seen.insert(s).inserted {
                out.append(s)
            }
            return
        }
        // const: single literal.
        if let constValue = schema["const"] {
            guard let s = constValue as? String else {
                throw SchemaError.unsupported(
                    "const with non-string value")
            }
            if seen.insert(s).inserted { out.append(s) }
            return
        }
        // oneOf / anyOf: recurse and union. anyOf is treated like
        // oneOf here because the constraint cares only about the
        // accepted *language*; OpenAI Structured Outputs additionally
        // require exclusivity at decode time, which we don't model.
        if let alternatives = (schema["oneOf"] as? [Any])
            ?? (schema["anyOf"] as? [Any])
        {
            guard !alternatives.isEmpty else {
                throw SchemaError.unsupported("empty oneOf/anyOf")
            }
            for alt in alternatives {
                guard let altDict = alt as? [String: Any] else {
                    throw SchemaError.unsupported(
                        "oneOf/anyOf member is not a JSON object")
                }
                try appendAllowedStrings(
                    from: altDict, out: &out, seen: &seen)
            }
            return
        }

        // Future: object / array / pattern / number ranges. Each
        // needs its own automaton; punt with a clear error so the
        // caller knows we haven't silently dropped the constraint
        // on the floor.
        throw SchemaError.unsupported(
            "JSON Schema feature not yet supported by SchemaCompiler "
            + "(this version handles enum / const / oneOf / anyOf — "
            + "see TODO §10.3 for object / array / pattern follow-ups)")
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
