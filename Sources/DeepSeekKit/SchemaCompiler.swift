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
        // type:"object" with `properties` whose values are themselves
        // reducible to finite string unions. We render the cartesian
        // product of (key, value) pairs as compact JSON, in
        // alphabetical key order — the model is constrained to emit
        // properties in that exact order. Free-form string values
        // are NOT supported (they'd blow up the enumeration); the
        // schema must be all-enum/const/oneOf at every property.
        //
        // For larger schemas (~5+ properties × ~5+ values each) the
        // product size grows fast — at construction we cap the
        // total enumerated outputs so the SchemaMask cache stays
        // tractable. Past the cap we throw `unsupported` and the
        // caller has to narrow the schema.
        if (schema["type"] as? String) == "object" {
            let props = (schema["properties"] as? [String: Any]) ?? [:]
            if props.isEmpty {
                // Empty object — single string output `{}`.
                if seen.insert("{}").inserted { out.append("{}") }
                return
            }
            let keys = props.keys.sorted()
            let propLimit = 8
            let valueLimit = 32
            let totalLimit = 4096
            guard keys.count <= propLimit else {
                throw SchemaError.unsupported(
                    "object with >\(propLimit) properties — cartesian "
                    + "enumeration would explode")
            }
            var perKeyAllowed: [[String]] = []
            for key in keys {
                guard let subSchema = props[key] as? [String: Any] else {
                    throw SchemaError.unsupported(
                        "properties.\(key) is not a JSON object")
                }
                let vals = try collectAllowedStrings(from: subSchema)
                guard !vals.isEmpty else {
                    throw SchemaError.unsupported(
                        "properties.\(key) admits no values")
                }
                guard vals.count <= valueLimit else {
                    throw SchemaError.unsupported(
                        "properties.\(key) has \(vals.count) values — "
                        + ">\(valueLimit) blows up the enumeration")
                }
                perKeyAllowed.append(vals)
            }
            // Estimate the product size before allocating.
            var product = 1
            for v in perKeyAllowed { product *= v.count }
            guard product <= totalLimit else {
                throw SchemaError.unsupported(
                    "object schema cartesian product = \(product) — "
                    + "exceeds the \(totalLimit) cap on enumerated outputs")
            }
            // Build the products. Compact JSON, alphabetical key
            // order, every value JSON-escaped.
            var rows: [String] = [""]
            for (i, key) in keys.enumerated() {
                let opener = i == 0 ? "{" : ","
                let prefix = "\(opener)\(jsonEncodeString(key)):"
                var expanded: [String] = []
                expanded.reserveCapacity(rows.count * perKeyAllowed[i].count)
                for cur in rows {
                    for val in perKeyAllowed[i] {
                        expanded.append(cur + prefix + jsonEncodeString(val))
                    }
                }
                rows = expanded
            }
            for s in rows {
                let closed = s + "}"
                if seen.insert(closed).inserted { out.append(closed) }
            }
            return
        }

        // Future: array / pattern / number ranges. Each needs its
        // own automaton; punt with a clear error so the caller
        // knows we haven't silently dropped the constraint on the
        // floor.
        throw SchemaError.unsupported(
            "JSON Schema feature not yet supported by SchemaCompiler "
            + "(this version handles enum / const / oneOf / anyOf / "
            + "type:object — see TODO §10.3 for array / pattern "
            + "follow-ups)")
    }

    /// JSON-escape a string and wrap it in double quotes. Matches
    /// `JSONSerialization`'s output for a single-string payload so
    /// downstream JSON parsers accept the enumerated outputs.
    private static func jsonEncodeString(_ s: String) -> String {
        let data = (try? JSONSerialization.data(
            withJSONObject: [s], options: [.withoutEscapingSlashes]))
            ?? Data("[\"\"]".utf8)
        let str = String(data: data, encoding: .utf8) ?? "[\"\"]"
        // `data` is `["..."]` — strip the surrounding brackets.
        let trimmed = str.dropFirst().dropLast()
        return String(trimmed)
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
