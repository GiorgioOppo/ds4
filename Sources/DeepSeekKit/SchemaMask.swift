import Foundation

/// Token-level constraint mask for `Sampling.sample(...)` (TODO §10.3
/// / T3). Built by `SchemaCompiler` and dropped into
/// `SamplingOptions.schemaMask`; the sampler then applies it as
/// stage 0 of the pipeline, zeroing out (`-INF`) the logits of any
/// token whose bytes would steer the running output off the schema's
/// accepted language.
///
/// First-cut scope: enum constraints (a fixed list of literal
/// strings — OpenAI's `response_format: {type:"json_schema",
/// json_schema:{schema:{enum:["RED","GREEN","BLUE"]}}}` case). Once
/// the running text matches one of the enum strings exactly, only
/// the tokenizer's stop tokens stay allowed so the model can't
/// keep emitting text past the constrained value.
///
/// Out of scope (T3 follow-up): nested `type:object` with
/// `properties`, `type:array` with `items`, `oneOf` / `anyOf`,
/// regex `pattern`. Each needs its own automaton; this class is
/// designed to grow into a generic NFA but starts narrow on purpose.
///
/// Performance: at construction we decode every token in vocab once
/// (`vocabSize` decode calls — ~100 ms on a 50k vocab) and cache the
/// `id → String` map. `allowedTokens()` then iterates the cache
/// once per sample (~100 µs for prefix matching). For long
/// generations a trie-keyed prefix index would shave that, but
/// constrained decoding is bounded by the enum length anyway.
public final class SchemaMask {
    /// The literal strings the output is allowed to be (one of).
    public let allowedStrings: [String]

    /// Token ids that terminate the generation. Returned as the
    /// only allowed set once an `allowedStrings` value has been
    /// fully emitted.
    public let stopTokenIDs: Set<Int32>

    /// Decoded string per token id (precomputed at init for fast
    /// per-sample masking).
    private let tokenStrings: [String]

    /// What the model has emitted under this mask so far. Updated
    /// by `advance(token:)`. Reset by `reset()`.
    private(set) var consumed: String = ""

    /// `tokenizer` is used once at init to populate the per-id
    /// string cache. `allowed` must be non-empty — an empty list
    /// would block every token. `vocabSize` is the model's logit
    /// vocabulary size; the precomputed cache is exactly that
    /// length so `allowedTokens()` can iterate by index.
    public init(tokenizer: any Tokenizer,
                allowed: [String],
                vocabSize: Int)
    {
        precondition(!allowed.isEmpty,
                      "SchemaMask: empty allowed list would block every token")
        precondition(vocabSize > 0)
        self.allowedStrings = allowed
        self.stopTokenIDs = Set(tokenizer.stopTokenIds.map(Int32.init))
        var strings = [String](repeating: "", count: vocabSize)
        for i in 0..<vocabSize {
            strings[i] = tokenizer.decode([i])
        }
        self.tokenStrings = strings
    }

    /// Reset the consumed-prefix state so the mask can be reused
    /// across independent generations. Cheap.
    public func reset() {
        consumed = ""
    }

    /// Set of token ids the sampler is allowed to pick this step.
    /// When the consumed text already equals one of the
    /// `allowedStrings`, returns just `stopTokenIDs` — the model
    /// must halt. Otherwise returns every token whose decoded
    /// string would keep `consumed` as a prefix of at least one
    /// allowed string.
    public func allowedTokens() -> Set<Int32> {
        if allowedStrings.contains(consumed) {
            return stopTokenIDs
        }
        var result = Set<Int32>()
        for (i, str) in tokenStrings.enumerated() {
            // Empty-decode tokens (unused vocab slots in some BPEs)
            // would otherwise look like a no-op extension that
            // never makes progress. Block them so the model can't
            // stall.
            if str.isEmpty { continue }
            let next = consumed + str
            for allowed in allowedStrings {
                if allowed.hasPrefix(next) {
                    result.insert(Int32(i))
                    break
                }
            }
        }
        return result
    }

    /// Update the consumed text with the just-sampled token. The
    /// sampler calls this after picking the next id. Stop tokens
    /// are not appended to `consumed` (their decode is typically
    /// empty / a control sequence the schema doesn't care about).
    public func advance(token: Int32) {
        if stopTokenIDs.contains(token) { return }
        let idx = Int(token)
        guard idx >= 0 && idx < tokenStrings.count else { return }
        consumed += tokenStrings[idx]
    }
}
