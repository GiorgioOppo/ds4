import CDS4

/// RAII wrapper around the C `ds4_tokens` value (a heap-backed int vector).
/// The engine's render helpers push into it; `deinit` frees the backing buffer.
final class Tokens {
    var raw = ds4_tokens()

    init() {}

    deinit {
        ds4_tokens_free(&raw)
    }

    var count: Int { Int(raw.len) }
}
