import Foundation

/// Tokenizer interface. Conforming type encodes UTF-8 strings to int token ids
/// and back. DeepSeek-V4 uses a byte-level BPE; the concrete implementation is
/// `BPETokenizer`.
public protocol Tokenizer {
    func encode(_ text: String) -> [Int]
    func decode(_ ids: [Int]) -> String
    var bosId: Int? { get }
    var eosId: Int? { get }
    /// All token ids whose sampling should terminate the generation loop.
    /// Always includes `eosId` (end-of-sentence = end-of-conversation) and
    /// `<|EOT|>` (end-of-turn) when present in the vocab. The decode loop
    /// must check this set, not just `eosId`, otherwise V4 chat
    /// generations run past the assistant's end-of-turn marker and the
    /// model — sampling whatever the LM head still ranks highest after
    /// EOT was already emitted — collapses into looped filler tokens.
    var stopTokenIds: Set<Int> { get }
}

/// Loads a HuggingFace `tokenizer.json` file.
public enum TokenizerLoader {
    public static func load(from url: URL) throws -> Tokenizer {
        let data = try Data(contentsOf: url)
        return try BPETokenizer(jsonData: data)
    }
}
