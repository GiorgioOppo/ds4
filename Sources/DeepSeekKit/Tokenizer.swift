import Foundation

/// Tokenizer interface. The actual implementation must read the BPE
/// merges and vocabulary from `tokenizer.json` shipped with the model
/// (HuggingFace `tokenizers` JSON format). DeepSeek uses a byte-level BPE
/// like GPT-2/Llama-3, with model-specific special tokens declared in
/// `added_tokens` / `pre_tokenizer`.
public protocol Tokenizer {
    func encode(_ text: String) -> [Int]
    func decode(_ ids: [Int]) -> String
    var bosId: Int? { get }
    var eosId: Int? { get }
}

/// Loader that parses HuggingFace `tokenizer.json`. Stub — only declares
/// the entry point, the BPE merge logic is not implemented here. The
/// expected approach is to port a byte-level BPE in pure Swift (no FFI),
/// keyed by the merges array, with byte-fallback for unknown bytes.
public enum TokenizerLoader {
    public static func load(from url: URL) throws -> Tokenizer {
        // Will parse `tokenizer.json` once the spec subset used by DeepSeek
        // is confirmed (model.type, model.vocab, model.merges, added_tokens,
        // pre_tokenizer.type, decoder.type).
        throw NSError(domain: "Tokenizer", code: -1,
                      userInfo: [NSLocalizedDescriptionKey:
                        "Tokenizer not implemented — see Sources/DeepSeekKit/Tokenizer.swift"])
    }
}
