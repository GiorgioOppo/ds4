import Foundation

/// Tokenizer interface for DeepSeek-V4.
///
/// The reference uses `transformers.AutoTokenizer.from_pretrained(ckpt_path)`,
/// which loads `tokenizer.json` (the HuggingFace `tokenizers` Rust-backed
/// format). Special tokens (from `tokenizer_config.json`):
///   bos:  `<｜begin▁of▁sentence｜>`   (id 0)
///   eos:  `<｜end▁of▁sentence｜>`     (id 1)
///
/// `add_bos_token` and `add_eos_token` are both false.
///
/// Chat encoding is NOT done by the tokenizer — it's a separate Python
/// pipeline in `Reference/encoding/encoding_dsv4.py` (744
/// lines) that maps OpenAI-style messages into a single string before the
/// tokenizer runs. Porting that is its own subproject; for now the CLI
/// expects an already-formatted prompt string.
public protocol Tokenizer {
    func encode(_ text: String) -> [Int]
    func decode(_ ids: [Int]) -> String
    var bosId: Int? { get }
    var eosId: Int? { get }
}

public enum TokenizerLoader {
    public static func load(from url: URL) throws -> Tokenizer {
        // Will parse tokenizer.json. Need byte-level BPE with merges array,
        // pre-tokenizer regex (DeepSeek uses GPT-2-style), and added_tokens
        // for the DeepSeek special tokens above. Pure-Swift port, no FFI.
        throw NSError(domain: "Tokenizer", code: -1, userInfo: [NSLocalizedDescriptionKey:
            "Tokenizer not implemented — port target: tokenizer.json + encoding_dsv4.py"])
    }
}
