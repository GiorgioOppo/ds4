import Foundation

/// Tokenizer interface. Conforming type encodes UTF-8 strings to int token ids
/// and back. DeepSeek-V4 uses a byte-level BPE; the concrete implementation is
/// `BPETokenizer`.
public protocol Tokenizer {
    func encode(_ text: String) -> [Int]
    func decode(_ ids: [Int]) -> String
    var bosId: Int? { get }
    var eosId: Int? { get }
}

/// Loads a HuggingFace `tokenizer.json` file.
public enum TokenizerLoader {
    public static func load(from url: URL) throws -> Tokenizer {
        let data = try Data(contentsOf: url)
        return try BPETokenizer(jsonData: data)
    }
}
