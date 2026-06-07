import Foundation
import DS4Core

// Native replacement for `ds4 --dump-tokens`: opens the GGUF's tokenizer and
// tokenizes a string exactly as written (recognizing DS4 protocol specials),
// using the pure-Swift DS4Core.Tokenizer (validated bit-for-bit against ./ds4
// in TokenizerTests). No subprocess.

public enum Diagnostics {
    /// Tokenize `text` with the model's tokenizer and return a readable dump
    /// (one "id<TAB>text" line per token, preceded by the token count).
    public static func dumpTokens(modelPath: String, text: String) throws -> String {
        let model = try GGUFModel(path: modelPath, metalMapping: false, prefetchCPU: false)
        let tok = try Tokenizer(model: model)
        let ids = tok.tokenizeRenderedChat(text)
        var out = "\(ids.count) token\n\n"
        for id in ids {
            let bytes = tok.tokenText(id)
            let s = String(bytes: bytes, encoding: .utf8)?
                .replacingOccurrences(of: "\n", with: "\\n") ?? "<\(bytes.count) byte>"
            out += "\(id)\t\(s)\n"
        }
        return out
    }
}
