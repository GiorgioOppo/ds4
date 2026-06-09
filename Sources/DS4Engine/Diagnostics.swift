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

    /// Dump everything needed to implement the model's real tool format: the raw
    /// `tokenizer.chat_template` (the authoritative wire format), whether each
    /// guessed tool-markup token actually exists in the vocab, and the prompt we
    /// currently render with tools (so the two can be compared).
    public static func dumpChatTemplate(modelPath: String) throws -> String {
        let model = try GGUFModel(path: modelPath, metalMapping: false, prefetchCPU: false)
        let tok = try Tokenizer(model: model)

        var out = "=== tokenizer.chat_template ===\n"
        if let tpl = model.string("tokenizer.chat_template"), !tpl.isEmpty {
            out += tpl + "\n"
        } else {
            out += "<assente nel GGUF>\n"
        }

        let markup = ToolMarkup.discover(in: tok)
        out += "\n=== Token speciali del protocollo (presenti nel vocab?) ===\n"
        let specials = ["｜DSML｜", "<｜begin▁of▁sentence｜>", "<｜end▁of▁sentence｜>",
                        "<｜User｜>", "<｜Assistant｜>", "<think>", "</think>",
                        "<｜action｜>", "<｜title｜>", "<｜query｜>", "<｜authority｜>",
                        "<｜domain｜>", "<｜extracted_url｜>", "<｜read_url｜>"]
        for s in specials {
            out += "\(s) -> \(tok.tokenId(s).map { "id \($0)" } ?? "ASSENTE")\n"
        }
        out += "\nDSML markup usato: \(markup.dsml)  (tag es. \(markup.callsOpen))\n"

        out += "\n=== Prompt che il GUI invia ORA con i tool abilitati ===\n"
        let tools = ToolRegistry.specs(enabled: Set(ToolRegistry.builtins.map { $0.spec.name }))
        out += ChatRenderer.render(turns: [.user("Ciao, come stai?")], tools: tools,
                                   think: .none, markup: markup)
        return out
    }
}
