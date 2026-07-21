import Foundation
import DS4Core

// Native replacement for `ds4 --dump-tokens`: opens the GGUF's tokenizer and
// tokenizes a string exactly as written (recognizing the backend's protocol
// specials). Architecture-aware: the GGUF's `general.architecture` selects the
// DeepSeek or GLM tokenizer/renderer, so a GLM file never hits the DeepSeek
// validator and its misleading `deepseek4.*` errors. No subprocess.

public enum Diagnostics {
    /// Tokenize `text` with the model's tokenizer and return a readable dump
    /// (one "id<TAB>text" line per token, preceded by the token count).
    public static func dumpTokens(modelPath: String, text: String) throws -> String {
        let model = try GGUFModel(path: modelPath, metalMapping: false, prefetchCPU: false)
        let ids: [Int32]
        let textOf: (Int32) -> [UInt8]
        if try ModelArchitectureDetector.detect(in: model).family == .glm {
            let tok = try GLM52Tokenizer(model: model)
            ids = tok.tokenizeRenderedChat(text)
            textOf = tok.tokenText
        } else {
            let tok = try Tokenizer(model: model)
            ids = tok.tokenizeRenderedChat(text)
            textOf = tok.tokenText
        }
        var out = "\(ids.count) token\n\n"
        for id in ids {
            let bytes = textOf(id)
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
        if try ModelArchitectureDetector.detect(in: model).family == .glm {
            return try dumpGLMChatTemplate(model: model)
        }
        let tok = try Tokenizer(model: model)

        var out = "=== tokenizer.chat_template ===\n"
        if let tpl = model.string("tokenizer.chat_template"), !tpl.isEmpty {
            out += tpl + "\n"
        } else {
            out += "<assente nel GGUF>\n"
        }

        let markup = ToolMarkup.discover(in: tok)
        out += "\n=== Token speciali del protocollo (vocab, tipo, atomico?) ===\n"
        out += "Legenda tipo: 1=NORMAL 2=UNKNOWN 3=CONTROL 4=USER_DEFINED 6=BYTE.\n"
        out += "'atomico' = la stringa tokenizza in un solo id (necessario per i tag DSML).\n"
        let types = model.intArray("tokenizer.ggml.token_type")
        let specials = ["｜DSML｜", "<｜begin▁of▁sentence｜>", "<｜end▁of▁sentence｜>",
                        "<｜User｜>", "<｜Assistant｜>", "<think>", "</think>",
                        "<｜action｜>", "<｜title｜>", "<｜query｜>", "<｜authority｜>",
                        "<｜domain｜>", "<｜extracted_url｜>", "<｜read_url｜>"]
        for s in specials {
            guard let id = tok.tokenId(s) else { out += "\(s) -> ASSENTE\n"; continue }
            let type = types.flatMap { Int(id) < $0.count ? $0[Int(id)] : nil }
            let typeStr = type.map { "tipo \($0)" } ?? "tipo ?"
            let atomic = tok.tokenizeRenderedChat(s) == [id]
            out += "\(s) -> id \(id), \(typeStr), \(atomic ? "atomico ✓" : "SPEZZATO ✗")\n"
        }
        out += "\nDSML markup usato: \(markup.dsml)  (tag es. \(markup.callsOpen))\n"

        let tools = ToolRegistry.specs(enabled: Set(ToolRegistry.builtins.map { $0.spec.name }))
        out += "\n=== Prompt con tool — formato COMPLETO (template) ===\n"
        out += ChatRenderer.render(turns: [.user("Ciao, come stai?")], tools: tools,
                                   think: .none, markup: markup, compactTools: false)
        out += "\n\n=== Prompt con tool — formato COMPATTO (default GUI) ===\n"
        out += ChatRenderer.render(turns: [.user("Ciao, come stai?")], tools: tools,
                                   think: .none, markup: markup, compactTools: true)
        return out
    }

    /// GLM counterpart: same sections (raw chat template, protocol-token
    /// vocab check, rendered tool prompt) over the native GLM control tokens
    /// and the `GLM52ChatRenderer` XML tool format. GLM has no compact tool
    /// variant — the XML prompt is the only wire format.
    private static func dumpGLMChatTemplate(model: GGUFModel) throws -> String {
        let tok = try GLM52Tokenizer(model: model)

        var out = "=== tokenizer.chat_template ===\n"
        if let tpl = model.string("tokenizer.chat_template"), !tpl.isEmpty {
            out += tpl + "\n"
        } else {
            out += "<assente nel GGUF>\n"
        }

        out += "\n=== Token speciali del protocollo GLM (vocab, tipo, atomico?) ===\n"
        out += "Legenda tipo: 1=NORMAL 2=UNKNOWN 3=CONTROL 4=USER_DEFINED 6=BYTE.\n"
        out += "'atomico' = la stringa tokenizza in un solo id (necessario per i marcatori di ruolo/tool).\n"
        let types = model.intArray("tokenizer.ggml.token_type")
        for s in GLM52ConversationProtocol.controlTokens {
            guard let id = tok.tokenID(s) else { out += "\(s) -> ASSENTE\n"; continue }
            let type = types.flatMap { Int(id) < $0.count ? $0[Int(id)] : nil }
            let typeStr = type.map { "tipo \($0)" } ?? "tipo ?"
            let atomic = tok.tokenizeRenderedChat(s) == [id]
            out += "\(s) -> id \(id), \(typeStr), \(atomic ? "atomico ✓" : "SPEZZATO ✗")\n"
        }

        let tools = ToolRegistry.specs(enabled: Set(ToolRegistry.builtins.map { $0.spec.name }))
        out += "\n=== Prompt con tool — formato XML nativo GLM ===\n"
        out += try GLM52ChatRenderer.render(turns: [.user("Ciao, come stai?")],
                                            tools: tools)
        return out
    }
}
