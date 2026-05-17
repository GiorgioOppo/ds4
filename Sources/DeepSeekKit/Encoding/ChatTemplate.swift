import Foundation

/// Renders a list of `Message`s into the prompt string a specific model
/// expects to see. Decouples the tokenizer from the chat surface so we
/// can load a HuggingFace `tokenizer_config.json` template (Mistral,
/// Llama, Gemma, Qwen, ChatML, …) without hand-porting each one.
///
/// The DSV4 path keeps using `DSV4Template`, which delegates to
/// `EncodingDSV4.encodeMessages` so existing call sites stay
/// byte-identical.
public protocol ChatTemplate: Sendable {
    func render(messages: [Message], options: ChatTemplateOptions) throws -> String
}

/// Runtime parameters that influence template rendering. The
/// HuggingFace standard scope (`bos_token`, `eos_token`,
/// `add_generation_prompt`, `tools`) is exposed via the corresponding
/// fields; DSV4-specific knobs (`thinkingMode`, `toolSchemasJSON`)
/// piggy-back on the same struct so the call site doesn't branch.
public struct ChatTemplateOptions: Sendable {
    /// Append a trailing role marker so the model continues as the
    /// assistant. HuggingFace Jinja templates check this directly.
    public var addGenerationPrompt: Bool
    /// DSV4-specific reasoning budget. Ignored by non-DSV4 templates.
    public var thinkingMode: ThinkingMode
    /// DSV4-specific tool schemas (already JSON-encoded). The Jinja
    /// path uses `tools` instead.
    public var toolSchemasJSON: String?
    /// OpenAI-style tool array exposed to Jinja templates as
    /// `tools`. Ogni entry è un `JSONValue.object(...)` con i
    /// campi attesi dal template (tipicamente `name`,
    /// `description`, `parameters`). Tipizzato su `JSONValue`
    /// perché `[String: Any]` non è `Sendable` in Swift 6.
    public var tools: [JSONValue]?
    public var bosToken: String
    public var eosToken: String

    public init(addGenerationPrompt: Bool = true,
                thinkingMode: ThinkingMode = .chat,
                toolSchemasJSON: String? = nil,
                tools: [JSONValue]? = nil,
                bosToken: String = "",
                eosToken: String = "") {
        self.addGenerationPrompt = addGenerationPrompt
        self.thinkingMode = thinkingMode
        self.toolSchemasJSON = toolSchemasJSON
        self.tools = tools
        self.bosToken = bosToken
        self.eosToken = eosToken
    }
}

/// Errors raised by `ChatTemplate.render` implementations.
public enum ChatTemplateError: Error, CustomStringConvertible {
    case unsupportedFeature(String)
    case templateRaise(String)
    case parseFailure(String)

    public var description: String {
        switch self {
        case .unsupportedFeature(let m): return "ChatTemplate: unsupported feature — \(m)"
        case .templateRaise(let m):      return "ChatTemplate: raise_exception(\(m))"
        case .parseFailure(let m):       return "ChatTemplate: parse error — \(m)"
        }
    }
}
