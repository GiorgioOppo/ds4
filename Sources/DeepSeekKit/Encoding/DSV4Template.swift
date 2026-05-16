import Foundation

/// Retro-compatible `ChatTemplate` for DeepSeek-V4: delegates `render`
/// straight to `EncodingDSV4.encodeMessages` so the byte stream is
/// identical to the one the engine has been emitting. Lives only to
/// give the new dispatcher path a uniform interface — every V4 call
/// site keeps using the same encoder under the hood.
public struct DSV4Template: ChatTemplate {
    public init() {}

    public func render(messages: [Message], options: ChatTemplateOptions) throws -> String {
        return EncodingDSV4.encodeMessages(messages,
                                           mode: options.thinkingMode,
                                           toolSchemasJSON: options.toolSchemasJSON)
    }
}
