import XCTest
@testable import DeepSeekKit

/// Tests for the `ChatTemplate` protocol implementations:
/// - `DSV4Template` must produce byte-identical output to
///   `EncodingDSV4.encodeMessages` (zero regression on the V4 path).
/// - `JinjaChatTemplate` must populate the standard HuggingFace scope
///   (`messages`, `bos_token`, `eos_token`, `add_generation_prompt`).
final class ChatTemplateTests: XCTestCase {

    private let conversation: [Message] = [
        Message(role: .system, content: "You are helpful."),
        Message(role: .user, content: "ciao"),
        Message(role: .assistant, content: "ciao a te"),
    ]

    func testDSV4TemplateMatchesEncodingDSV4Chat() throws {
        let template = DSV4Template()
        let opts = ChatTemplateOptions(thinkingMode: .chat)
        let viaTemplate = try template.render(messages: conversation, options: opts)
        let viaEncoding = EncodingDSV4.encodeMessages(conversation, mode: .chat)
        XCTAssertEqual(viaTemplate, viaEncoding, "DSV4Template must be byte-identical to EncodingDSV4")
    }

    func testDSV4TemplateMatchesEncodingDSV4ThinkHigh() throws {
        let template = DSV4Template()
        let opts = ChatTemplateOptions(thinkingMode: .high)
        let viaTemplate = try template.render(messages: conversation, options: opts)
        let viaEncoding = EncodingDSV4.encodeMessages(conversation, mode: .high)
        XCTAssertEqual(viaTemplate, viaEncoding)
    }

    func testDSV4TemplateMatchesEncodingDSV4ThinkMax() throws {
        let template = DSV4Template()
        let opts = ChatTemplateOptions(thinkingMode: .max)
        let viaTemplate = try template.render(messages: conversation, options: opts)
        let viaEncoding = EncodingDSV4.encodeMessages(conversation, mode: .max)
        XCTAssertEqual(viaTemplate, viaEncoding)
    }

    func testDSV4TemplateForwardsToolSchemas() throws {
        let schemas = #"{"name":"echo","description":"echo back"}"#
        let template = DSV4Template()
        let opts = ChatTemplateOptions(thinkingMode: .chat,
                                       toolSchemasJSON: schemas)
        let viaTemplate = try template.render(messages: conversation, options: opts)
        let viaEncoding = EncodingDSV4.encodeMessages(conversation,
                                                      mode: .chat,
                                                      toolSchemasJSON: schemas)
        XCTAssertEqual(viaTemplate, viaEncoding)
    }

    /// Render a minimal Mistral-style template via the dispatcher and
    /// verify the assistant marker is appended when
    /// `add_generation_prompt` is true.
    func testJinjaChatTemplateRendersMistral() throws {
        let mistralSrc = """
        {{ bos_token }}{%- for m in messages -%}{%- if m.role == 'user' -%}[INST] {{ m.content }} [/INST]{%- elif m.role == 'assistant' -%}{{ m.content }}{{ eos_token }}{%- endif -%}{%- endfor -%}
        """
        let template = try JinjaChatTemplate(mistralSrc)
        let opts = ChatTemplateOptions(addGenerationPrompt: false,
                                       thinkingMode: .chat,
                                       bosToken: "<s>", eosToken: "</s>")
        let messages = [
            Message(role: .user, content: "hi"),
            Message(role: .assistant, content: "hello"),
        ]
        let out = try template.render(messages: messages, options: opts)
        XCTAssertEqual(out, "<s>[INST] hi [/INST]hello</s>")
    }

    /// ChatML with add_generation_prompt: trailing `<|im_start|>assistant\n`.
    func testJinjaChatTemplateRendersChatML() throws {
        let chatml = """
        {%- for m in messages -%}<|im_start|>{{ m.role }}\n{{ m.content }}<|im_end|>\n{%- endfor -%}{%- if add_generation_prompt -%}<|im_start|>assistant\n{%- endif -%}
        """
        let template = try JinjaChatTemplate(chatml)
        let opts = ChatTemplateOptions(addGenerationPrompt: true)
        let messages = [Message(role: .user, content: "test")]
        let out = try template.render(messages: messages, options: opts)
        XCTAssertEqual(out, "<|im_start|>user\ntest<|im_end|>\n<|im_start|>assistant\n")
    }

    /// Tools array is exposed through the standard HF scope as `tools`.
    func testJinjaChatTemplateExposesTools() throws {
        let src = """
        {%- for t in tools -%}{{ t.name }}={{ t.description }};{%- endfor -%}
        """
        let template = try JinjaChatTemplate(src)
        let opts = ChatTemplateOptions(
            tools: [
                ["name": "echo", "description": "Echo back"],
                ["name": "now",  "description": "Current time"],
            ])
        let out = try template.render(messages: [], options: opts)
        XCTAssertEqual(out, "echo=Echo back;now=Current time;")
    }
}
