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

    /// `toolCallExample` shows only the required parameter, renders the
    /// tool/parameter names verbatim, and flags the string type.
    func testToolCallExampleRendersRequiredStringParam() {
        let schema: [String: Any] = [
            "type": "object",
            "properties": [
                "path": ["type": "string", "description": "A file path."],
                "limit": ["type": "integer", "description": "Max lines."],
            ],
            "required": ["path"],
        ]
        let example = EncodingDSV4.toolCallExample(name: "native__read",
                                                   inputSchema: schema)
        let dt = EncodingDSV4.dsmlToken
        XCTAssertTrue(example.contains("<\(dt)tool_calls>"))
        XCTAssertTrue(example.contains("name=\"native__read\""))
        XCTAssertTrue(example.contains("name=\"path\""))
        XCTAssertTrue(example.contains("string=\"true\""))
        // Optional `limit` is not in `required`, so it is omitted.
        XCTAssertFalse(example.contains("name=\"limit\""))
    }

    /// A schema with no properties yields a valid empty `inv` block —
    /// the shape a no-argument tool such as `__list_tools` needs.
    func testToolCallExampleNoParamsYieldsEmptyInvoke() {
        let example = EncodingDSV4.toolCallExample(
            name: "__list_tools", inputSchema: ["type": "object"])
        XCTAssertTrue(example.contains("name=\"__list_tools\""))
        XCTAssertFalse(example.contains("parameter"))
    }

    /// Non-string parameters render with `string="false"` and a JSON
    /// value; booleans must not collapse to `1`.
    func testToolCallExampleNonStringParam() {
        let schema: [String: Any] = [
            "type": "object",
            "properties": [
                "recursive": ["type": "boolean", "description": "Recurse."],
            ],
            "required": ["recursive"],
        ]
        let example = EncodingDSV4.toolCallExample(name: "native__glob",
                                                   inputSchema: schema)
        XCTAssertTrue(example.contains("string=\"false\""))
        XCTAssertTrue(example.contains(">true<"))
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
                .object([
                    "name": .string("echo"),
                    "description": .string("Echo back"),
                ]),
                .object([
                    "name": .string("now"),
                    "description": .string("Current time"),
                ]),
            ])
        let out = try template.render(messages: [], options: opts)
        XCTAssertEqual(out, "echo=Echo back;now=Current time;")
    }
}
