import XCTest
@testable import DeepSeekKit

/// Unit tests for the minimal Jinja2 subset shipped with the chat
/// template engine. Covers the slice the engine actually supports:
/// expressions, if/elif/else, for + loop.* variables, filters,
/// operators (incl. `in` / `not in`), negative subscripts,
/// whitespace trim markers, raise_exception, and a smoke test with
/// the real Mistral `[INST] … [/INST]` template.
final class JinjaTemplateTests: XCTestCase {

    private func render(_ src: String, scope: [String: JinjaValue] = [:]) throws -> String {
        let t = try JinjaTemplate(src)
        return try t.render(context: scope)
    }

    func testLiteralPassthrough() throws {
        let out = try render("hello world")
        XCTAssertEqual(out, "hello world")
    }

    func testVariableInterpolation() throws {
        let out = try render("hi {{ name }}!", scope: ["name": .string("Ada")])
        XCTAssertEqual(out, "hi Ada!")
    }

    func testIfElifElse() throws {
        let tpl = "{% if n == 0 %}zero{% elif n == 1 %}one{% else %}many{% endif %}"
        XCTAssertEqual(try render(tpl, scope: ["n": .int(0)]), "zero")
        XCTAssertEqual(try render(tpl, scope: ["n": .int(1)]), "one")
        XCTAssertEqual(try render(tpl, scope: ["n": .int(7)]), "many")
    }

    func testForLoopAndLoopVar() throws {
        let tpl = "{% for x in xs %}[{{ loop.index0 }}:{{ x }}{% if not loop.last %},{% endif %}]{% endfor %}"
        let scope: [String: JinjaValue] = [
            "xs": .list([.string("a"), .string("b"), .string("c")]),
        ]
        XCTAssertEqual(try render(tpl, scope: scope), "[0:a,][1:b,][2:c]")
    }

    func testFiltersTrimLowerDefaultLength() throws {
        let scope: [String: JinjaValue] = [
            "s": .string("  Hello  "),
            "empty": .string(""),
            "list": .list([.int(1), .int(2), .int(3)]),
        ]
        XCTAssertEqual(try render("{{ s | trim }}", scope: scope), "Hello")
        XCTAssertEqual(try render("{{ s | trim | lower }}", scope: scope), "hello")
        XCTAssertEqual(try render("{{ empty | default('--') }}", scope: scope), "--")
        XCTAssertEqual(try render("{{ list | length }}", scope: scope), "3")
    }

    func testOperatorsInAndNotIn() throws {
        let scope: [String: JinjaValue] = [
            "role": .string("user"),
            "roles": .list([.string("user"), .string("assistant")]),
        ]
        XCTAssertEqual(try render("{% if role in roles %}yes{% endif %}", scope: scope), "yes")
        XCTAssertEqual(try render("{% if 'system' not in roles %}absent{% endif %}", scope: scope), "absent")
    }

    func testNegativeSubscript() throws {
        let scope: [String: JinjaValue] = [
            "messages": .list([.string("a"), .string("b"), .string("c")]),
        ]
        XCTAssertEqual(try render("{{ messages[-1] }}", scope: scope), "c")
    }

    func testWhitespaceTrim() throws {
        let tpl = "<{%- if x -%}A{%- endif -%}>"
        XCTAssertEqual(try render(tpl, scope: ["x": .bool(true)]), "<A>")
        let tpl2 = "before  {{- name -}}  after"
        XCTAssertEqual(try render(tpl2, scope: ["name": .string("X")]), "beforeXafter")
    }

    func testRaiseExceptionPropagates() {
        XCTAssertThrowsError(try render("{{ raise_exception('boom') }}")) { error in
            guard case ChatTemplateError.templateRaise(let msg) = error else {
                return XCTFail("expected templateRaise, got \(error)")
            }
            XCTAssertEqual(msg, "boom")
        }
    }

    func testSetStatement() throws {
        let tpl = "{% set x = 'hi' %}{{ x }}"
        XCTAssertEqual(try render(tpl), "hi")
    }

    /// Smoke test against the Mistral-style chat template (simplified).
    func testMistralTemplateSmoke() throws {
        let mistral = """
        {{ bos_token }}{%- for m in messages -%}{%- if m.role == 'user' -%}[INST] {{ m.content }} [/INST]{%- elif m.role == 'assistant' -%}{{ m.content }}{{ eos_token }}{%- endif -%}{%- endfor -%}
        """
        let scope: [String: JinjaValue] = [
            "bos_token": .string("<s>"),
            "eos_token": .string("</s>"),
            "messages": .list([
                .dict(["role": .string("user"), "content": .string("hi")]),
                .dict(["role": .string("assistant"), "content": .string("hello")]),
            ]),
        ]
        let out = try render(mistral, scope: scope)
        XCTAssertEqual(out, "<s>[INST] hi [/INST]hello</s>")
    }

    /// ChatML template (used by Qwen and many newer models). Tests that
    /// `add_generation_prompt` produces a trailing assistant marker.
    func testChatMLTemplateWithGenerationPrompt() throws {
        let chatml = """
        {%- for m in messages -%}<|im_start|>{{ m.role }}\n{{ m.content }}<|im_end|>\n{%- endfor -%}{%- if add_generation_prompt -%}<|im_start|>assistant\n{%- endif -%}
        """
        let scope: [String: JinjaValue] = [
            "messages": .list([
                .dict(["role": .string("user"), "content": .string("hi")]),
            ]),
            "add_generation_prompt": .bool(true),
        ]
        let out = try render(chatml, scope: scope)
        XCTAssertEqual(out, "<|im_start|>user\nhi<|im_end|>\n<|im_start|>assistant\n")
    }

    func testMacroIsUnsupported() {
        XCTAssertThrowsError(try render("{% macro x() %}{% endmacro %}")) { error in
            guard case ChatTemplateError.unsupportedFeature = error else {
                return XCTFail("expected unsupportedFeature, got \(error)")
            }
        }
    }
}
