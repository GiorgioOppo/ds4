import XCTest
@testable import DS4Core

/// Pure-Swift checks of DeepSeek-V4 DSML tool rendering and parsing (no GGUF),
/// matched to the model's tokenizer.chat_template.
final class ChatToolsTests: XCTestCase {
    private let markup = ToolMarkup.dsv4

    private let weather = ToolSpec(
        name: "get_weather",
        description: "Get the current weather for a city.",
        parametersJSON: #"{"type":"object","properties":{"city":{"type":"string"}},"required":["city"]}"#)

    /// BOS is immediately followed by the system text (no spurious newline), and
    /// tools follow with a blank-line separator.
    func testRenderSystemAdjacency() {
        let s = ChatRenderer.render(turns: [.system("You are helpful."), .user("Hi")],
                                    tools: [weather], think: .none, markup: markup)
        XCTAssertTrue(s.hasPrefix("<｜begin▁of▁sentence｜>You are helpful.\n\n## Tools"))
        XCTAssertTrue(s.contains("get_weather"))
        XCTAssertTrue(s.contains("<｜User｜>Hi"))
        XCTAssertTrue(s.hasSuffix("<｜Assistant｜></think>"))
    }

    /// With no system prompt, BOS is immediately followed by "## Tools".
    func testRenderNoSystemNoLeadingNewline() {
        let s = ChatRenderer.render(turns: [.user("Hi")], tools: [weather], think: .none, markup: markup)
        XCTAssertTrue(s.hasPrefix("<｜begin▁of▁sentence｜>## Tools"))
    }

    func testRenderThinkMarker() {
        let s = ChatRenderer.render(turns: [.user("hi")], tools: [], think: .high, markup: markup)
        XCTAssertTrue(s.hasSuffix("<｜Assistant｜><think>"))
    }

    /// Past assistant turns get `<｜Assistant｜></think>…<eos>` (template form).
    func testMultiTurnAssistantThinkMarker() {
        let turns: [ChatTurn] = [.user("a"), .assistant(text: "answer", toolCalls: []), .user("b")]
        let s = ChatRenderer.render(turns: turns, tools: [], think: .none, markup: markup)
        XCTAssertTrue(s.contains("<｜User｜>a<｜Assistant｜></think>answer<｜end▁of▁sentence｜><｜User｜>b<｜Assistant｜></think>"))
    }

    /// Tool call rendered in DSML; tool result inside a user turn as <tool_result>.
    func testRenderToolCallAndResult() {
        let turns: [ChatTurn] = [
            .user("Weather in Paris?"),
            .assistant(text: "", toolCalls: [ToolCall(id: "call_0", name: "get_weather",
                                                      argumentsJSON: #"{"city":"Paris","days":3}"#)]),
            .toolResult(callId: "call_0", name: "get_weather", content: #"{"tempC":21}"#),
        ]
        let s = ChatRenderer.render(turns: turns, tools: [weather], think: .none, markup: markup)
        XCTAssertTrue(s.contains("\n\n" + markup.callsOpen))       // \n\n before the block
        XCTAssertTrue(s.contains(markup.invokeOpen("get_weather")))
        XCTAssertTrue(s.contains(#"<｜DSML｜parameter name="city" string="true">Paris</｜DSML｜parameter>"#))
        XCTAssertTrue(s.contains(#"<｜DSML｜parameter name="days" string="false">3</｜DSML｜parameter>"#))
        // Tool result: user turn + <tool_result> (NOT a DSML output block).
        XCTAssertTrue(s.contains(#"<｜User｜><tool_result>{"tempC":21}</tool_result>"#))
    }

    func testParseNoCallsReturnsText() {
        let (calls, visible) = ToolCallParser.parse("Just a normal answer.", markup: markup)
        XCTAssertTrue(calls.isEmpty)
        XCTAssertEqual(visible, "Just a normal answer.")
    }

    func testParseSingleCall() {
        let text = "Let me check.\n" + markup.callsOpen + "\n" +
            markup.invokeOpen("get_weather") + "\n" +
            #"<｜DSML｜parameter name="city" string="true">Paris</｜DSML｜parameter>"# + "\n" +
            #"<｜DSML｜parameter name="days" string="false">3</｜DSML｜parameter>"# + "\n" +
            markup.invokeClose + "\n" + markup.callsClose
        let (calls, visible) = ToolCallParser.parse(text, markup: markup)
        XCTAssertEqual(visible, "Let me check.")
        XCTAssertEqual(calls.count, 1)
        XCTAssertEqual(calls[0].name, "get_weather")
        XCTAssertEqual(calls[0].argumentsJSON, #"{"city":"Paris","days":3}"#)
    }

    /// render → parse round-trips the call arguments (string + numeric).
    func testRoundTrip() {
        let calls = [ToolCall(id: "call_0", name: "calc", argumentsJSON: #"{"a":2,"expr":"2+2"}"#)]
        let rendered = "Sure." + ChatRenderer.renderToolCalls(calls, markup: markup)
        let (parsed, visible) = ToolCallParser.parse(rendered, markup: markup)
        XCTAssertEqual(visible, "Sure.")
        XCTAssertEqual(parsed.count, 1)
        XCTAssertEqual(parsed[0].name, "calc")
        XCTAssertEqual(parsed[0].argumentsJSON, #"{"a":2,"expr":"2+2"}"#)
    }

    func testParseTwoCalls() {
        let text = markup.callsOpen + "\n" +
            markup.invokeOpen("now") + "\n" + markup.invokeClose + "\n" +
            markup.invokeOpen("ping") + "\n" +
            #"<｜DSML｜parameter name="host" string="true">a.com</｜DSML｜parameter>"# + "\n" +
            markup.invokeClose + "\n" + markup.callsClose
        let (calls, _) = ToolCallParser.parse(text, markup: markup)
        XCTAssertEqual(calls.map(\.name), ["now", "ping"])
        XCTAssertEqual(calls[0].argumentsJSON, "{}")
        XCTAssertEqual(calls[1].argumentsJSON, #"{"host":"a.com"}"#)
    }

    func testParseUnterminated() {
        let text = markup.callsOpen + "\n" + markup.invokeOpen("now")
        let (calls, _) = ToolCallParser.parse(text, markup: markup)
        XCTAssertEqual(calls.count, 1)
        XCTAssertEqual(calls[0].name, "now")
    }

    /// Compact mode lists only name(params) + a one-line format hint, and is much
    /// shorter than the full declaration.
    func testCompactToolsDeclaration() {
        let add = ToolSpec(name: "add", description: "Add a + b.",
                           parametersJSON: #"{"type":"object","properties":{"a":{"type":"number"},"b":{"type":"number"}}}"#)
        let full = ChatRenderer.render(turns: [.user("hi")], tools: [add, weather], think: .none,
                                       markup: markup, compactTools: false)
        let compact = ChatRenderer.render(turns: [.user("hi")], tools: [add, weather], think: .none,
                                          markup: markup, compactTools: true)
        XCTAssertTrue(compact.contains("## Tools"))
        XCTAssertTrue(compact.contains("- add(a, b)"))
        XCTAssertTrue(compact.contains("- get_weather(city)"))
        XCTAssertTrue(compact.contains(markup.callsOpen))           // keeps the format hint
        XCTAssertFalse(compact.contains("### Available Tool Schemas"))
        XCTAssertFalse(compact.contains("You MUST strictly follow"))
        XCTAssertLessThan(compact.count, full.count / 2)            // much smaller
    }

    /// The function schema uses sorted keys (≈ Jinja tojson) and nests parameters.
    func testFunctionSchema() {
        let s = ChatRenderer.functionJSON(weather)
        XCTAssertTrue(s.contains(#""name":"get_weather""#))
        XCTAssertTrue(s.contains(#""description":"Get the current weather for a city.""#))
        XCTAssertTrue(s.contains(#""parameters":{"#))
        // sorted keys: description before name before parameters
        let di = s.range(of: "\"description\"")!.lowerBound
        let ni = s.range(of: "\"name\"")!.lowerBound
        XCTAssertLessThan(di, ni)
    }
}
