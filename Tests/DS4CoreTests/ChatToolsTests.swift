import XCTest
@testable import DS4Core

/// Pure-Swift checks of DeepSeek-V4 DSML tool rendering and parsing (no GGUF).
final class ChatToolsTests: XCTestCase {
    private let markup = ToolMarkup.dsv4

    private let weather = ToolSpec(
        name: "get_weather",
        description: "Get the current weather for a city.",
        parametersJSON: #"{"type":"object","properties":{"city":{"type":"string"}},"required":["city"]}"#)

    func testRenderIncludesSystemToolsAndTags() {
        let turns: [ChatTurn] = [.system("You are helpful."), .user("Weather in Paris?")]
        let s = ChatRenderer.render(turns: turns, tools: [weather], think: .none, markup: markup)
        XCTAssertTrue(s.hasPrefix("<｜begin▁of▁sentence｜>"))
        XCTAssertTrue(s.contains("You are helpful."))
        XCTAssertTrue(s.contains("## Tools"))
        XCTAssertTrue(s.contains(markup.callsOpen))          // <｜DSML｜tool_calls>
        XCTAssertTrue(s.contains("get_weather"))
        XCTAssertTrue(s.contains("<｜User｜>Weather in Paris?"))
        XCTAssertTrue(s.hasSuffix("<｜Assistant｜></think>")) // think disabled
    }

    func testRenderThinkMarker() {
        let s = ChatRenderer.render(turns: [.user("hi")], tools: [], think: .high, markup: markup)
        XCTAssertTrue(s.hasSuffix("<｜Assistant｜><think>"))
    }

    func testRenderToolCallAndResult() {
        let turns: [ChatTurn] = [
            .user("Weather in Paris?"),
            .assistant(text: "", toolCalls: [ToolCall(id: "call_0", name: "get_weather",
                                                      argumentsJSON: #"{"city":"Paris","days":3}"#)]),
            .toolResult(callId: "call_0", name: "get_weather", content: #"{"tempC":21}"#),
        ]
        let s = ChatRenderer.render(turns: turns, tools: [weather], think: .none, markup: markup)
        XCTAssertTrue(s.contains(markup.invokeOpen("get_weather")))
        // string param rendered raw + string="true"; numeric param JSON + string="false".
        XCTAssertTrue(s.contains(#"<｜DSML｜parameter name="city" string="true">Paris</｜DSML｜parameter>"#))
        XCTAssertTrue(s.contains(#"<｜DSML｜parameter name="days" string="false">3</｜DSML｜parameter>"#))
        XCTAssertTrue(s.contains(#"{"tempC":21}"#))
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
        XCTAssertEqual(calls[0].argumentsJSON, #"{"city":"Paris","days":3}"#)  // keys sorted
    }

    /// render → parse round-trips the call arguments (string + numeric).
    func testRoundTrip() {
        let calls = [ToolCall(id: "call_0", name: "calc", argumentsJSON: #"{"a":2,"expr":"2+2"}"#)]
        let rendered = "Sure.\n" + ChatRenderer.renderToolCalls(calls, markup: markup)
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

    /// Missing trailing close tags (truncated generation) still parses the call.
    func testParseUnterminated() {
        let text = markup.callsOpen + "\n" + markup.invokeOpen("now")
        let (calls, _) = ToolCallParser.parse(text, markup: markup)
        XCTAssertEqual(calls.count, 1)
        XCTAssertEqual(calls[0].name, "now")
    }
}
