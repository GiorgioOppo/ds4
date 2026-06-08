import XCTest
@testable import DS4Core

/// Pure-Swift checks of tool rendering and tool-call parsing (no GGUF needed).
final class ChatToolsTests: XCTestCase {
    private let markup = ToolMarkup.deepSeekDefault

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
        XCTAssertTrue(s.contains("get_weather"))
        XCTAssertTrue(s.contains("<｜User｜>Weather in Paris?"))
        // Open assistant turn ends the prompt; think disabled -> </think>.
        XCTAssertTrue(s.hasSuffix("<｜Assistant｜></think>"))
    }

    func testRenderThinkMarker() {
        let s = ChatRenderer.render(turns: [.user("hi")], tools: [], think: .high, markup: markup)
        XCTAssertTrue(s.hasSuffix("<｜Assistant｜><think>"))
    }

    func testRenderToolResult() {
        let turns: [ChatTurn] = [
            .user("Weather in Paris?"),
            .assistant(text: "", toolCalls: [ToolCall(id: "call_0", name: "get_weather",
                                                      argumentsJSON: #"{"city":"Paris"}"#)]),
            .toolResult(callId: "call_0", name: "get_weather", content: #"{"tempC":21}"#),
        ]
        let s = ChatRenderer.render(turns: turns, tools: [weather], think: .none, markup: markup)
        XCTAssertTrue(s.contains(markup.callsBegin))
        XCTAssertTrue(s.contains(markup.sep + "get_weather"))
        XCTAssertTrue(s.contains(markup.outputsBegin))
        XCTAssertTrue(s.contains(#"{"tempC":21}"#))
    }

    func testParseNoCallsReturnsText() {
        let (calls, visible) = ToolCallParser.parse("Just a normal answer.", markup: markup)
        XCTAssertTrue(calls.isEmpty)
        XCTAssertEqual(visible, "Just a normal answer.")
    }

    func testParseSingleCall() {
        let text = "Let me check.\n" +
            markup.callsBegin + markup.callBegin + "function" + markup.sep +
            "get_weather\n```json\n{\"city\":\"Paris\"}\n```" + markup.callEnd + markup.callsEnd
        let (calls, visible) = ToolCallParser.parse(text, markup: markup)
        XCTAssertEqual(visible, "Let me check.")
        XCTAssertEqual(calls.count, 1)
        XCTAssertEqual(calls[0].name, "get_weather")
        XCTAssertEqual(calls[0].argumentsJSON, #"{"city":"Paris"}"#)
    }

    /// render → parse must recover the same calls (fenced JSON).
    func testRoundTrip() {
        let calls = [
            ToolCall(id: "call_0", name: "get_weather", argumentsJSON: #"{"city":"Paris"}"#),
            ToolCall(id: "call_1", name: "now", argumentsJSON: "{}"),
        ]
        let rendered = "Sure.\n" + ChatRenderer.renderToolCalls(calls, markup: markup)
        let (parsed, visible) = ToolCallParser.parse(rendered, markup: markup)
        XCTAssertEqual(visible, "Sure.")
        XCTAssertEqual(parsed.count, 2)
        XCTAssertEqual(parsed[0].name, "get_weather")
        XCTAssertEqual(parsed[0].argumentsJSON, #"{"city":"Paris"}"#)
        XCTAssertEqual(parsed[1].name, "now")
        XCTAssertEqual(parsed[1].argumentsJSON, "{}")
    }

    /// Missing trailing callsEnd (truncated generation) still parses the call.
    func testParseUnterminatedBlock() {
        let text = markup.callsBegin + markup.callBegin + "function" + markup.sep +
            "now\n```json\n{}\n```" + markup.callEnd
        let (calls, _) = ToolCallParser.parse(text, markup: markup)
        XCTAssertEqual(calls.count, 1)
        XCTAssertEqual(calls[0].name, "now")
    }
}
