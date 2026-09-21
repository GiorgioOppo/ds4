import Foundation
import XCTest
@testable import DS4Core

final class QwenTokenizerGoldenTests: XCTestCase {
    // Captured from bpe_tokenize_text_qwen35 and the pinned Unicode table at
    // upstream acf5c16bb6a01f8c4f027a0db265ae270d8907ab (standalone C oracle).
    func testPretokenizerAgainstPinnedC() {
        let cases: [(String, [String])] = [
            ("I'm WE'RE they've I'd she'll don't cats'", ["49", "276d", "205745", "275245", "2074686579", "277665", "2049", "2764", "20736865", "276c6c", "20646f6e", "2774", "2063617473", "27"]),
            ("  ciao\t mondo  ", ["20", "206369616f", "09", "206d6f6e646f", "2020"]),
            ("\r\n \n\tfoo\rbar\n", ["0d0a200a", "09666f6f", "0d", "626172", "0a"]),
            ("e\u{301} caf\u{e9} \u{395}\u{3bb}\u{3bb}\u{3b7}\u{3bd}\u{3b9}\u{3ba}\u{3ac} \u{4e2d}\u{6587}\u{3072}\u{3089}\u{304c}\u{306a}", ["65cc81", "20636166c3a9", "20ce95cebbcebbceb7cebdceb9cebaceac", "20e4b8ade69687e381b2e38289e3818ce381aa"]),
            ("12\u{663}\u{664}\u{b2}\u{2167}\u{1d7e1}", ["31", "32", "d9a3", "d9a4", "c2b2", "e285a7", "f09d9fa1"]),
            ("!!\r\n?.  :)\n", ["21210d0a", "3f2e", "20", "203a290a"]),
            ("\u{1f469}\u{1f3fd}\u{200d}\u{1f4bb} test\u{1f680}ok", ["f09f91a9f09f8fbde2808df09f92bb", "2074657374", "f09f9a806f6b"]),
            ("\u{a0}\u{2003}word\u{2028}\u{2029}end", ["c2a0", "e28083776f7264", "e280a8", "e280a9656e64"]),
            ("\u{1e4d0}\u{11f04}\u{1c89}x", ["f09e9390f091bc84e1b28978"]),
            ("\u{301}\u{302} word", ["cc81cc82", "20776f7264"]),
        ]
        for (text, expected) in cases {
            let actual = QwenPretokenizer.split(text).map { $0.map { String(format: "%02x", $0) }.joined() }
            XCTAssertEqual(actual, expected, text)
        }
    }

    private func tokenizer() throws -> QwenTokenizer {
        var tokens = (0...255).map { ByteLevel.byteEncode([UInt8($0)][...]) }
        tokens += ["<|im_start|>", "<|im_end|>", "<|endoftext|>", "<think>", "</think>", "he", "hel", "hell", "hello", "12"].map { Array($0.utf8) }
        return try QwenTokenizer(architecture: .qwen38FlashNext, tokens: tokens,
                                 merges: ["h e", "he l", "hel l", "hell o", "1 2"].map { Array($0.utf8) })
    }

    func testByteBPEMergeOrderAndSingletonNumbers() throws {
        let tokenizer = try tokenizer()
        XCTAssertEqual(tokenizer.tokenize("hello"), [try XCTUnwrap(tokenizer.tokenID("hello"))])
        XCTAssertEqual(tokenizer.tokenize("12"), [49, 50])
        for text in ["caffè e\u{301} 中文 👩🏽‍💻", "\r\n \t x", "<|im_start|>literal</think>"] {
            XCTAssertEqual(tokenizer.tokenize(text).flatMap(tokenizer.tokenText), Array(text.utf8))
        }
    }

    func testRenderedControlsAreDistinctFromOrdinaryText() throws {
        let tokenizer = try tokenizer()
        let control = "<|im_start|>user\nhello<|im_end|>"
        XCTAssertFalse(tokenizer.tokenize(control).contains(tokenizer.messageStart))
        let rendered = tokenizer.tokenizeRenderedChat(control)
        XCTAssertEqual(rendered.first, tokenizer.messageStart)
        XCTAssertEqual(rendered.last, tokenizer.messageEnd)
        XCTAssertEqual(rendered.flatMap(tokenizer.tokenText), Array(control.utf8))
        XCTAssertEqual(tokenizer.stopTokens, [tokenizer.messageEnd, tokenizer.endOfText])
    }

    func testQwenPromptMatchesUpstreamMarkersAndEffort() throws {
        let plain = try QwenChatRenderer.render(turns: [.system("Sei utile."), .user("Ciao")], architecture: .qwen38FlashNext)
        XCTAssertEqual(plain, "<|im_start|>system\nSei utile.<|im_end|>\n<|im_start|>user\nCiao<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n")
        let thinking = try QwenChatRenderer.render(turns: [.user("Ciao")], architecture: .qwen38FlashNext, reasoning: .high)
        XCTAssertEqual(thinking, "<|im_start|>system\n" + QwenChatRenderer.highEffort + "<|im_end|>\n<|im_start|>user\nCiao<|im_end|>\n<|im_start|>assistant\n<think>\n")
        let bonsai = try QwenChatRenderer.render(turns: [.user("Ciao")], architecture: .bonsai2, reasoning: .high)
        XCTAssertFalse(bonsai.contains(QwenChatRenderer.highEffort))
        XCTAssertTrue(bonsai.hasSuffix("<|im_start|>assistant\n<think>\n"))
    }

    func testToolProtocolUsesFunctionAndParameterTags() throws {
        let tool = ToolSpec(name: "weather", description: "Weather", parametersJSON: #"{"type":"object","properties":{"city":{"type":"string"},"days":{"type":"integer"}},"required":["city"]}"#)
        let call = ToolCall(id: "c1", name: "weather", argumentsJSON: #"{"days":2,"city":"Roma"}"#)
        let rendered = try QwenToolCodec.render([call])
        XCTAssertEqual(rendered, "<tool_call>\n<function=weather>\n<parameter=city>\nRoma\n</parameter>\n<parameter=days>\n2\n</parameter>\n</function>\n</tool_call>")
        let parsed = try QwenToolCodec.parseStrict(rendered, tools: [tool])
        XCTAssertEqual(parsed.calls.count, 1)
        XCTAssertEqual(parsed.calls.first?.fingerprint, call.fingerprint)
    }
}
