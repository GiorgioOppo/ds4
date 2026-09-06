import XCTest
@testable import DS4Metal

final class DeepSeekV4VisionAttentionTests: XCTestCase {
    func testParserSeparatesTextPaddingAndMultipleCompleteImages() throws {
        let v = 100
        let tokens = [7, v + 1, v + 1, v, v + 2, v + 3, v + 4, 8, v, v + 2, v + 4, 9]
        let spans = try DeepSeekV4VisionAttention.spans(tokens: tokens, vocabularySize: v)
        XCTAssertEqual(spans, [
            .init(block: 1..<7, image: 3...6),
            .init(block: 8..<11, image: 8...10),
        ])
    }

    func testParserRejectsTruncatedOrMalformedImageBlocks() {
        for tokens in [[100, 102], [101, 5], [102, 104], [100, 7, 104],
                       [100, 100, 104], [100, 105, 104], [-1]] {
            XCTAssertThrowsError(try DeepSeekV4VisionAttention.spans(tokens: tokens, vocabularySize: 100))
        }
        XCTAssertThrowsError(try DeepSeekV4VisionAttention.spans(tokens: [0], vocabularySize: 0))
    }

    func testVisualWindowMatchesUpstream383Left384Right() {
        let span = 64...1089
        func bounds(_ query: Int) -> ClosedRange<Int> {
            DeepSeekV4VisionAttention.bounds(query: query, startPosition: 1000,
                image: span, firstRawPosition: 873, lastRawPosition: 2089, window: 128)
        }
        XCTAssertEqual(bounds(0), 873...1000, "padding remains causal")
        XCTAssertEqual(bounds(63), 936...1063, "last alignment pad remains causal")
        XCTAssertEqual(bounds(64), 937...1448, "start sees 384 future rows and its text window")
        XCTAssertEqual(bounds(564), 1181...1948, "interior image query has a 768-row window")
        XCTAssertEqual(bounds(1089), 1706...2089, "image end never sees later text")
    }

    func testWindowClipsToLiveRawCacheAndPreservesLongerTextWindow() {
        let result = DeepSeekV4VisionAttention.bounds(query: 500, startPosition: 1000,
            image: 0...1000, firstRawPosition: 1200, lastRawPosition: 1800, window: 1024)
        XCTAssertEqual(result, 1200...1800)
        let unlimited = DeepSeekV4VisionAttention.bounds(query: 500, startPosition: 1000,
            image: 0...1000, firstRawPosition: 100, lastRawPosition: 2000, window: 0)
        XCTAssertEqual(unlimited, 100...1884)
    }
}
