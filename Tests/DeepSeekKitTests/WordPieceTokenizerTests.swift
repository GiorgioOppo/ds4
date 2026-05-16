import XCTest
@testable import DeepSeekKit

final class WordPieceTokenizerTests: XCTestCase {

    /// Build a tokenizer.json-like JSON with a minimal WordPiece vocab.
    private func makeTokenizerJSON(vocab: [String: Int]) -> Data {
        let payload: [String: Any] = [
            "model": [
                "type": "WordPiece",
                "vocab": vocab,
            ]
        ]
        return try! JSONSerialization.data(withJSONObject: payload, options: [])
    }

    func testGreedyLongestMatchWithContinuation() throws {
        let vocab: [String: Int] = [
            "[CLS]": 0, "[SEP]": 1, "[UNK]": 2, "[PAD]": 3,
            "play": 10, "##ing": 11, "##ed": 12,
            "fast": 20,
        ]
        let tok = try WordPieceTokenizer(jsonData: makeTokenizerJSON(vocab: vocab))
        XCTAssertEqual(tok.encode("playing"), [10, 11])
        XCTAssertEqual(tok.encode("played"), [10, 12])
    }

    func testUnknownWordFallsBackToUNK() throws {
        let vocab: [String: Int] = [
            "[CLS]": 0, "[SEP]": 1, "[UNK]": 2, "[PAD]": 3,
            "the": 10,
        ]
        let tok = try WordPieceTokenizer(jsonData: makeTokenizerJSON(vocab: vocab))
        XCTAssertEqual(tok.encode("the qwertyy"), [10, 2])
    }

    func testRoundTripDropsContinuationPrefix() throws {
        let vocab: [String: Int] = [
            "[CLS]": 0, "[SEP]": 1, "[UNK]": 2, "[PAD]": 3,
            "play": 10, "##ing": 11,
        ]
        let tok = try WordPieceTokenizer(jsonData: makeTokenizerJSON(vocab: vocab))
        let ids = tok.encode("playing")
        XCTAssertEqual(ids, [10, 11])
        let text = tok.decode(ids)
        // Output should be "playing" without "##".
        XCTAssertEqual(text, "playing")
    }

    func testStopTokensIncludesSEP() throws {
        let vocab: [String: Int] = [
            "[CLS]": 0, "[SEP]": 1, "[UNK]": 2, "[PAD]": 3,
            "hi": 5,
        ]
        let tok = try WordPieceTokenizer(jsonData: makeTokenizerJSON(vocab: vocab))
        XCTAssertEqual(tok.eosId, 1)
        XCTAssertTrue(tok.stopTokenIds.contains(1))
    }
}
