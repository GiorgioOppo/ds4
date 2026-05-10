import XCTest
@testable import DeepSeekKit

final class BPETokenizerTests: XCTestCase {

    /// Build a minimal HF-format tokenizer.json in memory: a tiny vocab with a
    /// single merge rule, the standard ByteLevel preset, and one special token.
    /// This lets us exercise the encode/decode pipeline without depending on
    /// the production tokenizer.json.
    func testEncodeDecodeRoundTripOnMiniVocab() throws {
        let json: [String: Any] = [
            "model": [
                "type": "BPE",
                "vocab": miniVocab(),
                "merges": ["h e", "he l", "hel lo"]
            ],
            "added_tokens": [
                ["id": 999, "content": "<eos>"]
            ]
        ]
        let data = try JSONSerialization.data(withJSONObject: json)
        let tok = try BPETokenizer(jsonData: data)

        let ids = tok.encode("hello")
        // The single word "hello" should consume the merges to become a single token.
        // (Exact id depends on vocab ordering; we assert round-trip instead.)
        XCTAssertFalse(ids.isEmpty)
        let back = tok.decode(ids)
        XCTAssertEqual(back, "hello")
    }

    /// Special tokens (`added_tokens`) are emitted literally and decode back exactly.
    func testSpecialTokenPassthrough() throws {
        let json: [String: Any] = [
            "model": [
                "type": "BPE",
                "vocab": miniVocab(),
                "merges": []
            ],
            "added_tokens": [
                ["id": 1000, "content": "<bos>"],
                ["id": 1001, "content": "<eos>"]
            ]
        ]
        let data = try JSONSerialization.data(withJSONObject: json)
        let tok = try BPETokenizer(jsonData: data)
        let ids = tok.encode("<bos>hello<eos>")
        XCTAssertTrue(ids.contains(1000), "bos id must be present")
        XCTAssertTrue(ids.contains(1001), "eos id must be present")
        let back = tok.decode(ids)
        XCTAssertEqual(back, "<bos>hello<eos>")
    }

    /// Round-trip arbitrary UTF-8 (bytes outside the printable ASCII range get
    /// remapped through the GPT-2 byte-to-unicode table).
    func testEncodeDecodeUTF8() throws {
        // vocab contains every single char so encode degenerates to one id per char.
        let vocab = makeAllSingleByteVocab()
        let json: [String: Any] = [
            "model": ["type": "BPE", "vocab": vocab, "merges": []]
        ]
        let data = try JSONSerialization.data(withJSONObject: json)
        let tok = try BPETokenizer(jsonData: data)

        let inputs = ["ciao", "héllo", "こんにちは", "🐢"]
        for s in inputs {
            let back = tok.decode(tok.encode(s))
            XCTAssertEqual(back, s, "round-trip failed for \(s)")
        }
    }

    private func miniVocab() -> [String: Int] {
        // Bytes 32..126 mapped to printable chars; "h", "e", "l", "o" plus merges.
        var v: [String: Int] = [:]
        var id = 0
        for b in (32 as UInt8)...(126 as UInt8) {
            v[String(UnicodeScalar(b))] = id; id += 1
        }
        v["he"] = id; id += 1
        v["hel"] = id; id += 1
        v["hello"] = id; id += 1
        return v
    }

    /// Build a vocab containing every byte mapped through the GPT-2 byte-to-unicode
    /// table so any UTF-8 string can be tokenized one byte at a time.
    private func makeAllSingleByteVocab() -> [String: Int] {
        var bs: [UInt8] = []
        bs.append(contentsOf: 33...126)
        bs.append(contentsOf: 161...172)
        bs.append(contentsOf: 174...255)
        var cs: [UInt32] = bs.map { UInt32($0) }
        var n: UInt32 = 0
        for b in 0...255 {
            if !bs.contains(UInt8(b)) { bs.append(UInt8(b)); cs.append(256 + n); n += 1 }
        }
        var v: [String: Int] = [:]
        for (i, c) in cs.enumerated() {
            v[String(UnicodeScalar(c)!)] = i
        }
        return v
    }
}
