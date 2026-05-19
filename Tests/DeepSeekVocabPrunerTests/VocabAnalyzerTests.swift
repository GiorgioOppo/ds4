import XCTest
@testable import DeepSeekVocabPruner
@testable import DeepSeekKit

/// Test della Fase 1 del pruner: VocabAnalyzer.
/// Costruisce mini-tokenizer + mini-corpus inline (no fixture su
/// disco) e verifica che keep_ids contenga gli ID giusti.
final class VocabAnalyzerTests: XCTestCase {

    /// Mini-tokenizer con vocab inline: 5 byte-base + 4 token italiani
    /// + 3 special. Niente merges (vocab "flat" — l'encoder usa solo
    /// vocab lookup per ogni char).
    /// I 256 byte-level chars sono iniettati come keep forced.
    private func makeMiniTokenizerJSON(extraVocab: [String: Int] = [:],
                                        addedTokens: [(id: Int, content: String)] = []
                                        ) throws -> URL {
        var vocab: [String: Int] = [:]
        let (b2u, _) = VocabAnalyzer.byteUnicodeMaps()
        // ID 0..255 = byte-level base.
        for b in 0...0xFF {
            vocab[b2u[UInt8(b)]!] = b
        }
        var nextId = 256
        for (k, _) in extraVocab.sorted(by: { $0.key < $1.key }) {
            vocab[k] = nextId
            nextId += 1
        }
        // Override IDs from extraVocab (only if user wants specific IDs).
        for (k, v) in extraVocab { vocab[k] = v }

        let added: [[String: Any]] = addedTokens.map {
            ["id": $0.id, "content": $0.content]
        }
        // pre_tokenizer minimale (ByteLevel pattern di default).
        let root: [String: Any] = [
            "model": ["type": "BPE", "vocab": vocab, "merges": []],
            "added_tokens": added,
            "pre_tokenizer": ["type": "ByteLevel"],
        ]
        let data = try JSONSerialization.data(withJSONObject: root)
        let url = FileManager.default.temporaryDirectory
            .appendingPathComponent("mini-tokenizer-\(UUID().uuidString).json")
        try data.write(to: url)
        return url
    }

    private func writeCorpus(_ lines: [String]) throws -> URL {
        let url = FileManager.default.temporaryDirectory
            .appendingPathComponent("mini-corpus-\(UUID().uuidString).txt")
        try lines.joined(separator: "\n").write(to: url, atomically: true, encoding: .utf8)
        return url
    }

    // MARK: - Tests

    func testBuildRemapPreservesAddedTokenIds() {
        // keep = {3, 5, 999}; preserve = {999} (special token).
        // → 3 → 0, 5 → 1, 999 → 999.
        let keep = Set([3, 5, 999])
        let preserve = Set([999])
        let map = VocabAnalyzer.buildRemap(keep: keep,
                                            preserveIds: preserve,
                                            totalVocab: 1000)
        XCTAssertEqual(map[999], 999, "special token ID must be preserved")
        XCTAssertEqual(map[3], 0)
        XCTAssertEqual(map[5], 1)
        XCTAssertEqual(map.count, 3)
    }

    func testBuildRemapSkipsPreservedSlots() {
        // keep = {0, 1, 2, 3, 4, 5}; preserve = {2}.
        // → 0 → 0, 1 → 1, 3 → 3, 4 → 4, 5 → 5; 2 → 2.
        // slot "2" è riservato all'ID 2; gli altri ID si riallocano
        // saltando lo slot 2.
        let keep = Set([0, 1, 2, 3, 4, 5])
        let preserve = Set([2])
        let map = VocabAnalyzer.buildRemap(keep: keep,
                                            preserveIds: preserve,
                                            totalVocab: 10)
        XCTAssertEqual(map[2], 2)
        // I non-preserved si distribuiscono nei primi slot liberi:
        // 0→0, 1→1, (skip 2), 3→3, 4→4, 5→5.
        XCTAssertEqual(map[0], 0)
        XCTAssertEqual(map[1], 1)
        XCTAssertEqual(map[3], 3)
        XCTAssertEqual(map[4], 4)
        XCTAssertEqual(map[5], 5)
    }

    func testIsAllLatinOrAscii() {
        XCTAssertTrue(VocabAnalyzer.isAllLatinOrAscii("hello"))
        XCTAssertTrue(VocabAnalyzer.isAllLatinOrAscii("perché"))
        XCTAssertTrue(VocabAnalyzer.isAllLatinOrAscii("naïve"))
        XCTAssertTrue(VocabAnalyzer.isAllLatinOrAscii(""))   // vacuously true
        XCTAssertFalse(VocabAnalyzer.isAllLatinOrAscii("こんにちは"))  // hiragana
        XCTAssertFalse(VocabAnalyzer.isAllLatinOrAscii("你好"))      // CJK
        XCTAssertFalse(VocabAnalyzer.isAllLatinOrAscii("ciao 你"))   // misto
    }

    func testContainsForeignScript() {
        XCTAssertFalse(VocabAnalyzer.containsForeignScript("hello world"))
        XCTAssertFalse(VocabAnalyzer.containsForeignScript("àèìòù"))
        // CJK Unified Ideographs (U+4E00)
        XCTAssertTrue(VocabAnalyzer.containsForeignScript("一"))
        // Hiragana (U+3042)
        XCTAssertTrue(VocabAnalyzer.containsForeignScript("あ"))
        // Hangul (U+AC00)
        XCTAssertTrue(VocabAnalyzer.containsForeignScript("가"))
        // Arabic (U+0627)
        XCTAssertTrue(VocabAnalyzer.containsForeignScript("ا"))
        // Hebrew (U+05D0)
        XCTAssertTrue(VocabAnalyzer.containsForeignScript("א"))
        // Devanagari (U+0905)
        XCTAssertTrue(VocabAnalyzer.containsForeignScript("अ"))
    }

    func testAnalyzeBasicCoverage() async throws {
        // Vocab: 256 byte-base + "ciao" id 300 + "mondo" id 301 +
        //        token CJK fittizio "noise" id 302 + special id 999.
        let tokURL = try makeMiniTokenizerJSON(
            extraVocab: ["ciao": 300, "mondo": 301, "noise": 302],
            addedTokens: [(id: 999, content: "<spec>")])
        defer { try? FileManager.default.removeItem(at: tokURL) }

        // Corpus: solo "ciao" e "mondo". "noise" non appare.
        let corpusURL = try writeCorpus(Array(repeating: "ciao mondo", count: 100))
        defer { try? FileManager.default.removeItem(at: corpusURL) }

        var events: [VocabPruneEvent] = []
        let decision = try await VocabAnalyzer.analyze(
            tokenizerJSON: tokURL,
            corpus: corpusURL,
            coverage: 0.99,
            onEvent: { events.append($0) })

        // Special token sopravvive col suo ID originale.
        XCTAssertEqual(decision.oldToNew[999], 999)
        // I 256 byte-base sono sempre keep.
        let keepSet = Set(decision.keepIds)
        for b in 0...0xFF {
            XCTAssertTrue(keepSet.contains(b),
                          "byte-base id \(b) deve essere keep")
        }
        // Coverage event emesso.
        let hasCoverage = events.contains { if case .coverage = $0 { return true } else { return false } }
        XCTAssertTrue(hasCoverage)
    }
}
