import XCTest
@testable import DeepSeekVocabPruner
@testable import DeepSeekKit

/// Test della Fase 2 del pruner: VocabRewriter.
/// Costruisce mini-safetensors inline e verifica:
/// - slicing riga-wise di embed.weight,
/// - pass-through degli altri tensor,
/// - ricostruzione di tokenizer.json con added_tokens preservati.
final class VocabRewriterTests: XCTestCase {

    /// Helper: scrive un mini-safetensors con embed [vocab, dim]
    /// F32 + un tensor "other" F32 [4]. Restituisce l'URL + i byte
    /// di embed per il confronto downstream.
    private func writeMiniSafetensors(vocab: Int, dim: Int, otherDim: Int = 4)
        throws -> (url: URL, embedBytes: [Float], otherBytes: [Float])
    {
        // Build embed: floats deterministici per riga = oldId * 100 + dimIdx.
        var embed = [Float]()
        embed.reserveCapacity(vocab * dim)
        for v in 0..<vocab {
            for d in 0..<dim {
                embed.append(Float(v) * 100.0 + Float(d))
            }
        }
        var other = [Float]()
        other.reserveCapacity(otherDim)
        for i in 0..<otherDim {
            other.append(Float(i) - 0.5)
        }

        let embedData = embed.withUnsafeBufferPointer {
            Data(buffer: $0)
        }
        let otherData = other.withUnsafeBufferPointer {
            Data(buffer: $0)
        }

        let writer = SafeTensorsWriter()
        writer.add(name: "embed.weight",
                   dtype: "F32",
                   shape: [vocab, dim],
                   source: .data(embedData))
        writer.add(name: "other.weight",
                   dtype: "F32",
                   shape: [otherDim],
                   source: .data(otherData))

        let url = FileManager.default.temporaryDirectory
            .appendingPathComponent("mini-safetensors-\(UUID().uuidString).safetensors")
        try writer.write(to: url)
        return (url, embed, other)
    }

    /// Scrive un mini-tokenizer.json inline.
    private func writeMiniTokenizerJSON(vocab: Int, addedTokens: [(id: Int, content: String)])
        throws -> URL
    {
        var vocabDict: [String: Int] = [:]
        // ID = token string "t<i>" per leggibilità.
        for i in 0..<vocab {
            vocabDict["t\(i)"] = i
        }
        let added: [[String: Any]] = addedTokens.map {
            ["id": $0.id, "content": $0.content]
        }
        let root: [String: Any] = [
            "model": ["type": "BPE", "vocab": vocabDict, "merges": []],
            "added_tokens": added,
            "pre_tokenizer": ["type": "ByteLevel"],
        ]
        let data = try JSONSerialization.data(withJSONObject: root)
        let url = FileManager.default.temporaryDirectory
            .appendingPathComponent("mini-tok-\(UUID().uuidString).json")
        try data.write(to: url)
        return url
    }

    /// Scrive l'index.json minimo.
    private func writeMiniIndex(at url: URL, weightMap: [String: String]) throws {
        let obj: [String: Any] = [
            "metadata": ["total_size": 0],
            "weight_map": weightMap,
        ]
        let data = try JSONSerialization.data(withJSONObject: obj)
        try data.write(to: url)
    }

    /// Scrive il config.json minimo.
    private func writeMiniConfig(at url: URL, vocabSize: Int) throws {
        let obj: [String: Any] = ["vocab_size": vocabSize, "dim": 128]
        let data = try JSONSerialization.data(withJSONObject: obj)
        try data.write(to: url)
    }

    // MARK: - Tests

    func testEmbedSlicingPreservesCorrectRows() throws {
        let vocab = 10
        let dim = 4
        let inputDir = FileManager.default.temporaryDirectory
            .appendingPathComponent("vptest-in-\(UUID().uuidString)")
        let outputDir = FileManager.default.temporaryDirectory
            .appendingPathComponent("vptest-out-\(UUID().uuidString)")
        try FileManager.default.createDirectory(at: inputDir, withIntermediateDirectories: true)
        defer {
            try? FileManager.default.removeItem(at: inputDir)
            try? FileManager.default.removeItem(at: outputDir)
        }

        // Crea mini-safetensors con embed [10, 4].
        let (shardURL, embed, _) = try writeMiniSafetensors(vocab: vocab, dim: dim)
        let inputShard = inputDir.appendingPathComponent("model-00.safetensors")
        try FileManager.default.moveItem(at: shardURL, to: inputShard)

        // Crea index + config + tokenizer minimali.
        try writeMiniIndex(at: inputDir.appendingPathComponent("model.safetensors.index.json"),
                            weightMap: ["embed.weight": "model-00.safetensors",
                                        "other.weight": "model-00.safetensors"])
        try writeMiniConfig(at: inputDir.appendingPathComponent("config.json"),
                             vocabSize: vocab)
        let tokURL = try writeMiniTokenizerJSON(vocab: vocab, addedTokens: [])
        try FileManager.default.moveItem(at: tokURL,
                                          to: inputDir.appendingPathComponent("tokenizer.json"))

        // Decisione: keep {0, 3, 7}, remap → {0:0, 3:1, 7:2}.
        let decision = KeepDecision(
            keepIds: [0, 3, 7],
            oldToNew: [0: 0, 3: 1, 7: 2],
            newVocabSize: 3,
            oldVocabSize: vocab,
            coveragePct: 1.0)

        _ = try VocabRewriter.rewrite(
            inputDir: inputDir,
            outputDir: outputDir,
            decision: decision,
            onEvent: { _ in })

        // Verifica: l'output ha embed [3, 4] con righe esatte di
        // {oldId=0, oldId=3, oldId=7}.
        let outShard = outputDir.appendingPathComponent("model-00.safetensors")
        let outFile = try SafeTensorsFile(url: outShard)
        guard let embedEntry = outFile.entries["embed.weight"] else {
            XCTFail("output missing embed.weight")
            return
        }
        XCTAssertEqual(embedEntry.shape, [3, 4])

        // Leggi le righe dal file di output.
        let dataStart = try readDataStart(outShard)
        let fh = try FileHandle(forReadingFrom: outShard)
        defer { try? fh.close() }
        try fh.seek(toOffset: UInt64(dataStart + embedEntry.dataOffsets[0]))
        let rowBytes = dim * MemoryLayout<Float>.size
        let bytes = try fh.read(upToCount: 3 * rowBytes)!
        let outFloats: [Float] = bytes.withUnsafeBytes { raw in
            let ptr = raw.bindMemory(to: Float.self).baseAddress!
            return Array(UnsafeBufferPointer(start: ptr, count: 3 * dim))
        }

        // Atteso: riga 0 = embed[0..4], riga 1 = embed[12..16] (oldId=3),
        // riga 2 = embed[28..32] (oldId=7).
        for d in 0..<dim {
            XCTAssertEqual(outFloats[0 * dim + d], embed[0 * dim + d], accuracy: 1e-5)
            XCTAssertEqual(outFloats[1 * dim + d], embed[3 * dim + d], accuracy: 1e-5)
            XCTAssertEqual(outFloats[2 * dim + d], embed[7 * dim + d], accuracy: 1e-5)
        }
    }

    func testAddedTokenIDIsPreservedInTokenizerJSON() throws {
        let inputDir = FileManager.default.temporaryDirectory
            .appendingPathComponent("vptok-in-\(UUID().uuidString)")
        let outputDir = FileManager.default.temporaryDirectory
            .appendingPathComponent("vptok-out-\(UUID().uuidString)")
        try FileManager.default.createDirectory(at: inputDir, withIntermediateDirectories: true)
        defer {
            try? FileManager.default.removeItem(at: inputDir)
            try? FileManager.default.removeItem(at: outputDir)
        }

        // Vocab = 1000, special token id=999.
        let tokURL = try writeMiniTokenizerJSON(
            vocab: 1000,
            addedTokens: [(id: 999, content: "<spec>")])
        try FileManager.default.moveItem(at: tokURL,
                                          to: inputDir.appendingPathComponent("tokenizer.json"))

        // Decisione: keep solo {5, 999}; remap → {5: 0, 999: 999}.
        let decision = KeepDecision(
            keepIds: [5, 999],
            oldToNew: [5: 0, 999: 999],
            newVocabSize: 1000,    // 999 + 1 (special preserva il suo ID alto)
            oldVocabSize: 1000,
            coveragePct: 1.0)

        try FileManager.default.createDirectory(at: outputDir, withIntermediateDirectories: true)
        try VocabRewriter.rewriteTokenizerJSON(
            inputDir: inputDir,
            outputDir: outputDir,
            decision: decision)

        let outURL = outputDir.appendingPathComponent("tokenizer.json")
        let data = try Data(contentsOf: outURL)
        let root = try JSONSerialization.jsonObject(with: data) as! [String: Any]
        let added = root["added_tokens"] as! [[String: Any]]
        XCTAssertEqual(added.count, 1)
        let entry = added[0]
        XCTAssertEqual(entry["id"] as? Int, 999,
                       "added_tokens id deve essere preservato")
        XCTAssertEqual(entry["content"] as? String, "<spec>")

        // E nel model.vocab: "t5" deve aver newId=0.
        let model = root["model"] as! [String: Any]
        let vocab = model["vocab"] as! [String: Int]
        XCTAssertEqual(vocab["t5"], 0)
        XCTAssertNil(vocab["t10"], "t10 (oldId=10) NON nei keep_ids")
    }

    func testIsVocabTensorClassification() {
        XCTAssertTrue(VocabRewriter.isVocabTensor("embed.weight"))
        XCTAssertTrue(VocabRewriter.isVocabTensor("head.weight"))
        XCTAssertTrue(VocabRewriter.isVocabTensor("mtp.0.embed.weight"))
        XCTAssertTrue(VocabRewriter.isVocabTensor("mtp.3.head.weight"))
        XCTAssertFalse(VocabRewriter.isVocabTensor("layers.0.attn.wq.weight"))
        XCTAssertFalse(VocabRewriter.isVocabTensor("norm.weight"))
    }

    func testIdempotencyRefusesInputEqualsOutput() {
        let dir = FileManager.default.temporaryDirectory
            .appendingPathComponent("vp-idem-\(UUID().uuidString)")
        try? FileManager.default.createDirectory(at: dir, withIntermediateDirectories: true)
        defer { try? FileManager.default.removeItem(at: dir) }

        let decision = KeepDecision(
            keepIds: [], oldToNew: [:], newVocabSize: 0,
            oldVocabSize: 0, coveragePct: 0)
        XCTAssertThrowsError(
            try VocabRewriter.rewrite(inputDir: dir, outputDir: dir,
                                       decision: decision, onEvent: { _ in })
        )
    }

    // MARK: - Helpers

    private func readDataStart(_ url: URL) throws -> Int {
        let fh = try FileHandle(forReadingFrom: url)
        defer { try? fh.close() }
        let head = try fh.read(upToCount: 8)!
        let headerLen = head.withUnsafeBytes { raw -> UInt64 in
            raw.load(as: UInt64.self).littleEndian
        }
        return 8 + Int(headerLen)
    }
}
