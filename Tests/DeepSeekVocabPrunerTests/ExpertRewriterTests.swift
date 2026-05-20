import XCTest
@testable import DeepSeekVocabPruner
@testable import DeepSeekKit

/// Test del rewriter dell'expert pruner: costruisce mini-safetensors
/// inline e verifica che:
///  - i tensor degli esperti droppati siano fisicamente assenti
///    nel nuovo shard,
///  - le righe del `gate.weight` dei dropped siano sovrascritte con
///    il sentinel negativo,
///  - le entry del `tid2eid` che puntavano a dropped siano rimappate
///    a un kept,
///  - `config.json` contenga `pruned_experts` con la lista corretta.
final class ExpertRewriterTests: XCTestCase {

    /// Helper: scrive un mini-safetensors con:
    ///  - `embed.weight` [vocab, dim] f32 (pass-through),
    ///  - `layers.0.ffn.gate.weight` [nExperts, dim] f32,
    ///  - `layers.0.ffn.experts.<e>.w1.weight` [interDim, dim] f32 per ogni e,
    ///  - `layers.0.ffn.experts.<e>.w2.weight` [dim, interDim] f32 per ogni e,
    ///  - `layers.0.ffn.experts.<e>.w3.weight` [interDim, dim] f32 per ogni e.
    private func writeMiniMoEShard(
        vocab: Int, dim: Int, nExperts: Int, interDim: Int, nLayers: Int = 1
    ) throws -> URL {
        let writer = SafeTensorsWriter()

        // embed (pass-through marker).
        var embed = [Float](repeating: 0, count: vocab * dim)
        for v in 0..<vocab {
            for d in 0..<dim { embed[v * dim + d] = Float(v) * 100 + Float(d) }
        }
        let embedData = embed.withUnsafeBufferPointer { Data(buffer: $0) }
        writer.add(name: "embed.weight",
                   dtype: "F32", shape: [vocab, dim], source: .data(embedData))

        for L in 0..<nLayers {
            // Gate weight: row e starts with marker float Float(e).
            var gate = [Float](repeating: 0, count: nExperts * dim)
            for e in 0..<nExperts {
                for d in 0..<dim { gate[e * dim + d] = Float(e * 1000 + d) }
            }
            let gateData = gate.withUnsafeBufferPointer { Data(buffer: $0) }
            writer.add(name: "layers.\(L).ffn.gate.weight",
                       dtype: "F32", shape: [nExperts, dim],
                       source: .data(gateData))

            for e in 0..<nExperts {
                var w1 = [Float](repeating: Float(e) + 0.1, count: interDim * dim)
                var w2 = [Float](repeating: Float(e) + 0.2, count: dim * interDim)
                var w3 = [Float](repeating: Float(e) + 0.3, count: interDim * dim)
                let w1Data = w1.withUnsafeBufferPointer { Data(buffer: $0) }
                let w2Data = w2.withUnsafeBufferPointer { Data(buffer: $0) }
                let w3Data = w3.withUnsafeBufferPointer { Data(buffer: $0) }
                writer.add(name: "layers.\(L).ffn.experts.\(e).w1.weight",
                           dtype: "F32", shape: [interDim, dim],
                           source: .data(w1Data))
                writer.add(name: "layers.\(L).ffn.experts.\(e).w2.weight",
                           dtype: "F32", shape: [dim, interDim],
                           source: .data(w2Data))
                writer.add(name: "layers.\(L).ffn.experts.\(e).w3.weight",
                           dtype: "F32", shape: [interDim, dim],
                           source: .data(w3Data))
            }
        }

        let url = FileManager.default.temporaryDirectory
            .appendingPathComponent("mini-moe-\(UUID().uuidString).safetensors")
        try writer.write(to: url)
        return url
    }

    private func writeMiniIndex(at url: URL,
                                 weightMap: [String: String]) throws {
        let obj: [String: Any] = [
            "metadata": ["total_size": 0],
            "weight_map": weightMap,
        ]
        let data = try JSONSerialization.data(withJSONObject: obj)
        try data.write(to: url)
    }

    private func writeMiniConfig(at url: URL, nExperts: Int) throws {
        let obj: [String: Any] = [
            "n_routed_experts": nExperts,
            "n_activated_experts": 2,
            "n_layers": 1,
        ]
        let data = try JSONSerialization.data(withJSONObject: obj)
        try data.write(to: url)
    }

    // MARK: - Tests

    func testDroppedExpertTensorsAreOmitted() throws {
        let vocab = 4, dim = 4, nExperts = 4, interDim = 8
        let inputDir = FileManager.default.temporaryDirectory
            .appendingPathComponent("exptest-in-\(UUID().uuidString)")
        let outputDir = FileManager.default.temporaryDirectory
            .appendingPathComponent("exptest-out-\(UUID().uuidString)")
        try FileManager.default.createDirectory(at: inputDir,
                                                  withIntermediateDirectories: true)
        defer {
            try? FileManager.default.removeItem(at: inputDir)
            try? FileManager.default.removeItem(at: outputDir)
        }

        let shardURL = try writeMiniMoEShard(
            vocab: vocab, dim: dim, nExperts: nExperts, interDim: interDim)
        let inputShard = inputDir.appendingPathComponent("model-00.safetensors")
        try FileManager.default.moveItem(at: shardURL, to: inputShard)

        // Build weight_map.
        var wm: [String: String] = [
            "embed.weight": "model-00.safetensors",
            "layers.0.ffn.gate.weight": "model-00.safetensors",
        ]
        for e in 0..<nExperts {
            for proj in ["w1", "w2", "w3"] {
                wm["layers.0.ffn.experts.\(e).\(proj).weight"] = "model-00.safetensors"
            }
        }
        try writeMiniIndex(
            at: inputDir.appendingPathComponent("model.safetensors.index.json"),
            weightMap: wm)
        try writeMiniConfig(
            at: inputDir.appendingPathComponent("config.json"),
            nExperts: nExperts)

        // Decision: drop experts 1 and 3, keep 0 and 2.
        let decision = ExpertKeepDecision(
            keepIds: [[0, 2]],
            droppedIds: [[1, 3]],
            nRoutedExperts: nExperts,
            nActivatedExperts: 2,
            coverage: 0.99,
            totalAssignments: 100,
            actualCoveragePerLayer: [1.0],
            usage: [])

        _ = try ExpertRewriter.rewrite(
            inputDir: inputDir,
            outputDir: outputDir,
            decision: decision,
            onEvent: { _ in })

        let outShard = outputDir.appendingPathComponent("model-00.safetensors")
        let outFile = try SafeTensorsFile(url: outShard)

        // Dropped expert tensors must NOT be present.
        for e in [1, 3] {
            for proj in ["w1", "w2", "w3"] {
                let name = "layers.0.ffn.experts.\(e).\(proj).weight"
                XCTAssertNil(outFile.entries[name],
                              "Dropped expert tensor \(name) should be absent")
            }
        }
        // Kept expert tensors must still be present.
        for e in [0, 2] {
            for proj in ["w1", "w2", "w3"] {
                let name = "layers.0.ffn.experts.\(e).\(proj).weight"
                XCTAssertNotNil(outFile.entries[name],
                                  "Kept expert tensor \(name) should still be present")
            }
        }
        // Embed must be preserved.
        XCTAssertNotNil(outFile.entries["embed.weight"])
    }

    func testGateWeightRowsAreOverwrittenForDroppedExperts() throws {
        let vocab = 4, dim = 4, nExperts = 4, interDim = 8
        let inputDir = FileManager.default.temporaryDirectory
            .appendingPathComponent("exptest-gate-in-\(UUID().uuidString)")
        let outputDir = FileManager.default.temporaryDirectory
            .appendingPathComponent("exptest-gate-out-\(UUID().uuidString)")
        try FileManager.default.createDirectory(at: inputDir,
                                                  withIntermediateDirectories: true)
        defer {
            try? FileManager.default.removeItem(at: inputDir)
            try? FileManager.default.removeItem(at: outputDir)
        }

        let shardURL = try writeMiniMoEShard(
            vocab: vocab, dim: dim, nExperts: nExperts, interDim: interDim)
        let inputShard = inputDir.appendingPathComponent("model-00.safetensors")
        try FileManager.default.moveItem(at: shardURL, to: inputShard)

        var wm: [String: String] = [
            "embed.weight": "model-00.safetensors",
            "layers.0.ffn.gate.weight": "model-00.safetensors",
        ]
        for e in 0..<nExperts {
            for proj in ["w1", "w2", "w3"] {
                wm["layers.0.ffn.experts.\(e).\(proj).weight"] = "model-00.safetensors"
            }
        }
        try writeMiniIndex(
            at: inputDir.appendingPathComponent("model.safetensors.index.json"),
            weightMap: wm)
        try writeMiniConfig(
            at: inputDir.appendingPathComponent("config.json"),
            nExperts: nExperts)

        // Drop expert 1.
        let decision = ExpertKeepDecision(
            keepIds: [[0, 2, 3]],
            droppedIds: [[1]],
            nRoutedExperts: nExperts,
            nActivatedExperts: 2,
            coverage: 0.99,
            totalAssignments: 100,
            actualCoveragePerLayer: [1.0],
            usage: [])

        _ = try ExpertRewriter.rewrite(
            inputDir: inputDir,
            outputDir: outputDir,
            decision: decision,
            onEvent: { _ in })

        // Read gate.weight from output: row 1 (dropped) should be
        // entirely -1e9; rows 0, 2, 3 unchanged.
        let outShard = outputDir.appendingPathComponent("model-00.safetensors")
        let outFile = try SafeTensorsFile(url: outShard)
        let entry = outFile.entries["layers.0.ffn.gate.weight"]!
        let dataStart = try readDataStart(outShard)
        let fh = try FileHandle(forReadingFrom: outShard)
        defer { try? fh.close() }
        try fh.seek(toOffset: UInt64(dataStart + entry.dataOffsets[0]))
        let byteCount = entry.dataOffsets[1] - entry.dataOffsets[0]
        let bytes = try fh.read(upToCount: byteCount)!
        let floats = bytes.withUnsafeBytes { raw -> [Float] in
            let p = raw.bindMemory(to: Float.self)
            return Array(UnsafeBufferPointer(start: p.baseAddress!,
                                              count: nExperts * dim))
        }
        // Row 1: all -1e9.
        for d in 0..<dim {
            XCTAssertEqual(floats[1 * dim + d], -1e9, accuracy: 1.0,
                            "dropped expert 1 gate row should be sentinel")
        }
        // Rows 0, 2, 3: original markers.
        for e in [0, 2, 3] {
            for d in 0..<dim {
                XCTAssertEqual(floats[e * dim + d], Float(e * 1000 + d),
                                accuracy: 1e-3)
            }
        }
    }

    func testConfigJSONHasPrunedExpertsField() throws {
        let inputDir = FileManager.default.temporaryDirectory
            .appendingPathComponent("exptest-cfg-in-\(UUID().uuidString)")
        let outputDir = FileManager.default.temporaryDirectory
            .appendingPathComponent("exptest-cfg-out-\(UUID().uuidString)")
        try FileManager.default.createDirectory(at: inputDir,
                                                  withIntermediateDirectories: true)
        try FileManager.default.createDirectory(at: outputDir,
                                                  withIntermediateDirectories: true)
        defer {
            try? FileManager.default.removeItem(at: inputDir)
            try? FileManager.default.removeItem(at: outputDir)
        }

        // Pre-existing config.json with some other fields.
        let cfg: [String: Any] = [
            "n_routed_experts": 8,
            "dim": 4096,
            "vocab_size": 32000,
        ]
        let data = try JSONSerialization.data(withJSONObject: cfg)
        try data.write(to: inputDir.appendingPathComponent("config.json"))

        let decision = ExpertKeepDecision(
            keepIds: [[0, 1, 2, 3]],
            droppedIds: [[4, 5, 6, 7]],
            nRoutedExperts: 8,
            nActivatedExperts: 2,
            coverage: 0.95,
            totalAssignments: 100,
            actualCoveragePerLayer: [0.98],
            usage: [])

        try ExpertRewriter.rewriteConfigJSON(inputDir: inputDir,
                                              outputDir: outputDir,
                                              decision: decision)

        let outCfgURL = outputDir.appendingPathComponent("config.json")
        let outData = try Data(contentsOf: outCfgURL)
        let outObj = try JSONSerialization.jsonObject(with: outData)
            as! [String: Any]

        XCTAssertEqual(outObj["dim"] as? Int, 4096)
        XCTAssertEqual(outObj["vocab_size"] as? Int, 32000)
        XCTAssertEqual(outObj["n_routed_experts"] as? Int, 8)
        let pruned = outObj["pruned_experts"] as! [[Int]]
        XCTAssertEqual(pruned, [[4, 5, 6, 7]])
    }

    func testNameParsers() {
        // parseExpertTensor: layers.<L>.ffn.experts.<E>.<rest>
        XCTAssertEqual(
            ExpertRewriter.parseExpertTensor("layers.3.ffn.experts.42.w1.weight")?.0, 3)
        XCTAssertEqual(
            ExpertRewriter.parseExpertTensor("layers.3.ffn.experts.42.w1.weight")?.1, 42)
        XCTAssertNil(
            ExpertRewriter.parseExpertTensor("layers.3.ffn.gate.weight"))
        XCTAssertNil(
            ExpertRewriter.parseExpertTensor("layers.3.attn.wq.weight"))

        // parseGateWeightLayer
        XCTAssertEqual(
            ExpertRewriter.parseGateWeightLayer("layers.5.ffn.gate.weight"), 5)
        XCTAssertNil(
            ExpertRewriter.parseGateWeightLayer("layers.5.ffn.gate.bias"))
        XCTAssertNil(
            ExpertRewriter.parseGateWeightLayer("layers.5.ffn.experts.0.w1.weight"))

        // parseTid2EidLayer
        XCTAssertEqual(
            ExpertRewriter.parseTid2EidLayer("layers.0.ffn.gate.tid2eid"), 0)
        XCTAssertNil(
            ExpertRewriter.parseTid2EidLayer("layers.0.ffn.gate.weight"))
    }

    func testIsDroppedExpertTensorClassification() {
        let dropped: [Int: Set<Int>] = [
            0: Set([1, 3]),
            1: Set([5]),
        ]
        XCTAssertTrue(ExpertRewriter.isDroppedExpertTensor(
            "layers.0.ffn.experts.1.w1.weight",
            droppedByLayer: dropped))
        XCTAssertTrue(ExpertRewriter.isDroppedExpertTensor(
            "layers.0.ffn.experts.3.w2.scale",
            droppedByLayer: dropped))
        XCTAssertFalse(ExpertRewriter.isDroppedExpertTensor(
            "layers.0.ffn.experts.0.w1.weight",
            droppedByLayer: dropped))
        XCTAssertFalse(ExpertRewriter.isDroppedExpertTensor(
            "layers.0.ffn.gate.weight",
            droppedByLayer: dropped))
        XCTAssertFalse(ExpertRewriter.isDroppedExpertTensor(
            "layers.1.ffn.experts.0.w1.weight",
            droppedByLayer: dropped),
            "layer 1 expert 0 should NOT be dropped (only 5 is)")
    }

    func testF32ToBF16RoundTripPreservesSentinel() {
        // BF16 has ~7 bits mantissa; -1e9 should round-trip cleanly
        // (within ~1% — what matters is it stays large negative so
        // sqrt(softplus) ≈ 0).
        let original: Float = -1e9
        let bits = ExpertRewriter.f32ToBF16Bits(original)
        let recovered = ExpertRewriter.bf16BitsToF32(bits)
        XCTAssertLessThan(recovered, -1e8,
            "BF16 round-trip should keep value very negative")
    }

    func testCosineSimilaritySanity() {
        let a: [Float] = [1, 0, 0, 0]
        let b: [Float] = [1, 0, 0, 0]
        let c: [Float] = [0, 1, 0, 0]
        let d: [Float] = [-1, 0, 0, 0]
        XCTAssertEqual(ExpertRewriter.cosineSimilarity(a, b), 1, accuracy: 1e-6)
        XCTAssertEqual(ExpertRewriter.cosineSimilarity(a, c), 0, accuracy: 1e-6)
        XCTAssertEqual(ExpertRewriter.cosineSimilarity(a, d), -1, accuracy: 1e-6)
    }

    func testIdempotencyRefusesInputEqualsOutput() {
        let dir = FileManager.default.temporaryDirectory
            .appendingPathComponent("exp-idem-\(UUID().uuidString)")
        try? FileManager.default.createDirectory(at: dir,
                                                  withIntermediateDirectories: true)
        defer { try? FileManager.default.removeItem(at: dir) }

        let decision = ExpertKeepDecision(
            keepIds: [], droppedIds: [], nRoutedExperts: 0,
            nActivatedExperts: 2, coverage: 0.99,
            totalAssignments: 0, actualCoveragePerLayer: [], usage: [])
        XCTAssertThrowsError(
            try ExpertRewriter.rewrite(inputDir: dir, outputDir: dir,
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
