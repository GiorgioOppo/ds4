import XCTest
@testable import DS4Core
@testable import DS4Metal

/// Round-trips the pure-Swift GGUFWriter against the GGUFModel reader: anything
/// written must parse back to identical metadata, tensor shapes and bytes. This
/// is the reader's inverse, so a passing round-trip pins the on-disk layout.
final class GGUFWriterTests: XCTestCase {

    private func tempURL() -> URL {
        FileManager.default.temporaryDirectory
            .appendingPathComponent("ggufw-\(UUID().uuidString).gguf")
    }

    func testRoundTripMetadataAndTensors() throws {
        // Two tiny f32 tensors with known bytes.
        let aVals: [Float] = [1, 2, 3, 4]
        let bVals: [Float] = [10, 20, 30, 40, 50, 60]
        func f32Data(_ xs: [Float]) -> Data {
            var d = Data()
            for x in xs { withUnsafeBytes(of: x.bitPattern.littleEndian) { d.append(contentsOf: $0) } }
            return d
        }

        var w = try GGUFWriter(alignment: 32)
        w.put("general.alignment", .uint32(32))
        w.put("general.name", .text("tiny"))
        w.put("general.architecture", .text("deepseek4"))
        w.put("deepseek4.block_count", .uint32(2))
        w.put("test.u64", .uint64(123_456_789))
        w.put("test.i32", .int32(-7))
        w.put("test.bool", .bool(true))
        w.put("test.f32", .float32(1.5))
        w.put("test.arr", .array(elementType: .int32,
                                 elements: [.int32(5), .int32(-6), .int32(7)]))
        w.add(.init(name: "a.weight", dims: [4], type: 0, data: f32Data(aVals)))
        w.add(.init(name: "b.weight", dims: [2, 3], type: 0, data: f32Data(bVals)))

        let url = tempURL()
        try w.write(to: url.path)
        defer { try? FileManager.default.removeItem(at: url) }

        let m = try GGUFModel(path: url.path, metalMapping: false)
        XCTAssertEqual(m.version, 3)
        XCTAssertEqual(m.n_kv, 9)
        XCTAssertEqual(m.n_tensors, 2)
        XCTAssertEqual(m.alignment, 32)

        XCTAssertEqual(m.string("general.name"), "tiny")
        XCTAssertEqual(m.string("general.architecture"), "deepseek4")
        XCTAssertEqual(m.u32("deepseek4.block_count"), 2)
        XCTAssertEqual(m.u64("test.u64"), 123_456_789)
        XCTAssertEqual(m.bool("test.bool"), true)
        XCTAssertEqual(m.f32Compat("test.f32"), 1.5)
        XCTAssertEqual(m.intArray("test.arr"), [5, -6, 7])

        let a = try XCTUnwrap(m.findTensor("a.weight"))
        XCTAssertEqual(a.dims, [4]); XCTAssertEqual(a.typeName, "f32"); XCTAssertEqual(a.bytes, 16)
        XCTAssertEqual(m.tensorData(a), f32Data(aVals))

        let b = try XCTUnwrap(m.findTensor("b.weight"))
        XCTAssertEqual(b.dims, [2, 3]); XCTAssertEqual(b.bytes, 24)
        XCTAssertEqual(m.tensorData(b), f32Data(bVals))

        // Each tensor's data offset is alignment-aligned.
        XCTAssertEqual(a.absOffset % 32, 0)
        XCTAssertEqual(b.absOffset % 32, 0)
    }

    /// build() (single Data) and write(to:) (streamed) must be byte-identical.
    func testBuildMatchesStreamedWrite() throws {
        var w = try GGUFWriter()
        w.put("general.name", .text("x"))
        w.add(.init(name: "t", dims: [8], type: 0, data: Data(count: 32)))
        let built = try w.build()
        let url = tempURL()
        try w.write(to: url.path)
        defer { try? FileManager.default.removeItem(at: url) }
        XCTAssertEqual(built, try Data(contentsOf: url))
    }

    /// Reading a file and re-writing its metadata + tensors reproduces a file the
    /// reader parses identically (the requantizer's read->write backbone).
    func testReadReExportRoundTrip() throws {
        var w = try GGUFWriter()
        w.put("general.alignment", .uint32(32))
        w.put("general.architecture", .text("deepseek4"))
        w.put("tokens", .array(elementType: .string,
                               elements: [.text("a"), .text("bb"), .text("ccc")]))
        w.add(.init(name: "w0", dims: [16], type: 0, data: Data(count: 64)))
        let src = tempURL()
        try w.write(to: src.path)
        defer { try? FileManager.default.removeItem(at: src) }

        let m0 = try GGUFModel(path: src.path, metalMapping: false)
        let meta = try m0.allMetadata()
        let tensors = m0.tensors.map { m0.tensorInputPassthrough($0) }
        let w2 = try GGUFWriter(metadata: meta, tensors: tensors)

        let dst = tempURL()
        try w2.write(to: dst.path)
        defer { try? FileManager.default.removeItem(at: dst) }

        let m1 = try GGUFModel(path: dst.path, metalMapping: false)
        XCTAssertEqual(m1.n_kv, m0.n_kv)
        XCTAssertEqual(m1.n_tensors, m0.n_tensors)
        XCTAssertEqual(m1.string("general.architecture"), "deepseek4")
        XCTAssertEqual(m1.stringArrayBytes("tokens")?.map { Array($0) },
                       [Array("a".utf8), Array("bb".utf8), Array("ccc".utf8)])
        let t = try XCTUnwrap(m1.findTensor("w0"))
        XCTAssertEqual(t.dims, [16]); XCTAssertEqual(t.bytes, 64)
    }

    func testRejectsWrongTensorDataSize() throws {
        var w = try GGUFWriter()
        w.add(.init(name: "bad", dims: [4], type: 0, data: Data(count: 8)))  // needs 16
        XCTAssertThrowsError(try w.build())
    }

    func testRejectsTensorShapeProductOverflow() throws {
        var w = try GGUFWriter()
        w.add(.init(name: "overflow", dims: [UInt64.max, 2], type: 0,
                    data: Data()))
        XCTAssertThrowsError(try w.build()) { error in
            guard case GGUFWriterError.tensorShapeOverflow("overflow") = error else {
                return XCTFail("unexpected error: \(error)")
            }
        }
    }

    func testTensorByteCountRoundingCannotOverflow() {
        XCTAssertNil(GGUF.tensorNBytes(type: 10, elements: UInt64.max))
    }

    func testDSparkSupportContractAcceptsCompleteThreeStageGGUF() throws {
        let url = tempURL()
        defer { try? FileManager.default.removeItem(at: url) }
        try makeSyntheticDSparkWriter().write(to: url.path)

        let support = try DSparkSupportModel(
            path: url.path,
            targetDims: syntheticDSparkDims,
            targetLayerCount: 4
        )
        XCTAssertTrue(support.isRunnable, support.validation.errors.map(\.message).joined(separator: "\n"))
        XCTAssertEqual(support.metadata.blockSize, 5)
        XCTAssertEqual(support.metadata.markovRank, 2)
        XCTAssertEqual(support.metadata.noiseTokenID, 7)
        XCTAssertEqual(support.metadata.targetLayerIDs, [0, 1, 2])
        XCTAssertEqual(support.metadata.stageCount, 3)
        XCTAssertEqual(support.stages.count, 3)
        XCTAssertEqual(support.model.tensors.count, 81)
    }

    func testDSparkSupportContractRejectsMissingConfidenceHead() throws {
        let url = tempURL()
        defer { try? FileManager.default.removeItem(at: url) }
        var writer = try makeSyntheticDSparkWriter()
        writer = try GGUFWriter(
            metadata: writer.metadata,
            tensors: writer.tensors.filter {
                $0.name != "mtp.2.confidence_head.proj.weight"
            }
        )
        try writer.write(to: url.path)

        let support = try DSparkSupportModel(
            path: url.path,
            targetDims: syntheticDSparkDims,
            targetLayerCount: 4
        )
        XCTAssertFalse(support.isRunnable)
        XCTAssertTrue(support.validation.errors.contains {
            $0.message.contains("mtp.2.confidence_head.proj.weight mancante")
        })
    }

    func testDSparkSupportLocateNeverMixes0730And0731() throws {
        let directory = FileManager.default.temporaryDirectory
            .appendingPathComponent("dspark-locate-\(UUID().uuidString)", isDirectory: true)
        try FileManager.default.createDirectory(at: directory, withIntermediateDirectories: true)
        defer { try? FileManager.default.removeItem(at: directory) }
        let original = directory.appendingPathComponent(
            "DeepSeek-V4-Flash-DSpark-support.gguf")
        let refresh = directory.appendingPathComponent(
            "DeepSeek-V4-Flash-DSpark-support-0731.gguf")
        try Data().write(to: original)
        try Data().write(to: refresh)

        XCTAssertEqual(
            DSparkSupportModel.locate(near: directory
                .appendingPathComponent("DeepSeek-V4-Flash-main.gguf").path),
            original.path
        )
        XCTAssertEqual(
            DSparkSupportModel.locate(near: directory
                .appendingPathComponent("DeepSeek-V4-Flash-main-0731.gguf").path),
            refresh.path
        )
    }

    func testDSparkGreedyVerifierUsesPreTokenLogitForFirstDraft() {
        XCTAssertEqual(
            DSparkGreedyVerifier.acceptedPrefix(
                proposal: [11, 12, 13], currentTarget: 11,
                verifiedNext: [12, 99, 13]),
            2
        )
        XCTAssertEqual(
            DSparkGreedyVerifier.acceptedPrefix(
                proposal: [11, 12], currentTarget: 99,
                verifiedNext: [12, 13]),
            0
        )
    }

    func testDSparkGreedyVerifierDoesNotAcceptWithoutVerifierRow() {
        XCTAssertEqual(
            DSparkGreedyVerifier.acceptedPrefix(
                proposal: [4, 5, 6], currentTarget: 4,
                verifiedNext: [5]),
            2
        )
        XCTAssertEqual(
            DSparkGreedyVerifier.acceptedPrefix(
                proposal: [], currentTarget: 4,
                verifiedNext: []),
            0
        )
    }

    func testDSparkProposalUsesStableConfidenceSigmoid() {
        XCTAssertEqual(DSparkProposal.sigmoid(0), 0.5, accuracy: 0.000_001)
        XCTAssertEqual(DSparkProposal.sigmoid(20), 1, accuracy: 0.000_001)
        XCTAssertEqual(DSparkProposal.sigmoid(-20), 0, accuracy: 0.000_001)

        let proposal = DSparkProposal(
            tokens: [7, 9], confidences: [0.91, 0.73], threshold: 0.7)
        XCTAssertEqual(proposal.tokens, [7, 9])
        XCTAssertEqual(proposal.confidences, [0.91, 0.73])
        XCTAssertEqual(proposal.threshold, 0.7)
        XCTAssertEqual(proposal.firstConfidence, 0.91)

        let rejected = DSparkProposal(
            tokens: [], confidences: [], threshold: 0.7,
            firstConfidence: 0.42)
        XCTAssertEqual(rejected.firstConfidence, 0.42)
    }

    func testDSparkKVWindowCropAppendAndRingEviction() {
        var window = DSparkKVWindow(capacity: 8)
        XCTAssertTrue(window.set(tokenStart: 10, length: 5))
        XCTAssertEqual(window.physicalStart, 2)
        XCTAssertTrue(window.ends(at: 15))
        XCTAssertTrue(window.crop(toPrefixLength: 13))
        XCTAssertEqual(window.length, 3)
        XCTAssertTrue(window.appendRow(at: 13))
        XCTAssertEqual(window.length, 4)
        XCTAssertFalse(window.appendRow(at: 13))

        XCTAssertTrue(window.set(tokenStart: 20, length: 8))
        XCTAssertTrue(window.appendRow(at: 28))
        XCTAssertEqual(window.tokenStart, 21)
        XCTAssertEqual(window.length, 8)
        XCTAssertEqual(window.physicalStart, 5)

        XCTAssertTrue(window.crop(toPrefixLength: 99))
        XCTAssertEqual(window.length, 0)
        XCTAssertTrue(window.ends(at: 99))
    }

    func testDSparkStageBlockPlanBuildsTargetAndNoiseRows() throws {
        var window = DSparkKVWindow(capacity: 16)
        XCTAssertTrue(window.set(tokenStart: 100, length: 7))
        let plan = try XCTUnwrap(DSparkStageBlockPlan.make(
            currentToken: 42, noiseToken: 3, position: 107,
            blockSize: 5, vocabularySize: 128, window: window))

        XCTAssertEqual(plan.draftTokens, [42, 3, 3, 3, 3])
        XCTAssertEqual(plan.positionIDs, [107, 107, 108, 109, 110, 111])
        XCTAssertEqual(plan.supportLength, 7)
        XCTAssertEqual(plan.attentionRawStart, 100)
        XCTAssertEqual(plan.appendPosition, 11)
        XCTAssertEqual(plan.visibleRows, 13)
    }

    func testDSparkStageBlockPlanRejectsCacheGapAndOverflow() {
        var gap = DSparkKVWindow(capacity: 16)
        XCTAssertTrue(gap.set(tokenStart: 5, length: 4))
        XCTAssertNil(DSparkStageBlockPlan.make(
            currentToken: 1, noiseToken: 2, position: 10,
            blockSize: 5, vocabularySize: 8, window: gap))

        var full = DSparkKVWindow(capacity: 8)
        XCTAssertTrue(full.set(tokenStart: 2, length: 4))
        XCTAssertNil(DSparkStageBlockPlan.make(
            currentToken: 1, noiseToken: 2, position: 6,
            blockSize: 5, vocabularySize: 8, window: full))
    }

    func testDSparkStage0CapturesEveryTargetLayerAndRunsMetalProjection() throws {
        let url = tempURL()
        defer { try? FileManager.default.removeItem(at: url) }
        try makeSyntheticDSparkWriter().write(to: url.path)

        let rt: MetalRuntime
        do { rt = try MetalRuntime() }
        catch MetalError.noDevice { throw XCTSkip("Metal device unavailable in this test environment") }
        let runtime = try DSparkStage0Runtime(
            rt: rt,
            supportPath: url.path,
            mainModelPath: "DeepSeek-V4-Flash-main.gguf",
            targetDims: syntheticDSparkDims,
            targetLayerCount: 4
        )
        runtime.beginCapture(position: 9)
        for layer in [0, 1, 2] {
            let hc = try GPUTensor.floats(rt, [
                Float(layer + 1), 2, 3, 4,
                Float(layer + 3), 4, 5, 6,
            ])
            try runtime.capture(layer: layer, hiddenHC: hc)
        }

        XCTAssertTrue(try runtime.finishCapture())
        XCTAssertEqual(runtime.position, 9)
        XCTAssertEqual(runtime.mainHidden(), [0, 0, 0, 0])
        XCTAssertGreaterThan(runtime.residentWeightBytes, 0)
        XCTAssertGreaterThan(runtime.privateKVBytes, 0)

        runtime.beginBatchCapture(startPosition: 20, nTokens: 3)
        let batchHC = try GPUTensor.floats(
            rt, Array(repeating: 1, count: 3 * 2 * 4))
        for layer in [0, 1, 2] {
            try runtime.captureBatch(layer: layer, hiddenHC: batchHC, nTokens: 3)
        }
        XCTAssertTrue(runtime.finishBatchCapture())
        XCTAssertEqual(runtime.capturedBatchRange, 20..<23)
    }

    func testDSparkStage0RejectsIncompleteCapture() throws {
        let url = tempURL()
        defer { try? FileManager.default.removeItem(at: url) }
        try makeSyntheticDSparkWriter().write(to: url.path)

        let rt: MetalRuntime
        do { rt = try MetalRuntime() }
        catch MetalError.noDevice { throw XCTSkip("Metal device unavailable in this test environment") }
        let runtime = try DSparkStage0Runtime(
            rt: rt,
            supportPath: url.path,
            mainModelPath: "DeepSeek-V4-Flash-main.gguf",
            targetDims: syntheticDSparkDims,
            targetLayerCount: 4
        )
        runtime.beginCapture(position: 3)
        let hc = try GPUTensor.floats(rt, Array(repeating: 1, count: 8))
        try runtime.capture(layer: 0, hiddenHC: hc)

        XCTAssertFalse(try runtime.finishCapture())
        XCTAssertFalse(runtime.isReady)
        XCTAssertNil(runtime.mainHidden())
    }

    func testDSparkMappedStageBindingKeepsExpertsMappedAndSmallWeightsResident() throws {
        let url = tempURL()
        defer { try? FileManager.default.removeItem(at: url) }
        try makeSyntheticDSparkWriter().write(to: url.path)

        let rt: MetalRuntime
        do { rt = try MetalRuntime() }
        catch MetalError.noDevice {
            throw XCTSkip("Metal device unavailable in this test environment")
        }
        let support = try DSparkSupportModel(
            path: url.path,
            targetDims: syntheticDSparkDims,
            targetLayerCount: 4,
            metalMapping: true)
        let stage = try support.loadMappedStageWeights(rt, stage: 2)

        XCTAssertEqual(stage.index, 2)
        XCTAssertEqual(stage.type(of: "ffn_gate_exps.weight"), 16)
        XCTAssertEqual(stage.type(of: "ffn_up_exps.weight"), 16)
        XCTAssertEqual(stage.type(of: "ffn_down_exps.weight"), 10)
        XCTAssertEqual(stage.block.gateQuant.kernel, MoEQuant.iq2_xxs.kernel)
        XCTAssertEqual(stage.block.upQuant.kernel, MoEQuant.iq2_xxs.kernel)
        XCTAssertEqual(stage.block.downQuant.kernel, MoEQuant.q2_K.kernel)
        XCTAssertNotNil(stage.block.expBias)
        XCTAssertNotNil(stage.finalHead)
        XCTAssertGreaterThan(stage.mappedBytes, stage.residentBytes)
    }

    // A tiny geometry keeps the fixture below a few KiB while retaining every
    // shape relation of the real 4096-wide, 81-tensor DSpark artifact.
    private var syntheticDSparkDims: DSV4Dims {
        DSV4Dims(
            nEmbd: 4, nHC: 2, headDim: 2, nHead: 2,
            qRank: 2, qDim: 4, sharedFfn: 4,
            nExperts: 2, expertFfn: 4, k: 1, nRot: 1, vocab: 8,
            nOutGroup: 1, nLoraO: 2
        )
    }

    private func makeSyntheticDSparkWriter() throws -> GGUFWriter {
        let d = syntheticDSparkDims
        var writer = try GGUFWriter(metadata: [
            ("general.architecture", .text("deepseek4-dspark")),
            ("general.name", .text("Synthetic DSpark support")),
            ("general.alignment", .uint32(32)),
            ("dspark.block_size", .uint32(5)),
            ("dspark.markov_rank", .uint32(2)),
            ("dspark.noise_token_id", .uint32(7)),
            ("dspark.target_layer_ids", .array(
                elementType: .uint32,
                elements: [.uint32(0), .uint32(1), .uint32(2)])),
            ("dspark.stage_count", .uint32(3)),
            ("dspark.n_layers", .uint32(3)),
        ])

        func add(_ name: String, _ dims: [UInt64], _ type: UInt32 = 0) {
            let elements = dims.reduce(UInt64(1), *)
            let count = Int(GGUF.tensorNBytes(type: type, elements: elements)!)
            writer.add(.init(name: name, dims: dims, type: type,
                             data: Data(count: count)))
        }

        let hcDim = d.nHC * d.nEmbd
        let hcMix = 2 * d.nHC + d.nHC * d.nHC
        let qDim = d.nHead * d.headDim
        let attentionGroup = d.headDim * (d.nHead / d.nOutGroup)
        let outputLow = d.nOutGroup * d.nLoraO
        let block: [(String, [UInt64], UInt32)] = [
            ("hc_attn_fn.weight", [UInt64(hcDim), UInt64(hcMix)], 0),
            ("hc_attn_scale.weight", [3], 0),
            ("hc_attn_base.weight", [UInt64(hcMix)], 0),
            ("attn_norm.weight", [UInt64(d.nEmbd)], 0),
            ("attn_q_a.weight", [UInt64(d.nEmbd), UInt64(d.qRank)], 0),
            ("attn_q_a_norm.weight", [UInt64(d.qRank)], 0),
            ("attn_q_b.weight", [UInt64(d.qRank), UInt64(qDim)], 0),
            ("attn_kv.weight", [UInt64(d.nEmbd), UInt64(d.headDim)], 0),
            ("attn_kv_a_norm.weight", [UInt64(d.headDim)], 0),
            ("attn_sinks.weight", [UInt64(d.nHead)], 0),
            ("attn_output_a.weight", [UInt64(attentionGroup), UInt64(outputLow)], 0),
            ("attn_output_b.weight", [UInt64(outputLow), UInt64(d.nEmbd)], 0),
            ("hc_ffn_fn.weight", [UInt64(hcDim), UInt64(hcMix)], 0),
            ("hc_ffn_scale.weight", [3], 0),
            ("hc_ffn_base.weight", [UInt64(hcMix)], 0),
            ("ffn_norm.weight", [UInt64(d.nEmbd)], 0),
            ("ffn_gate_inp.weight", [UInt64(d.nEmbd), UInt64(d.nExperts)], 0),
            ("exp_probs_b.bias", [UInt64(d.nExperts)], 0),
            ("ffn_gate_exps.weight", [UInt64(d.nEmbd), UInt64(d.expertFfn), UInt64(d.nExperts)], 16),
            ("ffn_up_exps.weight", [UInt64(d.nEmbd), UInt64(d.expertFfn), UInt64(d.nExperts)], 16),
            ("ffn_down_exps.weight", [UInt64(d.expertFfn), UInt64(d.nEmbd), UInt64(d.nExperts)], 10),
            ("ffn_gate_shexp.weight", [UInt64(d.nEmbd), UInt64(d.sharedFfn)], 0),
            ("ffn_up_shexp.weight", [UInt64(d.nEmbd), UInt64(d.sharedFfn)], 0),
            ("ffn_down_shexp.weight", [UInt64(d.sharedFfn), UInt64(d.nEmbd)], 0),
        ]
        for stage in 0..<3 {
            for (suffix, dims, type) in block {
                add("mtp.\(stage).\(suffix)", dims, type)
            }
        }
        add("mtp.0.main_proj.weight", [12, UInt64(d.nEmbd)])
        add("mtp.0.main_norm.weight", [UInt64(d.nEmbd)])
        add("mtp.2.norm.weight", [UInt64(d.nEmbd)])
        add("mtp.2.hc_head_base.weight", [UInt64(d.nHC)])
        add("mtp.2.hc_head_fn.weight", [UInt64(hcDim), UInt64(d.nHC)])
        add("mtp.2.hc_head_scale.weight", [1])
        add("mtp.2.markov_head.markov_w1.weight", [2, UInt64(d.vocab)])
        add("mtp.2.markov_head.markov_w2.weight", [2, UInt64(d.vocab)])
        add("mtp.2.confidence_head.proj.weight", [UInt64(d.nEmbd + 2), 1])
        return writer
    }
}
