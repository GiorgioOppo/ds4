import XCTest
@testable import DS4Engine

/// Distributed wire protocol: frame round-trips (with the v2 session echo),
/// STRICT activation unpacking (truncated payloads are rejected, never returned
/// short), the codec's 32/16/8-bit paths, and the route-length cap. Pure CPU.
final class DistProtocolTests: XCTestCase {

    // MARK: Frame round-trips

    func testHelloRoundTripCarriesVersion() throws {
        let hello = DistHello(modelName: "model.gguf", layerStart: 3, layerEnd: 17,
                              hasOutput: true, nLayers: 43, contextSize: 8192)
        let decoded = try XCTUnwrap(DistHello.decode(hello.encoded()))
        XCTAssertEqual(decoded.version, Dist.protocolVersion)
        XCTAssertEqual(decoded.modelName, "model.gguf")
        XCTAssertEqual(decoded.layerStart, 3)
        XCTAssertEqual(decoded.layerEnd, 17)
        XCTAssertTrue(decoded.hasOutput)
        XCTAssertEqual(decoded.nLayers, 43)
        XCTAssertEqual(decoded.contextSize, 8192)
    }

    func testIdleHelloRoundTrip() throws {
        let idle = DistHello.idle(localModelName: "local.gguf", nLayers: 43)
        let decoded = try XCTUnwrap(DistHello.decode(idle.encoded()))
        XCTAssertFalse(decoded.assigned)
        XCTAssertEqual(decoded.modelName, "local.gguf")
        XCTAssertEqual(decoded.nLayers, 43)
        XCTAssertEqual(decoded.contextSize, 0)
    }

    func testAssignRoundTrip() throws {
        let usage = Data(#"{"layers":{"0":[1,2,3]}}"#.utf8)
        let assign = DistAssign(modelPath: "/Models/ds4-flash.gguf", modelName: "ds4-flash.gguf",
                                contextSize: 8192, expertCacheSlots: 16,
                                diskKVBudgetTokens: 1_000_000, useExpertBundle: true,
                                useDenseQ4: true,
                                layerStart: 22, layerEnd: 42, hasOutput: true,
                                usageJSON: usage)
        let decoded = try XCTUnwrap(DistAssign.decode(assign.encoded()))
        XCTAssertEqual(decoded.modelPath, "/Models/ds4-flash.gguf")
        XCTAssertEqual(decoded.modelName, "ds4-flash.gguf")
        XCTAssertEqual(decoded.contextSize, 8192)
        XCTAssertEqual(decoded.expertCacheSlots, 16)
        XCTAssertEqual(decoded.diskKVBudgetTokens, 1_000_000)
        XCTAssertTrue(decoded.useExpertBundle)
        XCTAssertTrue(decoded.useDenseQ4)
        XCTAssertEqual(decoded.layerStart, 22)
        XCTAssertEqual(decoded.layerEnd, 42)
        XCTAssertTrue(decoded.hasOutput)
        XCTAssertEqual(decoded.usageJSON, usage)
        // Empty usage stays empty; truncated frames are rejected, not mis-decoded.
        let bare = DistAssign(modelPath: "/m.gguf", modelName: "m.gguf", contextSize: 4096,
                              expertCacheSlots: 0, diskKVBudgetTokens: 0,
                              layerStart: 0, layerEnd: 42, hasOutput: true)
        XCTAssertEqual(try XCTUnwrap(DistAssign.decode(bare.encoded())).usageJSON, Data())
        XCTAssertFalse(try XCTUnwrap(DistAssign.decode(bare.encoded())).useExpertBundle)
        XCTAssertFalse(try XCTUnwrap(DistAssign.decode(bare.encoded())).useDenseQ4)
        XCTAssertNil(DistAssign.decode(assign.encoded().prefix(42)))
        XCTAssertNil(DistAssign.decode(assign.encoded().dropLast(4)))   // usage blob truncated
    }

    // MARK: File distribution payloads

    func testFileOfferNeedChunkDoneRoundTrips() throws {
        let sha = Data((0..<32).map { UInt8($0) })
        let offer = DistFileOffer(entries: [
            DistFileEntry(kind: .gguf, name: "model.gguf", size: 123_456_789_012, sha256: sha),
            DistFileEntry(kind: .expertBundle, name: "model.gguf.expbundle",
                          size: 42, sha256: Data(repeating: 0xAB, count: 32)),
            DistFileEntry(kind: .q4Dense, name: "model.gguf.q4dense",
                          size: 1_400_000_000, sha256: Data(repeating: 0xCD, count: 32)),
        ])
        let decoded = try XCTUnwrap(DistFileOffer.decode(offer.encoded()))
        XCTAssertEqual(decoded.entries.count, 3)
        XCTAssertEqual(decoded.entries[0].kind, .gguf)
        XCTAssertEqual(decoded.entries[0].name, "model.gguf")
        XCTAssertEqual(decoded.entries[0].size, 123_456_789_012)   // > 4 GB: u64 on the wire
        XCTAssertEqual(decoded.entries[0].sha256, sha)
        XCTAssertEqual(decoded.entries[1].kind, .expertBundle)
        XCTAssertEqual(decoded.entries[2].kind, .q4Dense)
        XCTAssertNil(DistFileOffer.decode(offer.encoded().dropLast(1)))

        let need = try XCTUnwrap(DistFileNeed.decode(DistFileNeed(indices: [1]).encoded()))
        XCTAssertEqual(need.indices, [1])
        XCTAssertEqual(try XCTUnwrap(DistFileNeed.decode(DistFileNeed(indices: []).encoded())).indices, [])

        let chunk = DistFileChunk(index: 0, offset: 8_589_934_592, data: Data([1, 2, 3]))
        let dChunk = try XCTUnwrap(DistFileChunk.decode(chunk.encoded()))
        XCTAssertEqual(dChunk.index, 0)
        XCTAssertEqual(dChunk.offset, 8_589_934_592)               // > 4 GB offset
        XCTAssertEqual(dChunk.data, Data([1, 2, 3]))
        XCTAssertNil(DistFileChunk.decode(chunk.encoded().dropLast(1)))

        XCTAssertEqual(try XCTUnwrap(DistFileDone.decode(DistFileDone(index: 7).encoded())).index, 7)
    }

    func testFileStoreManifestAndSanitize() throws {
        XCTAssertEqual(DistFileStore.sanitize("../../etc/passwd"), "passwd")
        XCTAssertEqual(DistFileStore.sanitize("model.gguf"), "model.gguf")
        XCTAssertEqual(DistFileStore.sanitize(".."), "file")

        let dir = FileManager.default.temporaryDirectory
            .appendingPathComponent("dist-store-\(UUID().uuidString)")
        defer { try? FileManager.default.removeItem(at: dir) }
        let store = DistFileStore(directory: dir)
        let sha = Data(repeating: 7, count: 32)
        XCTAssertFalse(store.has(name: "m.gguf", size: 3, sha256: sha))
        // remember() without the file on disk still fails `has` (size check).
        store.remember(name: "m.gguf", size: 3, sha256: sha)
        XCTAssertFalse(store.has(name: "m.gguf", size: 3, sha256: sha))
        try Data([1, 2, 3]).write(to: store.url(for: "m.gguf"))
        XCTAssertTrue(store.has(name: "m.gguf", size: 3, sha256: sha))
        XCTAssertFalse(store.has(name: "m.gguf", size: 3, sha256: Data(repeating: 8, count: 32)))
        XCTAssertFalse(store.has(name: "m.gguf", size: 4, sha256: sha))
    }

    func testFileHashComputeKnownVector() throws {
        let url = FileManager.default.temporaryDirectory
            .appendingPathComponent("dist-hash-\(UUID().uuidString).bin")
        defer { try? FileManager.default.removeItem(at: url) }
        try Data("abc".utf8).write(to: url)
        // SHA-256("abc") — FIPS 180 test vector.
        XCTAssertEqual(DistFileHash.compute(path: url.path)?.hexString,
                       "ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad")
        // cachedOrCompute: same digest, and stable across a second (cached) call.
        let first = DistFileHash.cachedOrCompute(path: url.path)
        XCTAssertEqual(first?.hexString,
                       "ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad")
        XCTAssertEqual(DistFileHash.cachedOrCompute(path: url.path), first)
    }

    // MARK: KV control payloads

    func testKVTokenAndLengthPayloads() throws {
        let ids = Array(0..<300)
        XCTAssertEqual(try XCTUnwrap(DistKV.decodeTokens(DistKV.encodeTokens(ids))), ids)
        XCTAssertEqual(try XCTUnwrap(DistKV.decodeTokens(DistKV.encodeTokens([]))), [])
        XCTAssertNil(DistKV.decodeTokens(DistKV.encodeTokens(ids).dropLast(2)))   // truncated
        // Hostile count without payload → rejected.
        var hostile = Data(); hostile.appendLE(UInt32.max)
        XCTAssertNil(DistKV.decodeTokens(hostile))

        let lengths = [512, 256, 128]
        XCTAssertEqual(try XCTUnwrap(DistKV.decodeLengths(DistKV.encodeLengths(lengths))), lengths)

        let (savedTokens, cold) = try XCTUnwrap(DistKV.decodeSave(
            DistKV.encodeSave(tokens: ids, cold: true)))
        XCTAssertEqual(savedTokens, ids)
        XCTAssertTrue(cold)

        let ack = try XCTUnwrap(DistKV.decodeAck(DistKV.encodeAck(ok: false, message: "no entry")))
        XCTAssertFalse(ack.ok)
        XCTAssertEqual(ack.message, "no entry")
    }

    func testTurnStartFlagRoundTrip() throws {
        let work = DistWork(session: 9, pos: 512, nTokens: 1, layerStart: 0, layerEnd: 1,
                            flags: [.turnStart, .outputLogits], hcBits: 32,
                            hc: (0..<32).map(Float.init))
        let decoded = try XCTUnwrap(DistWork.decode(work.encoded()))
        XCTAssertTrue(decoded.flags.contains(.turnStart))
        XCTAssertTrue(decoded.flags.contains(.outputLogits))
        XCTAssertFalse(decoded.flags.contains(.resetSession))
    }

    // MARK: Coordinator-side layer partition

    func testPartitionEqualAndRemainder() throws {
        // 43 layers on 3 workers: 15 + 14 + 14, contiguous, full coverage.
        let p = try DistCoordinator.partition(nLayers: 43, workers: 3)
        XCTAssertEqual(p.map { $0.start }, [0, 15, 29])
        XCTAssertEqual(p.map { $0.end }, [14, 28, 42])
        // Single worker: the whole model.
        let one = try DistCoordinator.partition(nLayers: 43, workers: 1)
        XCTAssertEqual(one.count, 1)
        XCTAssertEqual(one[0].start, 0); XCTAssertEqual(one[0].end, 42)
        // Even split.
        let even = try DistCoordinator.partition(nLayers: 8, workers: 4)
        XCTAssertEqual(even.map { $0.start }, [0, 2, 4, 6])
        XCTAssertEqual(even.map { $0.end }, [1, 3, 5, 7])
        // Degenerate configs fail loudly.
        XCTAssertThrowsError(try DistCoordinator.partition(nLayers: 4, workers: 5))
        XCTAssertThrowsError(try DistCoordinator.partition(nLayers: 4, workers: 0))
    }

    // MARK: Worker-side gguf resolution

    func testResolveModelPathOrder() throws {
        let dir = FileManager.default.temporaryDirectory
            .appendingPathComponent("dist-resolve-\(UUID().uuidString)")
        try FileManager.default.createDirectory(at: dir, withIntermediateDirectories: true)
        defer { try? FileManager.default.removeItem(at: dir) }
        let coordPath = dir.appendingPathComponent("coord/model.gguf").path
        let localDir = dir.appendingPathComponent("local")
        try FileManager.default.createDirectory(at: localDir, withIntermediateDirectories: true)
        let sibling = localDir.appendingPathComponent("model.gguf").path
        let hint = localDir.appendingPathComponent("other.gguf").path
        FileManager.default.createFile(atPath: sibling, contents: Data("x".utf8))

        // Coordinator's path missing → the same-named sibling of the local hint wins.
        XCTAssertEqual(DistWorker.resolveModelPath(requestedPath: coordPath,
                                                   modelName: "model.gguf", localHint: hint), sibling)
        // Coordinator's exact path present → used verbatim.
        XCTAssertEqual(DistWorker.resolveModelPath(requestedPath: sibling,
                                                   modelName: "model.gguf", localHint: hint), sibling)
        // No coordinator path: resolved by NAME in the local hint's directory.
        XCTAssertEqual(DistWorker.resolveModelPath(requestedPath: "", modelName: "other.gguf",
                                                   localHint: sibling,
                                                   exists: { $0 == sibling || $0 == hint }),
                       hint)
        // Nothing matches → nil (the worker replies with an explicit error).
        XCTAssertNil(DistWorker.resolveModelPath(requestedPath: coordPath,
                                                 modelName: "missing.gguf", localHint: hint))
    }

    func testWorkRoundTripWithRouteAndSession() throws {
        let route = [DistRouteEntry(host: "10.0.0.2", port: 9100, layerStart: 0, layerEnd: 20, hasOutput: false),
                     DistRouteEntry(host: "10.0.0.3", port: 9101, layerStart: 21, layerEnd: 42, hasOutput: true)]
        let hc: [Float] = (0..<96).map { Float($0) * 0.25 - 8 }
        let work = DistWork(session: 7, pos: 128, nTokens: 3, layerStart: 0, layerEnd: 20,
                            flags: [.outputLogits], hcBits: 32, route: route, routeIndex: 1,
                            returnHost: "10.0.0.1", returnPort: 9099, hc: hc)
        let decoded = try XCTUnwrap(DistWork.decode(work.encoded()))
        XCTAssertEqual(decoded.session, 7)
        XCTAssertEqual(decoded.pos, 128)
        XCTAssertEqual(decoded.nTokens, 3)
        XCTAssertEqual(decoded.flags, [.outputLogits])
        XCTAssertEqual(decoded.route.count, 2)
        XCTAssertEqual(decoded.route[1].host, "10.0.0.3")
        XCTAssertEqual(decoded.route[1].port, 9101)
        XCTAssertTrue(decoded.route[1].hasOutput)
        XCTAssertEqual(decoded.routeIndex, 1)
        XCTAssertEqual(decoded.returnHost, "10.0.0.1")
        XCTAssertEqual(decoded.returnPort, 9099)
        XCTAssertEqual(decoded.hc, hc)
    }

    func testResultRoundTripEchoesSession() throws {
        let res = DistResult(session: 42, kind: .hidden, bits: 32, values: [1, -2, 3.5])
        let decoded = try XCTUnwrap(DistResult.decode(res.encoded()))
        XCTAssertEqual(decoded.session, 42)
        XCTAssertEqual(decoded.kind, .hidden)
        XCTAssertEqual(decoded.values, [1, -2, 3.5])

        let ack = DistResult(session: 43, kind: .ack, bits: 32, values: [])
        let dAck = try XCTUnwrap(DistResult.decode(ack.encoded()))
        XCTAssertEqual(dAck.session, 43)
        XCTAssertEqual(dAck.kind, .ack)
        XCTAssertTrue(dAck.values.isEmpty)
    }

    // MARK: Strictness (truncated payloads must be REJECTED, not shortened)

    func testTruncatedWorkPayloadIsRejected() throws {
        let work = DistWork(session: 1, pos: 0, nTokens: 2, layerStart: 0, layerEnd: 1,
                            flags: [], hcBits: 32, hc: (0..<64).map(Float.init))
        var encoded = work.encoded()
        encoded.removeLast(8)                       // chop part of the hc floats
        XCTAssertNil(DistWork.decode(encoded))
    }

    func testTruncatedResultPayloadIsRejected() throws {
        let res = DistResult(session: 1, kind: .hidden, bits: 16, values: (0..<32).map(Float.init))
        var encoded = res.encoded()
        encoded.removeLast(2)
        XCTAssertNil(DistResult.decode(encoded))
    }

    func testRouteCountCapRejectsHostileFrame() throws {
        // Hand-build a WORK frame declaring 4 billion route entries.
        var d = Data()
        d.appendLE(UInt32(1))                       // session
        d.appendLE(UInt32(0))                       // pos
        d.appendLE(UInt32(1))                       // nTokens
        d.appendLE(UInt32(0)); d.appendLE(UInt32(1))
        d.appendLE(UInt32(0))                       // flags
        d.appendLE(UInt32(32))                      // bits
        d.appendLE(UInt32.max)                      // routeCount: hostile
        d.appendLE(UInt32(0))                       // routeIndex
        XCTAssertNil(DistWork.decode(d))
    }

    // MARK: Activation codec

    func testCodec32RoundTripExact() throws {
        let v: [Float] = [0, 1, -1, .pi, 1e-7, -3.4e38, 3.4e38]
        let packed = ActivationCodec.pack(v, bits: 32)
        XCTAssertEqual(packed.count, v.count * 4)
        XCTAssertEqual(try XCTUnwrap(ActivationCodec.unpack(packed, count: v.count, bits: 32)), v)
    }

    func testCodec16RoundTripWithinHalfPrecision() throws {
        let v: [Float] = (0..<100).map { Float($0) * 0.37 - 18 }
        let packed = ActivationCodec.pack(v, bits: 16)
        XCTAssertEqual(packed.count, v.count * 2)
        let out = try XCTUnwrap(ActivationCodec.unpack(packed, count: v.count, bits: 16))
        for (a, b) in zip(v, out) {
            XCTAssertEqual(a, b, accuracy: max(0.01, abs(a) * 0.002))   // fp16 ulp
        }
    }

    func testCodec8RoundTripWithinScaledStep() throws {
        let v: [Float] = (0..<64).map { Float($0) - 32 }   // absmax 32 → step 32/127
        let packed = ActivationCodec.pack(v, bits: 8)
        XCTAssertEqual(packed.count, 4 + v.count)
        let out = try XCTUnwrap(ActivationCodec.unpack(packed, count: v.count, bits: 8))
        let step = 32.0 as Float / 127.0
        for (a, b) in zip(v, out) { XCTAssertEqual(a, b, accuracy: step * 0.51) }
        // All-zero vector: scale falls back to 1 and round-trips exactly.
        let zeros = [Float](repeating: 0, count: 8)
        XCTAssertEqual(try XCTUnwrap(ActivationCodec.unpack(
            ActivationCodec.pack(zeros, bits: 8), count: 8, bits: 8)), zeros)
    }

    func testUnpackStrictOnShortData() throws {
        let packed = ActivationCodec.pack([1, 2, 3], bits: 32)
        XCTAssertNil(ActivationCodec.unpack(packed, count: 4, bits: 32))     // asks for more
        XCTAssertNil(ActivationCodec.unpack(packed.prefix(11), count: 3, bits: 32))
        XCTAssertNil(ActivationCodec.unpack(Data(), count: 1, bits: 8))      // missing scale
        XCTAssertEqual(try XCTUnwrap(ActivationCodec.unpack(Data(), count: 0, bits: 32)), [])
    }
}
