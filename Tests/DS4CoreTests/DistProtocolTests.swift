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
