import XCTest
@testable import DS4Core

/// Swift-only checks of the disk KV-cache file format and policy bits in
/// `KVCFile`: little-endian helpers, header fill/parse round-trips, SHA1 naming,
/// reason/key-kind mapping, default options, and the eviction score. (The
/// original bit-for-bit cross-check against ds4_kvstore.c was dropped with the
/// C engine; expected values here are well-known constants or hand-derived.)
final class KVCFileTests: XCTestCase {

    func testLEHelpersRoundTrip() {
        let values: [UInt32] = [0, 1, 255, 256, 0x1234_5678, 0xFFFF_FFFF, 0xDEAD_BEEF]
        for v in values {
            var buf = [UInt8](repeating: 0, count: 4)
            KVCFile.lePut32(&buf, 0, v)
            // Little-endian byte order.
            XCTAssertEqual(buf[0], UInt8(v & 0xff))
            XCTAssertEqual(buf[3], UInt8((v >> 24) & 0xff))
            XCTAssertEqual(KVCFile.leGet32(buf, 0), v, "round-trip \(v)")
        }
    }

    func testFillHeaderRoundTrip() {
        let h = KVCFile.Header(quantBits: 4, reason: 1, extFlags: 0b101, modelId: 0,
                               tokens: 12345, hits: 7, ctxSize: 100000,
                               createdAt: 1_700_000_000, lastUsed: 1_700_009_999,
                               payloadBytes: 987_654_321)
        let b = KVCFile.fillHeader(h)
        XCTAssertEqual(b.count, KVCFile.fixedHeader)
        // Magic "KVC", version, and payload ABI are at fixed offsets.
        XCTAssertEqual(Array(b[0..<4]), [KVCFile.magic0, KVCFile.magic1, KVCFile.magic2, KVCFile.version])
        XCTAssertEqual(b[20], KVCFile.payloadABI)
        // Fields round-trip through parse.
        XCTAssertEqual(KVCFile.parseHeader(b), h)
    }

    func testParseRejectsBadHeaders() {
        let b = KVCFile.fillHeader(KVCFile.Header(quantBits: 4, reason: 1, extFlags: 0, modelId: 0,
                                                  tokens: 10, hits: 0, ctxSize: 1, createdAt: 0,
                                                  lastUsed: 0, payloadBytes: 0))
        var bad = b; bad[0] = 0x00
        XCTAssertNil(KVCFile.parseHeader(bad))         // wrong magic
        bad = b; bad[3] = 9
        XCTAssertNil(KVCFile.parseHeader(bad))         // wrong version
        bad = b; KVCFile.lePut32(&bad, 8, 0)
        XCTAssertNil(KVCFile.parseHeader(bad))         // tokens == 0
        bad = b; bad[4] = 3
        XCTAssertNil(KVCFile.parseHeader(bad))         // quant not 2/4
    }

    func testReasonCode() {
        XCTAssertEqual(KVCFile.reasonCode("cold"), 1)
        XCTAssertEqual(KVCFile.reasonCode("continued"), 2)
        XCTAssertEqual(KVCFile.reasonCode("evict"), 3)
        XCTAssertEqual(KVCFile.reasonCode("shutdown"), 4)
        XCTAssertEqual(KVCFile.reasonCode("agent-system"), 5)
        XCTAssertEqual(KVCFile.reasonCode("agent-session"), 6)
        XCTAssertEqual(KVCFile.reasonCode("bogus"), 0)
        XCTAssertEqual(KVCFile.reasonCode(""), 0)
        XCTAssertEqual(KVCFile.reasonCode(nil), 0)
    }

    func testKeyKind() {
        XCTAssertEqual(KVCFile.keyKind(extFlags: 0), "token-text")
        XCTAssertEqual(KVCFile.keyKind(extFlags: 1), "token-text")            // tool-map only
        XCTAssertEqual(KVCFile.keyKind(extFlags: 2), "responses-visible")    // responses bit
        XCTAssertEqual(KVCFile.keyKind(extFlags: 4), "thinking-visible")     // thinking bit
        XCTAssertEqual(KVCFile.keyKind(extFlags: 6), "responses-visible")    // both -> responses wins
        XCTAssertEqual(KVCFile.keyKind(extFlags: 7), "responses-visible")
    }

    func testSHA1Known() {
        // RFC 3174 / well-known SHA1 digests.
        XCTAssertEqual(KVCFile.sha1Hex([]), "da39a3ee5e6b4b0d3255bfef95601890afd80709")
        XCTAssertEqual(KVCFile.sha1Hex(Array("abc".utf8)), "a9993e364706816aba3e25717850c26c9cd0d89d")
        // Any output is 40 lowercase hex chars.
        let hex = KVCFile.sha1Hex(Array("The quick brown fox".utf8))
        XCTAssertEqual(hex.count, 40)
        XCTAssertTrue(hex.allSatisfy { $0.isHexDigit && (!$0.isLetter || $0.isLowercase) })
    }

    func testShaHexName() {
        XCTAssertEqual(KVCFile.shaHexName("0123456789abcdef0123456789abcdef01234567.kv"),
                       "0123456789abcdef0123456789abcdef01234567")
        // Uppercase hex is accepted and lowercased.
        XCTAssertEqual(KVCFile.shaHexName("0123456789ABCDEF0123456789ABCDEF01234567.kv"),
                       "0123456789abcdef0123456789abcdef01234567")
        // Rejected: too short, wrong extension, non-hex, wrong length.
        for bad in ["short.kv", "0123456789abcdef0123456789abcdef01234567.txt",
                    "0123456789abcdef0123456789abcdef0123456g.kv",
                    "0123456789abcdef0123456789abcdef012345678.kv"] {
            XCTAssertNil(KVCFile.shaHexName(bad), "should reject \(bad)")
        }
    }

    func testDefaultOptions() {
        let o = KVCFile.defaultOptions()
        XCTAssertEqual(o.minTokens, 512)
        XCTAssertEqual(o.coldMaxTokens, 30000)
        XCTAssertEqual(o.continuedIntervalTokens, 10000)
        XCTAssertEqual(o.boundaryTrimTokens, 32)
        XCTAssertEqual(o.boundaryAlignTokens, 2048)
    }

    func testEvictionScore() {
        let now: UInt64 = 2_000_000_000
        // fileSize 0 -> score 0.
        XCTAssertEqual(KVCFile.evictionScore(
            .init(hits: 5, tokens: 100, fileSize: 0, createdAt: now, lastUsed: now, reason: 1), now: now), 0.0)

        // lastUsed == now -> no time decay; reason 2 (continued) is not an anchor.
        // score = (hits + 1) * tokens / fileSize = 1 * 1000 / 1_000_000 = 0.001
        XCTAssertEqual(KVCFile.evictionScore(
            .init(hits: 0, tokens: 1000, fileSize: 1_000_000, createdAt: now, lastUsed: now, reason: 2),
            now: now), 0.001, accuracy: 1e-12)

        // reason 1 (cold) is an anchor -> score doubled: 0.001 * 2 = 0.002
        XCTAssertEqual(KVCFile.evictionScore(
            .init(hits: 0, tokens: 1000, fileSize: 1_000_000, createdAt: now, lastUsed: now, reason: 1),
            now: now), 0.002, accuracy: 1e-12)

        // usedAt == 0 -> effectiveHits forced to 0: (0+1)*2000/1000 = 2.0
        XCTAssertEqual(KVCFile.evictionScore(
            .init(hits: 9, tokens: 2000, fileSize: 1000, createdAt: 0, lastUsed: 0, reason: 2),
            now: now), 2.0, accuracy: 1e-12)
    }

    func testDSV4HeaderRoundTrip() {
        let h = DSV4PayloadHeader(savedContextSize: 100000, prefillChunk: 4096, rawKVCapacity: 128,
                                  rawSlidingWindow: 128, compressedKVCapacity: 1026,
                                  checkpointTokenCount: 1234, layerCount: 43, rawHeadKVDim: 512,
                                  indexerHeadDim: 128, vocabSize: 129280, liveRawRows: 128)
        let bytes = h.serialize()
        XCTAssertEqual(bytes.count, DSV4PayloadHeader.u32Fields * 4)
        XCTAssertEqual(KVCFile.leGet32(bytes, 0), DSV4PayloadHeader.magic)
        XCTAssertEqual(KVCFile.leGet32(bytes, 4), DSV4PayloadHeader.version)
        XCTAssertEqual(DSV4PayloadHeader(bytes), h)
        // Reject wrong magic.
        var bad = bytes; bad[0] = 0
        XCTAssertNil(DSV4PayloadHeader(bad))
    }
}
