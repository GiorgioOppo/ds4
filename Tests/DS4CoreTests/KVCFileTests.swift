import XCTest
import CDS4
@testable import DS4Core

/// Cross-checks the KVC file-format port against the C ds4_kvstore_* functions.
final class KVCFileTests: XCTestCase {

    func testLEHelpersMatchC() {
        let values: [UInt32] = [0, 1, 255, 256, 0x1234_5678, 0xFFFF_FFFF, 0xDEAD_BEEF]
        for v in values {
            var cbuf = [UInt8](repeating: 0, count: 4)
            cbuf.withUnsafeMutableBufferPointer { ds4_kvstore_le_put32($0.baseAddress, v) }
            var sbuf = [UInt8](repeating: 0, count: 4)
            KVCFile.lePut32(&sbuf, 0, v)
            XCTAssertEqual(sbuf, cbuf, "le_put32 \(v)")
            let cget = cbuf.withUnsafeBufferPointer { ds4_kvstore_le_get32($0.baseAddress) }
            XCTAssertEqual(KVCFile.leGet32(sbuf, 0), cget, "le_get32 \(v)")
        }
    }

    func testFillHeaderMatchesC() {
        let h = KVCFile.Header(quantBits: 4, reason: 1, extFlags: 0b101, modelId: 0,
                               tokens: 12345, hits: 7, ctxSize: 100000,
                               createdAt: 1_700_000_000, lastUsed: 1_700_009_999,
                               payloadBytes: 987_654_321)
        var cbuf = [UInt8](repeating: 0, count: KVCFile.fixedHeader)
        cbuf.withUnsafeMutableBufferPointer {
            ds4_kvstore_fill_header($0.baseAddress, h.modelId, h.quantBits, h.reason, h.extFlags,
                                    h.tokens, h.hits, h.ctxSize, h.createdAt, h.lastUsed, h.payloadBytes)
        }
        XCTAssertEqual(KVCFile.fillHeader(h), cbuf)

        // Parse the C-produced bytes back; fields must round-trip.
        let parsed = KVCFile.parseHeader(cbuf)
        XCTAssertEqual(parsed, h)
    }

    func testParseRejectsBadHeaders() {
        var b = KVCFile.fillHeader(KVCFile.Header(quantBits: 4, reason: 1, extFlags: 0, modelId: 0,
                                                  tokens: 10, hits: 0, ctxSize: 1, createdAt: 0,
                                                  lastUsed: 0, payloadBytes: 0))
        var bad = b; bad[0] = 0x00
        XCTAssertNil(KVCFile.parseHeader(bad))         // wrong magic
        bad = b; bad[3] = 9
        XCTAssertNil(KVCFile.parseHeader(bad))         // wrong version
        bad = b; KVCFile.lePut32(&bad, 8, 0)
        XCTAssertNil(KVCFile.parseHeader(bad))         // tokens == 0
        b[4] = 3
        XCTAssertNil(KVCFile.parseHeader(b))           // quant not 2/4
    }

    func testReasonCodeMatchesC() {
        for r in ["cold", "continued", "evict", "shutdown", "agent-system", "agent-session", "bogus", ""] {
            let c = r.withCString { ds4_kvstore_reason_code($0) }
            XCTAssertEqual(KVCFile.reasonCode(r), c, "reason \(r)")
        }
        XCTAssertEqual(KVCFile.reasonCode(nil), ds4_kvstore_reason_code(nil))
    }

    func testKeyKindMatchesC() {
        for flags: UInt8 in [0, 1, 2, 4, 6, 7, 3] {
            let c = String(cString: ds4_kvstore_key_kind(flags))
            XCTAssertEqual(KVCFile.keyKind(extFlags: flags), c, "flags \(flags)")
        }
    }

    func testSHA1MatchesC() {
        let inputs: [[UInt8]] = [
            [], Array("".utf8), Array("abc".utf8), Array("The quick brown fox".utf8),
            Array(repeating: 0x41, count: 1000), Array(0..<UInt8(255)),
        ]
        for input in inputs {
            var out = [CChar](repeating: 0, count: 41)
            input.withUnsafeBytes { raw in
                _ = out.withUnsafeMutableBufferPointer {
                    ds4_kvstore_sha1_bytes_hex(raw.baseAddress, raw.count, $0.baseAddress)
                }
            }
            let cHex = String(cString: out)
            XCTAssertEqual(KVCFile.sha1Hex(input), cHex, "sha1 of \(input.count) bytes")
        }
    }

    func testShaHexNameMatchesC() {
        let names = [
            "0123456789abcdef0123456789abcdef01234567.kv",   // valid
            "0123456789ABCDEF0123456789ABCDEF01234567.kv",   // valid, uppercase
            "short.kv", "0123456789abcdef0123456789abcdef01234567.txt",
            "0123456789abcdef0123456789abcdef0123456g.kv",   // non-hex
            "0123456789abcdef0123456789abcdef012345678.kv",  // wrong length
        ]
        for name in names {
            var csha = [CChar](repeating: 0, count: 41)
            let cok = name.withCString { n in
                csha.withUnsafeMutableBufferPointer { ds4_kvstore_sha_hex_name(n, $0.baseAddress) }
            }
            let swift = KVCFile.shaHexName(name)
            if cok {
                XCTAssertEqual(swift, String(cString: csha), "sha_hex_name \(name)")
            } else {
                XCTAssertNil(swift, "sha_hex_name should reject \(name)")
            }
        }
    }

    func testDefaultOptionsMatchC() {
        let c = ds4_kvstore_default_options()
        let s = KVCFile.defaultOptions()
        XCTAssertEqual(Int(c.min_tokens), s.minTokens)
        XCTAssertEqual(Int(c.cold_max_tokens), s.coldMaxTokens)
        XCTAssertEqual(Int(c.continued_interval_tokens), s.continuedIntervalTokens)
        XCTAssertEqual(Int(c.boundary_trim_tokens), s.boundaryTrimTokens)
        XCTAssertEqual(Int(c.boundary_align_tokens), s.boundaryAlignTokens)
    }

    func testEvictionScoreMatchesC() {
        let now: UInt64 = 2_000_000_000
        let entries: [KVCFile.Entry] = [
            .init(hits: 0, tokens: 1000, fileSize: 1_000_000, createdAt: now, lastUsed: now, reason: 1),
            .init(hits: 5, tokens: 8000, fileSize: 5_000_000, createdAt: now - 3600, lastUsed: now - 3600, reason: 2),
            .init(hits: 100, tokens: 30000, fileSize: 80_000_000, createdAt: now - 86400, lastUsed: now - 86400, reason: 3),
            .init(hits: 3, tokens: 500, fileSize: 200_000, createdAt: now - 100000, lastUsed: 0, reason: 0),
            .init(hits: 0, tokens: 1, fileSize: 0, createdAt: 0, lastUsed: 0, reason: 4),
            .init(hits: 50, tokens: 12000, fileSize: 9_000_000, createdAt: now - 21600, lastUsed: now - 21600, reason: 1),
        ]
        for e in entries {
            var ce = ds4_kvstore_entry()
            ce.hits = e.hits; ce.tokens = e.tokens; ce.file_size = e.fileSize
            ce.created_at = e.createdAt; ce.last_used = e.lastUsed; ce.reason = e.reason
            let cscore = ds4_kvstore_entry_eviction_score(&ce, nil, now, nil)
            let sscore = KVCFile.evictionScore(e, now: now)
            XCTAssertEqual(sscore, cscore, accuracy: 1e-12, "eviction score hits=\(e.hits) reason=\(e.reason)")
        }
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
        let parsed = DSV4PayloadHeader(bytes)
        XCTAssertEqual(parsed, h)
        // Reject wrong magic.
        var bad = bytes; bad[0] = 0
        XCTAssertNil(DSV4PayloadHeader(bad))
    }
}
