import XCTest
import CDS4
@testable import DS4Core

/// Cross-checks the Swift SSD planning port against the C originals in
/// ds4_ssd.c for a wide range of inputs. If these pass, the Swift behavior is
/// byte-for-byte faithful.
final class SSDCachePlanTests: XCTestCase {

    // MARK: parseGiBArg vs ds4_parse_gib_arg

    func testParseGiBArgMatchesC() {
        let inputs = ["", "0", "10", "10GB", "10gb", "10Gb", "10gB", "GB", "10G",
                      "G", "B", "abc", "1.5GB", " 10", "10 ", "32GB", "1",
                      "4096", "18446744073709551615", "999999999999GB",
                      "18446744073709551616", "00010", "007"]
        for s in inputs {
            var cBytes: UInt64 = 0
            let cOk = s.withCString { ds4_parse_gib_arg($0, &cBytes) }
            let swift = SSDStreaming.parseGiBArg(s)
            if cOk {
                XCTAssertEqual(swift, cBytes, "parseGiBArg mismatch for \"\(s)\"")
            } else {
                XCTAssertNil(swift, "parseGiBArg should be nil for \"\(s)\" (C rejected)")
            }
        }
    }

    // MARK: parseCacheExpertsArg vs ds4_parse_streaming_cache_experts_arg

    func testParseCacheExpertsArgMatchesC() {
        let inputs = ["", "0", "32GB", "4854", "16GB", "100", "GB", "abc",
                      "4294967295", "4294967296", "1gb", "2GB", "999",
                      "18446744073709551615", "64gb"]
        for s in inputs {
            var cExperts: UInt32 = 0
            var cBytes: UInt64 = 0
            let cOk = s.withCString {
                ds4_parse_streaming_cache_experts_arg($0, &cExperts, &cBytes)
            }
            let swift = SSDStreaming.parseCacheExpertsArg(s)
            if cOk {
                switch swift {
                case .bytes(let b):
                    XCTAssertEqual(cBytes, b, "bytes mismatch for \"\(s)\"")
                    XCTAssertEqual(cExperts, 0, "C experts should be 0 for byte spec \"\(s)\"")
                case .experts(let e):
                    XCTAssertEqual(cExperts, e, "experts mismatch for \"\(s)\"")
                    XCTAssertEqual(cBytes, 0, "C bytes should be 0 for expert spec \"\(s)\"")
                case nil:
                    XCTFail("Swift returned nil but C accepted \"\(s)\"")
                }
            } else {
                XCTAssertNil(swift, "parseCacheExpertsArg should be nil for \"\(s)\"")
            }
        }
    }

    // MARK: cacheExpertsForByteBudget vs C

    func testCacheExpertsForByteBudgetMatchesC() {
        let cases: [(UInt64, UInt64)] = [
            (0, 0), (0, 100), (100, 0),
            (32 * SSDStreaming.gib, 13_500_000),
            (2 * SSDStreaming.gib, 13_500_000),
            (1, 1), (1000, 7), (UInt64.max, 1), (UInt64.max, UInt64.max),
            (14_000_000, 13_500_000),
        ]
        for (bytes, per) in cases {
            let c = ds4_ssd_cache_experts_for_byte_budget(bytes, per)
            let swift = SSDStreaming.cacheExpertsForByteBudget(bytes: bytes, perExpertBytes: per)
            XCTAssertEqual(c, swift, "cacheExperts mismatch for bytes=\(bytes) per=\(per)")
        }
    }

    // MARK: autoCachePlan vs ds4_ssd_auto_cache_plan

    func testAutoCachePlanMatchesC() {
        let cases: [(UInt64, UInt64, UInt64, UInt64)] = [
            (0, 0, 0, 0),
            (64 * SSDStreaming.gib, 10 * SSDStreaming.gib, 13_500_000, 256_000),
            (16 * SSDStreaming.gib, 8 * SSDStreaming.gib, 13_500_000, 0),
            (128 * SSDStreaming.gib, 20 * SSDStreaming.gib, 6_750_000, 100_000),
            (10 * SSDStreaming.gib, 20 * SSDStreaming.gib, 13_500_000, 256_000), // nonRouted > target
            (100, 0, 13_500_000, 0),
            (64 * SSDStreaming.gib, 0, 13_500_000, 4),
            (UInt64.max, 1 * SSDStreaming.gib, 13_500_000, 0),
        ]
        for (rec, nonRouted, per, maxExp) in cases {
            var cPlan = ds4_ssd_cache_plan()
            let cOk = ds4_ssd_auto_cache_plan(rec, nonRouted, per, maxExp, &cPlan)
            let swift = SSDStreaming.autoCachePlan(recommendedBytes: rec,
                                                   nonRoutedBytes: nonRouted,
                                                   perExpertBytes: per,
                                                   maxModelExperts: maxExp)
            let label = "rec=\(rec) nonRouted=\(nonRouted) per=\(per) max=\(maxExp)"
            if cOk {
                guard let s = swift else { XCTFail("Swift nil but C ok: \(label)"); continue }
                XCTAssertEqual(cPlan.model_target_bytes, s.modelTargetBytes, "modelTarget \(label)")
                XCTAssertEqual(cPlan.cache_bytes, s.cacheBytes, "cacheBytes \(label)")
                XCTAssertEqual(cPlan.effective_cache_bytes, s.effectiveCacheBytes, "effective \(label)")
                XCTAssertEqual(cPlan.cache_experts, s.cacheExperts, "experts \(label)")
            } else {
                XCTAssertNil(swift, "Swift should be nil when C rejects: \(label)")
            }
        }
    }
}
