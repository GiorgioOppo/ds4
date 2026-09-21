import XCTest
import Foundation
@testable import DS4Metal

/// CPU contracts for the production-only specialization. GPU parity, guard
/// zones and repeated scratch reuse also run in check_attention_port.swift.
final class GraphQ4PrefillPairTests: XCTestCase {
    func testDeviceGateDoesNotAdmitOtherGenerationsOrVendors() {
        for device in ["Apple M1", "Apple M1 Pro", "Apple M2 Max", "Apple M3 Ultra", "Apple M4"] {
            XCTAssertTrue(eligible(device: device, tokens: 128), device)
        }
        for device in ["Apple M5", "Apple M6", "Apple M10", "Apple M1X",
                       "Apple M0", "M1 Pro", "AMD Apple M1", "Intel", "Apple M", ""] {
            XCTAssertFalse(eligible(device: device, tokens: 128), device)
        }
    }

    func testOnlyCompleteProductionTilesAreEligible() {
        for tokens in stride(from: 32, through: 256, by: 32) {
            XCTAssertTrue(eligible(tokens: tokens))
        }
        for tokens in [Int.min, -32, 0, 1, 8, 16, 31, 33, 63, 127, 255, 257, 512, Int.max] {
            XCTAssertFalse(eligible(tokens: tokens))
        }
        for shape in [(4095, 1024, 512), (4097, 1024, 512), (8192, 1024, 512),
                      (4096, 512, 1024), (4096, 2048, 512), (4096, 1024, 256),
                      (Int.max, 1024, 512), (4096, Int.max, 512), (4096, 1024, Int.min)] {
            XCTAssertFalse(GraphContext.q4PrefillPairEligible(deviceName: "Apple M1 Pro",
                inDim: shape.0, qOutDim: shape.1, kvOutDim: shape.2, nTok: 128))
        }
    }

    func testMatmulABIUsesHalfRHSAndCanonicalQ4Rows() {
        for tokens in [32, 128, 256] {
            for width in [512, 1024] {
                let bytes = GraphContext.q4PrefillPairMMArgs(inDim: 4096, outDim: width, nTok: tokens)
                XCTAssertEqual(bytes.count, 88)
                func word32(_ offset: Int) -> Int32 {
                    bytes.withUnsafeBytes { Int32(littleEndian: $0.loadUnaligned(fromByteOffset: offset, as: Int32.self)) }
                }
                func word64(_ offset: Int) -> UInt64 {
                    bytes.withUnsafeBytes { UInt64(littleEndian: $0.loadUnaligned(fromByteOffset: offset, as: UInt64.self)) }
                }
                XCTAssertEqual(word32(0), 4096)               // reduction dimension
                XCTAssertEqual(word64(8), 2304)               // 16 canonical 144-byte Q4_K blocks
                XCTAssertEqual(word64(40), 2)                 // sizeof(half), not sizeof(float)
                XCTAssertEqual(word64(48), 8192)              // contiguous half activation row
                XCTAssertEqual(word64(56), UInt64(tokens * 8192))
                XCTAssertEqual(word64(64), UInt64(tokens * 8192))
                XCTAssertEqual(word32(72), Int32(width))      // float output row stride
                XCTAssertEqual(word32(76), Int32(tokens))
            }
        }
    }

    private func eligible(device: String = "Apple M1 Pro", tokens: Int) -> Bool {
        GraphContext.q4PrefillPairEligible(deviceName: device,
            inDim: 4096, qOutDim: 1024, kvOutDim: 512, nTok: tokens)
    }
}
