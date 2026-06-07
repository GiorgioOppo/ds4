import XCTest
import Foundation
@testable import DS4Metal

/// Stage D (expert-cache): validates that running the routed MoE matvec over a
/// PACKED buffer of only the K selected experts (ids remapped to 0..<K) gives the
/// EXACT same result as running over the full expert set with the real ids. This
/// is the core of the expert-cache: stream ~K/256 of each layer's expert weight.
final class ExpertCacheTests: XCTestCase {
    static let metalDir = "/Users/oppog/Downloads/ds4-main/DS4-gui/metal"

    private func makeRuntime() throws -> MetalRuntime {
        try XCTSkipUnless(FileManager.default.fileExists(atPath: Self.metalDir + "/moe.metal"),
                          "vendored metal kernels not present")
        do { return try MetalRuntime(metalDir: Self.metalDir) }
        catch { throw XCTSkip("Metal unavailable: \(error)") }
    }

    func testPackedExpertsMatchFull() throws {
        let rt = try makeRuntime()
        var seed: UInt64 = 0xEC0
        func nb() -> UInt8 { seed = seed &* 6364136223846793005 &+ 1442695040888963407; return UInt8(truncatingIfNeeded: seed >> 40) }
        func rf() -> Float { seed = seed &* 6364136223846793005 &+ 1442695040888963407; return Float(Int32(truncatingIfNeeded: seed >> 33)) / Float(1 << 30) }

        let inDim = 512, outDim = 128, nExperts = 8
        let rowBytes = (inDim / 256) * 144
        let expertBytes = rowBytes * outDim

        var full = [UInt8](repeating: 0, count: expertBytes * nExperts)
        var off = 0
        for _ in 0..<(nExperts * outDim * (inDim / 256)) {
            let d = Float16(abs(rf()) * 0.05), dmin = Float16(abs(rf()) * 0.02)
            withUnsafeBytes(of: d.bitPattern.littleEndian) { full[off] = $0[0]; full[off+1] = $0[1] }
            withUnsafeBytes(of: dmin.bitPattern.littleEndian) { full[off+2] = $0[0]; full[off+3] = $0[1] }
            for i in 0..<12 { full[off+4+i] = nb() }
            for i in 0..<128 { full[off+16+i] = nb() }
            off += 144
        }
        var activation = [Float](repeating: 0, count: inDim)
        for i in 0..<inDim { activation[i] = rf() }

        let ids: [Int32] = [3, 0, 7, 5, 1, 6]
        let gpuFull = try rt.moeMatvecQ4_K(experts: full, expertIds: ids, activation: activation,
                                           nExperts: nExperts, inDim: inDim, outDim: outDim)

        // Pack only the selected experts (mirrors GGUFWeights.gatherExperts).
        var packed = [UInt8](repeating: 0, count: ids.count * expertBytes)
        for (i, e) in ids.enumerated() {
            let src = Int(e) * expertBytes
            for b in 0..<expertBytes { packed[i * expertBytes + b] = full[src + b] }
        }
        let remapped: [Int32] = Array(0..<Int32(ids.count))
        let gpuPacked = try rt.moeMatvecQ4_K(experts: packed, expertIds: remapped, activation: activation,
                                             nExperts: ids.count, inDim: inDim, outDim: outDim)

        XCTAssertEqual(gpuFull.count, gpuPacked.count)
        for i in 0..<gpuFull.count {
            XCTAssertEqual(gpuFull[i], gpuPacked[i], "packed vs full expert \(i)")
        }
    }
}
