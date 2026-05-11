import XCTest
import Foundation
@testable import DeepSeekKit

/// Round-trip validation of the converter-side INT8 quantizer:
///   1. write a BF16 tensor to a temporary file in raw layout,
///   2. run `quantizeBF16ToInt8` on it,
///   3. write the produced I8 weight + F16 scale through `SafeTensorsWriter`,
///   4. read both back via `SafeTensorsFile`,
///   5. dequantize on CPU and verify max relative error per row is bounded by
///      the worst-case INT8 RTN error of `1 / 254`.
final class Int8ConverterTests: XCTestCase {

    func testShouldQuantizeWhitelist() {
        XCTAssertTrue(shouldQuantizeToInt8("layers.3.attn.wq.weight", lastDim: 128))
        XCTAssertTrue(shouldQuantizeToInt8("layers.0.ffn.experts.5.w1.weight", lastDim: 256))
        XCTAssertTrue(shouldQuantizeToInt8("layers.3.attn.wo_b.weight", lastDim: 512))
        XCTAssertTrue(shouldQuantizeToInt8("layers.3.attn.wkv.weight", lastDim: 4096))
        XCTAssertTrue(shouldQuantizeToInt8("layers.3.attn.compressor.wgate.weight", lastDim: 4096))

        XCTAssertFalse(shouldQuantizeToInt8("layers.0.attn_norm.weight", lastDim: 4096))
        XCTAssertFalse(shouldQuantizeToInt8("embed.weight", lastDim: 4096))
        XCTAssertFalse(shouldQuantizeToInt8("head.weight", lastDim: 4096))
        XCTAssertFalse(shouldQuantizeToInt8("layers.3.attn.attn_sink", lastDim: 128))
        XCTAssertFalse(shouldQuantizeToInt8("layers.3.attn.wq.scale", lastDim: 128))
        // wo_a is excluded — its weight feeds Einsum.bsgdGrd which needs F32.
        XCTAssertFalse(shouldQuantizeToInt8("layers.3.attn.wo_a.weight", lastDim: 128))
        // gate / weights_proj kept BF16 — small routing scores, RTN noise hurts topk.
        XCTAssertFalse(shouldQuantizeToInt8("layers.3.ffn.gate.weight", lastDim: 4096))
        XCTAssertFalse(shouldQuantizeToInt8("layers.3.attn.indexer.weights_proj.weight", lastDim: 4096))
        // K not divisible by 128 → fall back.
        XCTAssertFalse(shouldQuantizeToInt8("layers.3.attn.wq.weight", lastDim: 100))
    }

    func testQuantizeBF16RoundTrip() throws {
        let outDim = 16
        let inDim = 256
        let blocksIn = inDim / 128

        // Generate deterministic BF16 source.
        let raw = randomArray(outDim * inDim, seed: 7, scale: 0.5)
        let bf16: [UInt16] = raw.map { floatToBF16Local($0) }
        let bf16AsFloat: [Float] = bf16.map { Float(bitPattern: UInt32($0) << 16) }

        // Write the BF16 array to a tmp file as raw bytes; the quantizer
        // mmaps/seeks at `srcOffset` — we put nothing before the payload.
        let tmpURL = try writeTempBytes(bf16.withUnsafeBytes { Data($0) })
        defer { try? FileManager.default.removeItem(at: tmpURL) }

        let (weightBytes, scaleBytes) = try quantizeBF16ToInt8(
            srcURL: tmpURL, srcOffset: 0, outDim: outDim, inDim: inDim)

        XCTAssertEqual(weightBytes.count, outDim * inDim)
        XCTAssertEqual(scaleBytes.count, outDim * blocksIn * 2)

        // Dequantize on CPU using the produced bytes and verify max abs
        // error per element is bounded by `s/2` (RTN worst case) in its
        // own group.
        let weight: [Int8] = weightBytes.withUnsafeBytes {
            Array($0.bindMemory(to: Int8.self))
        }
        let scaleF16: [UInt16] = scaleBytes.withUnsafeBytes {
            Array($0.bindMemory(to: UInt16.self))
        }
        for i in 0..<outDim {
            for kb in 0..<blocksIn {
                let s = halfToFloat(scaleF16[i * blocksIn + kb])
                let k0 = kb * 128
                // Compute the original row max-abs in this group for the
                // expected per-element error bound.
                var maxAbs: Float = 0
                for k in 0..<128 {
                    maxAbs = max(maxAbs, abs(bf16AsFloat[i * inDim + k0 + k]))
                }
                let bound = max(maxAbs / 254.0, 1e-6) * 1.01
                for k in 0..<128 {
                    let dq = Float(weight[i * inDim + k0 + k]) * s
                    let orig = bf16AsFloat[i * inDim + k0 + k]
                    let err = abs(dq - orig)
                    XCTAssertLessThanOrEqual(
                        err, bound,
                        "i=\(i) k=\(k0+k) dq=\(dq) orig=\(orig) bound=\(bound)")
                }
            }
        }
    }

    /// Writer + reader round-trip with `dtype="I8"` and `dtype="F16"`.
    /// Verifies SafeTensorsWriter emits valid headers for these dtypes and
    /// SafeTensorsFile parses them back into `.i8` / `.f16` tensors with
    /// identical bytes.
    func testSafetensorsRoundTripI8AndF16() throws {
        let outDim = 8, inDim = 128, blocksIn = 1
        let raw = randomArray(outDim * inDim, seed: 13, scale: 0.5)
        let bf16: [UInt16] = raw.map { floatToBF16Local($0) }
        let srcURL = try writeTempBytes(bf16.withUnsafeBytes { Data($0) })
        defer { try? FileManager.default.removeItem(at: srcURL) }

        let (wBytes, sBytes) = try quantizeBF16ToInt8(
            srcURL: srcURL, srcOffset: 0, outDim: outDim, inDim: inDim)

        let writer = SafeTensorsWriter()
        writer.add(name: "x.weight", dtype: "I8", shape: [outDim, inDim],
                   source: .data(wBytes))
        writer.add(name: "x.scale", dtype: "F16", shape: [outDim, blocksIn],
                   source: .data(sBytes))
        let stURL = FileManager.default.temporaryDirectory
            .appendingPathComponent("int8_roundtrip_\(UUID().uuidString).safetensors")
        try writer.write(to: stURL)
        defer { try? FileManager.default.removeItem(at: stURL) }

        let f = try SafeTensorsFile(url: stURL)
        let w = try f.load("x.weight")
        let s = try f.load("x.scale")
        XCTAssertEqual(w.dtype, .i8)
        XCTAssertEqual(w.shape, [outDim, inDim])
        XCTAssertEqual(s.dtype, .f16)
        XCTAssertEqual(s.shape, [outDim, blocksIn])

        // Byte-for-byte equality with the produced buffers.
        let wPtr = w.buffer.contents().advanced(by: w.offset)
            .bindMemory(to: UInt8.self, capacity: outDim * inDim)
        let sPtr = s.buffer.contents().advanced(by: s.offset)
            .bindMemory(to: UInt8.self, capacity: outDim * blocksIn * 2)
        for i in 0..<(outDim * inDim) {
            XCTAssertEqual(wPtr[i], wBytes[i], "weight byte \(i)")
        }
        for i in 0..<(outDim * blocksIn * 2) {
            XCTAssertEqual(sPtr[i], sBytes[i], "scale byte \(i)")
        }
    }

    // ---- helpers ----

    private func writeTempBytes(_ data: Data) throws -> URL {
        let url = FileManager.default.temporaryDirectory
            .appendingPathComponent("int8_conv_\(UUID().uuidString).bin")
        try data.write(to: url)
        return url
    }

    private func randomArray(_ count: Int, seed: UInt64, scale: Float = 1.0) -> [Float] {
        var state = seed | 1
        var out = [Float](repeating: 0, count: count)
        for i in 0..<count {
            state = state &* 6364136223846793005 &+ 1442695040888963407
            let frac = Float(Double(state >> 11) / Double(1 << 53))
            out[i] = (frac - 0.5) * 2 * scale
        }
        return out
    }

    private func floatToBF16Local(_ f: Float) -> UInt16 {
        let bits = f.bitPattern
        let rounded = bits &+ ((bits >> 16) & 1) &+ 0x7FFF
        return UInt16(truncatingIfNeeded: rounded >> 16)
    }

    private func halfToFloat(_ h: UInt16) -> Float {
        let sign = UInt32(h >> 15) & 0x1
        let exp = UInt32(h >> 10) & 0x1F
        let mant = UInt32(h) & 0x3FF
        var f: UInt32
        if exp == 0 {
            if mant == 0 {
                f = sign << 31
            } else {
                var e: UInt32 = 1
                var m = mant
                while m & 0x400 == 0 { m <<= 1; e += 1 }
                m &= 0x3FF
                f = (sign << 31) | ((127 - 15 - e + 1) << 23) | (m << 13)
            }
        } else if exp == 0x1F {
            f = (sign << 31) | (0xFF << 23) | (mant << 13)
        } else {
            f = (sign << 31) | ((exp + 127 - 15) << 23) | (mant << 13)
        }
        return Float(bitPattern: f)
    }
}
