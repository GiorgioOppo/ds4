import Foundation
import Metal

/// Block-wise activation quantization. Mirrors `act_quant` (FP8) and
/// `fp4_act_quant` (FP4) in
/// `Reference/inference/kernel.py` lines 105–200.
///
/// Two output modes:
///   - `inplace == true`:  round-trip quant→dequant back into the input
///                         buffer (QAT noise injection)
///   - `inplace == false`: write the quantized bytes (fp8 or packed fp4)
///                         and the per-block scales (stored as f32 here;
///                         to be repacked to E8M0 once we have a pipeline
///                         that consumes E8M0 directly).
public final class ActQuant {
    public enum Format { case fp8, fp4 }

    public let format: Format
    public let blockSize: Int
    private let pipeline: MTLComputePipelineState

    public init(format: Format) {
        self.format = format
        self.blockSize = format == .fp8 ? Quant.actBlockSizeFP8 : Quant.actBlockSizeFP4
        let consts = MTLFunctionConstantValues()
        var bs = UInt32(self.blockSize)
        consts.setConstantValue(&bs, type: .uint, index: format == .fp8 ? 0 : 10)
        let lib = Device.shared.library
        let name = format == .fp8 ? "act_quant_fp8" : "act_quant_fp4"
        do {
            let fn = try lib.makeFunction(name: name, constantValues: consts)
            self.pipeline = try Device.shared.mtl.makeComputePipelineState(function: fn)
        } catch {
            fatalError("ActQuant pipeline failed for \(name): \(error)")
        }
    }

    public struct Output {
        public let inplace: Tensor?      // [M, N] f32 (round-tripped) when inplace
        public let qbytes: Tensor?       // [M, N] u8 fp8 or [M, N/2] u8 fp4 packed
        public let scales: Tensor        // [M, N/blockSize] f32 (E8M0-rounded but stored as f32)
    }

    public func quant(_ x: Tensor, inplace: Bool, in cmd: MTLCommandBuffer) -> Output {
        precondition(x.dtype == .f32 && x.shape.count == 2)
        let M = x.shape[0]
        let N = x.shape[1]
        precondition(N % blockSize == 0, "N must be a multiple of \(blockSize)")
        let blocksPerRow = N / blockSize

        let yIp: Tensor? = inplace ? Tensor.empty(shape: [M, N], dtype: .f32) : nil
        let qbytes: Tensor?
        if inplace {
            qbytes = nil
        } else {
            switch format {
            case .fp8: qbytes = Tensor.empty(shape: [M, N], dtype: .i8)
            case .fp4: qbytes = Tensor.empty(shape: [M, N / 2], dtype: .i8)
            }
        }
        let scales = Tensor.empty(shape: [M, blocksPerRow], dtype: .f32)

        // Dummy buffers for the unused output slot — Metal requires every
        // bound buffer index to point at something valid.
        let zero = Device.shared.mtl.makeBuffer(length: 16, options: .storageModeShared)!

        let enc = cmd.makeComputeCommandEncoder()!
        enc.setComputePipelineState(pipeline)
        enc.setBuffer(x.buffer, offset: x.offset, index: 0)
        enc.setBuffer(qbytes?.buffer ?? zero, offset: 0, index: 1)
        enc.setBuffer(yIp?.buffer ?? zero, offset: 0, index: 2)
        enc.setBuffer(scales.buffer, offset: 0, index: 3)
        var dims = SIMD2<UInt32>(UInt32(M), UInt32(N))
        enc.setBytes(&dims, length: MemoryLayout.size(ofValue: dims), index: 4)
        var ip: UInt32 = inplace ? 1 : 0
        enc.setBytes(&ip, length: 4, index: 5)
        enc.setThreadgroupMemoryLength(blockSize * MemoryLayout<Float>.size, index: 0)

        let tg = MTLSize(width: blockSize, height: 1, depth: 1)
        let grid = MTLSize(width: M, height: blocksPerRow, depth: 1)
        enc.dispatchThreadgroups(grid, threadsPerThreadgroup: tg)
        enc.endEncoding()

        return Output(inplace: yIp, qbytes: qbytes, scales: scales)
    }

    // MARK: - Pure-Swift reference

    public static func referenceCPU(_ row: [Float], format: Format,
                                    blockSize: Int) -> (rt: [Float], scales: [Float]) {
        precondition(row.count % blockSize == 0)
        var out = [Float](repeating: 0, count: row.count)
        var scales = [Float](repeating: 0, count: row.count / blockSize)
        let maxV: Float = format == .fp8 ? 448.0 : 6.0
        let amaxFloor: Float = format == .fp8 ? 1e-4 : 6.0 * pow(2.0, -126)

        for b in 0..<scales.count {
            let lo = b * blockSize
            let hi = lo + blockSize
            var amax: Float = 0
            for i in lo..<hi { amax = max(amax, abs(row[i])) }
            amax = max(amax, amaxFloor)
            // round-pow2: 2^ceil(log2(amax / maxV))
            let v = amax / maxV
            let bits = v.bitPattern
            let exp = Int((bits >> 23) & 0xFF)
            let mant = bits & 0x7FFFFF
            let log2ceil = exp - 127 + (mant != 0 ? 1 : 0)
            let scale = Float(bitPattern: UInt32(log2ceil + 127) << 23)
            scales[b] = scale

            for i in lo..<hi {
                let clipped = max(-maxV, min(maxV, row[i] / scale))
                let rounded: Float
                switch format {
                case .fp8:
                    rounded = roundToE4M3(clipped)
                case .fp4:
                    rounded = roundToE2M1(clipped)
                }
                out[i] = rounded * scale
            }
        }
        return (out, scales)
    }

    private static func roundToE4M3(_ x: Float) -> Float {
        let h = Float16(x)
        let bits = h.bitPattern
        let sign = (bits >> 15) & 1
        let exp16 = Int((bits >> 10) & 0x1F)
        let mant10 = UInt32(bits & 0x3FF)

        if exp16 == 0 && mant10 == 0 { return sign == 1 ? -0.0 : 0.0 }

        var newExp = exp16 - 15 + 7
        if newExp <= 0 {
            if exp16 == 0 { return sign == 1 ? -0.0 : 0.0 }
            let shift = 17 - exp16
            let full = 1024 + Int(mant10)
            var result = full >> shift
            let roundBit = (full >> (shift - 1)) & 1
            let sticky = full & ((1 << (shift - 1)) - 1)
            if roundBit != 0 && (sticky != 0 || (result & 1) != 0) { result += 1 }
            if result >= 8 {
                let bits = (UInt32(sign) << 31) | ((121) << 23)
                let v = Float(bitPattern: bits)
                return sign == 1 ? -v : v
            }
            let v = Float(result) * 0x1p-9
            return sign == 1 ? -v : v
        }

        var mant3 = Int(mant10 >> 7)
        let roundBit = Int((mant10 >> 6) & 1)
        let sticky = Int(mant10 & 0x3F)
        if roundBit != 0 && (sticky != 0 || (mant3 & 1) != 0) {
            mant3 += 1
            if mant3 == 8 { mant3 = 0; newExp += 1 }
        }
        if newExp >= 16 || (newExp == 15 && mant3 == 7) {
            let v: Float = 448.0
            return sign == 1 ? -v : v
        }
        let bits = (UInt32(sign) << 31) | (UInt32(newExp + 120) << 23) | (UInt32(mant3) << 20)
        return Float(bitPattern: bits)
    }

    private static func roundToE2M1(_ x: Float) -> Float {
        let a = abs(x)
        let mag: Float
        if      a < 0.25 { mag = 0 }
        else if a < 0.75 { mag = 0.5 }
        else if a < 1.25 { mag = 1 }
        else if a < 1.75 { mag = 1.5 }
        else if a < 2.5  { mag = 2 }
        else if a < 3.5  { mag = 3 }
        else if a < 5.0  { mag = 4 }
        else             { mag = 6 }
        return x < 0 ? -mag : mag
    }
}
