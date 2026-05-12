import Foundation
import Metal

public enum DType: Int, Sendable {
    case f32 = 0
    case f16 = 1
    case bf16 = 2
    case i32 = 3
    case i8 = 4
    /// FP8 E4M3 (1 sign + 4 exp + 3 mantissa). Storage: 1 byte per value.
    case fp8E4M3 = 5
    /// FP4 E2M1 packed two-per-byte. Storage: 1 byte per 2 values.
    case fp4E2M1 = 6
    /// E8M0 unbiased exponent format used as block scale. Storage: 1 byte per value.
    case e8m0 = 7
    /// 64-bit signed/unsigned integer (only used as a load-time staging
    /// dtype for tensors that the downstream code wants as i32 — see
    /// AssemblyHelpers.castIntToI32).
    case i64 = 8
    /// 4-bit signed integer (two's complement, range [-8, 7]) packed
    /// two-per-byte (low nibble = index 2k, high nibble = index 2k+1).
    /// Storage: 1 byte per 2 values. Symmetric per-row × per-128-block
    /// F16 scales, mirroring the INT8 layout. See Int4Quant.swift.
    case i4 = 9
    /// 2-bit signed integer (two's complement, range [-2, 1]) packed
    /// four-per-byte (LSB-first: index 4k in bits [1:0], 4k+1 in [3:2],
    /// 4k+2 in [5:4], 4k+3 in [7:6]). Storage: 1 byte per 4 values.
    /// Symmetric per-row × per-128-block F16 scales. See Int2Quant.swift.
    case i2 = 10

    public var bitsPerElement: Int {
        switch self {
        case .f32, .i32: return 32
        case .f16, .bf16: return 16
        case .i8, .fp8E4M3, .e8m0: return 8
        case .fp4E2M1, .i4: return 4
        case .i64: return 64
        case .i2: return 2
        }
    }
}

/// Row-major n-dim tensor backed by an `MTLBuffer`.
public final class Tensor {
    public let shape: [Int]
    public let dtype: DType
    public let buffer: MTLBuffer
    public let offset: Int

    public init(shape: [Int], dtype: DType, buffer: MTLBuffer, offset: Int = 0) {
        self.shape = shape
        self.dtype = dtype
        self.buffer = buffer
        self.offset = offset
    }

    public var count: Int { shape.reduce(1, *) }

    public var byteCount: Int { (count * dtype.bitsPerElement + 7) / 8 }

    public static func empty(shape: [Int], dtype: DType, on device: Device = .shared) -> Tensor {
        let bytes = max(((shape.reduce(1, *) * dtype.bitsPerElement) + 7) / 8, 16)
        guard let buf = device.mtl.makeBuffer(length: bytes, options: .storageModeShared) else {
            fatalError("MTLBuffer allocation failed for \(bytes) bytes")
        }
        return Tensor(shape: shape, dtype: dtype, buffer: buf)
    }

    public static func from(bytes: UnsafeRawBufferPointer, shape: [Int], dtype: DType,
                            on device: Device = .shared) -> Tensor {
        let needed = (shape.reduce(1, *) * dtype.bitsPerElement + 7) / 8
        precondition(bytes.count >= needed, "byte buffer smaller than tensor")
        guard let buf = device.mtl.makeBuffer(bytes: bytes.baseAddress!,
                                              length: needed,
                                              options: .storageModeShared) else {
            fatalError("MTLBuffer creation failed")
        }
        return Tensor(shape: shape, dtype: dtype, buffer: buf)
    }

    public func reshape(_ newShape: [Int]) -> Tensor {
        precondition(newShape.reduce(1, *) == count, "reshape size mismatch")
        return Tensor(shape: newShape, dtype: dtype, buffer: buffer, offset: offset)
    }

    /// Copy contents to a host array of `Float`. For inspection / tests only —
    /// quantized dtypes (fp8/fp4/e8m0) require a dequant pass that has not been
    /// written yet, so this path traps for them.
    public func toFloatArray() -> [Float] {
        let n = count
        let raw = buffer.contents().advanced(by: offset)
        switch dtype {
        case .f32:
            let p = raw.bindMemory(to: Float.self, capacity: n)
            return Array(UnsafeBufferPointer(start: p, count: n))
        case .f16:
            let p = raw.bindMemory(to: UInt16.self, capacity: n)
            return (0..<n).map { Self.halfToFloat(p[$0]) }
        case .bf16:
            let p = raw.bindMemory(to: UInt16.self, capacity: n)
            return (0..<n).map { Float(bitPattern: UInt32(p[$0]) << 16) }
        default:
            fatalError("toFloatArray not supported for \(dtype) — needs dequant kernel")
        }
    }

    private static func halfToFloat(_ h: UInt16) -> Float {
        let sign = UInt32(h >> 15) & 0x1
        let exp = UInt32(h >> 10) & 0x1F
        let mant = UInt32(h) & 0x3FF
        var f: UInt32
        if exp == 0 {
            if mant == 0 {
                f = sign << 31
            } else {
                var e: UInt32 = 0
                var m = mant
                while (m & 0x400) == 0 { m <<= 1; e &+= 1 }
                m &= 0x3FF
                f = (sign << 31) | ((127 &- 15 &- e &+ 1) << 23) | (m << 13)
            }
        } else if exp == 31 {
            f = (sign << 31) | (0xFF << 23) | (mant << 13)
        } else {
            f = (sign << 31) | ((exp &+ 127 &- 15) << 23) | (mant << 13)
        }
        return Float(bitPattern: f)
    }
}
