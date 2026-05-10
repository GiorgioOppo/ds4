import Foundation
import Metal

public enum DType: Int, Sendable {
    case f32 = 0
    case f16 = 1
    case bf16 = 2
    case i8 = 3
    case i4 = 4   // packed two-per-byte

    public var bitsPerElement: Int {
        switch self {
        case .f32: return 32
        case .f16, .bf16: return 16
        case .i8: return 8
        case .i4: return 4
        }
    }
}

/// Row-major n-dim tensor backed by an `MTLBuffer`. No autograd, no broadcasting.
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

    public var byteCount: Int {
        (count * dtype.bitsPerElement + 7) / 8
    }

    public static func empty(shape: [Int], dtype: DType, on device: Device = .shared) -> Tensor {
        let bytes = (shape.reduce(1, *) * dtype.bitsPerElement + 7) / 8
        guard let buf = device.mtl.makeBuffer(length: max(bytes, 16), options: .storageModeShared) else {
            fatalError("MTLBuffer allocation failed for \(bytes) bytes")
        }
        return Tensor(shape: shape, dtype: dtype, buffer: buf)
    }

    public static func from(bytes: UnsafeRawBufferPointer, shape: [Int], dtype: DType,
                            on device: Device = .shared) -> Tensor {
        let bytesNeeded = (shape.reduce(1, *) * dtype.bitsPerElement + 7) / 8
        precondition(bytes.count >= bytesNeeded, "byte buffer smaller than tensor shape")
        guard let buf = device.mtl.makeBuffer(bytes: bytes.baseAddress!,
                                              length: bytesNeeded,
                                              options: .storageModeShared) else {
            fatalError("MTLBuffer creation failed")
        }
        return Tensor(shape: shape, dtype: dtype, buffer: buf)
    }

    public func reshape(_ newShape: [Int]) -> Tensor {
        precondition(newShape.reduce(1, *) == count, "reshape size mismatch")
        return Tensor(shape: newShape, dtype: dtype, buffer: buffer, offset: offset)
    }

    /// Copy the contents to a host array of `Float`. Only for f32/f16/bf16.
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
            fatalError("toFloatArray not supported for \(dtype)")
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
