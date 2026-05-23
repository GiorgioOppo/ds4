import Foundation
import MLX

public enum DType: Int, Sendable {
    case f32 = 0
    case f16 = 1
    case bf16 = 2
    case i32 = 3
    case i8 = 4
    case fp8E4M3 = 5
    case fp4E2M1 = 6
    case e8m0 = 7
    case i64 = 8
    case i4 = 9
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

    public var mlxDType: MLX.DType {
        switch self {
        case .f32: return .float32
        case .f16: return .float16
        case .bf16: return .bfloat16
        case .i32: return .int32
        case .i8: return .int8
        case .i64: return .int64
        case .fp8E4M3, .fp4E2M1, .e8m0, .i4, .i2: return .uint8
        }
    }
}

public final class Tensor {
    public var array: MLXArray
    public let dtype: DType

    public var shape: [Int] {
        array.shape
    }

    public init(array: MLXArray, dtype: DType) {
        self.array = array
        self.dtype = dtype
    }

    public var count: Int { array.size }

    public var byteCount: Int { (count * dtype.bitsPerElement + 7) / 8 }

    public static func empty(shape: [Int], dtype: DType) -> Tensor {
        let arr = MLXArray.zeros(shape).asType(dtype.mlxDType)
        return Tensor(array: arr, dtype: dtype)
    }

    public static func from(bytes: UnsafeRawBufferPointer, shape: [Int], dtype: DType) -> Tensor {
        let data = Data(bytes: bytes.baseAddress!, count: bytes.count)
        let arr = MLXArray(data, shape, dtype: dtype.mlxDType)
        return Tensor(array: arr, dtype: dtype)
    }

    public func reshape(_ newShape: [Int]) -> Tensor {
        return Tensor(array: array.reshaped(newShape), dtype: dtype)
    }

    public func readBytes() -> Data {
        MLX.eval(array)
        let n = byteCount
        var data = Data(count: n)
        data.withUnsafeMutableBytes { ptr in
            // Fallback since mlx-swift might not have a direct bytes copy like this,
            // but we assume it has an asData() or we can copy from asArray().
            // Let's use asData() if available or rely on compiler errors later
        }
        // Actually MLXArray has a `asData()` method
        return array.asData(access: .copy).data
    }

    public func writeBytes(_ data: Data) {
        self.array = MLXArray(data, shape, dtype: dtype.mlxDType)
    }

    public func toFloatArray() -> [Float] {
        MLX.eval(array)
        if dtype == .f32 || dtype == .f16 || dtype == .bf16 {
            return array.asArray(Float.self)
        }
        fatalError("toFloatArray not supported for \(dtype)")
    }
}
