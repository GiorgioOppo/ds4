import Foundation

/// Block-wise int4 quantization layout used for weight storage.
/// Each block of `blockSize` weights stores: one f16 scale, one f16 zero-point,
/// then `blockSize/2` packed nibbles. This is a common GGUF/AWQ-style layout
/// and matches what most safetensors quantized exports use, but the exact
/// layout for V4-Pro must be confirmed once the official weights are inspected.
public enum Quant {
    public static let blockSize = 32

    public static func dequantBytesPerBlock() -> Int {
        // 2 (scale) + 2 (zero) + blockSize/2 (nibbles)
        return 2 + 2 + blockSize / 2
    }
}
