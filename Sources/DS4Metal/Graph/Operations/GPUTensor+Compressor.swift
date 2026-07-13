import Foundation
import Metal

extension GPUTensor {
    /// Overwrite the first `floatCount` floats with `value` (CPU upload).
    func fill(_ rt: MetalRuntime, value: Float, floatCount: Int) throws {
        let p = buffer.contents().advanced(by: byteOffset).bindMemory(to: Float.self, capacity: floatCount)
        for i in 0..<floatCount { p[i] = value }
    }
    /// A sub-view starting at row `row` of width `cols` floats (same backing buffer).
    func rowView(row: Int, cols: Int) -> GPUTensor {
        GPUTensor(buffer: buffer, byteLength: cols * 4, count: cols, byteOffset: byteOffset + row * cols * 4)
    }
}
