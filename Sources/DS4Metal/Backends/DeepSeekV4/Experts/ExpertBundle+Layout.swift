import Foundation
import Metal
import DS4Core

extension ExpertBundle {
    // MARK: Layout

    static func headerBytes(layerCount: Int) -> Int { 56 + layerCount * 8 }

    static func recordStride(_ gate: Int, _ up: Int, _ down: Int) -> Int {
        (gate + up + down + align - 1) / align * align
    }

    /// FNV-1a over the first 4 KB of a layer's gate-experts tensor (source fd):
    /// the bundle must match the MODEL BYTES, not just its size and shape.
    static func layerHash(fd: Int32, model: GGUFModel, layer: Int) -> UInt64 {
        guard let t = model.findTensor("blk.\(layer).ffn_gate_exps.weight") else { return 0 }
        let n = min(4096, Int(t.bytes))
        var buf = [UInt8](repeating: 0, count: max(1, n))
        let ok = buf.withUnsafeMutableBytes {
            GGUFWeights.preadFull(fd, into: $0.baseAddress!, bytes: n, offset: Int(t.absOffset))
        }
        guard ok else { return 0 }
        var h: UInt64 = 0xcbf2_9ce4_8422_2325
        for b in buf { h = (h ^ UInt64(b)) &* 0x1_0000_0000_01b3 }
        return h
    }
}
