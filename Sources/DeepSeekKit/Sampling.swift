import Foundation
import Metal

public enum Sampler {
    private static let argmaxP = Device.shared.makePipeline("argmax_f32")
    private static let tempP = Device.shared.makePipeline("apply_temperature")

    /// Greedy sampling. For top-k / top-p / penalties, build them on top.
    public static func argmax(_ logits: Tensor) -> Int {
        precondition(logits.dtype == .f32 && logits.shape.count == 2 && logits.shape[0] == 1)
        let V = logits.shape[1]
        let outBuf = Device.shared.mtl.makeBuffer(length: 4, options: .storageModeShared)!

        let cmd = Device.shared.queue.makeCommandBuffer()!
        let enc = cmd.makeComputeCommandEncoder()!
        enc.setComputePipelineState(argmaxP)
        enc.setBuffer(logits.buffer, offset: logits.offset, index: 0)
        enc.setBuffer(outBuf, offset: 0, index: 1)
        var v = UInt32(V)
        enc.setBytes(&v, length: 4, index: 2)
        enc.dispatchThreadgroups(MTLSize(width: 1, height: 1, depth: 1),
                                 threadsPerThreadgroup: MTLSize(width: 256, height: 1, depth: 1))
        enc.endEncoding()
        cmd.commit(); cmd.waitUntilCompleted()
        return Int(outBuf.contents().load(as: UInt32.self))
    }

    public static func applyTemperature(_ logits: Tensor, _ T: Float) {
        precondition(logits.dtype == .f32)
        if T == 0.0 { return }
        let V = logits.count
        let cmd = Device.shared.queue.makeCommandBuffer()!
        let enc = cmd.makeComputeCommandEncoder()!
        enc.setComputePipelineState(tempP)
        enc.setBuffer(logits.buffer, offset: logits.offset, index: 0)
        var v = UInt32(V); var t = T
        enc.setBytes(&v, length: 4, index: 1)
        enc.setBytes(&t, length: 4, index: 2)
        enc.dispatchThreads(MTLSize(width: V, height: 1, depth: 1),
                            threadsPerThreadgroup: MTLSize(width: 256, height: 1, depth: 1))
        enc.endEncoding()
        cmd.commit(); cmd.waitUntilCompleted()
    }
}
