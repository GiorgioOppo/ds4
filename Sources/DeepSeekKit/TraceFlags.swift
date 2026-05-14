import Foundation

/// Runtime-toggleable diagnostic switches. Kept separate from
/// `MemoryLogger` so the two traces can be enabled independently and
/// neither imposes overhead when the other is on.
public enum TraceFlags {
    /// Print L2 norm + min/max/mean + NaN/Inf counts of the residual
    /// stream at strategic points in `Transformer.forward`. Off by
    /// default. Enable via `--trace-norms` on the CLI to diagnose
    /// where the activations diverge / collapse / NaN.
    nonisolated(unsafe) public static var normTrace: Bool = false
}

/// Materialize an f32 tensor to the host and emit one diagnostic
/// line on stderr. No-op when `TraceFlags.normTrace` is false.
///
/// Assumes `t` is already host-readable (the caller must have
/// `commit + waitUntilCompleted`'d the command buffer that wrote to
/// it). All `MTLBuffer`s in this codebase use `.storageModeShared`,
/// so reading directly from `contents()` is safe after the wait.
public func traceTensorStats(_ name: String, _ t: Tensor) {
    guard TraceFlags.normTrace else { return }
    precondition(t.dtype == .f32, "traceTensorStats supports f32 only; got \(t.dtype)")
    let n = t.count
    guard n > 0 else { return }
    let p = t.buffer.contents().advanced(by: t.offset)
        .bindMemory(to: Float.self, capacity: n)
    var sumSq: Double = 0
    var sum: Double = 0
    var mn: Float = .greatestFiniteMagnitude
    var mx: Float = -.greatestFiniteMagnitude
    var nanCount = 0
    var infCount = 0
    var finiteCount = 0
    for i in 0..<n {
        let v = p[i]
        if v.isNaN { nanCount += 1; continue }
        if v.isInfinite { infCount += 1; continue }
        let d = Double(v)
        sumSq += d * d
        sum += d
        if v < mn { mn = v }
        if v > mx { mx = v }
        finiteCount += 1
    }
    let l2 = sumSq.squareRoot()
    let mean = finiteCount > 0 ? sum / Double(finiteCount) : 0
    let minOut: Double = finiteCount > 0 ? Double(mn) : 0
    let maxOut: Double = finiteCount > 0 ? Double(mx) : 0
    let line = String(format:
        "[trace %@] shape=%@ L2=%.4e mean=%+.4e min=%+.4e max=%+.4e nans=%d infs=%d\n",
        name as CVarArg,
        "\(t.shape)" as CVarArg,
        l2, mean, minOut, maxOut,
        nanCount, infCount)
    FileHandle.standardError.write(Data(line.utf8))
}
