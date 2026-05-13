import Foundation
import Metal

public enum ScoreFunc: Int, Sendable {
    case softmax = 0
    case sigmoid = 1
    case sqrtsoftplus = 2

    public init(_ s: String) {
        switch s {
        case "softmax": self = .softmax
        case "sigmoid": self = .sigmoid
        case "sqrtsoftplus": self = .sqrtsoftplus
        default: fatalError("unknown score_func: \(s)")
        }
    }
}

/// MoE Gate. Mirrors `Gate` in
/// `Reference/inference/model.py` lines 546–584.
public final class Gate {
    public let topK: Int
    public let nExperts: Int
    public let scoreFunc: ScoreFunc
    public let routeScale: Float
    public let hashRouting: Bool
    public let weight: Linear              // [n_experts, hidden] f32
    public let bias: Tensor?               // [n_experts] f32
    public let tid2eid: Tensor?            // [vocab, top_k] i32 (hash routing)

    private let pipeline: MTLComputePipelineState

    public init(config: ModelConfig, layerId: Int,
                weight: Linear, bias: Tensor?, tid2eid: Tensor?) {
        self.topK = config.nActivatedExperts
        self.nExperts = config.nRoutedExperts
        self.scoreFunc = ScoreFunc(config.scoreFunc)
        self.routeScale = config.routeScale
        self.hashRouting = layerId < config.nHashLayers
        self.weight = weight
        self.bias = bias
        self.tid2eid = tid2eid

        // Specialise the moe_gate kernel with this gate's score func + scale.
        let consts = MTLFunctionConstantValues()
        var sc = UInt32(self.scoreFunc.rawValue)
        var rs = self.routeScale
        consts.setConstantValue(&sc, type: .uint, index: 2)
        consts.setConstantValue(&rs, type: .float, index: 3)
        let lib = Device.shared.library
        do {
            let fn = try lib.makeFunction(name: "moe_gate", constantValues: consts)
            self.pipeline = try Device.shared.mtl.makeComputePipelineState(function: fn)
        } catch {
            fatalError("Gate pipeline failed: \(error)")
        }
    }

    /// Returns `(weights, indices)` of shape `[N, topK]` each.
    /// `cmd` is `inout` because the hash-routing path needs to commit
    /// the caller's buffer (to materialise the input `x`, which is the
    /// product of preceding kernels) and swap in a fresh one so the
    /// caller doesn't double-commit. The score-based path treats it
    /// as a regular non-mutating reference.
    public func callAsFunction(_ x: Tensor, inputIds: [Int32],
                                in cmd: inout MTLCommandBuffer) -> (weights: Tensor, indices: Tensor) {
        precondition(x.dtype == .f32 && x.shape.count == 2)
        let N = x.shape[0]

        let weights = Tensor.empty(shape: [N, topK], dtype: .f32)
        let indices = Tensor.empty(shape: [N, topK], dtype: .i32)

        if hashRouting {
            guard let tid = tid2eid else { fatalError("hash routing requires tid2eid") }
            // Hash-routed layer: expert indices come from a precomputed
            // tid2eid table (lookup keyed on the input token id), but
            // the per-expert routing WEIGHTS still come from
            //     softplus(linear(x, gate.weight)).sqrt()
            // gathered at those indices, normalised across the topK and
            // multiplied by route_scale. Mirrors Reference/inference/
            // model.py:576-583. Earlier this branch used a uniform
            // 1/topK weight, which silently degraded the first
            // n_hash_layers (=3 for V4-Flash) by replacing learned
            // gating with a flat average.
            //
            // `x` was produced by kernels queued on the caller's `cmd`
            // (RMSNorm + hc.pre at least) which hasn't been committed
            // yet, so its bytes aren't valid host-side until we drain
            // that buffer. We commit + wait, then run the gate
            // projection on a fresh cmd, then swap a brand-new cmd
            // back to the caller so its later `cmd.commit()` doesn't
            // try to commit an already-committed buffer.
            cmd.commit(); cmd.waitUntilCompleted()
            let scoreCmd = Device.shared.queue.makeCommandBuffer()!
            let logits = weight(x, in: scoreCmd)
            scoreCmd.commit(); scoreCmd.waitUntilCompleted()
            let logitsPtr = logits.buffer.contents()
                .bindMemory(to: Float.self, capacity: N * nExperts)
            let tidPtr = tid.buffer.contents()
                .bindMemory(to: Int32.self, capacity: tid.count)
            let idxPtr = indices.buffer.contents()
                .bindMemory(to: Int32.self, capacity: N * topK)
            let weightsPtr = weights.buffer.contents()
                .bindMemory(to: Float.self, capacity: N * topK)
            let vocabSize = tid.shape[0]
            // sqrt(softplus(x)) — numerically stable form matching the
            // moe_gate Metal kernel's score_fn for SCORE==2.
            @inline(__always)
            func sqrtSoftplus(_ x: Float) -> Float {
                let sp = max(x, 0) + log(1 + expf(-abs(x)))
                return sqrt(sp)
            }
            for n in 0..<N {
                let id = Int(inputIds[n])
                precondition(id >= 0 && id < vocabSize)
                var sum: Float = 0
                for k in 0..<topK {
                    let e = Int(tidPtr[id * topK + k])
                    let l = logitsPtr[n * nExperts + e]
                    let w = sqrtSoftplus(l)
                    weightsPtr[n * topK + k] = w
                    idxPtr[n * topK + k] = Int32(e)
                    sum += w
                }
                // Renormalise + scale (the reference normalises only
                // when score_func != "softmax" — V4 uses sqrtsoftplus
                // so we always normalise here).
                let inv = 1.0 / max(sum, 1e-12)
                for k in 0..<topK {
                    weightsPtr[n * topK + k] *= inv * routeScale
                }
            }
            // Replace the caller's already-committed cmd so its
            // upcoming `cmd.commit()` (the MoEFFN drain at gate +1)
            // runs against a fresh buffer.
            cmd = Device.shared.queue.makeCommandBuffer()!
            return (weights, indices)
        }

        // Score-based routing — needs `logits = x @ weight^T` on GPU.
        let logits = weight(x, in: cmd)

        let enc = cmd.makeComputeCommandEncoder()!
        enc.setComputePipelineState(pipeline)
        enc.setBuffer(logits.buffer, offset: 0, index: 0)
        if let b = bias {
            enc.setBuffer(b.buffer, offset: b.offset, index: 1)
        } else {
            // Bind a dummy buffer so the pipeline state is valid.
            let zero = Device.shared.mtl.makeBuffer(length: 16, options: .storageModeShared)!
            enc.setBuffer(zero, offset: 0, index: 1)
        }
        enc.setBuffer(indices.buffer, offset: 0, index: 2)
        enc.setBuffer(weights.buffer, offset: 0, index: 3)
        var dims = SIMD3<UInt32>(UInt32(N), UInt32(nExperts), UInt32(topK))
        enc.setBytes(&dims, length: MemoryLayout.size(ofValue: dims), index: 4)
        enc.dispatchThreads(MTLSize(width: N, height: 1, depth: 1),
                            threadsPerThreadgroup: MTLSize(width: min(N, 64), height: 1, depth: 1))
        enc.endEncoding()
        return (weights, indices)
    }
}

/// Single MoE expert: SwiGLU FFN (w1 = gate_proj, w3 = up_proj, w2 = down_proj).
public final class Expert {
    public let w1: Linear
    public let w2: Linear
    public let w3: Linear
    public let swigluLimit: Float

    public init(w1: Linear, w2: Linear, w3: Linear, swigluLimit: Float) {
        self.w1 = w1; self.w2 = w2; self.w3 = w3
        self.swigluLimit = swigluLimit
    }

    public func callAsFunction(_ x: Tensor, in cmd: MTLCommandBuffer) -> Tensor {
        let g = w1(x, in: cmd)
        let u = w3(x, in: cmd)
        // Pass swigluLimit through so the kernel applies the V4 clamp.
        let h = Elementwise.siluMul(g, u, swigluLimit: swigluLimit, in: cmd)
        // Mirrors model.py:606 — the clamped + SiLU activation is cast back
        // to the activation dtype (BF16) before flowing into w2. Round-trip
        // in place to inject the per-block QAT noise.
        Elementwise.bf16RoundTripInplace(h, in: cmd)
        return w2(h, in: cmd)
    }
}

/// MoE container. Mirrors `MoE` in model.py:609–644.
public final class MoEFFN {
    public let gate: Gate
    public let experts: [Expert?]
    public let sharedExpert: Expert
    public let dim: Int
    public let nExperts: Int
    public let topK: Int
    public var layerId: Int = -1

    public init(config: ModelConfig, gate: Gate, experts: [Expert?], shared: Expert) {
        self.gate = gate
        self.experts = experts
        self.sharedExpert = shared
        self.dim = config.dim
        self.nExperts = config.nRoutedExperts
        self.topK = config.nActivatedExperts
    }

    /// `x`: [B, S, D] f32. `inputIds`: [B*S] (used for hash routing only).
    ///
    /// `cmd` is `inout`: the gate output has to be read back to host to
    /// build the dispatch plan, so the original command buffer is
    /// committed and replaced with a fresh one. Caller's subsequent work
    /// goes onto the swapped buffer.
    public func callAsFunction(_ x: Tensor, inputIds: [Int32],
                                in cmd: inout MTLCommandBuffer) -> Tensor {
        precondition(x.dtype == .f32)
        let shape = x.shape
        let N = shape.dropLast().reduce(1, *)
        let xFlat = x.reshape([N, dim])

        // 1. Gate.
        let (weights, indices) = gate(xFlat, inputIds: inputIds, in: &cmd)
        // We need indices+weights on host to build the dispatch plan.
        cmd.commit(); cmd.waitUntilCompleted()
        let idxPtr = indices.buffer.contents().bindMemory(to: Int32.self, capacity: N * topK)
        let wPtr = weights.buffer.contents().bindMemory(to: Float.self, capacity: N * topK)
        let idxArr = Array(UnsafeBufferPointer(start: idxPtr, count: N * topK))
        let wArr = Array(UnsafeBufferPointer(start: wPtr, count: N * topK))

        let plan = MoEDispatch.prepare(indices: idxArr, weights: wArr,
                                        N: N, topK: topK, nExperts: nExperts)

        // Layer 5/6 diagnostic: which experts get chosen, with what weight?
        if TraceFlags.normTrace && (layerId >= 2 && layerId <= 7) {
            var msg = "[trace moe[\(layerId)] routing] indices/weights (N=\(N), topK=\(topK)):\n"
            for n in 0..<N {
                var parts: [String] = []
                for k in 0..<topK {
                    parts.append("e\(idxArr[n * topK + k])(w=\(String(format: "%.4f", wArr[n * topK + k])))")
                }
                msg += "  tok\(n): " + parts.joined(separator: " ") + "\n"
            }
            FileHandle.standardError.write(Data(msg.utf8))
        }

        // Swap in a fresh command buffer for the rest of the work; the
        // committed one above can no longer accept encoders.
        cmd = Device.shared.queue.makeCommandBuffer()!

        // 2. Gather tokens per expert.
        let gathered = MoEDispatch.gather(xFlat, plan: plan, in: cmd)

        // 3. Forward each active expert on its slice; write into a flat
        //    [T, D] output buffer.
        let T = plan.totalAssignments
        let outs = Tensor.empty(shape: [max(T, 1), dim], dtype: .f32)
        // Zero outs first.
        let blit = cmd.makeBlitCommandEncoder()!
        blit.fill(buffer: outs.buffer, range: 0..<outs.byteCount, value: 0)
        blit.endEncoding()

        let bytesPerRow = dim * MemoryLayout<Float>.size
        let perExpertTrace = TraceFlags.normTrace && (layerId == 5 || layerId == 6)
        for e in 0..<nExperts {
            let lo = plan.perExpertOffsets[e]
            let hi = plan.perExpertOffsets[e + 1]
            let count = hi - lo
            if count == 0 { continue }
            guard let expert = experts[e] else { continue }    // sharded out
            // Slice gathered rows [lo..hi].
            let slice = Tensor(shape: [count, dim], dtype: .f32,
                                buffer: gathered.buffer,
                                offset: gathered.offset + lo * bytesPerRow)
            let outSlice = expert(slice, in: cmd)
            if perExpertTrace {
                cmd.commit(); cmd.waitUntilCompleted()
                traceTensorStats("moe[\(layerId)] expert[\(e)] out (count=\(count))", outSlice)
                cmd = Device.shared.queue.makeCommandBuffer()!
            }
            // Copy into outs.
            let blit2 = cmd.makeBlitCommandEncoder()!
            blit2.copy(from: outSlice.buffer, sourceOffset: 0,
                       to: outs.buffer, destinationOffset: lo * bytesPerRow,
                       size: count * bytesPerRow)
            blit2.endEncoding()
        }

        // 4. Scatter back into a dense [N, D] tensor.
        let y = Tensor.empty(shape: [N, dim], dtype: .f32)
        let blitY = cmd.makeBlitCommandEncoder()!
        blitY.fill(buffer: y.buffer, range: 0..<y.byteCount, value: 0)
        blitY.endEncoding()
        MoEDispatch.scatter(y: y, outs: outs, plan: plan, in: cmd)

        // Diagnostic trace: routed-expert contribution before adding shared.
        if TraceFlags.normTrace && (layerId >= 2 && layerId <= 7) {
            cmd.commit(); cmd.waitUntilCompleted()
            traceTensorStats("moe[\(layerId)] routed-only (post-scatter)", y)
            cmd = Device.shared.queue.makeCommandBuffer()!
        }

        // 5. Add shared expert output. Leave cmd uncommitted; the caller
        // (Block) continues encoding hc.post into the same buffer.
        let sharedOut = sharedExpert(xFlat, in: cmd)

        if TraceFlags.normTrace && (layerId >= 2 && layerId <= 7) {
            cmd.commit(); cmd.waitUntilCompleted()
            traceTensorStats("moe[\(layerId)] shared-only", sharedOut)
            cmd = Device.shared.queue.makeCommandBuffer()!
        }

        Elementwise.addInPlace(y, sharedOut, in: cmd)

        if TraceFlags.normTrace && (layerId >= 2 && layerId <= 7) {
            cmd.commit(); cmd.waitUntilCompleted()
            traceTensorStats("moe[\(layerId)] final (routed+shared)", y)
            cmd = Device.shared.queue.makeCommandBuffer()!
        }

        return y.reshape(shape)
    }
}
