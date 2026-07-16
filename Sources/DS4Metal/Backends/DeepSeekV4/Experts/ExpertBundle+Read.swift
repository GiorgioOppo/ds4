import Foundation
import Metal
import DS4Core

extension ExpertBundle {
    // MARK: Read path

    /// Batch-load expert records directly from the bundle into Metal buffers.
    /// `slotStride == nil` means three tightly-packed destination buffers;
    /// otherwise gate/up/down are views of one interleaved pool and every view
    /// advances by the common record stride. Returns false when MetalIO is off
    /// or fails; callers then execute the byte-identical pread path.
    public func copyExpertsMetalIO(layer: Int,
                                   pairs: [(id: Int32, slot: Int)],
                                   gateDst: GPUTensor, upDst: GPUTensor, downDst: GPUTensor,
                                   slotStride: Int? = nil) -> Bool {
        guard let io = metalIOSnapshot(), layers.contains(layer), !pairs.isEmpty else { return false }
        let destinations: [(fileOff: Int, bytes: Int, dst: GPUTensor)] = [
            (0, gateBytes, gateDst),
            (gateBytes, upBytes, upDst),
            (gateBytes + upBytes, downBytes, downDst),
        ]
        for pair in pairs {
            guard pair.id >= 0, Int(pair.id) < nExpert, pair.slot >= 0 else { return false }
            for d in destinations {
                let stride = slotStride ?? d.bytes
                guard d.dst.byteOffset + pair.slot * stride + d.bytes <= d.dst.buffer.length else {
                    return false
                }
            }
        }

        let cb = io.queue.makeCommandBuffer()
        cb.label = "DS4 expert bundle MetalIO layer \(layer) × \(pairs.count)"
        let payloadBytes = gateBytes + upBytes + downBytes
        if let stride = slotStride {
            // Bundle record and interleaved pool slot are both gate|up|down:
            // ONE contiguous DMA command per expert, not three sub-loads.
            for pair in pairs {
                let sourceBase = dataBase + ((layer - layers.lowerBound) * nExpert + Int(pair.id)) * record
                cb.load(gateDst.buffer,
                        offset: gateDst.byteOffset + pair.slot * stride,
                        size: payloadBytes,
                        sourceHandle: io.handle,
                        sourceHandleOffset: sourceBase)
            }
        } else {
            for pair in pairs {
                let sourceBase = dataBase + ((layer - layers.lowerBound) * nExpert + Int(pair.id)) * record
                for d in destinations {
                    cb.load(d.dst.buffer,
                            offset: d.dst.byteOffset + pair.slot * d.bytes,
                            size: d.bytes,
                            sourceHandle: io.handle,
                            sourceHandleOffset: sourceBase + d.fileOff)
                }
            }
        }
        let t0 = Date()
        cb.commit()
        cb.waitUntilCompleted()
        guard cb.status == .complete else {
            disableMetalIO(cb.error?.localizedDescription ?? "stato \(cb.status.rawValue)")
            return false
        }
        let seconds = max(1e-9, Date().timeIntervalSince(t0))
        let gbs = Double(pairs.count * payloadBytes) / seconds / 1e9
        noteMetalIOSuccess(experts: pairs.count, gbs: gbs)
        // Successful bytes are valid, so serve this batch. Evaluate aggregate
        // windows rather than individual commands: a 1-expert fill is too
        // small for its GB/s figure to amortize MetalIO submission latency.
        switch evaluateMetalIO(bytes: pairs.count * payloadBytes, seconds: seconds) {
        case .keep:
            break
        case let .slowWindow(windowGBs, count):
            Self.log(String(format: "MetalIO campione aggregato lento: %.2f GB/s (verifica %d/2)",
                            windowGBs, count))
        case let .disable(windowGBs):
            disableMetalIO(String(format: "banda aggregata persistentemente lenta: %.2f < %.2f GB/s",
                                  windowGBs, metalIOBreaker.minimumGBs))
        }
        for _ in pairs { noteUse() }
        return true
    }

    /// pread expert (layer, id)'s three slabs into the pool slots. The slabs
    /// are ADJACENT in the sidecar: the three concurrent preads form one ~7 MB
    /// sequential burst instead of three scattered ~2 MB random reads.
    /// Returns false on any error (caller falls back to the GGUF reads).
    public func copyExpert(layer: Int, id: Int32,
                           gateDst: GPUTensor, upDst: GPUTensor, downDst: GPUTensor,
                           slot: Int) -> Bool {
        guard layers.contains(layer), id >= 0, Int(id) < nExpert else { return false }
        let base = dataBase + ((layer - layers.lowerBound) * nExpert + Int(id)) * record
        // nonisolated(unsafe): i 3 pread scrivono slab DISGIUNTI (gate/up/down
        // del proprio slot); il flag di esito e' protetto dal lock.
        nonisolated(unsafe) let jobs: [(fileOff: Int, bytes: Int, dst: GPUTensor)] = [
            (0, gateBytes, gateDst),
            (gateBytes, upBytes, upDst),
            (gateBytes + upBytes, downBytes, downDst)]
        for j in jobs where j.dst.byteOffset + (slot + 1) * j.bytes > j.dst.buffer.length {
            return false
        }
        let lock = NSLock()
        nonisolated(unsafe) var ok = true
        DispatchQueue.concurrentPerform(iterations: jobs.count) { i in
            let j = jobs[i]
            let p = j.dst.buffer.contents().advanced(by: j.dst.byteOffset + slot * j.bytes)
            if !GGUFWeights.preadFull(fd, into: p, bytes: j.bytes, offset: base + j.fileOff) {
                lock.lock(); ok = false; lock.unlock()
            }
        }
        if ok { noteUse() }
        return ok
    }

    /// Single-pread fill for the INTERLEAVED pool: the record layout in the
    /// sidecar (gate|up|down contiguous) matches the pool slot layout, so a
    /// miss is ONE ~7 MB pread straight into the slot — one syscall instead of
    /// three, larger I/O at the same queue depth. `dst` is the pool's combined
    /// buffer (gate view, byteOffset 0); `stride` is the slot record size.
    public func copyExpertInterleaved(layer: Int, id: Int32,
                                      dst: GPUTensor, slot: Int, stride: Int) -> Bool {
        let bytes = gateBytes + upBytes + downBytes
        guard layers.contains(layer), id >= 0, Int(id) < nExpert,
              bytes <= stride,
              dst.byteOffset + slot * stride + bytes <= dst.buffer.length else { return false }
        let base = dataBase + ((layer - layers.lowerBound) * nExpert + Int(id)) * record
        let p = dst.buffer.contents().advanced(by: dst.byteOffset + slot * stride)
        guard GGUFWeights.preadFull(fd, into: p, bytes: bytes, offset: base) else { return false }
        noteUse()
        return true
    }

    /// Gather the selected `ids` into three freshly packed K-expert tensors
    /// (same shape gatherLayerExperts returns) — used by the batched PREFILL
    /// path too, where each expert of the union becomes one sequential burst.
    /// nil on any error (caller falls back to the GGUF gather).
    public func gatherPacked(_ rt: MetalRuntime, layer: Int,
                             ids: [Int32]) -> (GPUTensor, GPUTensor, GPUTensor)? {
        guard layers.contains(layer), !ids.isEmpty,
              ids.allSatisfy({ $0 >= 0 && Int($0) < nExpert }) else { return nil }
        guard let g = try? GPUTensor.zerosBytes(rt, byteLength: ids.count * gateBytes),
              let u = try? GPUTensor.zerosBytes(rt, byteLength: ids.count * upBytes),
              let dn = try? GPUTensor.zerosBytes(rt, byteLength: ids.count * downBytes) else { return nil }
        let lock = NSLock()
        // nonisolated(unsafe): ogni iterazione scrive SOLO lo slot k dei tre
        // tensori packed; il flag di esito e' protetto dal lock.
        nonisolated(unsafe) var ok = true
        nonisolated(unsafe) let gRef = g, uRef = u, dnRef = dn
        DispatchQueue.concurrentPerform(iterations: ids.count) { k in
            if !copyExpert(layer: layer, id: ids[k], gateDst: gRef, upDst: uRef, downDst: dnRef, slot: k) {
                lock.lock(); ok = false; lock.unlock()
            }
        }
        return ok ? (g, u, dn) : nil
    }
}
