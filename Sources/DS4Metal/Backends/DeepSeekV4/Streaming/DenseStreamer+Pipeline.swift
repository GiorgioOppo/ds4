import Foundation
import Metal
import DS4Core

extension DenseStreamer {
    /// Join every read-ahead `pread` before the decoder releases its staging
    /// buffers/model mapping. `pending` is decode-thread-owned, so callers must
    /// invoke this from the same serialized engine executor as `weights(_:)`.
    public func quiesceForTeardown() {
        for item in pending {
            item.sem.wait()
            if item.error == nil { slotLayer[item.slot] = item.layer }
        }
        pending.removeAll(keepingCapacity: false)
    }

    /// The next `count` layers of the pass after `il` (wrapping to the range
    /// start), i.e. the layers whose staging slots must not be overwritten.
    private func upcoming(after il: Int, count: Int) -> [Int] {
        var out: [Int] = []
        var next = il
        for _ in 0..<count {
            next = next + 1 < layers.upperBound ? next + 1 : layers.lowerBound
            if next == il { break }
            out.append(next)
        }
        return out
    }

    /// LayerProvider entry point (DECODE thread only). Returns layer `il`'s
    /// weights with the big fields as views into a ready staging slot, then
    /// keeps the read-ahead pipeline `ahead` layers deep: with the default 1
    /// this is the classic 2-slot ring (read i+1 while computing i); with
    /// DS4_DENSE_AHEAD=2 the SSD moves on to i+2 as soon as i+1 lands instead
    /// of idling for the rest of layer i's compute. Overwrite safety: a slot is
    /// reused only when its occupant is neither the current layer nor one of
    /// the next `ahead` layers; the only async cb that can still be in flight
    /// here (the routed FFN) reads no staged slab — see the class contract.
    public func weights(_ il: Int) throws -> LayerWeights {
        let slot: Int
        if let pi = pending.firstIndex(where: { $0.layer == il }) {
            let p = pending.remove(at: pi)
            p.sem.wait()                        // usually already signalled (read ran during layer i-1)
            if let e = p.error { throw e }
            slotLayer[p.slot] = il
            slot = p.slot
        } else if let s = slotLayer.firstIndex(of: il) {
            slot = s                            // already staged (e.g. retry after an error)
        } else {
            // Cold start or out-of-order request: drain every in-flight load,
            // then read synchronously into a slot we won't need imminently.
            for p in pending {
                p.sem.wait()
                if p.error == nil { slotLayer[p.slot] = p.layer }
            }
            pending.removeAll()
            let wanted = Set(upcoming(after: il, count: ahead))
            slot = (0..<slots.count).first { !wanted.contains(slotLayer[$0]) } ?? 0
            try load(il, into: slot)
            slotLayer[slot] = il
        }
        // Top up the pipeline: for each of the next `ahead` layers not already
        // staged or in flight, start a background read into a reusable slot.
        let wanted = Set(upcoming(after: il, count: ahead) + [il])
        for next in upcoming(after: il, count: ahead) {
            if slotLayer.contains(next) || pending.contains(where: { $0.layer == next }) { continue }
            guard let free = (0..<slots.count).first(where: { s in
                s != slot && !wanted.contains(slotLayer[s]) && !pending.contains(where: { $0.slot == s })
            }) else { break }
            let p = Pending(layer: next, slot: free)
            slotLayer[free] = -1                // being overwritten
            pending.append(p)
            DispatchQueue.global(qos: .userInitiated).async {
                do { try self.load(next, into: p.slot) } catch { p.error = error }
                p.sem.signal()
            }
        }
        return makeWeights(il, slot: slot)
    }


    /// pread every tensor of layer `il` into `slot`, all slabs CONCURRENTLY
    /// (10-17 reads of 2-36 MB — real queue depth for the NVMe). F_NOCACHE:
    /// zero page-cache footprint. Runs on the caller's thread.
    private func load(_ il: Int, into slot: Int) throws {
        guard let plan = entries[il] else {
            throw GGUFWeights.LoadError.message("DenseStreamer: layer \(il) outside streamed range")
        }
        // nonisolated(unsafe): ogni pread scrive un range DISGIUNTO dello slot
        // (stageOffset per-tensore); il flag di errore e' protetto dal lock.
        nonisolated(unsafe) let base = slots[slot].contents()
        let lock = NSLock()
        nonisolated(unsafe) var failed = false
        DispatchQueue.concurrentPerform(iterations: plan.count) { i in
            let e = plan[i]
            if !GGUFWeights.preadFull(fd, into: base + e.stageOffset, bytes: e.bytes, offset: e.fileOffset) {
                lock.lock(); failed = true; lock.unlock()
            }
        }
        if failed {
            throw GGUFWeights.LoadError.message("DenseStreamer: pread failed on layer \(il)")
        }
    }

    /// Skeleton (small resident fields) + staging views for the big ones.
    private func makeWeights(_ il: Int, slot: Int) -> LayerWeights {
        var w = skeleton[il]!
        let buf = slots[slot]
        for e in entries[il]! {
            let t = GPUTensor(buffer: buf, byteLength: e.bytes, count: e.bytes, byteOffset: e.stageOffset)
            switch e.field {
            case .hcAttnFn: w.hcAttnFn = t
            case .qA: w.qA = t
            case .qB: w.qB = t
            case .kvW: w.kvW = t
            case .attnOutA: w.attnOutA = t
            case .attnOut: w.attnOut = t
            case .hcFfnFn: w.hcFfnFn = t
            case .sharedGate: w.sharedGate = t
            case .sharedUp: w.sharedUp = t
            case .sharedDown: w.sharedDown = t
            case .routerW: w.routerW = t
            case .compKv: w.compKv = t
            case .compGate: w.compGate = t
            case .idxQB: w.idxQB = t
            case .idxProj: w.idxProj = t
            case .idxKv: w.idxKv = t
            case .idxGate: w.idxGate = t
            }
        }
        return w
    }
}
