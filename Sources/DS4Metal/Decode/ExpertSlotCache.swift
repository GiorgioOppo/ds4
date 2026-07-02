import Foundation

// "Persistent + changing" experts: an LRU slot-cache for routed-expert weights.
//
// Per layer, a fixed pool of S slots (packed gate/up/down slabs in shared GPU
// buffers). Hot experts stay RESIDENT in the pool across tokens (persistent);
// a miss evicts the least-recently-used slot and memcpy's just that expert from
// the mmap (changing). The matvec then runs on the pool with SLOT indices as
// ids, so the same validated/fused kernels are used — a cache hit costs zero
// copies and zero kernel changes.
//
// Memory: S × (gate+up+down expert bytes) per layer, allocated lazily per layer
// (2-bit model ≈ 6.9 MB/slot → S=8 ≈ 2.4 GB across 43 layers). The buffers are
// wired (not evictable like the page cache), so on tight-RAM machines start
// small and watch the hit rate in the decode profile.
public final class ExpertSlotCache {
    public struct LayerPool {
        public let gate: GPUTensor    // S x gateExpertBytes, packed by slot
        public let up: GPUTensor      // S x upExpertBytes
        public let down: GPUTensor    // S x downExpertBytes
        var owner: [Int32]            // slot -> expert id (-1 = free)
        var lastUse: [UInt64]         // slot -> LRU tick
        var slotOf: [Int32: Int]      // expert id -> slot
    }

    public let slotsPerLayer: Int
    /// gate+up+down bytes of one expert — stats only (the decode profile turns
    /// miss counts into gathered bytes / effective SSD bandwidth).
    public let bytesPerExpert: Int
    public private(set) var hits = 0
    public private(set) var misses = 0
    private var pools: [Int: LayerPool] = [:]
    private var tick: UInt64 = 0
    private let makePool: () throws -> (gate: GPUTensor, up: GPUTensor, down: GPUTensor)
    /// Copy expert `id` of layer `layer` into pool slot `slot` (all 3 matrices).
    /// MUST be safe to call concurrently for distinct slots (misses are filled
    /// in parallel — each fill writes only its own slot's slabs).
    private let fill: (_ layer: Int, _ id: Int32, _ pool: LayerPool, _ slot: Int) throws -> Void
    /// Optional readahead hint, called with ALL the ids about to be filled BEFORE
    /// the copies (e.g. madvise(WILLNEED) on their mmap slabs) so the SSD serves
    /// the regions concurrently instead of fault-by-fault.
    private let prefetch: ((_ layer: Int, _ ids: [Int32]) -> Void)?
    /// Optional warm-set provider: historically hottest experts of a layer (from
    /// the persisted usage stats); pre-filled into the pool on first use.
    private let warm: ((_ layer: Int) -> [Int32])?

    public init(slotsPerLayer: Int,
                bytesPerExpert: Int = 0,
                makePool: @escaping () throws -> (gate: GPUTensor, up: GPUTensor, down: GPUTensor),
                fill: @escaping (_ layer: Int, _ id: Int32, _ pool: LayerPool, _ slot: Int) throws -> Void,
                prefetch: ((_ layer: Int, _ ids: [Int32]) -> Void)? = nil,
                warm: ((_ layer: Int) -> [Int32])? = nil) {
        self.slotsPerLayer = max(8, slotsPerLayer)   // ≥ k+2 so this tick's ids never starve eviction
        self.bytesPerExpert = bytesPerExpert
        self.makePool = makePool
        self.fill = fill
        self.prefetch = prefetch
        self.warm = warm
    }

    /// Run `fill` for every (id, slot) pair CONCURRENTLY (distinct slots write
    /// disjoint pool ranges). Throws the first fill error, if any.
    private func fillAll(layer: Int, pairs: [(id: Int32, slot: Int)], pool: LayerPool) throws {
        guard !pairs.isEmpty else { return }
        prefetch?(layer, pairs.map { $0.id })
        if pairs.count == 1 {
            try fill(layer, pairs[0].id, pool, pairs[0].slot)
            return
        }
        let lock = NSLock()
        var firstError: Error? = nil
        DispatchQueue.concurrentPerform(iterations: pairs.count) { j in
            do { try fill(layer, pairs[j].id, pool, pairs[j].slot) }
            catch {
                lock.lock()
                if firstError == nil { firstError = error }
                lock.unlock()
            }
        }
        if let e = firstError { throw e }
    }

    /// Drop all pools (frees the wired buffers) and reset the hit/miss counters.
    /// Pools rebuild lazily and re-warm from the CURRENT usage profile on next
    /// use — called on agent switch, when the warm prior changes.
    public func invalidate() {
        pools.removeAll()
        hits = 0
        misses = 0
    }

    /// Ensure all `ids` are resident in layer `layer`'s pool; returns the pool and
    /// each id's slot index (same order — route weights stay aligned). Misses
    /// evict LRU slots, never a slot already touched by this call.
    public func acquire(layer: Int, ids: [Int32]) throws -> (pool: LayerPool, slots: [Int32]) {
        if pools[layer] == nil {
            let p = try makePool()
            var fresh = LayerPool(gate: p.gate, up: p.up, down: p.down,
                                  owner: Array(repeating: -1, count: slotsPerLayer),
                                  lastUse: Array(repeating: 0, count: slotsPerLayer),
                                  slotOf: [:])
            // Pre-warm with the historically hottest experts (usage-stats prior):
            // they start as the oldest entries, so a wrong prior is evicted fast.
            if let warm {
                let warmIds = Array(warm(layer).prefix(slotsPerLayer))
                try fillAll(layer: layer,
                            pairs: warmIds.enumerated().map { (id: $1, slot: $0) }, pool: fresh)
                for (s, id) in warmIds.enumerated() {
                    fresh.owner[s] = id
                    fresh.slotOf[id] = s
                }
            }
            pools[layer] = fresh
        }
        tick += 1
        var pool = pools[layer]!
        var slots = [Int32](repeating: -1, count: ids.count)
        var missIdx: [Int] = []
        for (j, id) in ids.enumerated() {
            if let s = pool.slotOf[id] {
                pool.lastUse[s] = tick
                slots[j] = Int32(s)
                hits += 1
            } else {
                missIdx.append(j)
            }
        }
        // Assign a victim slot to every miss FIRST (serial LRU bookkeeping), then
        // fill all the assigned slots concurrently — the SSD sees every missing
        // slab at once instead of one page-fault stream at a time.
        var toFill: [(id: Int32, slot: Int)] = []
        for j in missIdx {
            let id = ids[j]
            // Victim: a FREE slot if any, else the least-recently-used one —
            // never a slot already touched by this call.
            var victim = -1
            var best = UInt64.max
            for s in 0..<slotsPerLayer where pool.lastUse[s] != tick {
                if pool.owner[s] < 0 { victim = s; break }
                if pool.lastUse[s] < best { best = pool.lastUse[s]; victim = s }
            }
            precondition(victim >= 0, "expert cache: no evictable slot (S too small)")
            if pool.owner[victim] >= 0 { pool.slotOf.removeValue(forKey: pool.owner[victim]) }
            pool.owner[victim] = id
            pool.slotOf[id] = victim
            pool.lastUse[victim] = tick
            slots[j] = Int32(victim)
            misses += 1
            toFill.append((id: id, slot: victim))
        }
        do {
            try fillAll(layer: layer, pairs: toFill, pool: pool)
        } catch {
            // A fill failed: those slots hold garbage — mark every slot of this
            // batch free so a later acquire refills instead of serving a bad hit.
            for (id, slot) in toFill {
                pool.owner[slot] = -1
                pool.slotOf.removeValue(forKey: id)
            }
            pools[layer] = pool
            throw error
        }
        pools[layer] = pool
        return (pool, slots)
    }
}
