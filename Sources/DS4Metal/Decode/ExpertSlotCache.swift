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
    public private(set) var hits = 0
    public private(set) var misses = 0
    private var pools: [Int: LayerPool] = [:]
    private var tick: UInt64 = 0
    private let makePool: () throws -> (gate: GPUTensor, up: GPUTensor, down: GPUTensor)
    /// Copy expert `id` of layer `layer` into pool slot `slot` (all 3 matrices).
    private let fill: (_ layer: Int, _ id: Int32, _ pool: LayerPool, _ slot: Int) throws -> Void
    /// Optional warm-set provider: historically hottest experts of a layer (from
    /// the persisted usage stats); pre-filled into the pool on first use.
    private let warm: ((_ layer: Int) -> [Int32])?

    public init(slotsPerLayer: Int,
                makePool: @escaping () throws -> (gate: GPUTensor, up: GPUTensor, down: GPUTensor),
                fill: @escaping (_ layer: Int, _ id: Int32, _ pool: LayerPool, _ slot: Int) throws -> Void,
                warm: ((_ layer: Int) -> [Int32])? = nil) {
        self.slotsPerLayer = max(8, slotsPerLayer)   // ≥ k+2 so this tick's ids never starve eviction
        self.makePool = makePool
        self.fill = fill
        self.warm = warm
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
                for (s, id) in warm(layer).prefix(slotsPerLayer).enumerated() {
                    try fill(layer, id, fresh, s)
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
            try fill(layer, id, pool, victim)
            pool.owner[victim] = id
            pool.slotOf[id] = victim
            pool.lastUse[victim] = tick
            slots[j] = Int32(victim)
            misses += 1
        }
        pools[layer] = pool
        return (pool, slots)
    }
}
