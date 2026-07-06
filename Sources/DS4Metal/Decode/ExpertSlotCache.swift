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
//
// CONCURRENCY: operations are serialized PER LAYER (one lock per layer), so the
// decode thread's `acquire(layer: i)` can run while the look-ahead queue
// `prefill(layer: i+1)`s the NEXT layer — that is the whole point (the C engine
// hides expert I/O the same way, with in-flight tracking on its stream cache).
// The demand path has PRIORITY: an acquire announces itself (demandWaiting)
// before taking the layer lock, and a running prefill yields between fill
// chunks — the speculative I/O can delay the critical path by at most ~one
// chunk, never by a whole speculative batch. Shared bookkeeping (pools map,
// tick, counters, epoch, demandWaiting) is guarded by `stateLock`; the GPU
// never reads a slot that a concurrent op is writing because a slot's bytes
// change only under its layer's lock, the decode thread encodes the layer's
// matvec strictly after its own acquire(layer:) returned, and a speculative
// fill never evicts the last demand-acquire's slots (lastDemand).
public final class ExpertSlotCache: @unchecked Sendable {
    public struct LayerPool {
        public let gate: GPUTensor    // S x gateExpertBytes, packed by slot
        public let up: GPUTensor      // S x upExpertBytes
        public let down: GPUTensor    // S x downExpertBytes
        var owner: [Int32]            // slot -> expert id (-1 = free)
        var lastUse: [UInt64]         // slot -> LRU tick
        var slotOf: [Int32: Int]      // expert id -> slot
        /// Tick of the most recent DEMAND acquire: those slots may still be
        /// read by an in-flight command buffer — a late speculative prefill
        /// must never evict them (see ensureResident).
        var lastDemand: UInt64 = 0
    }

    public let slotsPerLayer: Int
    /// gate+up+down bytes of one expert — stats only (the decode profile turns
    /// miss counts into gathered bytes / effective SSD bandwidth).
    public let bytesPerExpert: Int
    private var _hits = 0
    private var _misses = 0
    /// Slabs filled by the speculative look-ahead (they don't count as misses:
    /// their I/O ran hidden under compute, not on the decode critical path).
    private var _prefilled = 0
    public var hits: Int { stateLock.lock(); defer { stateLock.unlock() }; return _hits }
    public var misses: Int { stateLock.lock(); defer { stateLock.unlock() }; return _misses }
    public var prefilled: Int { stateLock.lock(); defer { stateLock.unlock() }; return _prefilled }
    private var pools: [Int: LayerPool] = [:]
    private var tick: UInt64 = 0
    /// Bumped by invalidate(): an operation that started against a dropped
    /// generation discards its result instead of resurrecting a stale pool.
    private var epoch = 0
    private let stateLock = NSLock()
    private var layerLocks: [Int: NSLock] = [:]
    /// Layers with a demand acquire blocked on (or about to take) the layer
    /// lock: a running prefill checks this between fill chunks and yields.
    private var demandWaiting = Set<Int>()
    /// Optional per-layer slot-count override (e.g. the usage-driven allocation:
    /// same total budget, more slots where the routing concentrates). Consulted
    /// once per layer at pool creation — so also after every invalidate().
    /// nil (or a nil provider) = the uniform slotsPerLayer.
    private let slotsFor: ((_ layer: Int) -> Int)?
    private let makePool: (_ slots: Int) throws -> (gate: GPUTensor, up: GPUTensor, down: GPUTensor)
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
                makePool: @escaping (_ slots: Int) throws -> (gate: GPUTensor, up: GPUTensor, down: GPUTensor),
                fill: @escaping (_ layer: Int, _ id: Int32, _ pool: LayerPool, _ slot: Int) throws -> Void,
                prefetch: ((_ layer: Int, _ ids: [Int32]) -> Void)? = nil,
                warm: ((_ layer: Int) -> [Int32])? = nil,
                slotsFor: ((_ layer: Int) -> Int)? = nil) {
        self.slotsPerLayer = max(8, slotsPerLayer)   // ≥ k+2 so this tick's ids never starve eviction
        self.bytesPerExpert = bytesPerExpert
        self.makePool = makePool
        self.fill = fill
        self.prefetch = prefetch
        self.warm = warm
        self.slotsFor = slotsFor
    }

    /// Slot count for a layer's pool: per-layer override, floored at 8 (≥ k+2,
    /// so one acquire's ids can never starve the LRU eviction).
    private func slotCount(_ layer: Int) -> Int {
        max(8, slotsFor?(layer) ?? slotsPerLayer)
    }

    private func lockFor(layer: Int) -> NSLock {
        stateLock.lock(); defer { stateLock.unlock() }
        if let l = layerLocks[layer] { return l }
        let l = NSLock()
        layerLocks[layer] = l
        return l
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
        // nonisolated(unsafe): ogni fill scrive SOLO gli slab del proprio slot
        // (contratto di `fill`), l'errore e' protetto dal lock, e la closure/
        // pool catturati sono usati in sola lettura durante il fan-out.
        nonisolated(unsafe) var firstError: Error? = nil
        nonisolated(unsafe) let fillFn = fill
        nonisolated(unsafe) let poolRef = pool
        DispatchQueue.concurrentPerform(iterations: pairs.count) { j in
            do { try fillFn(layer, pairs[j].id, poolRef, pairs[j].slot) }
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
    /// use — called on agent switch, when the warm prior changes. An operation
    /// in flight against the old generation discards its result (epoch check).
    public func invalidate() {
        stateLock.lock(); defer { stateLock.unlock() }
        pools.removeAll()
        _hits = 0
        _misses = 0
        _prefilled = 0
        epoch += 1
    }

    /// Ensure all `ids` are resident in layer `layer`'s pool; returns the pool and
    /// each id's slot index (same order — route weights stay aligned). Misses
    /// evict LRU slots, never a slot already touched by this call. A prefill in
    /// flight on the SAME layer yields within ~one fill chunk (demandWaiting).
    public func acquire(layer: Int, ids: [Int32]) throws -> (pool: LayerPool, slots: [Int32]) {
        let l = lockFor(layer: layer)
        stateLock.lock(); demandWaiting.insert(layer); stateLock.unlock()
        l.lock()
        stateLock.lock(); demandWaiting.remove(layer); stateLock.unlock()
        defer { l.unlock() }
        return try ensureResident(layer: layer, ids: ids, speculative: false)
    }

    /// Speculative look-ahead fill (best-effort, never throws): make `ids`
    /// resident in `layer`'s pool so the decode thread's acquire finds hits.
    /// Runs on a background queue WHILE the GPU computes the previous layer —
    /// the expert I/O moves off the decode critical path (SSD idle window).
    /// Never creates the pool (the demand path pays the one-time warm fill),
    /// skips outright if a demand is already waiting, and YIELDS between fill
    /// chunks when one arrives. Errors mark the batch's slots free and are
    /// swallowed: the later demand acquire retries and reports the real error.
    public func prefill(layer: Int, ids: [Int32]) {
        guard !ids.isEmpty else { return }
        stateLock.lock()
        let ready = pools[layer] != nil && !demandWaiting.contains(layer)
        stateLock.unlock()
        guard ready else { return }
        let l = lockFor(layer: layer)
        l.lock(); defer { l.unlock() }
        _ = try? ensureResident(layer: layer, ids: ids, speculative: true)
    }

    private func ensureResident(layer: Int, ids: [Int32],
                                speculative: Bool) throws -> (pool: LayerPool, slots: [Int32]) {
        // Snapshot shared state under stateLock; the pool VALUE is then mutated
        // privately (this layer's lock is held) and published at the end.
        stateLock.lock()
        let startEpoch = epoch
        var poolOpt = pools[layer]
        stateLock.unlock()

        if poolOpt == nil {
            let S = slotCount(layer)
            let p = try makePool(S)
            var fresh = LayerPool(gate: p.gate, up: p.up, down: p.down,
                                  owner: Array(repeating: -1, count: S),
                                  lastUse: Array(repeating: 0, count: S),
                                  slotOf: [:])
            // Pre-warm with the historically hottest experts (usage-stats prior):
            // they start as the oldest entries, so a wrong prior is evicted fast.
            if let warm {
                let warmIds = Array(warm(layer).prefix(S))
                try fillAll(layer: layer,
                            pairs: warmIds.enumerated().map { (id: $1, slot: $0) }, pool: fresh)
                for (s, id) in warmIds.enumerated() {
                    fresh.owner[s] = id
                    fresh.slotOf[id] = s
                }
            }
            poolOpt = fresh
        }
        var pool = poolOpt!

        stateLock.lock()
        tick += 1
        let now = tick        // snapshot: a concurrent op on ANOTHER layer bumps tick too
        stateLock.unlock()

        if !speculative { pool.lastDemand = now }
        var slots = [Int32](repeating: -1, count: ids.count)
        var missIdx: [Int] = []
        for (j, id) in ids.enumerated() {
            if let s = pool.slotOf[id] {
                pool.lastUse[s] = now
                slots[j] = Int32(s)
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
            // never a slot already touched by this call. A SPECULATIVE fill
            // additionally must never evict the last demand-acquire's slots:
            // a late prefill (queue backlog) can run while the FFN command
            // buffer reading those slots is still on the GPU. Best-effort:
            // when no safe victim exists the id is simply skipped.
            var victim = -1
            var best = UInt64.max
            for s in 0..<pool.owner.count
            where pool.lastUse[s] != now && !(speculative && pool.lastUse[s] == pool.lastDemand) {
                if pool.owner[s] < 0 { victim = s; break }
                if pool.lastUse[s] < best { best = pool.lastUse[s]; victim = s }
            }
            if speculative && victim < 0 { continue }
            precondition(victim >= 0, "expert cache: no evictable slot (S too small)")
            if pool.owner[victim] >= 0 { pool.slotOf.removeValue(forKey: pool.owner[victim]) }
            pool.owner[victim] = id
            pool.slotOf[id] = victim
            pool.lastUse[victim] = now
            slots[j] = Int32(victim)
            toFill.append((id: id, slot: victim))
        }
        func publish(_ p: LayerPool) {
            stateLock.lock(); defer { stateLock.unlock() }
            // Don't resurrect a pool dropped by invalidate() mid-operation.
            if epoch == startEpoch { pools[layer] = p }
        }
        func abandon(_ pairs: ArraySlice<(id: Int32, slot: Int)>) {
            // Unfilled/failed slots hold garbage — mark them free so a later
            // acquire refills instead of serving a bad hit.
            for (id, slot) in pairs {
                pool.owner[slot] = -1
                pool.slotOf.removeValue(forKey: id)
            }
        }
        if speculative {
            // Chunked fill with DEMAND PREEMPTION: between chunks, yield if the
            // decode thread is waiting on this layer's lock — speculative I/O
            // may cost the critical path at most ~one chunk.
            var filled = 0
            var idx = 0
            while idx < toFill.count {
                stateLock.lock()
                let yield = demandWaiting.contains(layer)
                stateLock.unlock()
                if yield { break }
                let next = min(idx + 2, toFill.count)
                do { try fillAll(layer: layer, pairs: Array(toFill[idx..<next]), pool: pool) }
                catch { break }
                filled += next - idx
                idx = next
            }
            abandon(toFill[idx...])
            publish(pool)
            stateLock.lock(); _prefilled += filled; stateLock.unlock()
            return (pool, slots)
        }
        do {
            try fillAll(layer: layer, pairs: toFill, pool: pool)
        } catch {
            abandon(toFill[...])
            publish(pool)
            throw error
        }
        publish(pool)
        stateLock.lock()
        _hits += ids.count - missIdx.count
        _misses += missIdx.count
        stateLock.unlock()
        return (pool, slots)
    }
}
