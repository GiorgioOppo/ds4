import Foundation

/// Per-phase wall-clock accumulator for the decode forward pass. Each phase is
/// timed around a committed (and waited) command buffer / a CPU gather, so the
/// numbers reflect real elapsed time and answer "I/O vs compute". Times are
/// totals over all forward() calls; `report()` averages per token.
public struct DecodeProfile: Sendable {
    public var embedS = 0.0       // token embedding
    public var routeS = 0.0       // attention + router (compute)
    // DS4_PROFILE_ROUTE: detailed route/attn split (ratios meaningful).
    public var routeCompS = 0.0       // hc-pre + NSA compressor (attn + indexer)
    public var routeQS = 0.0          // Q projection (q_a, q_b)
    public var routeKvS = 0.0         // KV projection + indexer scoring
    public var routeAttnPhaseS = 0.0  // flash-attn
    public var routeOutProjS = 0.0    // inverse RoPE + output_a/b + HC expand
    public var routeHcFfnS = 0.0      // pre-FFN HyperConnection reduce
    public var routeRouterS = 0.0     // router projection + probabilities/top-k
    public var gatherS = 0.0      // gather the 6 selected experts from the mmap (EXPERT I/O)
    public var expertsS = 0.0     // shared FFN + routed MoE matvec (compute)
    public var layerOtherS = 0.0  // non-split decode path (resident experts)
    public var headS = 0.0        // output head
    public var forwards = 0       // number of forward() calls (= tokens)
    public var layers = 0         // total per-layer iterations
    public var expertHits = 0     // expert slot-cache hits (persistent experts)
    public var expertMisses = 0   // expert slot-cache misses (changed experts)
    public var expertHitBytes = 0 // bytes not gathered because of cache hits
    public var expertMissBytes = 0 // bytes gathered for cacheable misses
    public var expertWarmed = 0    // synchronous history-driven pool warm fills
    public var expertWarmedBytes = 0 // critical-path bytes read by those fills
    /// Routed selections that could not use the slot cache (cache disabled or
    /// unsupported layer/layout). Kept separate so a mixed-quant bypass cannot
    /// inflate the displayed hit-rate by disappearing from its denominator.
    public var expertBypasses = 0
    public var expertPrefilled = 0  // slabs filled by the look-ahead (I/O hidden under compute)
    public var expertPrefilledBytes = 0 // physical look-ahead I/O omitted from gatherS/gatherBytes
    public var gatherBytes = 0    // expert bytes copied from the mmap (EXPERT I/O volume)
    public var expertBypassBytes = 0 // subset of gatherBytes caused by bypass selections

    public init() {}

    /// Engine-side seconds accounted by the per-phase counters (what report()
    /// calls "totale"). Wall-clock minus this = time spent OUTSIDE the engine
    /// (sampler, streaming, UI) — the GUI logs that split per turn.
    public var totalS: Double { embedS + routeS + gatherS + expertsS + layerOtherS + headS }

    public var expertCacheableHitRate: Double? {
        let total = expertHits + expertMisses
        return total > 0 ? Double(expertHits) / Double(total) : nil
    }

    public var expertGlobalHitRate: Double? {
        let total = expertHits + expertMisses + expertBypasses
        return total > 0 ? Double(expertHits) / Double(total) : nil
    }

    public var expertCacheableByteHitRate: Double? {
        let total = expertHitBytes + expertMissBytes
        return total > 0 ? Double(expertHitBytes) / Double(total) : nil
    }

    public var expertGlobalByteHitRate: Double? {
        let total = expertHitBytes + expertMissBytes + expertBypassBytes
        return total > 0 ? Double(expertHitBytes) / Double(total) : nil
    }

    public func report(title: String = "Profilo decode") -> String {
        guard forwards > 0 else { return "\(title): nessun forward registrato." }
        let f = Double(forwards)
        let total = embedS + routeS + gatherS + expertsS + layerOtherS + headS
        func ms(_ s: Double) -> String { String(format: "%6.1f", s / f * 1000) }
        func pct(_ s: Double) -> String { String(format: "%2.0f%%", total > 0 ? s / total * 100 : 0) }
        let tps = total > 0 ? f / total : 0
        var cacheLine = ""
        if let rate = expertCacheableHitRate {
            let ahead: String
            if expertPrefilled > 0, expertPrefilledBytes > 0 {
                ahead = String(format: " — %d slab da look-ahead, %.1f MB/token I/O nascosti",
                               expertPrefilled, Double(expertPrefilledBytes) / f / 1_048_576)
            } else if expertPrefilled > 0 {
                ahead = " — \(expertPrefilled) slab da look-ahead"
            } else {
                ahead = ""
            }
            cacheLine = "\n  cache expert \(expertHits) hit / \(expertMisses) miss  (\(String(format: "%.0f", rate * 100))% sui cacheabili)\(ahead)"
        }
        if expertBypasses > 0, let globalRate = expertGlobalHitRate {
            let bypassVolume: String
            if expertBypassBytes > 0 {
                bypassVolume = String(format: ", %.1f MB/token",
                                      Double(expertBypassBytes) / f / 1_048_576)
            } else {
                bypassVolume = ""
            }
            cacheLine += "\n  cache bypass \(expertBypasses) selezioni\(bypassVolume) — \(String(format: "%.0f", globalRate * 100))% hit globale"
        }
        if expertWarmed > 0 {
            cacheLine += String(format: "\n  cache warm   %d slab iniziali, %.1f MB/token I/O sincroni",
                                expertWarmed, Double(expertWarmedBytes) / f / 1_048_576)
        }
        if let cacheableBytes = expertCacheableByteHitRate {
            let globalBytes = expertGlobalByteHitRate ?? cacheableBytes
            cacheLine += String(format: "\n  cache byte    %.0f%% hit sui cacheabili / %.0f%% globale",
                                cacheableBytes * 100, globalBytes * 100)
        }
        // Effective gather bandwidth: how fast the expert slabs actually leave the
        // SSD/page cache. Compare against the raw sequential bandwidth of the disk
        // to see the streaming headroom (bytes/gatherS; page-cache hits inflate it).
        if gatherBytes > 0 && gatherS > 0 {
            let mbTok = Double(gatherBytes) / f / 1_048_576
            let gbs = Double(gatherBytes) / gatherS / 1e9
            cacheLine += "\n  gather IO    \(String(format: "%6.1f", mbTok)) MB/token — banda effettiva \(String(format: "%.2f", gbs)) GB/s"
        }
        var routeSplit = ""   // DS4_PROFILE_ROUTE: ratios meaningful, absolutes inflated (extra commits)
        if routeCompS + routeQS + routeKvS + routeAttnPhaseS + routeOutProjS + routeHcFfnS + routeRouterS > 0 {
            routeSplit = "\n     ├ comp \(ms(routeCompS)) ms/token (hc-pre + NSA compressor)"
                       + "\n     ├ q    \(ms(routeQS)) ms/token (q_a + q_b proj)"
                       + "\n     ├ kv   \(ms(routeKvS)) ms/token (kv proj + indexer)"
                       + "\n     ├ attn \(ms(routeAttnPhaseS)) ms/token (flash-attn)"
                       + "\n     ├ out  \(ms(routeOutProjS)) ms/token (output_a/b + HC expand)"
                       + "\n     ├ hc   \(ms(routeHcFfnS)) ms/token (pre-FFN HC reduce)"
                       + "\n     └ rtr  \(ms(routeRouterS)) ms/token (router proj + top-k)"
        }
        return """
        \(title) — \(forwards) token, \(layers) iterazioni-layer
          embed        \(ms(embedS)) ms/token  (\(pct(embedS)))
          route/attn   \(ms(routeS)) ms/token  (\(pct(routeS)))   compute\(routeSplit)
          gather IO    \(ms(gatherS)) ms/token  (\(pct(gatherS)))   <- streaming esperti (SSD/page cache)
          experts      \(ms(expertsS)) ms/token  (\(pct(expertsS)))   compute
          layer (alt)  \(ms(layerOtherS)) ms/token  (\(pct(layerOtherS)))
          output head  \(ms(headS)) ms/token  (\(pct(headS)))\(cacheLine)
          ----------------------------------------
          totale       \(ms(total)) ms/token  (~\(String(format: "%.2f", tps)) tok/s)
        """
    }
}
