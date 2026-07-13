import Foundation
import DS4Core

extension DistCoordinator {
    /// Benchmark della route verticale (prompt sintetico + decode). Chiamare
    /// via Task.detached: il decode BLOCCA sui round-trip di rete.
    public func verticalBenchmark(contextTokens: Int, genTokens: Int) throws -> InferenceService.BenchPoint {
        guard let engine = verticalEngine else {
            throw DistError.sliceGap("verticale non connesso (connectVertical prima)")
        }
        return try engine.benchmarkVertical(contextTokens: contextTokens, genTokens: genTokens)
    }

    // MARK: Benchmark

    /// Distributed benchmark: prefill a synthetic prompt of `contextTokens` tokens
    /// across the already-connected cluster and decode `genTokens` from it, returning
    /// prefill / generation throughput at that context frontier. Reuses the live
    /// route (no reconnect) and runs with a fresh cluster KV (posBase 0 resets the
    /// workers), so it must NOT overlap a chat `send`. Mirrors
    /// `InferenceService.benchmark` so local and distributed numbers are comparable.
    public func benchmark(contextTokens: Int, genTokens: Int) async throws -> InferenceService.BenchPoint {
        guard !entries.isEmpty else { throw DistError.closed }
        sessionCounter &+= 1
        let session = sessionCounter
        kvValid = false          // the run rewrites the cluster KV from pos 0
        let ctx = max(8, min(contextTokens, config.contextSize - genTokens - 4))
        // Synthetic prompt: BOS + tiled filler. Output quality is irrelevant for
        // timing; the per-token work (embed · slice forward · expert gather) is the same.
        var ids: [Int] = [engine.bosId]
        let filler = engine.tokenize("The quick brown fox jumps over the lazy dog. ")
        let pad = filler.isEmpty ? [engine.eosId] : filler
        var i = 0
        while ids.count < ctx { ids.append(pad[i % pad.count]); i += 1 }
        ids = Array(ids.prefix(ctx))

        // PREFILL the whole prompt in chunks (posBase 0 resets the workers' KV).
        let t0 = Date()
        var pos = 0
        var lastLogits: [Float] = []
        var start = 0
        var firstChunk = true
        while start < ids.count {
            try Task.checkCancellation()
            let end = min(start + config.prefillChunk, ids.count)
            var hcs: [[Float]] = []
            for (k, id) in ids[start..<end].enumerated() {
                hcs.append(try engine.embed(token: id, pos: pos + k))
            }
            if let logits = try await runChunk(session: session, hcs: hcs, posBase: pos,
                                               tokens: Array(ids[start..<end]),
                                               wantLogits: end == ids.count, turnStart: firstChunk) {
                lastLogits = logits
            }
            firstChunk = false
            pos += end - start
            start = end
        }
        let prefillDt = Date().timeIntervalSince(t0)
        guard !lastLogits.isEmpty else { throw DistError.badFrame }

        // DECODE genTokens token-by-token (content discarded; only timing matters).
        var rng: UInt64 = 0xD54
        let samp = SamplingParams()
        var produced = 0
        let g0 = Date()
        while produced < genTokens {
            try Task.checkCancellation()
            let next = engine.sample(lastLogits, params: samp, rng: &rng)
            let hc = try engine.embed(token: next, pos: pos)
            guard let logits = try await runChunk(session: session, hcs: [hc], posBase: pos,
                                                  tokens: [next], wantLogits: true) else {
                throw DistError.badFrame
            }
            lastLogits = logits
            pos += 1; produced += 1
        }
        let genDt = Date().timeIntervalSince(g0)
        let kv = UInt64(engine.nLayers) * UInt64(ctx) * UInt64(engine.headDim) * 4
        return InferenceService.BenchPoint(
            contextTokens: ctx,
            prefillTps: prefillDt > 0 ? Double(ctx) / prefillDt : 0,
            genTps: genDt > 0 && produced > 0 ? Double(produced) / genDt : 0,
            kvBytes: kv)
    }
}
