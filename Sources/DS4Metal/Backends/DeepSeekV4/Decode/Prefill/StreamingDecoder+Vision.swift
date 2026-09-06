import Foundation
import Metal
import DS4Core

extension StreamingDecoder {
    /// Configure the encoder's checkpoint-specific selection biases. Visual
    /// rows use bias_vl alone, including the first three hash-routed layers.
    /// The ordinary hash biases are validated for sidecar integrity; upstream
    /// visual routing never adds them to bias_vl.
    public func configureVision(visualRouterBias: [[Float]], hashRouterBias: [[Float]]) throws {
        guard visualRouterBias.count == nLayers, hashRouterBias.count == 3,
              (visualRouterBias + hashRouterBias).allSatisfy({ row in
                  row.count == d.nExperts && row.allSatisfy(\.isFinite)
              }) else {
            throw MetalError.unsupported("Vision: bias del router incompatibili con il modello")
        }
        drainFFN()
        visionRouterBias = try visualRouterBias.map { values in
            let tensor = try GPUTensor.zerosBytes(rt, byteLength: values.count * 4)
            writeFloats(values, into: tensor)
            return tensor
        }
    }

    func validateVisionOverrides(tokens: [Int], startPos: Int,
                                 overrides: [Int: [Float]]) throws {
        for (index, token) in tokens.enumerated() {
            if token >= d.vocab {
                guard visionRouterBias.count == nLayers,
                      let row = overrides[startPos + index], row.count == d.nEmbd,
                      row.allSatisfy(\.isFinite) else {
                    throw MetalError.unsupported("Vision: encoder o embedding mancante alla posizione \(startPos + index)")
                }
            } else if overrides[startPos + index] != nil {
                throw MetalError.unsupported("Vision: un embedding non può sostituire un token di testo")
            }
        }
    }

    /// Prefill a complete image block, including alignment padding. A temporary
    /// raw cache keeps the complete block alive until every query consumed its
    /// future keys; only committed rows are then copied back into the usual
    /// ring. Compressor updates and compressed visibility remain causal.
    func prefillVisionBlock(_ tokens: [Int], startPos: Int,
                            embeddingOverrides: [Int: [Float]]) throws -> [GPUTensor] {
        guard remoteExperts == nil, kvRange == 0..<nLayers else {
            throw MetalError.unsupported("Vision richiede il decoder locale completo")
        }
        let spans = try DeepSeekV4VisionAttention.spans(tokens: tokens, vocabularySize: d.vocab)
        guard spans.count == 1, spans[0].block == tokens.indices else {
            throw MetalError.unsupported("Vision: il prefill richiede un blocco immagine intero")
        }
        let n = tokens.count
        let endPos = startPos + n
        drainFullLayerGather()
        drainFFN()
        try prepareLiveContext(nKeys: endPos)
        let savedScratch = scratch
        let savedDirtyCount = maskDirtyCount
        let firstRaw = max(0, startPos + 1 - d.nSWA)
        let rawRows = endPos - firstRaw
        let maximumCompressed = compStates.compactMap { $0 }.map { endPos / $0.ratio + 8 }.max() ?? 0
        scratch = try DecodeScratch(rt, d, maxKeys: max(rawRows + maximumCompressed + 64,
                                                      savedScratch.attentionRows))
        maskDirtyCount = 0
        defer {
            drainFFN()
            scratch = savedScratch
            maskDirtyCount = savedDirtyCount
        }

        let hcWidth = d.nHC * d.nEmbd
        var cur = try PrefillStage.slabViews(rt, n: n, rowBytes: hcWidth * 4, rowCount: hcWidth).views
        var other = try PrefillStage.slabViews(rt, n: n, rowBytes: hcWidth * 4, rowCount: hcWidth).views
        for j in 0..<n {
            guard let values = embeddingOverrides[startPos + j] else {
                throw MetalError.unsupported("Vision: embedding del blocco non disponibile")
            }
            for h in 0..<d.nHC {
                values.withUnsafeBytes { bytes in
                    _ = memcpy(cur[j].buffer.contents() + cur[j].byteOffset + h * d.nEmbd * 4,
                               bytes.baseAddress!, bytes.count)
                }
            }
        }
        let queries = try PrefillStage.slabViews(rt, n: n, rowBytes: scratch.q.count * 4,
                                               rowCount: scratch.q.count).views
        let splits = try PrefillStage.slabViews(rt, n: n, rowBytes: scratch.split.count * 4,
                                              rowCount: scratch.split.count).views
        let raw = try GPUTensor.zeros(rt, floatCount: rawRows * d.headDim)
        var visibleCompressed = [Int](repeating: 0, count: n)
        var selectedCompressed = [[Bool]?](repeating: nil, count: n)

        for layer in 0..<nLayers {
            try Task.checkCancellation()
            prefillLayerProgress?(layer, nLayers, n)
            let weights = try layerProvider(layer)
            if layer + 1 < nLayers { prefetch?(layer + 1) }
            let layerRope = ropeParams(layer: layer)
            let oldRaw = rawCaches[layer]
            let oldRows = oldRaw.count / d.headDim
            // Absolute positions determine ring slots on both caches.
            for position in firstRaw..<startPos {
                copyVisionRawRow(position: position, from: oldRaw, rows: oldRows,
                                 to: raw, destinationRows: rawRows)
            }
            for j in 0..<n {
                try Task.checkCancellation()
                let position = startPos + j
                let indexer = indexStates[layer]
                let scoring = indexerActive(layer, pos: position)
                    && weights.idxQB != nil && weights.idxProj != nil
                let context = GraphContext(rt)
                try context.begin()
                visibleCompressed[j] = try context.decodeRoutePre(
                    curHc: cur[j], w: weights, s: scratch, d: d,
                    rope: layerRope, rawCache: raw, pos: position,
                    rmsEps: rmsEps, hcEps: hcEps, comp: compStates[layer],
                    idx: weights.idxKv != nil && weights.idxGate != nil ? indexer : nil,
                    indexerScoring: scoring)
                context.commit()
                copyFloats(from: scratch.q, to: queries[j], count: scratch.q.count)
                copyFloats(from: scratch.split, to: splits[j], count: scratch.split.count)
                if scoring, let indexer {
                    let scores = (scratch.idxScores.buffer.contents() + scratch.idxScores.byteOffset)
                        .bindMemory(to: Float.self, capacity: indexer.count)
                    selectedCompressed[j] = IndexerSelect.allowedTopK(
                        scores: scores, count: indexer.count, k: d.indexerTopK)
                } else {
                    selectedCompressed[j] = nil
                }
            }

            // Visual rows always use score-based selection, including the hash
            // layers. Reusing the existing top-k graph preserves route weights
            // normalized on UNBIASED probabilities.
            var visualWeights = weights
            visualWeights.tid2eid = nil
            visualWeights.tid2eidRows = 0
            visualWeights.expBias = visionRouterBias[layer]
            for j in 0..<n {
                try Task.checkCancellation()
                let position = startPos + j
                let bounds = DeepSeekV4VisionAttention.bounds(
                    query: j, startPosition: startPos, image: spans[0].image,
                    firstRawPosition: firstRaw, lastRawPosition: endPos - 1, window: d.nSWA)
                let rawCount = bounds.count
                let compressedCount = visibleCompressed[j]
                let mask = (scratch.mask.buffer.contents() + scratch.mask.byteOffset)
                    .bindMemory(to: UInt16.self, capacity: rawCount + compressedCount)
                for k in 0..<rawCount { mask[k] = 0 }
                for k in 0..<compressedCount {
                    let allowed = selectedCompressed[j].map { k < $0.count ? $0[k] : true } ?? true
                    mask[rawCount + k] = allowed ? 0 : 0xFC00
                }
                copyFloats(from: queries[j], to: scratch.q, count: scratch.q.count)
                let context = GraphContext(rt)
                try context.begin()
                try context.flashAttnCore(
                    q: scratch.q, kvF32: raw, kvF16: scratch.kvF16, mask: scratch.mask,
                    sinks: weights.attnSinks, pad: scratch.pad, tmp: scratch.tmp,
                    heads: scratch.heads, nHead: d.nHead, nKeys: rawCount,
                    rawStartRow: bounds.lowerBound, hasSinks: true,
                    comp: compStates[layer]?.cache, nComp: compressedCount)
                try context.decodeRouteAttnTail(
                    curHc: cur[j], w: visualWeights, s: scratch, d: d,
                    rope: layerRope, pos: position, token: tokens[j],
                    rmsEps: rmsEps, hcEps: hcEps, attnSplit: splits[j])
                context.commit()
                try finishVisionExperts(layer: layer, weights: weights, output: other[j])
                profile.layers += 1
            }
            for position in firstRaw..<endPos {
                copyVisionRawRow(position: position, from: raw, rows: rawRows,
                                 to: oldRaw, destinationRows: oldRows)
            }
            swap(&cur, &other)
        }
        return cur
    }

    private func copyVisionRawRow(position: Int, from source: GPUTensor, rows: Int,
                                  to destination: GPUTensor, destinationRows: Int) {
        memcpy(destination.buffer.contents() + destination.byteOffset + (position % destinationRows) * d.headDim * 4,
               source.buffer.contents() + source.byteOffset + (position % rows) * d.headDim * 4,
               d.headDim * 4)
    }

    /// Consume a committed route. Deliberately synchronous: query snapshots and
    /// the temporary raw cache can be released as soon as the block finishes.
    private func finishVisionExperts(layer: Int, weights: LayerWeights, output: GPUTensor) throws {
        guard let gather = expertGather else {
            let context = GraphContext(rt)
            try context.begin()
            try context.decodeExperts(w: weights, s: scratch, d: d,
                gateExp: weights.expGate, upExp: weights.expUp, downExp: weights.expDown,
                ids: scratch.selected, outHc: output)
            context.commit()
            return
        }
        let (ids, routeWeights) = readRouteSelection(layer: layer)
        writeFloats(routeWeights, into: scratch.rw)
        zeroDown6(from: ids.count)
        let context = GraphContext(rt)
        try context.begin()
        try context.decodeSharedFFN(w: weights, s: scratch, d: d)
        let onClass = weights.gateQuant == d.gateQuant && weights.upQuant == d.upQuant
            && weights.downQuant == d.downQuant
        if let cache = slotCache, onClass || cache.supports(layer: layer) {
            let acquired = try cache.acquireLeased(layer: layer, ids: ids)
            defer { acquired.lease.release() }
            acquired.slots.withUnsafeBytes { bytes in
                _ = memcpy(slotsScratch.buffer.contents(), bytes.baseAddress!, bytes.count)
            }
            try context.decodeRoutedExperts(w: weights, s: scratch, d: d,
                gateExp: acquired.pool.gate, upExp: acquired.pool.up, downExp: acquired.pool.down,
                ids: slotsScratch, outHc: output, activeK: ids.count,
                expertStride: acquired.pool.expertStride ?? slotCacheStride)
            context.commit()
        } else {
            let (gate, up, down) = try gather(layer, ids)
            try context.decodeRoutedExperts(w: weights, s: scratch, d: d,
                gateExp: gate, upExp: up, downExp: down, ids: idsPacked,
                outHc: output, activeK: ids.count)
            context.commit()
        }
    }
}
