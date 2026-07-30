import DS4Core
import Foundation
import Metal

// Leva 1 del prefill GLM (port del route-batch DeepSeek, 2026-07-22).
//
// Il percorso storico di runLayerMajor chiama glm52ChainedDecodeLayer per
// OGNI (layer, token): ogni chiamata paga 2-3 commit+waitUntilCompleted
// (trunk/indexer, router, shared FFN). Su un prompt lungo sono centinaia di
// migliaia di sincronizzazioni CPU↔GPU — su DeepSeek la stessa struttura
// valeva la differenza 3.6 → 16 t/s.
//
// Qui un GRUPPO di R token attraversa il layer con DUE soli commit:
//   cb1  per ogni token: trunk pre-attention + store KV/chiavi indexer
//        (+ score e top-k su device quando visible > topK)
//   tap  CPU: risoluzione selezioni (identità / top-k / IndexShare)
//   cb2  per ogni token: attention + proiezioni + router fuso + FFN
//        (dense, oppure shared — gli esperti routed restano differiti alla
//        fase B expert-major esistente)
//   tap  CPU: lettura routing dai buffer del router fuso
//
// NUMERICA INVARIATA: stessi kernel, stessi argomenti, stesso ordine di
// dispatch per token del percorso per-token (l'hazard tracking Metal
// serializza gli encoder nello stesso ordine in cui sono codificati) —
// parità bit-per-bit per costruzione, fissata da GLM52PrefillBatchTests.
// L'unica differenza strutturale: la shared FFN sparse viaggia nel command
// buffer del tail invece che in un terzo commit (nessuna dipendenza dal
// routing: legge ffnIn e accumula hidden come prima).
//
// I kernel di decode NON sono toccati (percorso per-token intatto come
// riferimento e fallback), secondo la regola del progetto: prefill e
// generazione restano percorsi separati.

/// Gate della leva. Opt-in (DS4_GLM_PREFILL_BATCH=1) finché la parità
/// sul modello reale non è certificata dai test di integrazione
/// (DS4_GLM52_SPARSE_GGUF); il default storico resta il per-token.
enum GLM52PrefillBatchDispatch {
    nonisolated(unsafe) static var enabled = DS4RuntimeEnvironment.flag(
        "DS4_PREFILL_BATCH",
        overrides: ["DS4_GLM_PREFILL_BATCH"],
        default: false)
    /// Token per gruppo (DS4_GLM_PREFILL_ROUTE_BATCH, default 16): ogni
    /// token del gruppo ha il SUO set di scratch, quindi il costo è
    /// ~R × footprint di GLM52DecodeScratch (qualche MB a set).
    nonisolated(unsafe) static var groupSize = max(2,
        DS4RuntimeEnvironment.integer(
            "DS4_PREFILL_ROUTE_BATCH",
            overrides: ["DS4_GLM_PREFILL_ROUTE_BATCH"]) ?? 16)
}

/// R set di scratch indipendenti: dentro un gruppo ogni token scrive solo
/// nel proprio set, le uniche risorse condivise sono le cache KV/indexer
/// (append a righe distinte, ordinate dall'ordine di encode).
final class GLM52PrefillScratchPool {
    let sets: [GLM52DecodeScratch]
    init(runtime: MetalRuntime, geometry: GLM52DecodeGeometry,
         scoreCapacity: Int, count: Int) throws {
        sets = try (0..<count).map { _ in
            try GLM52DecodeScratch(runtime: runtime, geometry: geometry,
                                   scoreCapacity: scoreCapacity)
        }
    }
}

/// Esito di un gruppo: selezioni e routing PER TOKEN (ordine del gruppo),
/// fasi aggregate del gruppo intero (i confini di misura sono i due commit).
struct GLM52PrefillGroupOutcome {
    var selections: [[UInt32]]
    var routings: [GLM52RouterOutput?]
    var phases: GLM52LayerPhases
}

extension MetalRuntime {
    /// Un layer per un gruppo di `count` token consecutivi
    /// (posizioni basePosition..<basePosition+count), due commit totali.
    /// Richiede il router fuso (GLM52GpuRouterDispatch) sui layer sparse:
    /// il chiamante gata il percorso. Gli esperti routed NON sono applicati
    /// (equivalente di deferSparseFFN: la fase B expert-major li applica
    /// sui piani hidden/ffnIn dei set del pool).
    func glm52PrefillGroupLayer(
        weights: GLM52ResidentDecodeWeights,
        ffn: GLM52ResidentFFN,
        caches: GLM52ResidentDecodeCaches,
        pool: GLM52PrefillScratchPool,
        count: Int,
        hiddenAt: (Int) -> [Float],
        reusedSelectionAt: (Int) throws -> [UInt32]?,
        basePosition: Int) throws -> GLM52PrefillGroupOutcome {
        let g = weights.geometry
        let layer = g.layer
        let headsWidth = layer.headCount * layer.valueDimension
        var phases = GLM52LayerPhases()
        let tStart = Date()
        let gpuStart = GLM52GraphTelemetry.gpuSeconds
        guard count >= 1, count <= pool.sets.count else {
            throw MetalError.unsupported(
                "GLM 5.2 prefill batch: gruppo \(count) fuori dal pool "
                + "\(pool.sets.count)")
        }
        guard basePosition == caches.rows,
              basePosition + count <= caches.capacity else {
            throw MetalError.unsupported(
                "GLM 5.2 prefill batch: posizioni \(basePosition)+\(count) "
                + "non allineate alle \(caches.rows) righe vive / capacità "
                + "\(caches.capacity)")
        }
        let lastVisible = basePosition + count
        guard let anySet = pool.sets.first,
              lastVisible <= anySet.scores.length
                  / MemoryLayout<Float>.stride else {
            throw MetalError.unsupported(
                "GLM 5.2 prefill batch: visible \(lastVisible) oltre la "
                + "capacità degli score del pool")
        }

        // ── cb1: trunk pre-attention + store cache per tutti i token.
        guard let cb1 = queue.makeCommandBuffer() else {
            throw MetalError.bufferAlloc
        }
        // Il proj F32 dell'indexer è identico per tutti i token del gruppo:
        // un solo upload (il per-token ne creava uno a chiamata).
        var projBuffer: MTLBuffer?
        enum SelectionPlan {
            case identity(Int)      // visible <= topK: 0..<visible
            case device             // top-k su GPU: leggere indexerTopKOut
            case reused([UInt32])   // IndexShare: dal layer sorgente
        }
        var plans: [SelectionPlan] = []
        plans.reserveCapacity(count)

        for i in 0..<count {
            let set = pool.sets[i]
            let position = basePosition + i
            let visible = position + 1
            set.loadHidden(hiddenAt(i))

            try glm52EncodeRMSNorm(into: cb1, input: set.hidden,
                                   weight: weights.attnNorm,
                                   output: set.normed,
                                   width: layer.embeddingWidth)
            try glm52EncodeMatvecQ8Pair(into: cb1,
                                        input: set.normed,
                                        weightsA: weights.qA,
                                        outputA: set.qRank,
                                        rowsA: g.qLoraRank,
                                        typeA: weights.types.qA,
                                        weightsB: weights.kvA,
                                        outputB: set.kvRaw,
                                        rowsB: layer.kvRawWidth,
                                        typeB: weights.types.kvA,
                                        inputWidth: layer.embeddingWidth)
            try glm52EncodeRMSNorm(into: cb1, input: set.qRank,
                                   weight: weights.qANorm,
                                   output: set.qRankNorm,
                                   width: g.qLoraRank)
            try glm52GraphEncode(
                into: cb1,
                pipelineName: "kernel_glm52_kv_lora_norm_cache_ready_f32",
                arguments: [1, 0, 0, 0],
                buffers: [set.kvRaw, weights.kvANorm, set.cacheReady],
                threadgroups: MTLSize(width: 1, height: 1, depth: 1),
                threadsPerThreadgroup: MTLSize(width: 128, height: 1,
                                               depth: 1),
                threadgroupMemoryLength: 128 * MemoryLayout<Float>.stride)
            try glm52GraphEncode(
                into: cb1,
                pipelineName: "kernel_glm52_store_compact_row_f16",
                arguments: [UInt32(position), 1, UInt32(caches.capacity), 0],
                buffers: [set.cacheReady, caches.compact],
                threadgroups: MTLSize(width: 1, height: 1, depth: 1),
                threadsPerThreadgroup: MTLSize(width: 64, height: 1,
                                               depth: 1))
            try glm52EncodeMatvecQ8(into: cb1,
                                    input: set.qRankNorm,
                                    weights: weights.qB, output: set.query,
                                    rowCount: g.queryWidth,
                                    inputWidth: g.qLoraRank,
                                    weightType: weights.types.qB)
            try glm52EncodeRope(into: cb1,
                                pipelineName: "kernel_glm52_rope_tail_f32",
                                values: set.query,
                                headCount: layer.headCount,
                                headDimension: g.qkDimension,
                                rotationDimension: layer.ropeDimension,
                                position: position)

            if let indexer = weights.indexer,
               let keyCache = caches.indexerKeys {
                guard try reusedSelectionAt(i) == nil else {
                    throw MetalError.unsupported(
                        "GLM 5.2 prefill batch: layer full-indexer con "
                        + "selezione riusata")
                }
                try glm52EncodeMatvecQ8(into: cb1,
                                        input: set.hidden,
                                        weights: indexer.key,
                                        output: set.indexerRaw,
                                        rowCount: g.indexerHeadDimension,
                                        inputWidth: layer.embeddingWidth,
                                        weightType: weights.types.indexerKey)
                try glm52GraphEncode(
                    into: cb1,
                    pipelineName: "kernel_glm52_store_indexer_k_f16",
                    arguments: [UInt32(position), 1,
                                UInt32(caches.capacity), 0],
                    buffers: [set.indexerRaw, indexer.keyNorm,
                              indexer.keyNormBias, keyCache],
                    threadgroups: MTLSize(width: 1, height: 1, depth: 1),
                    threadsPerThreadgroup: MTLSize(width: 32, height: 1,
                                                   depth: 1),
                    threadgroupMemoryLength: 32 * MemoryLayout<Float>.stride)
                if visible <= g.indexerTopK {
                    plans.append(.identity(visible))
                } else {
                    try glm52EncodeMatvecQ8(
                        into: cb1,
                        input: set.qRankNorm,
                        weights: indexer.queryB,
                        output: set.indexerQuery,
                        rowCount: g.indexerQueryWidth,
                        inputWidth: g.qLoraRank,
                        weightType: weights.types.indexerQueryB)
                    try glm52EncodeRope(
                        into: cb1,
                        pipelineName: "kernel_glm52_rope_prefix_f32",
                        values: set.indexerQuery,
                        headCount: g.indexerHeadCount,
                        headDimension: g.indexerHeadDimension,
                        rotationDimension: g.indexerRotationDimension,
                        position: position)
                    if projBuffer == nil {
                        projBuffer = try glm52GraphBuffer(indexer.proj)
                    }
                    try glm52EncodeMatvecF32(into: cb1,
                                             rows: projBuffer!,
                                             input: set.hidden,
                                             output: set.indexerWeights,
                                             rowCount: g.indexerHeadCount,
                                             inputWidth: layer.embeddingWidth)
                    try glm52GraphEncode(
                        into: cb1,
                        pipelineName: "kernel_glm52_indexer_scores_f16",
                        arguments: [UInt32(visible), 1, UInt32(position),
                                    g.indexerScale.bitPattern],
                        buffers: [set.indexerQuery, set.indexerWeights,
                                  keyCache, set.scores],
                        threadgroups: MTLSize(width: visible, height: 1,
                                              depth: 1),
                        threadsPerThreadgroup: MTLSize(width: 32, height: 4,
                                                       depth: 1),
                        threadgroupMemoryLength:
                            (128 + 4) * MemoryLayout<Float>.stride)
                    try glm52EncodeIndexerTopK(
                        into: cb1, scores: set.scores,
                        rowCount: visible, topK: g.indexerTopK,
                        output: set.indexerTopKOut,
                        sortScratch: set.indexerSortScratch)
                    plans.append(.device)
                }
            } else {
                guard let reused = try reusedSelectionAt(i) else {
                    throw MetalError.unsupported(
                        "GLM 5.2 prefill batch: layer IndexShare senza "
                        + "selezione dal layer sorgente")
                }
                plans.append(.reused(reused))
            }
        }
        try glm52GraphCommit(cb1)

        // ── tap: risoluzione delle selezioni.
        var selections: [[UInt32]] = []
        selections.reserveCapacity(count)
        for i in 0..<count {
            let visible = basePosition + i + 1
            let selection: [UInt32]
            switch plans[i] {
            case .identity(let n):
                selection = (0..<n).map(UInt32.init)
            case .device:
                let top = pool.sets[i].indexerTopKOut.contents()
                    .bindMemory(to: UInt32.self, capacity: g.indexerTopK)
                selection = Array(UnsafeBufferPointer(start: top,
                                                      count: g.indexerTopK))
            case .reused(let rows):
                selection = rows
            }
            guard !selection.isEmpty, selection.count <= visible,
                  Set(selection).count == selection.count,
                  selection.allSatisfy({ Int($0) < visible }) else {
                throw MetalError.unsupported(
                    "GLM 5.2 prefill batch: selezione non valida per il "
                    + "token \(i) (visible \(visible))")
            }
            selections.append(selection)
        }
        phases.routeS = Date().timeIntervalSince(tStart)
        let routeGpu = GLM52GraphTelemetry.gpuSeconds
        phases.routeGpuS = routeGpu - gpuStart
        let tTail = Date()

        // ── cb2: attention + proiezioni + router fuso + FFN.
        guard let cb2 = queue.makeCommandBuffer() else {
            throw MetalError.bufferAlloc
        }
        let attentionGeometry = GLM52AttentionGeometry.v5_2
        for i in 0..<count {
            let set = pool.sets[i]
            let visible = basePosition + i + 1
            let selection = selections[i]
            guard let selectionBuffer = device.makeBuffer(
                bytes: selection,
                length: selection.count * MemoryLayout<UInt32>.stride,
                options: .storageModeShared) else {
                throw MetalError.bufferAlloc
            }
            try glm52GraphEncode(
                into: cb2,
                pipelineName: "kernel_glm52_qk_lowrank_q8_0",
                arguments: [0, 0, 0, 0],
                buffers: [set.query, weights.keyB, set.qLow],
                threadgroups: MTLSize(width: layer.kvLoraRank / 4,
                                      height: layer.headCount, depth: 1),
                threadsPerThreadgroup: MTLSize(width: 32, height: 4,
                                               depth: 1))
            try glm52GraphEncode(
                into: cb2,
                pipelineName: "kernel_glm52_attention_indexed_f16",
                arguments: [UInt32(visible), UInt32(selection.count),
                            attentionGeometry.scale.bitPattern, 1],
                buffers: [set.qLow, set.query, caches.compact,
                          selectionBuffer, set.attnLora],
                threadgroups: MTLSize(width: layer.headCount, height: 1,
                                      depth: 1),
                threadsPerThreadgroup: MTLSize(width: 32, height: 4,
                                               depth: 1),
                threadgroupMemoryLength:
                    (selection.count + 5) * MemoryLayout<Float>.stride)
            try glm52GraphEncode(
                into: cb2,
                pipelineName: "kernel_glm52_value_project_q8_0",
                arguments: [0, 0, 0, 0],
                buffers: [set.attnLora, weights.valueB, set.heads],
                threadgroups: MTLSize(width: layer.valueDimension / 4,
                                      height: layer.headCount, depth: 1),
                threadsPerThreadgroup: MTLSize(width: 32, height: 4,
                                               depth: 1))
            try glm52EncodeMatvecQ8(into: cb2, input: set.heads,
                                    weights: weights.attnOutput,
                                    output: set.hidden,
                                    rowCount: layer.embeddingWidth,
                                    inputWidth: headsWidth,
                                    weightType: weights.types.attnOutput,
                                    accumulate: true)
            try glm52EncodeRMSNorm(into: cb2, input: set.hidden,
                                   weight: ffn.ffnNorm, output: set.ffnIn,
                                   width: layer.embeddingWidth)

            switch ffn.kind {
            case .dense(let gate, let up, let down):
                try glm52EncodePairSwiGLU(
                    into: cb2, input: set.ffnIn, gate: gate,
                    up: up, mid: set.mid,
                    hiddenWidth: layer.denseHiddenWidth,
                    inputWidth: layer.embeddingWidth, routeWeight: 1)
                try glm52EncodeMatvecQ8(into: cb2, input: set.mid,
                                        weights: down, output: set.hidden,
                                        rowCount: layer.embeddingWidth,
                                        inputWidth: layer.denseHiddenWidth,
                                        accumulate: true)
            case .sparse(let routerRows, let routerBias, let sharedGate,
                         let sharedUp, let sharedDown, _):
                guard GLM52GpuRouterDispatch.enabled else {
                    throw MetalError.unsupported(
                        "GLM 5.2 prefill batch richiede il router fuso "
                        + "(DS4_GLM_GPU_ROUTER)")
                }
                try glm52EncodeMatvecF32(
                    into: cb2, rows: routerRows,
                    input: set.ffnIn, output: set.routerLogits,
                    rowCount: GLM52RouterReference.expertCount,
                    inputWidth: layer.embeddingWidth)
                try glm52GraphEncode(
                    into: cb2,
                    pipelineName: "kernel_glm52_router_select",
                    arguments: [UInt32(GLM52RouterReference.expertCount),
                                UInt32(GLM52RouterReference.expertsUsed),
                                GLM52RouterReference.expertWeightScale
                                    .bitPattern, 0],
                    buffers: [set.routerLogits, routerBias,
                              set.routerSelected, set.routerWeights,
                              set.routerProbs],
                    threadgroups: MTLSize(width: 1, height: 1, depth: 1),
                    threadsPerThreadgroup: MTLSize(width: 256, height: 1,
                                                   depth: 1),
                    threadgroupMemoryLength: 256 * 2
                        * MemoryLayout<Float>.stride)
                // La shared FFN non dipende dal routing: viaggia nello
                // stesso commit (il per-token la pagava come terzo wait).
                let hidden = layer.expertHiddenWidth
                try glm52EncodePairSwiGLU(
                    into: cb2, input: set.ffnIn, gate: sharedGate,
                    up: sharedUp, mid: set.mid, hiddenWidth: hidden,
                    inputWidth: layer.embeddingWidth, routeWeight: 1,
                    weightType: ffn.sharedWeightTypes.gateUp)
                try glm52EncodeMatvecQ8(into: cb2, input: set.mid,
                                        weights: sharedDown,
                                        output: set.hidden,
                                        rowCount: layer.embeddingWidth,
                                        inputWidth: hidden,
                                        weightType: ffn.sharedWeightTypes.down,
                                        accumulate: true)
            }
        }
        try glm52GraphCommit(cb2)

        // ── tap: routing dai buffer del router fuso.
        var routings: [GLM52RouterOutput?] = []
        routings.reserveCapacity(count)
        for i in 0..<count {
            if case .sparse = ffn.kind {
                let set = pool.sets[i]
                let used = GLM52RouterReference.expertsUsed
                let ids = set.routerSelected.contents()
                    .bindMemory(to: Int32.self, capacity: used)
                let ws = set.routerWeights.contents()
                    .bindMemory(to: Float.self, capacity: used)
                let probs = set.routerProbs.contents()
                    .bindMemory(to: Float.self,
                                capacity: GLM52RouterReference.expertCount)
                routings.append(GLM52RouterOutput(
                    selected: Array(UnsafeBufferPointer(start: ids,
                                                        count: used)),
                    weights: Array(UnsafeBufferPointer(start: ws,
                                                       count: used)),
                    probabilities: Array(UnsafeBufferPointer(
                        start: probs,
                        count: GLM52RouterReference.expertCount))))
            } else {
                routings.append(nil)
            }
        }
        for _ in 0..<count { caches.appendedRow() }
        // expertsS del gruppo = fase tail (attention+router+FFN in cb2):
        // con la leva la shared FFN non è più separabile dal trunk senza
        // reintrodurre il commit che la leva elimina.
        phases.expertsS = Date().timeIntervalSince(tTail)
        phases.expertsGpuS = GLM52GraphTelemetry.gpuSeconds - routeGpu
        return GLM52PrefillGroupOutcome(selections: selections,
                                        routings: routings,
                                        phases: phases)
    }
}
