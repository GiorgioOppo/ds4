import DS4Core
import Foundation
import Metal

// Chained decode — the GLM GraphContext analog. The per-layer functions in
// GLM52DecodeGraph read activations back to the host between stages; here
// the hidden state LIVES on the GPU across the whole layer and the whole
// token, every activation buffer comes from a persistent scratch reused
// across layers and tokens, and the host is tapped only where a decision is
// genuinely CPU-side: the top-k readback on scoring full-indexer layers and
// the 24 KiB ffnIn readback that feeds each sparse layer's F32 router. A
// dense layer is ONE command buffer end to end; a sparse fill-range layer
// is two. Numerically identical to the per-stage path by construction: the
// same kernels encode in the same order on the same buffers.

/// Persistent activation buffers for the chained decode, sized once from the
/// geometry. One instance per engine; never read on the host except through
/// the explicit taps.
public final class GLM52DecodeScratch {
    let hidden: MTLBuffer
    let normed: MTLBuffer
    let attnOut: MTLBuffer
    let ffnIn: MTLBuffer
    let ffnOut: MTLBuffer
    let contribution: MTLBuffer
    let qRank: MTLBuffer
    let qRankNorm: MTLBuffer
    let kvRaw: MTLBuffer
    let cacheReady: MTLBuffer
    let query: MTLBuffer
    let qLow: MTLBuffer
    let attnLora: MTLBuffer
    let heads: MTLBuffer
    let indexerRaw: MTLBuffer
    let indexerQuery: MTLBuffer
    let indexerWeights: MTLBuffer
    let scores: MTLBuffer
    let mid: MTLBuffer
    /// Piani mid del percorso MoE batched: [8 esperti][expertHiddenWidth].
    let midBatch: MTLBuffer
    /// Piani di wave del prefill fase B multi-token (mids/contribs per
    /// applicazione), cresciuti on demand fino al cap di wave.
    let prefillWave = GLM52PrefillWaveScratch()
    /// Scratch del ROUTER FUSO: logits (256), top-8 selezionati (id int32),
    /// pesi normalizzati e probabilità — il readback per layer scende a
    /// 64 byte + 1 KB di probabilità per la usage imatrix.
    let routerLogits: MTLBuffer
    let routerSelected: MTLBuffer
    let routerWeights: MTLBuffer
    let routerProbs: MTLBuffer
    /// Scratch del top-k indexer su device (contesti > topK): indici
    /// selezionati e ping-pong dei run del merge — niente readback degli
    /// score né secondo command buffer.
    let indexerTopKOut: MTLBuffer
    let indexerSortScratch: MTLBuffer
    let geometry: GLM52DecodeGeometry

    public init(runtime: MetalRuntime,
                geometry: GLM52DecodeGeometry,
                scoreCapacity: Int) throws {
        self.geometry = geometry
        let layer = geometry.layer
        func make(_ floats: Int) throws -> MTLBuffer {
            guard let buffer = runtime.device.makeBuffer(
                length: max(floats, 1) * MemoryLayout<Float>.stride,
                options: .storageModeShared) else {
                throw MetalError.bufferAlloc
            }
            return buffer
        }
        hidden = try make(layer.embeddingWidth)
        normed = try make(layer.embeddingWidth)
        attnOut = try make(layer.embeddingWidth)
        ffnIn = try make(layer.embeddingWidth)
        ffnOut = try make(layer.embeddingWidth)
        contribution = try make(layer.embeddingWidth)
        qRank = try make(geometry.qLoraRank)
        qRankNorm = try make(geometry.qLoraRank)
        kvRaw = try make(layer.kvRawWidth)
        cacheReady = try make(layer.kvRawWidth)
        query = try make(geometry.queryWidth)
        qLow = try make(layer.headCount * layer.kvLoraRank)
        attnLora = try make(layer.headCount * layer.kvLoraRank)
        heads = try make(layer.headCount * layer.valueDimension)
        indexerRaw = try make(geometry.indexerHeadDimension)
        indexerQuery = try make(geometry.indexerQueryWidth)
        indexerWeights = try make(geometry.indexerHeadCount)
        scores = try make(scoreCapacity)
        mid = try make(max(layer.denseHiddenWidth, layer.expertHiddenWidth))
        midBatch = try make(8 * layer.expertHiddenWidth)
        routerLogits = try make(GLM52RouterReference.expertCount)
        routerSelected = try make(GLM52RouterReference.expertsUsed)
        routerWeights = try make(GLM52RouterReference.expertsUsed)
        routerProbs = try make(GLM52RouterReference.expertCount)
        indexerTopKOut = try make(Int(geometry.indexerTopK))
        indexerSortScratch = try make(2 * scoreCapacity)
    }

    /// Load the embedded token into the resident hidden state.
    func loadHidden(_ values: [Float]) {
        precondition(values.count * MemoryLayout<Float>.stride
                         <= hidden.length)
        values.withUnsafeBytes {
            hidden.contents().copyMemory(
                from: $0.baseAddress!, byteCount: $0.count)
        }
    }

    func readHidden(count: Int) -> [Float] {
        let pointer = hidden.contents().bindMemory(
            to: Float.self, capacity: count)
        return Array(UnsafeBufferPointer(start: pointer, count: count))
    }
}

/// Tempi wall-clock per fase di UN layer chained, nello stesso spirito del
/// DecodeProfile DeepSeek. I confini coincidono con i commit sincroni già
/// presenti (trunk attn/router → staging esperti → FFN): misurarli non
/// aggiunge alcun punto di sincronizzazione. Un layer dense è UN solo
/// command buffer end-to-end, quindi non separabile: finisce in denseS
/// (la voce "layer (alt)" del report).
public struct GLM52LayerPhases: Sendable {
    public var routeS = 0.0    // trunk attenzione + indexer + router (CPU incluso)
    public var gatherS = 0.0   // lettura record esperti (arena/SSD, sincrona)
    public var expertsS = 0.0  // FFN shared + esperti routed (encode + commit)
    public var denseS = 0.0    // layer dense a commit unico (non splittabile)
    // Quota di ESECUZIONE GPU dentro le fasi wall-clock qui sopra (dai
    // timestamp gpuStart/gpuEnd dei commit): la differenza è host —
    // encode, round-trip di sync, router CPU, allocazioni.
    public var routeGpuS = 0.0
    public var expertsGpuS = 0.0
    public var denseGpuS = 0.0
    public init() {}
}

extension MetalRuntime {
    func glm52EncodeMatvecF32(into commandBuffer: MTLCommandBuffer,
                              rows: MTLBuffer, input: MTLBuffer,
                              output: MTLBuffer, rowCount: Int,
                              inputWidth: Int) throws {
        let width = 64
        try glm52GraphEncode(
            into: commandBuffer, pipelineName: "kernel_glm52_matvec_f32",
            arguments: [UInt32(rowCount), UInt32(inputWidth), 0, 0],
            buffers: [rows, input, output],
            threadgroups: MTLSize(width: (rowCount + width - 1) / width,
                                  height: 1, depth: 1),
            threadsPerThreadgroup: MTLSize(width: width, height: 1, depth: 1))
    }

    /// Validation wrapper for the F32 matvec kernel.
    public func glm52MatvecF32(rows: [Float], input: [Float],
                               rowCount: Int) throws -> [Float] {
        guard rowCount > 0, !input.isEmpty,
              rows.count == rowCount * input.count else {
            throw MetalError.unsupported(
                "GLM 5.2 F32 matvec expects [\(rowCount)]x\(input.count) rows")
        }
        let rowsBuffer = try glm52GraphBuffer(rows)
        let inputBuffer = try glm52GraphBuffer(input)
        let output = try glm52GraphOutputBuffer(floats: rowCount)
        guard let commandBuffer = queue.makeCommandBuffer() else {
            throw MetalError.bufferAlloc
        }
        try glm52EncodeMatvecF32(into: commandBuffer, rows: rowsBuffer,
                                 input: inputBuffer, output: output,
                                 rowCount: rowCount,
                                 inputWidth: input.count)
        try glm52GraphCommit(commandBuffer)
        return glm52GraphReadback(output, count: rowCount)
    }

    /// One full decode layer with the hidden state resident in
    /// `scratch.hidden` (in/out). Encodes attention, residual, FFN norm and
    /// — on dense layers — the whole FFN into as few command buffers as the
    /// host taps allow. Returns the routing (sparse) and the selection.
    public func glm52ChainedDecodeLayer(
        weights: GLM52ResidentDecodeWeights,
        ffn: GLM52ResidentFFN,
        caches: GLM52ResidentDecodeCaches,
        scratch: GLM52DecodeScratch,
        reusedSelection: [UInt32]?,
        position: Int,
        activeExperts: Int? = nil,
        deferSparseFFN: Bool = false,
        fusedDecode: Bool = false,
        carry: inout MTLCommandBuffer?) throws
        -> (routing: GLM52RouterOutput?, selection: [UInt32],
            phases: GLM52LayerPhases) {
        let tLayer = Date()
        var phases = GLM52LayerPhases()
        let gpuStart = GLM52GraphTelemetry.gpuSeconds
        let g = weights.geometry
        let layer = g.layer
        let headsWidth = layer.headCount * layer.valueDimension
        let visible = position + 1
        guard position == caches.rows, visible <= caches.capacity,
              visible <= scratch.scores.length / MemoryLayout<Float>.stride
        else {
            throw MetalError.unsupported(
                "GLM 5.2 chained decode position \(position) does not match "
                + "\(caches.rows) live rows / capacity \(caches.capacity)")
        }
        if weights.indexer != nil {
            guard reusedSelection == nil, caches.indexerKeys != nil else {
                throw MetalError.unsupported(
                    "full-indexer chained layer needs indexer caches and no "
                    + "reused selection")
            }
        } else {
            guard reusedSelection != nil, caches.indexerKeys == nil else {
                throw MetalError.unsupported(
                    "IndexShare chained layer requires the preceding "
                    + "full-indexer selection and owns no indexer keys")
            }
        }

        var commandBuffer: MTLCommandBuffer
        if let carried = carry {
            // FUSIONE COMMIT: il buffer arriva col FFN (o il layer dense) del
            // layer precedente già codificato e non ancora committato — il
            // trunk di QUESTO layer vi si accoda e le due metà pagano UNA
            // sola attesa. L'ordinamento è garantito dall'hazard tracking di
            // Metal sugli scratch condivisi (encoder in sequenza nello
            // stesso command buffer).
            commandBuffer = carried
            carry = nil
        } else {
            guard let fresh = queue.makeCommandBuffer() else {
                throw MetalError.bufferAlloc
            }
            commandBuffer = fresh
        }

        // Shared trunk (all on scratch buffers, hidden untouched so far).
        try glm52EncodeRMSNorm(into: commandBuffer, input: scratch.hidden,
                               weight: weights.attnNorm,
                               output: scratch.normed,
                               width: layer.embeddingWidth)
        // qA e kvA leggono lo stesso vettore normato: UN dispatch a coppia.
        try glm52EncodeMatvecQ8Pair(into: commandBuffer,
                                    input: scratch.normed,
                                    weightsA: weights.qA,
                                    outputA: scratch.qRank,
                                    rowsA: g.qLoraRank,
                                    typeA: weights.types.qA,
                                    weightsB: weights.kvA,
                                    outputB: scratch.kvRaw,
                                    rowsB: layer.kvRawWidth,
                                    typeB: weights.types.kvA,
                                    inputWidth: layer.embeddingWidth)
        try glm52EncodeRMSNorm(into: commandBuffer, input: scratch.qRank,
                               weight: weights.qANorm,
                               output: scratch.qRankNorm,
                               width: g.qLoraRank)
        try glm52GraphEncode(
            into: commandBuffer,
            pipelineName: "kernel_glm52_kv_lora_norm_cache_ready_f32",
            arguments: [1, 0, 0, 0],
            buffers: [scratch.kvRaw, weights.kvANorm, scratch.cacheReady],
            threadgroups: MTLSize(width: 1, height: 1, depth: 1),
            threadsPerThreadgroup: MTLSize(width: 128, height: 1, depth: 1),
            threadgroupMemoryLength: 128 * MemoryLayout<Float>.stride)
        try glm52GraphEncode(
            into: commandBuffer,
            pipelineName: "kernel_glm52_store_compact_row_f16",
            arguments: [UInt32(position), 1, UInt32(caches.capacity), 0],
            buffers: [scratch.cacheReady, caches.compact],
            threadgroups: MTLSize(width: 1, height: 1, depth: 1),
            threadsPerThreadgroup: MTLSize(width: 64, height: 1, depth: 1))
        try glm52EncodeMatvecQ8(into: commandBuffer,
                                input: scratch.qRankNorm,
                                weights: weights.qB, output: scratch.query,
                                rowCount: g.queryWidth,
                                inputWidth: g.qLoraRank,
                                weightType: weights.types.qB)
        try glm52EncodeRope(into: commandBuffer,
                            pipelineName: "kernel_glm52_rope_tail_f32",
                            values: scratch.query,
                            headCount: layer.headCount,
                            headDimension: g.qkDimension,
                            rotationDimension: layer.ropeDimension,
                            position: position)

        var selection: [UInt32]
        if let indexer = weights.indexer, let keyCache = caches.indexerKeys {
            try glm52EncodeMatvecQ8(into: commandBuffer,
                                    input: scratch.hidden,
                                    weights: indexer.key,
                                    output: scratch.indexerRaw,
                                    rowCount: g.indexerHeadDimension,
                                    inputWidth: layer.embeddingWidth,
                                    weightType: weights.types.indexerKey)
            try glm52GraphEncode(
                into: commandBuffer,
                pipelineName: "kernel_glm52_store_indexer_k_f16",
                arguments: [UInt32(position), 1, UInt32(caches.capacity), 0],
                buffers: [scratch.indexerRaw, indexer.keyNorm,
                          indexer.keyNormBias, keyCache],
                threadgroups: MTLSize(width: 1, height: 1, depth: 1),
                threadsPerThreadgroup: MTLSize(width: 32, height: 1, depth: 1),
                threadgroupMemoryLength: 32 * MemoryLayout<Float>.stride)
            if visible <= g.indexerTopK {
                selection = (0..<visible).map(UInt32.init)
            } else {
                // Score on GPU (indexer proj included), then the single
                // host tap of the branch: the top-k over the score row.
                try glm52EncodeMatvecQ8(
                    into: commandBuffer,
                    input: scratch.qRankNorm,
                    weights: indexer.queryB,
                    output: scratch.indexerQuery,
                    rowCount: g.indexerQueryWidth,
                    inputWidth: g.qLoraRank,
                    weightType: weights.types.indexerQueryB)
                try glm52EncodeRope(
                    into: commandBuffer,
                    pipelineName: "kernel_glm52_rope_prefix_f32",
                    values: scratch.indexerQuery,
                    headCount: g.indexerHeadCount,
                    headDimension: g.indexerHeadDimension,
                    rotationDimension: g.indexerRotationDimension,
                    position: position)
                let projBuffer = try glm52GraphBuffer(indexer.proj)
                try glm52EncodeMatvecF32(into: commandBuffer,
                                         rows: projBuffer,
                                         input: scratch.hidden,
                                         output: scratch.indexerWeights,
                                         rowCount: g.indexerHeadCount,
                                         inputWidth: layer.embeddingWidth)
                try glm52GraphEncode(
                    into: commandBuffer,
                    pipelineName: "kernel_glm52_indexer_scores_f16",
                    arguments: [UInt32(visible), 1, UInt32(position),
                                g.indexerScale.bitPattern],
                    buffers: [scratch.indexerQuery, scratch.indexerWeights,
                              keyCache, scratch.scores],
                    threadgroups: MTLSize(width: visible, height: 1,
                                          depth: 1),
                    threadsPerThreadgroup: MTLSize(width: 32, height: 4,
                                                   depth: 1),
                    threadgroupMemoryLength:
                        (128 + 4) * MemoryLayout<Float>.stride)
                // TOP-K sul DEVICE: gli score non tornano più sul host (e
                // niente re-upload + secondo command buffer come faceva il
                // wrapper standalone); il commit resta — gli indici servono
                // alla CPU per il binding e il riuso IndexShare — ma
                // rilegge solo topK int32.
                try glm52EncodeIndexerTopK(
                    into: commandBuffer, scores: scratch.scores,
                    rowCount: visible, topK: g.indexerTopK,
                    output: scratch.indexerTopKOut,
                    sortScratch: scratch.indexerSortScratch)
                try glm52GraphCommit(commandBuffer)
                let top = scratch.indexerTopKOut.contents()
                    .bindMemory(to: UInt32.self, capacity: g.indexerTopK)
                selection = Array(UnsafeBufferPointer(start: top,
                                                      count: g.indexerTopK))
                guard let fresh = queue.makeCommandBuffer() else {
                    throw MetalError.bufferAlloc
                }
                commandBuffer = fresh
            }
        } else {
            selection = reusedSelection ?? []
        }
        guard !selection.isEmpty, selection.count <= visible,
              Set(selection).count == selection.count,
              selection.allSatisfy({ Int($0) < visible }) else {
            throw MetalError.unsupported(
                "GLM 5.2 chained decode selection must be unique rows "
                + "inside 0..<\(visible)")
        }
        guard let selectionBuffer = device.makeBuffer(
            bytes: selection,
            length: selection.count * MemoryLayout<UInt32>.stride,
            options: .storageModeShared) else {
            throw MetalError.bufferAlloc
        }

        // Attention tail, residual and FFN norm — hidden updated in place.
        let attentionGeometry = GLM52AttentionGeometry.v5_2
        try glm52GraphEncode(
            into: commandBuffer,
            pipelineName: "kernel_glm52_qk_lowrank_q8_0",
            arguments: [0, 0, 0, 0],
            buffers: [scratch.query, weights.keyB, scratch.qLow],
            threadgroups: MTLSize(width: layer.kvLoraRank / 4,
                                  height: layer.headCount, depth: 1),
            threadsPerThreadgroup: MTLSize(width: 32, height: 4, depth: 1))
        try glm52GraphEncode(
            into: commandBuffer,
            pipelineName: "kernel_glm52_attention_indexed_f16",
            arguments: [UInt32(visible), UInt32(selection.count),
                        attentionGeometry.scale.bitPattern, 1],
            buffers: [scratch.qLow, scratch.query, caches.compact,
                      selectionBuffer, scratch.attnLora],
            threadgroups: MTLSize(width: layer.headCount, height: 1,
                                  depth: 1),
            threadsPerThreadgroup: MTLSize(width: 32, height: 4, depth: 1),
            threadgroupMemoryLength:
                (selection.count + 5) * MemoryLayout<Float>.stride)
        try glm52GraphEncode(
            into: commandBuffer,
            pipelineName: "kernel_glm52_value_project_q8_0",
            arguments: [0, 0, 0, 0],
            buffers: [scratch.attnLora, weights.valueB, scratch.heads],
            threadgroups: MTLSize(width: layer.valueDimension / 4,
                                  height: layer.headCount, depth: 1),
            threadsPerThreadgroup: MTLSize(width: 32, height: 4, depth: 1))
        try glm52EncodeMatvecQ8(into: commandBuffer, input: scratch.heads,
                                weights: weights.attnOutput,
                                output: scratch.hidden,
                                rowCount: layer.embeddingWidth,
                                inputWidth: headsWidth,
                                weightType: weights.types.attnOutput,
                                accumulate: true)
        try glm52EncodeRMSNorm(into: commandBuffer, input: scratch.hidden,
                               weight: ffn.ffnNorm, output: scratch.ffnIn,
                               width: layer.embeddingWidth)

        var routing: GLM52RouterOutput?
        switch ffn.kind {
        case .dense(let gate, let up, let down):
            // No host tap: the dense FFN rides the same command buffer.
            try glm52EncodePairSwiGLU(
                into: commandBuffer, input: scratch.ffnIn, gate: gate,
                up: up, mid: scratch.mid,
                hiddenWidth: layer.denseHiddenWidth,
                inputWidth: layer.embeddingWidth, routeWeight: 1)
            try glm52EncodeMatvecQ8(into: commandBuffer, input: scratch.mid,
                                    weights: down, output: scratch.hidden,
                                    rowCount: layer.embeddingWidth,
                                    inputWidth: layer.denseHiddenWidth,
                                    accumulate: true)
            if fusedDecode {
                // Nessun tap host in un layer dense: l'INTERO layer viaggia
                // col commit del trunk successivo — zero attese qui.
                carry = commandBuffer
            } else {
                try glm52GraphCommit(commandBuffer)
            }
            phases.denseS = Date().timeIntervalSince(tLayer)
            phases.denseGpuS = GLM52GraphTelemetry.gpuSeconds - gpuStart

        case .sparse(let routerRows, let routerBias, let sharedGate,
                     let sharedUp, let sharedDown, let expertProvider):
            let routed: GLM52RouterOutput
            if GLM52GpuRouterDispatch.enabled {
                // ROUTER FUSO: matvec F32 + sigmoid/bias/top-8 viaggiano nel
                // commit del trunk; il tap host si riduce a leggere id, pesi
                // e probabilità dagli scratch condivisi.
                try glm52EncodeMatvecF32(
                    into: commandBuffer, rows: routerRows,
                    input: scratch.ffnIn, output: scratch.routerLogits,
                    rowCount: GLM52RouterReference.expertCount,
                    inputWidth: layer.embeddingWidth)
                try glm52GraphEncode(
                    into: commandBuffer,
                    pipelineName: "kernel_glm52_router_select",
                    arguments: [UInt32(GLM52RouterReference.expertCount),
                                UInt32(GLM52RouterReference.expertsUsed),
                                GLM52RouterReference.expertWeightScale
                                    .bitPattern, 0],
                    buffers: [scratch.routerLogits, routerBias,
                              scratch.routerSelected, scratch.routerWeights,
                              scratch.routerProbs],
                    threadgroups: MTLSize(width: 1, height: 1, depth: 1),
                    threadsPerThreadgroup: MTLSize(width: 256, height: 1,
                                                   depth: 1),
                    threadgroupMemoryLength: 256 * 2
                        * MemoryLayout<Float>.stride)
                try glm52GraphCommit(commandBuffer)
                let used = GLM52RouterReference.expertsUsed
                let ids = scratch.routerSelected.contents()
                    .bindMemory(to: Int32.self, capacity: used)
                let weights = scratch.routerWeights.contents()
                    .bindMemory(to: Float.self, capacity: used)
                let probs = scratch.routerProbs.contents()
                    .bindMemory(to: Float.self,
                                capacity: GLM52RouterReference.expertCount)
                routed = GLM52RouterOutput(
                    selected: Array(UnsafeBufferPointer(start: ids,
                                                        count: used)),
                    weights: Array(UnsafeBufferPointer(start: weights,
                                                       count: used)),
                    probabilities: Array(UnsafeBufferPointer(
                        start: probs,
                        count: GLM52RouterReference.expertCount)))
            } else {
                try glm52GraphCommit(commandBuffer)
                // Fallback di riferimento: router su CPU dai buffer.
                let ffnInHost = glm52GraphReadback(
                    scratch.ffnIn, count: layer.embeddingWidth)
                routed = try glm52RouteFromBuffers(
                    rows: routerRows, bias: routerBias, ffnIn: ffnInHost)
            }
            routing = routed
            let tRoute = Date()
            phases.routeS = tRoute.timeIntervalSince(tLayer)
            let routeGpuEnd = GLM52GraphTelemetry.gpuSeconds
            phases.routeGpuS = routeGpuEnd - gpuStart

            guard let ffnBuffer = queue.makeCommandBuffer() else {
                throw MetalError.bufferAlloc
            }
            let hidden = layer.expertHiddenWidth
            try glm52EncodePairSwiGLU(
                into: ffnBuffer, input: scratch.ffnIn, gate: sharedGate,
                up: sharedUp, mid: scratch.mid, hiddenWidth: hidden,
                inputWidth: layer.embeddingWidth, routeWeight: 1,
                weightType: ffn.sharedWeightTypes.gateUp)
            try glm52EncodeMatvecQ8(into: ffnBuffer, input: scratch.mid,
                                    weights: sharedDown,
                                    output: scratch.hidden,
                                    rowCount: layer.embeddingWidth,
                                    inputWidth: hidden,
                                    weightType: ffn.sharedWeightTypes.down,
                                    accumulate: true)
            if deferSparseFFN {
                // Two-phase prefill, phase A: the shared expert is already
                // encoded above; the ROUTED experts are applied later by
                // the caller EXPERT-MAJOR (union staging reads each unique
                // record once for the whole prompt). scratch.ffnIn and
                // scratch.hidden are the per-token planes it snapshots.
                try glm52GraphCommit(ffnBuffer)
                phases.expertsS = Date().timeIntervalSince(tRoute)
                phases.expertsGpuS =
                    GLM52GraphTelemetry.gpuSeconds - routeGpuEnd
                caches.appendedRow()
                return (routing, selection, phases)
            }
            let usedExperts = min(routed.selected.count,
                                  max(1, activeExperts
                                          ?? routed.selected.count))
            if let stage = ffn.stagedSelection {
                // Zero-copy path: the WHOLE selection lands in one reusable
                // staging buffer with ONE concurrent read burst, and each
                // record is bound by offset — no per-expert allocations or
                // copies. Byte-identical to the provider path by
                // construction (same packed record layout).
                let ids = (0..<usedExperts).map {
                    UInt32(bitPattern: routed.selected[$0])
                }
                let tStage = Date()
                let staged = try stage(ids)
                phases.gatherS = Date().timeIntervalSince(tStage)
                if GLM52MatvecDispatch.cooperative,
                   GLM52MoEBatchDispatch.enabled, usedExperts <= 8 {
                    // MoE BATCHED: tutti gli esperti in due dispatch —
                    // l'analogo del kernel moe DeepSeek. Il fallback
                    // per-esperto sotto resta il riferimento (DS4_GLM_SG=0
                    // o DS4_GLM_MOE_BATCH=0).
                    try glm52EncodeMoEBatch(
                        into: ffnBuffer, staged: staged,
                        weights: Array(routed.weights[0..<usedExperts]),
                        input: scratch.ffnIn, mids: scratch.midBatch,
                        accumulate: scratch.hidden,
                        hiddenWidth: hidden,
                        inputWidth: layer.embeddingWidth)
                } else {
                for rank in 0..<usedExperts {
                    let base = staged.recordOffsets[rank]
                    try glm52EncodePairSwiGLU(
                        into: ffnBuffer, input: scratch.ffnIn,
                        gate: staged.buffer, up: staged.buffer,
                        mid: scratch.mid, hiddenWidth: hidden,
                        inputWidth: layer.embeddingWidth,
                        routeWeight: routed.weights[rank],
                        weightType: staged.gateUpType,
                        gateOffset: base,
                        upOffset: base + staged.gateBytes)
                    try glm52EncodeMatvecQ8(
                        into: ffnBuffer, input: scratch.mid,
                        weights: staged.buffer,
                        output: scratch.hidden,
                        rowCount: layer.embeddingWidth,
                        inputWidth: hidden,
                        weightType: staged.downType,
                        weightsOffset: base + staged.gateBytes
                            + staged.upBytes,
                        accumulate: true)
                }
                }
            } else {
                for rank in 0..<usedExperts {
                    let tRead = Date()
                    let record = try expertProvider(
                        UInt32(bitPattern: routed.selected[rank]))
                    phases.gatherS += Date().timeIntervalSince(tRead)
                    let gate = try glm52GraphBuffer(record.gate)
                    let up = try glm52GraphBuffer(record.up)
                    let down = try glm52GraphBuffer(record.down)
                    try glm52EncodePairSwiGLU(
                        into: ffnBuffer, input: scratch.ffnIn, gate: gate,
                        up: up, mid: scratch.mid, hiddenWidth: hidden,
                        inputWidth: layer.embeddingWidth,
                        routeWeight: routed.weights[rank],
                        weightType: record.gateUpType)
                    try glm52EncodeMatvecQ8(into: ffnBuffer,
                                            input: scratch.mid,
                                            weights: down,
                                            output: scratch.hidden,
                                            rowCount: layer.embeddingWidth,
                                            inputWidth: hidden,
                                            weightType: record.downType,
                                            accumulate: true)
                }
            }
            if fusedDecode {
                // Il FFN non committa: sarà il trunk del layer successivo (o
                // il flush di fine token) a pagarne l'attesa. Gli slot arena
                // della selezione restano validi: il prossimo stage() avviene
                // DOPO quel commit fuso, mai prima.
                carry = ffnBuffer
            } else {
                try glm52GraphCommit(ffnBuffer)
            }
            phases.expertsS = Date().timeIntervalSince(tRoute) - phases.gatherS
            phases.expertsGpuS = GLM52GraphTelemetry.gpuSeconds - routeGpuEnd
        }

        caches.appendedRow()
        return (routing, selection, phases)
    }

    /// Two-phase prefill, phase B: apply ONE staged batch of routed experts
    /// to every (token, weight) that selected them, expert-major.
    /// `hiddenAll`/`ffnInAll` are `[token][embeddingWidth]` F32 planes
    /// snapshotted by phase A; each token's hidden accumulates its
    /// contributions in place. Note the float-order caveat: per-token
    /// contributions land in union order instead of router rank order, so
    /// prefill and token-by-token decode agree within accumulation
    /// tolerance, no longer bit-exactly.
    ///
    /// Default path (DS4_GLM_PREFILL_MOE): the token loop lives INSIDE the
    /// multi-token kernels — expert weights cross DRAM once per 4-token
    /// tile instead of once per token, and one wave is 3 dispatches instead
    /// of 3 per application. Per-token math and per-token accumulation
    /// order are identical to the legacy per-application path, so the two
    /// agree bit-for-bit (pinned by GLM52MoEPrefillTests).
    public func glm52ApplyRoutedExperts(
        staged: GLM52StagedExpertSelection,
        applications: [(slot: Int, token: Int, weight: Float)],
        hiddenAll: MTLBuffer, ffnInAll: MTLBuffer,
        scratch: GLM52DecodeScratch,
        embeddingWidth: Int, expertHiddenWidth: Int) throws {
        guard !applications.isEmpty else { return }
        guard GLM52PrefillMoEDispatch.enabled,
              GLM52MatvecDispatch.cooperative else {
            try glm52ApplyRoutedExpertsLegacy(
                staged: staged, applications: applications,
                hiddenAll: hiddenAll, ffnInAll: ffnInAll, scratch: scratch,
                embeddingWidth: embeddingWidth,
                expertHiddenWidth: expertHiddenWidth)
            return
        }
        guard let commandBuffer = queue.makeCommandBuffer() else {
            throw MetalError.bufferAlloc
        }
        // Wave cap: bounds the mids/contribs planes (~32 MB at 1024). A
        // token's applications may split across waves; each wave reduces
        // its own contributions in ascending application order, so the
        // per-token order stays the legacy one.
        let waveCap = max(1, GLM52PrefillMoEDispatch.waveCap)
        var index = 0
        while index < applications.count {
            let wave = Array(applications[index..<min(
                index + waveCap, applications.count)])
            try glm52EncodeRoutedExpertsWave(
                into: commandBuffer, staged: staged, wave: wave,
                hiddenAll: hiddenAll, ffnInAll: ffnInAll, scratch: scratch,
                embeddingWidth: embeddingWidth,
                expertHiddenWidth: expertHiddenWidth)
            index += wave.count
        }
        try glm52GraphCommit(commandBuffer)
    }

    private func glm52EncodeRoutedExpertsWave(
        into commandBuffer: MTLCommandBuffer,
        staged: GLM52StagedExpertSelection,
        wave: [(slot: Int, token: Int, weight: Float)],
        hiddenAll: MTLBuffer, ffnInAll: MTLBuffer,
        scratch: GLM52DecodeScratch,
        embeddingWidth: Int, expertHiddenWidth: Int) throws {
        // Expert entries: consecutive equal slots (the wave is expert-major
        // by construction).
        var expertMeta: [UInt32] = []
        var appTokens: [UInt32] = []
        var appWeights: [Float] = []
        var entryStart = 0
        for (i, application) in wave.enumerated() {
            appTokens.append(UInt32(application.token))
            appWeights.append(application.weight)
            let isLast = i == wave.count - 1
            if isLast || wave[i + 1].slot != application.slot {
                expertMeta.append(UInt32(
                    staged.recordOffsets[application.slot]))
                expertMeta.append(UInt32(entryStart))
                expertMeta.append(UInt32(i - entryStart + 1))
                entryStart = i + 1
            }
        }
        let expertEntries = expertMeta.count / 3
        // Token entries: each wave token with its ascending application
        // list (ascending == expert-major == the legacy add order).
        var byToken: [Int: [Int]] = [:]
        for (i, application) in wave.enumerated() {
            byToken[application.token, default: []].append(i)
        }
        var tokenMeta: [UInt32] = []
        var tokenApps: [UInt32] = []
        for (token, apps) in byToken.sorted(by: { $0.key < $1.key }) {
            tokenMeta.append(UInt32(token))
            tokenMeta.append(UInt32(tokenApps.count))
            tokenMeta.append(UInt32(apps.count))
            tokenApps.append(contentsOf: apps.map(UInt32.init))
        }
        let tokenEntries = tokenMeta.count / 3

        let device = staged.buffer.device
        let mids = try scratch.prefillWave.mids(
            device: device,
            bytes: wave.count * expertHiddenWidth
                * MemoryLayout<Float>.stride)
        let contribs = try scratch.prefillWave.contribs(
            device: device,
            bytes: wave.count * embeddingWidth * MemoryLayout<Float>.stride)
        func upload(_ words: [UInt32]) throws -> MTLBuffer {
            guard let buffer = device.makeBuffer(
                bytes: words, length: words.count * 4,
                options: .storageModeShared) else {
                throw MetalError.bufferAlloc
            }
            return buffer
        }
        let expertMetaBuffer = try upload(expertMeta)
        let appTokensBuffer = try upload(appTokens)
        let appWeightsBuffer = try upload(appWeights.map(\.bitPattern))
        let tokenMetaBuffer = try upload(tokenMeta)
        let tokenAppsBuffer = try upload(tokenApps)

        let arguments: [UInt32] = [
            staged.gateUpType, UInt32(expertHiddenWidth),
            UInt32(embeddingWidth), UInt32(expertEntries),
            UInt32(staged.gateBytes),
            UInt32(staged.gateBytes + staged.upBytes),
            staged.downType, UInt32(tokenEntries),
        ]
        let rows = GLM52MatvecDispatch.rowsPerThreadgroup
        try glm52GraphEncode(
            into: commandBuffer,
            pipelineName: "kernel_glm52_moe_prefill_swiglu_sg",
            arguments: arguments,
            buffers: [expertMetaBuffer, appTokensBuffer, appWeightsBuffer,
                      ffnInAll, staged.buffer, mids],
            threadgroups: MTLSize(
                width: (expertHiddenWidth + rows - 1) / rows,
                height: 1, depth: expertEntries),
            threadsPerThreadgroup: MTLSize(width: 32, height: rows,
                                           depth: 1))
        try glm52GraphEncode(
            into: commandBuffer,
            pipelineName: "kernel_glm52_moe_prefill_down_sg",
            arguments: arguments,
            buffers: [expertMetaBuffer, mids, staged.buffer, contribs],
            threadgroups: MTLSize(
                width: (embeddingWidth + rows - 1) / rows,
                height: 1, depth: expertEntries),
            threadsPerThreadgroup: MTLSize(width: 32, height: rows,
                                           depth: 1))
        try glm52GraphEncode(
            into: commandBuffer,
            pipelineName: "kernel_glm52_moe_prefill_reduce",
            arguments: arguments,
            buffers: [tokenMetaBuffer, tokenAppsBuffer, contribs, hiddenAll],
            threadgroups: MTLSize(width: (embeddingWidth + 255) / 256,
                                  height: tokenEntries, depth: 1),
            threadsPerThreadgroup: MTLSize(width: 256, height: 1, depth: 1))
    }

    /// Legacy per-application path (DS4_GLM_PREFILL_MOE=0 or scalar-dot
    /// mode): three dispatches per (expert, token), shared scratch.
    func glm52ApplyRoutedExpertsLegacy(
        staged: GLM52StagedExpertSelection,
        applications: [(slot: Int, token: Int, weight: Float)],
        hiddenAll: MTLBuffer, ffnInAll: MTLBuffer,
        scratch: GLM52DecodeScratch,
        embeddingWidth: Int, expertHiddenWidth: Int) throws {
        guard !applications.isEmpty else { return }
        guard let commandBuffer = queue.makeCommandBuffer() else {
            throw MetalError.bufferAlloc
        }
        let planeStride = embeddingWidth * MemoryLayout<Float>.stride
        for application in applications {
            let base = staged.recordOffsets[application.slot]
            let tokenOffset = application.token * planeStride
            try glm52EncodePairSwiGLU(
                into: commandBuffer, input: ffnInAll,
                gate: staged.buffer, up: staged.buffer,
                mid: scratch.mid, hiddenWidth: expertHiddenWidth,
                inputWidth: embeddingWidth,
                routeWeight: application.weight,
                weightType: staged.gateUpType,
                gateOffset: base,
                upOffset: base + staged.gateBytes,
                inputOffset: tokenOffset)
            try glm52EncodeMatvecQ8(
                into: commandBuffer, input: scratch.mid,
                weights: staged.buffer, output: scratch.contribution,
                rowCount: embeddingWidth, inputWidth: expertHiddenWidth,
                weightType: staged.downType,
                weightsOffset: base + staged.gateBytes + staged.upBytes)
            try glm52EncodeAdd(
                into: commandBuffer, a: hiddenAll,
                b: scratch.contribution, output: hiddenAll,
                count: embeddingWidth,
                aOffset: tokenOffset, outputOffset: tokenOffset)
        }
        try glm52GraphCommit(commandBuffer)
    }
}

/// Lazily-grown wave planes for the multi-token prefill phase B. Grown to
/// the wave high-water mark (~32 MB at the 1024-application cap) and kept
/// for the engine's lifetime; the single-driver decode discipline makes the
/// unguarded mutation safe.
public final class GLM52PrefillWaveScratch {
    private var midsBuffer: MTLBuffer?
    private var contribsBuffer: MTLBuffer?

    func mids(device: MTLDevice, bytes: Int) throws -> MTLBuffer {
        if let midsBuffer, midsBuffer.length >= bytes { return midsBuffer }
        guard let grown = device.makeBuffer(
            length: bytes, options: .storageModeShared) else {
            throw MetalError.bufferAlloc
        }
        midsBuffer = grown
        return grown
    }

    func contribs(device: MTLDevice, bytes: Int) throws -> MTLBuffer {
        if let contribsBuffer, contribsBuffer.length >= bytes {
            return contribsBuffer
        }
        guard let grown = device.makeBuffer(
            length: bytes, options: .storageModeShared) else {
            throw MetalError.bufferAlloc
        }
        contribsBuffer = grown
        return grown
    }
}
