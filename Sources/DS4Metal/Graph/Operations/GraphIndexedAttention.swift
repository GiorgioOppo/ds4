import Foundation
import Metal

// Attention DSA INDICIZZATA per il decode (DS4_INDEXED_ATTN=1) — il "Stage B"
// dei kernel già portati in dsv4_misc.metal. Il percorso storico paga
// O(contesto): staging F16 dell'intero span raw+comp (flash_kv_stage) e flash
// mascherata su tutte le righe. Qui, quando l'indexer è attivo, si attende
// SOLO sulle top-K righe compresse selezionate (512 sul Flash) più la
// finestra raw SWA — il costo per token diventa O(nSWA + topK) e resta
// costante al crescere del contesto (il muro DSA oltre ~4096 token).
//
// Selezione: kernel_dsv4_indexer_topk_indices_one — lo STESSO min-heap del
// percorso a maschera (set identico per costruzione) ma con uscita a indici;
// poi kernel_dsv4_sort_i32_rows_asc li riordina per id crescente, così la
// scansione segue l'ordine di cache del grafo denso (semantica --quality di
// ds4; il costo su 512 id è trascurabile).
extension GraphContext {
    /// Top-K dell'indexer a INDICI (una riga di score, decode): scrive in
    /// `out` i topK id di riga compressa, -1 di riempimento.
    func indexerTopKIndices(scores: GPUTensor, out: GPUTensor,
                            nScores: Int, topK: Int) throws {
        precondition(nScores >= 0 && topK > 0)
        let keep = min(topK, nScores)
        var args = [UInt8](repeating: 0, count: 16)
        func u32(_ off: Int, _ value: Int) {
            var v = UInt32(value).littleEndian
            withUnsafeBytes(of: &v) { args.replaceSubrange(off..<(off + 4), with: $0) }
        }
        u32(0, 0); u32(4, nScores); u32(8, nScores); u32(12, topK)
        let pso = try rt.pipeline("kernel_dsv4_indexer_topk_indices_one")
        let nth = min(max(32, min(topK, 256)), pso.maxTotalThreadsPerThreadgroup)
        guard nth >= 32 else {
            throw MetalError.unsupported(
                "indexer GPU top-k richiede almeno 32 thread, limite \(pso.maxTotalThreadsPerThreadgroup)")
        }
        let e = encoder
        e.setComputePipelineState(pso)
        args.withUnsafeBytes { e.setBytes($0.baseAddress!, length: args.count, index: 0) }
        e.setBuffer(scores.buffer, offset: scores.byteOffset, index: 1)
        e.setBuffer(out.buffer, offset: out.byteOffset, index: 2)
        e.setThreadgroupMemoryLength((max(1, keep) * 4 + 15) & ~15, index: 0)
        e.dispatchThreadgroups(MTLSize(width: 1, height: 1, depth: 1),
                               threadsPerThreadgroup: MTLSize(width: nth, height: 1, depth: 1))
    }

    /// Ordina per id crescente i topK indici selezionati (bitonic, topK
    /// potenza di due — 512 sul Flash), una riga per token. I -1 di
    /// riempimento finiscono in testa e il kernel di attention li salta.
    func sortTopKAsc(indices: GPUTensor, sorted: GPUTensor, topK: Int,
                     nTokens: Int = 1) throws {
        precondition(topK > 0 && topK & (topK - 1) == 0,
                     "sort bitonic: topK deve essere potenza di due")
        let args = MetalRuntime.topkMaskArgs(
            ne00: topK, ne01: nTokens, nb00: 4, nb01: UInt64(topK) * 4,
            ne0: 0, ne1: 0, nb0: 0, nb1: 0)
        let pso = try rt.pipeline("kernel_dsv4_sort_i32_rows_asc")
        let e = encoder
        e.setComputePipelineState(pso)
        args.withUnsafeBytes { e.setBytes($0.baseAddress!, length: args.count, index: 0) }
        e.setBuffer(indices.buffer, offset: indices.byteOffset, index: 1)
        e.setBuffer(sorted.buffer, offset: sorted.byteOffset, index: 2)
        e.setThreadgroupMemoryLength((topK * 4 + 15) & ~15, index: 0)
        e.dispatchThreadgroups(MTLSize(width: nTokens, height: 1, depth: 1),
                               threadsPerThreadgroup: MTLSize(width: topK, height: 1, depth: 1))
    }

    /// Leva 9 v2.1 — score dell'indexer TILED per un run (port fedele di
    /// ds4_gpu_indexer_scores_batch_tensor, variante quality f32): UNA
    /// dispatch per l'intero run. La visibilità causale per token è
    /// calcolata IN-KERNEL ((pos0+t+1)/ratio, cap n_comp) e le posizioni
    /// oltre sono riempite a -inf: righe UNIFORMI di nComp float, pronte
    /// per l'argsort batch.
    func indexerScoresTiledBatch(q: GPUTensor, weights: GPUTensor,
                                 indexComp: GPUTensor, scores: GPUTensor,
                                 nComp: Int, nTokens: Int, pos0: Int,
                                 ratio: Int, nHead: Int, headDim: Int,
                                 scale: Float) throws {
        precondition(nComp > 0 && nTokens > 0 && headDim == 128)
        var args = [UInt8](repeating: 0, count: 72)
        args.withUnsafeMutableBytes { p in
            p.storeBytes(of: UInt32(nComp), toByteOffset: 0, as: UInt32.self)
            p.storeBytes(of: UInt32(nTokens), toByteOffset: 4, as: UInt32.self)
            p.storeBytes(of: UInt32(nHead), toByteOffset: 8, as: UInt32.self)
            p.storeBytes(of: UInt32(headDim), toByteOffset: 12, as: UInt32.self)
            p.storeBytes(of: UInt32(pos0), toByteOffset: 16, as: UInt32.self)
            p.storeBytes(of: UInt32(ratio), toByteOffset: 20, as: UInt32.self)
            p.storeBytes(of: UInt64(nHead * headDim * 4), toByteOffset: 24, as: UInt64.self)
            p.storeBytes(of: UInt64(headDim * 4), toByteOffset: 32, as: UInt64.self)
            p.storeBytes(of: UInt64(nHead * 4), toByteOffset: 40, as: UInt64.self)
            p.storeBytes(of: UInt64(headDim * 4), toByteOffset: 48, as: UInt64.self)
            p.storeBytes(of: UInt64(nComp * 4), toByteOffset: 56, as: UInt64.self)
            p.storeBytes(of: scale, toByteOffset: 64, as: Float.self)
        }
        let pso = try rt.pipeline("kernel_dsv4_indexer_scores_tiled_f32")
        let e = encoder
        e.setComputePipelineState(pso)
        args.withUnsafeBytes { e.setBytes($0.baseAddress!, length: args.count, index: 0) }
        e.setBuffer(q.buffer, offset: q.byteOffset, index: 1)
        e.setBuffer(weights.buffer, offset: weights.byteOffset, index: 2)
        e.setBuffer(indexComp.buffer, offset: indexComp.byteOffset, index: 3)
        e.setBuffer(scores.buffer, offset: scores.byteOffset, index: 4)
        // q [8][128] + k [32][128] + dot [8][32], tutti f32 (21504 byte).
        e.setThreadgroupMemoryLength((8 * 128 + 32 * 128 + 8 * 32) * 4, index: 0)
        e.dispatchThreadgroups(MTLSize(width: (nComp + 31) / 32,
                                       height: (nTokens + 7) / 8, depth: 1),
                               threadsPerThreadgroup: MTLSize(width: 32, height: 4, depth: 1))
    }

    /// Leva 9 v2.1 — top-K a INDICI per un run (port fedele di
    /// ds4_gpu_indexer_topk_tensor): argsort discendente per blocchi +
    /// catena di merge, una riga di `topK` id per token in `out`. Le righe
    /// di score sono uniformi (nComp, -inf oltre la visibilità). Il
    /// tie-break è quello dell'argsort del motore C in produzione.
    func indexerTopKIndicesBatch(scores: GPUTensor, nComp: Int, nTokens: Int,
                                 topK: Int, out: GPUTensor,
                                 scratch: GPUTensor) throws {
        precondition(nComp > 0 && nTokens > 0 && topK > 0 && topK <= nComp)
        let sortPso = try rt.pipeline("kernel_argsort_f32_i32_desc")
        var maxThreads = sortPso.maxTotalThreadsPerThreadgroup
        if maxThreads == 0 { maxThreads = 256 }
        var nth = 1
        while nth < nComp && 2 * nth <= maxThreads { nth *= 2 }
        let npr = (nComp + nth - 1) / nth
        let blockTopK = min(topK, nth)
        var workWidth = topK
        if npr > 1 {
            let lastBlock = nComp - (npr - 1) * nth
            workWidth = (npr - 1) * blockTopK + min(lastBlock, blockTopK)
        }
        let rowBytes = workWidth * 4
        let onePass = npr <= 1
        precondition(scratch.byteLength >= (onePass ? 0 : 2 * rowBytes * nTokens),
                     "leva 9 v2.1: scratch argsort insufficiente")

        func sortArgs(topKArg: Int) -> [UInt8] {
            var a = [UInt8](repeating: 0, count: 72)
            a.withUnsafeMutableBytes { p in
                p.storeBytes(of: Int32(nComp), toByteOffset: 0, as: Int32.self)
                p.storeBytes(of: Int32(nTokens), toByteOffset: 4, as: Int32.self)
                p.storeBytes(of: Int32(1), toByteOffset: 8, as: Int32.self)
                p.storeBytes(of: Int32(1), toByteOffset: 12, as: Int32.self)
                p.storeBytes(of: UInt64(4), toByteOffset: 16, as: UInt64.self)
                p.storeBytes(of: UInt64(nComp * 4), toByteOffset: 24, as: UInt64.self)
                p.storeBytes(of: UInt64(nComp * nTokens * 4), toByteOffset: 32, as: UInt64.self)
                p.storeBytes(of: UInt64(nComp * nTokens * 4), toByteOffset: 40, as: UInt64.self)
                p.storeBytes(of: Int32(workWidth), toByteOffset: 48, as: Int32.self)
                p.storeBytes(of: Int32(nTokens), toByteOffset: 52, as: Int32.self)
                p.storeBytes(of: Int32(1), toByteOffset: 56, as: Int32.self)
                p.storeBytes(of: Int32(1), toByteOffset: 60, as: Int32.self)
                p.storeBytes(of: Int32(topKArg), toByteOffset: 64, as: Int32.self)
            }
            return a
        }
        let e = encoder
        e.setComputePipelineState(sortPso)
        let a0 = sortArgs(topKArg: blockTopK)
        a0.withUnsafeBytes { e.setBytes($0.baseAddress!, length: $0.count, index: 0) }
        e.setBuffer(scores.buffer, offset: scores.byteOffset, index: 1)
        if onePass {
            e.setBuffer(out.buffer, offset: out.byteOffset, index: 2)
        } else {
            e.setBuffer(scratch.buffer, offset: scratch.byteOffset, index: 2)
        }
        e.setThreadgroupMemoryLength((nth * 8 + 15) & ~15, index: 0)
        e.dispatchThreadgroups(MTLSize(width: npr * nTokens, height: 1, depth: 1),
                               threadsPerThreadgroup: MTLSize(width: nth, height: 1, depth: 1))

        var len = blockTopK
        var curOff = 0
        var nextOff = rowBytes * nTokens
        let mergePso = try rt.pipeline("kernel_argsort_merge_f32_i32_desc")
        while len < workWidth {
            let nm = (workWidth + 2 * len - 1) / (2 * len)
            let finalMerge = nm == 1
            var mergeThreads = mergePso.maxTotalThreadsPerThreadgroup
            if mergeThreads == 0 || mergeThreads > 512 { mergeThreads = 512 }
            if mergeThreads > len { mergeThreads = len }
            if mergeThreads == 0 { mergeThreads = 1 }
            var margs = [UInt8](repeating: 0, count: 88)
            margs.withUnsafeMutableBytes { p in
                p.storeBytes(of: Int64(nComp), toByteOffset: 0, as: Int64.self)
                p.storeBytes(of: Int64(nTokens), toByteOffset: 8, as: Int64.self)
                p.storeBytes(of: Int64(1), toByteOffset: 16, as: Int64.self)
                p.storeBytes(of: Int64(1), toByteOffset: 24, as: Int64.self)
                p.storeBytes(of: UInt64(4), toByteOffset: 32, as: UInt64.self)
                p.storeBytes(of: UInt64(nComp * 4), toByteOffset: 40, as: UInt64.self)
                p.storeBytes(of: UInt64(nComp * nTokens * 4), toByteOffset: 48, as: UInt64.self)
                p.storeBytes(of: UInt64(nComp * nTokens * 4), toByteOffset: 56, as: UInt64.self)
                p.storeBytes(of: Int32(workWidth), toByteOffset: 64, as: Int32.self)
                p.storeBytes(of: Int32(nTokens), toByteOffset: 68, as: Int32.self)
                p.storeBytes(of: Int32(1), toByteOffset: 72, as: Int32.self)
                p.storeBytes(of: Int32(1), toByteOffset: 76, as: Int32.self)
                p.storeBytes(of: Int32(finalMerge ? topK : workWidth), toByteOffset: 80, as: Int32.self)
                p.storeBytes(of: Int32(len), toByteOffset: 84, as: Int32.self)
            }
            e.setComputePipelineState(mergePso)
            margs.withUnsafeBytes { e.setBytes($0.baseAddress!, length: $0.count, index: 0) }
            e.setBuffer(scores.buffer, offset: scores.byteOffset, index: 1)
            e.setBuffer(scratch.buffer, offset: scratch.byteOffset + curOff, index: 2)
            if finalMerge {
                e.setBuffer(out.buffer, offset: out.byteOffset, index: 3)
            } else {
                e.setBuffer(scratch.buffer, offset: scratch.byteOffset + nextOff, index: 3)
            }
            e.dispatchThreadgroups(MTLSize(width: nm * nTokens, height: 1, depth: 1),
                                   threadsPerThreadgroup: MTLSize(width: mergeThreads, height: 1, depth: 1))
            swap(&curOff, &nextOff)
            len <<= 1
        }
    }

    /// Leva 9 v2 — attention mista indicizzata MULTI-TOKEN (prefill, run di
    /// nq query): UN dispatch di kernel_dsv4_indexed_mixed_attention_heads8.
    /// Ogni token del run attende la SUA finestra raw causale (clamp per
    /// qpos dentro il kernel) + le SUE topK righe compresse da `topk`
    /// (riga per token, id crescenti, -1 di riempimento) + sink. Niente
    /// staging F16 dello span né maschera CPU.
    func indexedMixedAttentionBatch(q: GPUTensor, rawKv: GPUTensor, comp: GPUTensor,
                                    topk: GPUTensor, sinks: GPUTensor,
                                    heads: GPUTensor, nTokens: Int, nHead: Int,
                                    nRaw: Int, rawCap: Int, rawStart: Int,
                                    nComp: Int, topK: Int, pos0: Int,
                                    window: Int, ratio: Int) throws {
        let headDim = 512
        precondition(nTokens > 0 && nRaw > 0 && rawCap >= nRaw && rawStart < rawCap)
        precondition(nComp > 0 && topK > 0)
        var args = [UInt8](repeating: 0, count: 112)
        func u32(_ off: Int, _ value: Int) {
            var v = UInt32(value).littleEndian
            withUnsafeBytes(of: &v) { args.replaceSubrange(off..<(off + 4), with: $0) }
        }
        func u64(_ off: Int, _ value: Int) {
            var v = UInt64(value).littleEndian
            withUnsafeBytes(of: &v) { args.replaceSubrange(off..<(off + 8), with: $0) }
        }
        u32(0, nTokens)
        u32(4, nHead)
        u32(8, nRaw)
        u32(12, rawCap)
        u32(16, rawStart)
        u32(20, nComp)
        u32(24, topK)
        u32(28, pos0)
        u32(32, window)
        u32(36, ratio)
        u32(40, 0)                 // comp_kv_f16: cache compressa F32
        u32(44, 0)                 // pad0
        u64(48, nHead * headDim * 4)   // q_token_stride
        u64(56, headDim * 4)           // q_head_stride
        u64(64, headDim * 4)           // raw_row_stride
        u64(72, headDim * 4)           // comp_row_stride (F32)
        u64(80, topK * 4)              // topk_token_stride
        u64(88, nHead * headDim * 4)   // dst_token_stride
        u64(96, headDim * 4)           // dst_head_stride
        var scale = (1.0 / Float(headDim).squareRoot()).bitPattern.littleEndian
        withUnsafeBytes(of: &scale) { args.replaceSubrange(104..<108, with: $0) }

        let pso = try rt.pipeline("kernel_dsv4_indexed_mixed_attention_heads8")
        let e = encoder
        e.setComputePipelineState(pso)
        args.withUnsafeBytes { e.setBytes($0.baseAddress!, length: args.count, index: 0) }
        e.setBuffer(q.buffer, offset: q.byteOffset, index: 1)
        e.setBuffer(rawKv.buffer, offset: rawKv.byteOffset, index: 2)
        e.setBuffer(comp.buffer, offset: comp.byteOffset, index: 3)
        e.setBuffer(topk.buffer, offset: topk.byteOffset, index: 4)
        e.setBuffer(sinks.buffer, offset: sinks.byteOffset, index: 5)
        e.setBuffer(heads.buffer, offset: heads.byteOffset, index: 6)
        // Una riga K/V condivisa per threadgroup: 128 half4.
        e.setThreadgroupMemoryLength(128 * 8, index: 0)
        e.dispatchThreadgroups(MTLSize(width: nTokens, height: (nHead + 7) / 8, depth: 1),
                               threadsPerThreadgroup: MTLSize(width: 256, height: 1, depth: 1))
    }

    /// Attention mista indicizzata (decode, 1 token): finestra raw SWA dal
    /// ring + SOLO le topK righe compresse selezionate + sink, in un unico
    /// dispatch (kernel_dsv4_indexed_mixed_attention_heads8_rb16). Legge il
    /// raw F32 e la cache compressa F32 DIRETTAMENTE: niente staging F16
    /// dell'intero span, niente maschera.
    func indexedMixedAttention(q: GPUTensor, rawKv: GPUTensor, comp: GPUTensor,
                               topk: GPUTensor, sinks: GPUTensor,
                               heads: GPUTensor, nHead: Int,
                               nRaw: Int, rawCap: Int, rawStart: Int,
                               nComp: Int, topK: Int, pos: Int,
                               window: Int, ratio: Int) throws {
        let headDim = 512
        precondition(nRaw > 0 && rawCap >= nRaw && rawStart < rawCap)
        precondition(nComp > 0 && topK > 0)
        // ds4_metal_args_dsv4_indexed_attention: 12×u32, 7×u64, float scale
        // (allineamento 8 → 112 byte totali).
        var args = [UInt8](repeating: 0, count: 112)
        func u32(_ off: Int, _ value: Int) {
            var v = UInt32(value).littleEndian
            withUnsafeBytes(of: &v) { args.replaceSubrange(off..<(off + 4), with: $0) }
        }
        func u64(_ off: Int, _ value: Int) {
            var v = UInt64(value).littleEndian
            withUnsafeBytes(of: &v) { args.replaceSubrange(off..<(off + 8), with: $0) }
        }
        u32(0, 1)                 // n_tokens (decode)
        u32(4, nHead)
        u32(8, nRaw)
        u32(12, rawCap)
        u32(16, rawStart)
        u32(20, nComp)
        u32(24, topK)
        u32(28, pos)              // pos0
        u32(32, window)
        u32(36, ratio)
        u32(40, 0)                // comp_kv_f16: cache compressa F32
        u32(44, 0)                // pad0
        u64(48, nHead * headDim * 4)   // q_token_stride
        u64(56, headDim * 4)           // q_head_stride
        u64(64, headDim * 4)           // raw_row_stride
        u64(72, headDim * 4)           // comp_row_stride (F32)
        u64(80, topK * 4)              // topk_token_stride
        u64(88, nHead * headDim * 4)   // dst_token_stride
        u64(96, headDim * 4)           // dst_head_stride
        var scale = (1.0 / Float(headDim).squareRoot()).bitPattern.littleEndian
        withUnsafeBytes(of: &scale) { args.replaceSubrange(104..<108, with: $0) }

        let pso = try rt.pipeline("kernel_dsv4_indexed_mixed_attention_heads8_rb16")
        let e = encoder
        e.setComputePipelineState(pso)
        args.withUnsafeBytes { e.setBytes($0.baseAddress!, length: args.count, index: 0) }
        e.setBuffer(q.buffer, offset: q.byteOffset, index: 1)
        e.setBuffer(rawKv.buffer, offset: rawKv.byteOffset, index: 2)
        e.setBuffer(comp.buffer, offset: comp.byteOffset, index: 3)
        e.setBuffer(topk.buffer, offset: topk.byteOffset, index: 4)
        e.setBuffer(sinks.buffer, offset: sinks.byteOffset, index: 5)
        e.setBuffer(heads.buffer, offset: heads.byteOffset, index: 6)
        // 16 righe × 128 half4 di staging condiviso (multiplo di 16 byte).
        e.setThreadgroupMemoryLength(16 * 128 * MemoryLayout<UInt16>.stride * 4, index: 0)
        e.dispatchThreadgroups(MTLSize(width: 1, height: (nHead + 7) / 8, depth: 1),
                               threadsPerThreadgroup: MTLSize(width: 256, height: 1, depth: 1))
    }
}
