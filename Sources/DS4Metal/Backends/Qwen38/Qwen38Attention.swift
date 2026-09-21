import Foundation
import DS4Core

extension Qwen38Model {
    static func denseRows(position: Int, count: Int, blockBudget: Int) -> Int {
        min(count, max(0, (blockBudget + 1) * 4 - 1 - position))
    }

    func attention(_ graph: GraphContext, layer: Int, count T: Int) throws {
        let p = "blk.\(layer).", state = states[layer]
        for (name, target) in [("attn_q", "qg"), ("attn_k", "kp"), ("attn_v", "vp"), ("indexer.q_proj", "iq"), ("indexer.k_proj", "ik")] {
            try project(graph, p + name + ".weight", s("mixed"), s(target), T)
        }
        var frequencies = [UInt32](repeating: 0, count: 32)
        for i in 0..<(c.rotaryDimension / 2) { frequencies[i] = Float(pow(Double(c.ropeBase), -Double(2 * i) / Double(c.rotaryDimension))).bitPattern }
        let args: [UInt32] = [UInt32(T), UInt32(c.heads), UInt32(c.kvHeads), UInt32(c.headDimension), UInt32(c.rotaryDimension), UInt32(c.indexHeads), UInt32(c.indexDimension), UInt32(position), UInt32(contextCapacity), c.ropeBase.bitPattern, c.epsilon.bitPattern, 0, Float(1).bitPattern] + frequencies
        try dispatch(graph, "kernel_qwen4_attn_prep", args,
                     [s("qg"), s("kp"), s("vp"), s("iq"), s("ik"), try weight(p + "attn_q_norm.weight"), try weight(p + "attn_k_norm.weight"), try weight(p + "indexer.q_norm.weight"), s("q"), s("gate"), state["k"]!, state["v"]!, s("iqn"), state["ik"]!, s("positions")],
                     grid: (c.heads + c.kvHeads + c.indexHeads + 1, T, 1), threads: 32)
        let firstBlock = position / 4, blocksAfter = (position + T) / 4
        if blocksAfter > firstBlock {
            let blockArgs: [UInt32] = [UInt32(firstBlock), UInt32(blocksAfter - firstBlock), 4, UInt32(c.indexDimension), UInt32(c.rotaryDimension), c.ropeBase.bitPattern, c.epsilon.bitPattern, 0, Float(1).bitPattern] + frequencies
            try dispatch(graph, "kernel_qwen4_idx_block_key", blockArgs,
                         [state["ik"]!, try weight(p + "indexer.k_norm.weight"), s("positions"), state["blocks"]!], grid: (blocksAfter - firstBlock, 1, 1), threads: 32)
        }
        let dense = Self.denseRows(position: position, count: T, blockBudget: c.indexTopK / 4)
        if dense > 0 { try attentionCore(graph, state: state, offset: 0, rows: dense, parentRows: T, selected: false) }
        if dense < T {
            let sparse = T - dense, p0 = position + dense
            let iq = rows(s("iqn"), start: dense, count: sparse, width: c.indexHeads * c.indexDimension)
            let scoreArgs: [UInt32] = [UInt32(sparse), UInt32(blocksAfter), UInt32(c.indexHeads), UInt32(c.indexDimension), UInt32(p0), 4, 0, 0]
            let useMM = sparse > 8 && c.indexDimension == 128
            try dispatch(graph, useMM ? "kernel_qwen4_idx_score_mm" : "kernel_qwen4_idx_score", scoreArgs,
                         [iq, state["blocks"]!, s("scores")], grid: (useMM ? (blocksAfter + 63) / 64 : (blocksAfter + 127) / 128, useMM ? (sparse + 15) / 16 : sparse, 1))
            try dispatch(graph, "kernel_qwen4_idx_select", [UInt32(sparse), UInt32(blocksAfter), UInt32(c.indexTopK / 4), 0],
                         [s("scores"), s("selectedBlocks")], grid: (sparse, 1, 1), threads: 1024)
            try dispatch(graph, "kernel_qwen4_idx_expand", [UInt32(sparse), UInt32(c.indexTopK / 4), 4, UInt32(p0), UInt32(c.indexTopK + 4), 0, 0, 0],
                         [s("selectedBlocks"), s("selectedTokens"), s("selectedCount")], grid: (sparse, 1, 1), threads: 256)
            try attentionCore(graph, state: state, offset: dense, rows: sparse, parentRows: T, selected: true)
        }
        try project(graph, p + "attn_output.weight", s("attentionOut"), s("blk"), T)
    }

    func rows(_ tensor: GPUTensor, start: Int, count: Int, width: Int) -> GPUTensor {
        tensor.subview(byteOffset: start * width * 4, byteLength: count * width * 4, count: count * width)
    }

    private func attentionCore(_ graph: GraphContext, state: [String: GPUTensor], offset: Int, rows count: Int, parentRows: Int, selected: Bool) throws {
        let p0 = position + offset
        let keys = selected ? c.indexTopK + 4 : p0 + count
        let splits = parentRows <= 2 ? min(64, max(1, (keys + 31) / 32)) : 1
        let args: [UInt32] = [UInt32(count), UInt32(c.heads), UInt32(c.kvHeads), UInt32(c.headDimension), UInt32(p0), selected ? 1 : 0, UInt32(c.indexTopK + 4), (1 / sqrt(Float(c.headDimension))).bitPattern, UInt32(splits), UInt32((keys + splits - 1) / splits), 0, 0]
        let q = rows(s("q"), start: offset, count: count, width: c.queryDimension)
        let gate = rows(s("gate"), start: offset, count: count, width: c.queryDimension)
        let out = rows(s("attentionOut"), start: offset, count: count, width: c.queryDimension)
        let buffers = [q, gate, state["k"]!, state["v"]!, s("selectedTokens"), s("selectedCount"), out]
        if parentRows > 8 && c.headDimension == 256 {
            try dispatch(graph, "kernel_qwen4_attn_mm", args, buffers, grid: (c.kvHeads, count, 1))
        } else {
            let suffix = String(c.headDimension / 32)
            try dispatch(graph, "kernel_qwen4_attn_decode_npt" + suffix, args, buffers + [s("attentionPart")], grid: (splits, c.kvHeads, count))
            if splits > 1 {
                try dispatch(graph, "kernel_qwen4_attn_merge_npt" + suffix, args, [s("attentionPart"), gate, out], grid: (c.heads, count, 1), threads: 32)
            }
        }
    }
}
