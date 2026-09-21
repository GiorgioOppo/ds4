import Foundation
import DS4Core

extension Qwen38Model {
    /// Stable first-use compaction keeps the per-token top-k slot order and
    /// therefore the reduction arithmetic unchanged. No host pointer enters a
    /// shader. Lists contain IDs in the compact, contiguous weight slabs.
    static func compactExperts(_ selected: [Int32], slots: Int, experts: Int) throws -> (ids: [Int], remapped: [Int32]) {
        guard slots > 0, selected.count % slots == 0 else { throw Qwen38Error.invalidModel("Invalid routing shape") }
        var lookup = [Int: Int32](), ids = [Int](), remapped = [Int32]()
        for start in stride(from: 0, to: selected.count, by: slots) {
            var seen = Set<Int32>()
            for raw in selected[start..<(start + slots)] {
                guard raw >= 0, Int(raw) < experts, seen.insert(raw).inserted else { throw Qwen38Error.invalidModel("Invalid or duplicate routed expert ID") }
                let id = Int(raw)
                if lookup[id] == nil { lookup[id] = Int32(ids.count); ids.append(id) }
                remapped.append(lookup[id]!)
            }
        }
        return (ids, remapped)
    }

    func expertSlab(_ name: String, ids: [Int], cancelled: @Sendable () -> Bool) throws -> GPUTensor {
        let t = tensor(name), expertBytes = Int(t.bytes) / c.experts
        let length = ids.count * expertBytes
        guard !ids.isEmpty, length <= 1 << 30, length <= rt.device.maxBufferLength else {
            throw Qwen38Error.invalidModel("Active expert slab exceeds the bounded staging limit")
        }
        let result = try GPUTensor.uninitializedBytes(rt, byteLength: length, elementCount: length)
        for (slot, id) in ids.enumerated() {
            if cancelled() { throw SwiftModelDecoderError.cancelled }
            try read(result.buffer.contents() + slot * expertBytes, bytes: expertBytes, offset: t.absOffset + UInt64(id * expertBytes))
        }
        return result
    }

    func moeMMArgs(_ weight: GGUFModel.Tensor, count: Int, active: Int, input: Int, output: Int) throws -> [UInt32] {
        let bytes = weight.bytes / UInt64(c.experts)
        let row = try Qwen38Weights.rowBytes(type: weight.type, width: Int(weight.dims[0]))
        var args: [UInt32] = [UInt32(count), UInt32(c.expertsUsed), UInt32(c.expertsUsed), UInt32(input), UInt32(output), weight.type, UInt32(row), UInt32(count), UInt32(truncatingIfNeeded: bytes), UInt32(bytes >> 32), UInt32(active), UInt32((count + 31) / 32), 0, 0, 0]
        args += [UInt32](repeating: 0, count: 528 - args.count)
        return args
    }

    func moeMVArgs(_ weight: GGUFModel.Tensor, count: Int, active: Int, input: Int, output: Int) throws -> [UInt32] {
        let bytes = weight.bytes / UInt64(c.experts)
        let row = try Qwen38Weights.rowBytes(type: weight.type, width: Int(weight.dims[0]))
        return [UInt32(count), UInt32(c.expertsUsed), UInt32(input), UInt32(output), weight.type, UInt32(row), UInt32(truncatingIfNeeded: bytes), UInt32(bytes >> 32), 0, 0, 0, UInt32(active), 0, 0, 0, 0, 0, 0]
    }

    func moe(_ graph: GraphContext, prefix p: String, count T: Int, cancelled: @Sendable () -> Bool) throws {
        try project(graph, p + "ffn_gate_inp.weight", s("mixed"), s("router"), T)
        try dispatch(graph, "kernel_qwen4_router_topk", [UInt32(T), UInt32(c.experts), UInt32(c.expertsUsed), 0, UInt32(c.embedding), 0, 0, 0],
                     [s("router"), s("selected"), s("routeWeights"), s("mixed"), try weight(p + "ffn_gate_inp_shexp.weight"), s("sharedGate")], grid: (T, 1, 1), threads: 256)
        // Run the always-active expert while the command buffer computes routing.
        try project(graph, p + "ffn_gate_shexp.weight", s("mixed"), s("sharedA"), T)
        try project(graph, p + "ffn_up_shexp.weight", s("mixed"), s("sharedB"), T)
        try dispatch(graph, "kernel_qwen38_swiglu", [UInt32(T * c.expertWidth)], [s("sharedA"), s("sharedB"), s("sharedMid")], grid: ((T * c.expertWidth + 255) / 256, 1, 1), threads: 256)
        try project(graph, p + "ffn_down_shexp.weight", s("sharedMid"), s("sharedOut"), T)
        try finish(graph)
        if cancelled() { throw SwiftModelDecoderError.cancelled }
        let selected = s("selected").buffer.contents().assumingMemoryBound(to: Int32.self)
        let compact = try Self.compactExperts(Array(UnsafeBufferPointer(start: selected, count: T * c.expertsUsed)), slots: c.expertsUsed, experts: c.experts)
        let gateName = p + "ffn_gate_exps.weight", upName = p + "ffn_up_exps.weight", downName = p + "ffn_down_exps.weight"
        let gateWeight = tensor(gateName), downWeight = tensor(downName)
        let gate = try expertSlab(gateName, ids: compact.ids, cancelled: cancelled)
        let up = try expertSlab(upName, ids: compact.ids, cancelled: cancelled)
        let down = try expertSlab(downName, ids: compact.ids, cancelled: cancelled)
        _ = compact.remapped.withUnsafeBytes { memcpy(selected, $0.baseAddress!, $0.count) }
        try graph.begin()
        let mmTypes: Set<UInt32> = [8, 10, 12, 16, 39]
        let useMM = T > 8 && mmTypes.contains(gateWeight.type) && mmTypes.contains(downWeight.type)
            && c.embedding % 64 == 0 && c.expertWidth % 64 == 0
        if useMM {
            let args = try moeMMArgs(gateWeight, count: T, active: compact.ids.count, input: c.embedding, output: c.expertWidth)
            try dispatch(graph, "kernel_qwen4_moe_build_lists", args, [s("selected"), s("lists"), s("listCounts")], grid: (1, 1, 1), threads: 512)
            try dispatch(graph, "kernel_qwen4_moe_mm_mid", args, [gate, up, s("lists"), s("listCounts"), s("mixed"), s("mid")],
                         grid: ((c.expertWidth + 31) / 32, compact.ids.count, (T + 31) / 32), quant: gateWeight.type)
            let downArgs = try moeMMArgs(downWeight, count: T, active: compact.ids.count, input: c.expertWidth, output: c.embedding)
            try dispatch(graph, "kernel_qwen4_moe_mm_down", downArgs, [down, s("lists"), s("listCounts"), s("mid"), s("part")],
                         grid: ((c.embedding + 31) / 32, compact.ids.count, (T + 31) / 32), quant: downWeight.type)
        } else {
            let args = try moeMVArgs(gateWeight, count: T, active: compact.ids.count, input: c.embedding, output: c.expertWidth)
            try dispatch(graph, "kernel_qwen4_moe_mid", args, [gate, up, s("selected"), s("mixed"), s("mid"), gate, up], grid: ((c.expertWidth + 7) / 8, c.expertsUsed, T))
            let downArgs = try moeMVArgs(downWeight, count: T, active: compact.ids.count, input: c.expertWidth, output: c.embedding)
            try dispatch(graph, "kernel_qwen4_moe_down", downArgs, [down, s("selected"), s("mid"), s("part"), down], grid: ((c.embedding + 7) / 8, c.expertsUsed, T))
        }
        try dispatch(graph, "kernel_qwen4_moe_reduce", [UInt32(T), UInt32(c.expertsUsed), UInt32(c.embedding), 2, UInt32(c.hcCount), UInt32(c.expertsUsed), 0, 0],
                     [s("part"), s("routeWeights"), s("sharedGate"), s("blk"), s("R"), s("inj"), s("sharedOut")], grid: ((c.embedding + 255) / 256, T, 1), threads: 256)
        // Explicit completion owns each slab through its final GPU read and
        // releases the layer's staging before allocating the next layer's set.
        try withExtendedLifetime([gate, up, down]) { try finish(graph) }
        try graph.begin()
    }
}
