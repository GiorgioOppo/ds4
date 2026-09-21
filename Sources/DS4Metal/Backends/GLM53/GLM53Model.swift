import DS4Core
import Foundation
import Metal

/// Pure Swift orchestration over resident activation/recurrent state and
/// bounded mmap views of GGUF weights. No C engine or subprocess is involved.
public final class GLM53Model: SwiftModelDecoder {
    public let contextCapacity: Int
    public let vocabularySize = 154880
    public private(set) var position = 0
    private let model: GGUFModel
    private let configuration: GLM53Configuration
    private let weights: GLM53Weights
    private let runtime: MetalRuntime
    private let batchCapacity: Int
    private var valid = true
    private var scratch: [String: GPUTensor] = [:]
    private var states: [Int: [String: GPUTensor]] = [:]
    private let ones: GPUTensor
    private let meanWeights: GPUTensor

    public init(model: GGUFModel, contextSize: Int) throws {
        let config = try GLM53Configuration(model: model)
        guard contextSize > 0, contextSize <= config.maximumContextLength else {
            throw SwiftModelDecoderError.invalidInput("GLM 5.3 context must be 1...\(config.maximumContextLength)")
        }
        self.model = model
        self.configuration = config
        self.weights = try GLM53Weights(model: model)
        self.contextCapacity = contextSize
        self.batchCapacity = min(contextSize, 32)
        let rt = try MetalRuntime(additionalSources: GLM53KernelSources.all)
        self.runtime = rt
        self.ones = try .floats(rt, Array(repeating: 1, count: min(contextSize, 32)))
        self.meanWeights = try .floats(rt, Array(repeating: 0.25, count: 4))
        // Cache allocations are lazy and only the written causal prefix is
        // read. KDA recurrence is fixed-size and explicitly zero initialized.
        for layer in 0..<45 {
            if GLM53Configuration.isKDALayer(layer) {
                states[layer] = [
                    "conv": try .zeros(rt, floatCount: 9 * 8192),
                    "recurrent": try .zeros(rt, floatCount: 64 * 128 * 128)
                ]
            } else {
                states[layer] = [
                    "kv": try .uninitializedBytes(rt, byteLength: contextSize * 512 * 2, elementCount: contextSize * 512),
                    "pool": try .uninitializedBytes(rt, byteLength: ((contextSize + 3) / 4) * 128 * 2, elementCount: ((contextSize + 3) / 4) * 128),
                    "tailK": try .zeros(rt, floatCount: 4 * 128),
                    "tailGate": try .zeros(rt, floatCount: 4 * 128)
                ]
            }
        }
        let widths: [String: Int] = [
            "hc0":16384, "hc1":16384, "hc2":16384, "flat":16384,
            "mix":24, "split":24, "collapsed":4096, "norm":4096,
            "attnOut":4096, "ffnOut":4096, "gate":12288, "up":12288, "mid":12288,
            "q":16384, "k":8192, "v":8192, "decay":8192, "beta":64,
            "outputGate":8192, "lowrank":128, "kdaOut":8192,
            "qRank":1536, "qNorm":1536, "kvRaw":512, "kvNorm":512,
            "qLow":32768, "latent":32768, "heads":16384,
            "indexQ":4096, "indexK":128, "poolGate":128, "indexWeights":32,
            "logitsRouter":288, "routeIDs":8, "routeWeights":8,
            "expertX":4096, "expertGate":2048, "expertUp":2048, "expertMid":2048, "expertOut":4096,
            "expertContributions":32768
        ]
        for (name, width) in widths { scratch[name] = try .zeros(rt, floatCount: batchCapacity * width) }
        scratch["scores"] = try .zeros(rt, floatCount: max(1, (contextSize + 3) / 4))
        scratch["pools"] = try .zeros(rt, floatCount: 512)
        scratch["sortedPools"] = try .zeros(rt, floatCount: 512)
        scratch["selected"] = try .zeros(rt, floatCount: 2051)
        scratch["logits"] = try .zeros(rt, floatCount: vocabularySize)
    }

    public func reset() throws {
        for state in states.values {
            for (name, tensor) in state where name != "kv" && name != "pool" { tensor.zero() }
        }
        position = 0
        valid = true
    }

    public func evaluate(tokens: [Int], cancelled: @Sendable () -> Bool) throws -> [Float] {
        guard valid else { throw SwiftModelDecoderError.invalidState }
        guard !tokens.isEmpty, tokens.allSatisfy({ $0 >= 0 && $0 < vocabularySize }) else {
            throw SwiftModelDecoderError.invalidInput("GLM 5.3 requires nonempty in-vocabulary token IDs")
        }
        guard tokens.count <= contextCapacity - position else {
            throw SwiftModelDecoderError.contextOverflow(requested: position + tokens.count, capacity: contextCapacity)
        }
        do {
            var result: [Float] = []
            for start in stride(from: 0, to: tokens.count, by: batchCapacity) {
                if cancelled() { throw SwiftModelDecoderError.cancelled }
                let chunk = Array(tokens[start..<min(tokens.count, start + batchCapacity)])
                result = try evaluateBatch(chunk, final: start + chunk.count == tokens.count, cancelled: cancelled)
                position += chunk.count
            }
            return result
        } catch {
            // An interrupted layer may already have appended KDA/KV state.
            // Explicit reset/replay is mandatory before another evaluation.
            valid = false
            throw error
        }
    }

    private func s(_ key: String) -> GPUTensor { scratch[key]! }
    private func row(_ tensor: GPUTensor, _ index: Int, _ width: Int) -> GPUTensor {
        tensor.subview(byteOffset: index * width * 4, byteLength: width * 4, count: width)
    }
    private func complete(_ graph: GraphContext) throws {
        graph.commit()
        if let error = graph.lastError { throw SwiftModelDecoderError.gpu(error.localizedDescription) }
    }
    private func uints(_ values: [UInt32]) throws -> GPUTensor {
        try values.withUnsafeBytes { try .raw(runtime, ptr: $0.baseAddress!, byteLength: $0.count, elementCount: values.count) }
    }

    private func evaluateBatch(_ tokens: [Int], final: Bool, cancelled: @Sendable () -> Bool) throws -> [Float] {
        let rows = tokens.count
        var current = s("hc0"), next = s("hc2")
        let afterAttention = s("hc1")
        var graph = GraphContext(runtime)
        try graph.begin()
        let embed = try weights.embedding.map(runtime, model: model)
        let ids = try uints(tokens.map(UInt32.init))
        let rowBytes = Int(weights.embedding.tensor.bytes) / vocabularySize
        try graph.glm53Dispatch("kernel_glm53_embedding_hc",
            [weights.embedding.type, 4096, UInt32(vocabularySize), UInt32(rows), 1, UInt32(rowBytes)],
            [embed, ids, current], groups: .init(width: 32, height: rows, depth: 1))
        try complete(graph)
        for layer in 0..<45 {
            if cancelled() { throw SwiftModelDecoderError.cancelled }
            try autoreleasepool {
            let description = weights.layers[layer]
            var mapped: [String: GPUTensor] = [:]
            // No expert-bank mapping: mapped views here are only signal-path
            // matrices and small norms. Layer scope bounds their GPU lifetime.
            for (name, weight) in description where !name.contains("_exps.") {
                mapped[name] = try weight.map(runtime, model: model)
            }
            graph = GraphContext(runtime); try graph.begin()
            try hcPre(graph, part: "attn", input: current, weights: mapped, descriptions: description, rows: rows)
            if GLM53Configuration.isKDALayer(layer) {
                try kda(graph, weights: mapped, descriptions: description, state: states[layer]!, rows: rows)
            } else {
                try dsa(graph, weights: mapped, descriptions: description, state: states[layer]!, rows: rows)
            }
            try graph.hcExpand4(blockOut: s("attnOut"), residual: current,
                post: s("split"), comb: s("split"), blockAdd: nil, out: afterAttention,
                nEmbd: 4096, nTokens: rows, postByteOffset: 16, combByteOffset: 32, splitTokenStride: 96)
            try hcPre(graph, part: "ffn", input: afterAttention, weights: mapped, descriptions: description, rows: rows)
            try denseFFN(graph, prefix: "ffn_", shared: layer >= 3,
                         weights: mapped, descriptions: description, rows: rows)
            if layer >= 3 {
                try project(graph, "ffn_gate_inp.weight", s("norm"), s("logitsRouter"), mapped, description, rows)
                try graph.glm53Dispatch("kernel_glm53_route", [UInt32(rows)],
                    [s("logitsRouter"), mapped["exp_probs_b.bias"]!, s("routeIDs"), s("routeWeights")],
                    groups: .init(width: (rows + 31) / 32, height: 1, depth: 1), threads: 32)
                try complete(graph)
                try routedFFN(descriptions: description, rows: rows, cancelled: cancelled)
                graph = GraphContext(runtime); try graph.begin()
            }
            try graph.hcExpand4(blockOut: s("ffnOut"), residual: afterAttention,
                post: s("split"), comb: s("split"), blockAdd: nil, out: next,
                nEmbd: 4096, nTokens: rows, postByteOffset: 16, combByteOffset: 32, splitTokenStride: 96)
            try complete(graph)
            swap(&current, &next)
            }
        }
        guard final else { return [] }
        graph = GraphContext(runtime); try graph.begin()
        try graph.hcWeightedSum(x: row(current, rows - 1, 16384), weights: meanWeights,
                               out: s("collapsed"), nEmbd: 4096, nHC: 4, nTokens: 1)
        let norm = try weights.outputNorm.map(runtime, model: model)
        try graph.rmsNorm(s("collapsed"), weight: norm, out: s("norm"), rows: 1, n: 4096, eps: 1e-5)
        let head = try weights.output.map(runtime, model: model)
        try graph.glm53Projection(head, type: weights.output.type, input: s("norm"), output: s("logits"),
                                 columns: 4096, width: vocabularySize, rows: 1)
        try complete(graph)
        let logits = s("logits").floatArray()
        guard logits.allSatisfy(\.isFinite) else { throw SwiftModelDecoderError.gpu("GLM 5.3 produced non-finite logits") }
        return logits
    }

    private func project(_ graph: GraphContext, _ name: String, _ input: GPUTensor, _ output: GPUTensor,
                         _ mapped: [String:GPUTensor], _ descriptions: [String:GLM53Weights.Weight], _ rows: Int) throws {
        let w = descriptions[name]!
        try graph.glm53Projection(mapped[name]!, type: w.type, input: input, output: output,
                                 columns: w.columns, width: w.rows, rows: rows,
                                 heads: w.tensor.dims.count == 3 ? Int(w.tensor.dims[2]) : 1)
    }

    private func hcPre(_ graph: GraphContext, part: String, input: GPUTensor,
                       weights w: [String:GPUTensor], descriptions d: [String:GLM53Weights.Weight], rows: Int) throws {
        try graph.rmsNorm(input, weight: nil, out: s("flat"), rows: rows, n: 16384, eps: 1e-5)
        try project(graph, "hc_\(part)_fn.weight", s("flat"), s("mix"), w, d, rows)
        try graph.hcSplitWeightedSumNorm4(mix: s("mix"), scale: w["hc_\(part)_scale.weight"]!,
            base: w["hc_\(part)_base.weight"]!, x: input, split: s("split"), embd: s("collapsed"),
            normWeight: w["\(part)_norm.weight"]!, normOut: s("norm"), nEmbd: 4096, nRows: rows,
            sinkhornIters: 20, eps: 1e-6, normEps: 1e-5)
    }

    private func kda(_ graph: GraphContext, weights w: [String:GPUTensor],
                     descriptions d: [String:GLM53Weights.Weight], state: [String:GPUTensor], rows: Int) throws {
        for n in ["q", "k", "v"] { try project(graph, "kda_\(n).weight", s("norm"), s(n), w, d, rows) }
        try project(graph, "kda_f_a.weight", s("norm"), s("lowrank"), w, d, rows)
        try project(graph, "kda_f_b.weight", s("lowrank"), s("decay"), w, d, rows)
        try project(graph, "kda_beta.weight", s("norm"), s("beta"), w, d, rows)
        try project(graph, "kda_g_a.weight", s("norm"), s("lowrank"), w, d, rows)
        try project(graph, "kda_g_b.weight", s("lowrank"), s("outputGate"), w, d, rows)
        try graph.glm53KDA(q: s("q"), k: s("k"), v: s("v"), decay: s("decay"), beta: s("beta"), gate: s("outputGate"),
            qConv: w["kda_q_conv.weight"]!, kConv: w["kda_k_conv.weight"]!, vConv: w["kda_v_conv.weight"]!,
            aLog: w["kda_a_log.weight"]!, dtBias: w["kda_dt_bias.weight"]!, norm: w["kda_o_norm.weight"]!,
            convolutionState: state["conv"]!, recurrentState: state["recurrent"]!, out: s("kdaOut"), rows: rows)
        try project(graph, "kda_output.weight", s("kdaOut"), s("attnOut"), w, d, rows)
    }

    private func denseFFN(_ graph: GraphContext, prefix: String, shared: Bool,
                          weights w: [String:GPUTensor], descriptions d: [String:GLM53Weights.Weight], rows: Int) throws {
        let suffix = shared ? "_shexp.weight" : ".weight", width = shared ? 2048 : 12288
        try project(graph, prefix + "gate" + suffix, s("norm"), s("gate"), w, d, rows)
        try project(graph, prefix + "up" + suffix, s("norm"), s("up"), w, d, rows)
        try graph.glm53Rows("kernel_glm53_swiglu_route", width: width, rows: rows,
            buffers: [s("gate"), s("up"), ones, s("mid")])
        try project(graph, prefix + "down" + suffix, s("mid"), s("ffnOut"), w, d, rows)
    }

    private func dsa(_ graph: GraphContext, weights w: [String:GPUTensor],
                     descriptions d: [String:GLM53Weights.Weight], state: [String:GPUTensor], rows: Int) throws {
        try project(graph, "attn_q_a.weight", s("norm"), s("qRank"), w, d, rows)
        try graph.rmsNorm(s("qRank"), weight: w["attn_q_a_norm.weight"]!, out: s("qNorm"), rows: rows, n: 1536, eps: 1e-5)
        try project(graph, "attn_q_b.weight", s("qNorm"), s("q"), w, d, rows)
        try project(graph, "attn_kv_a_mqa.weight", s("norm"), s("kvRaw"), w, d, rows)
        try graph.rmsNorm(s("kvRaw"), weight: w["attn_kv_a_norm.weight"]!, out: s("kvNorm"), rows: rows, n: 512, eps: 1e-5)
        let kvDestination = state["kv"]!.subview(byteOffset: position * 512 * 2,
            byteLength: rows * 512 * 2, count: rows * 512)
        try graph.glm53Dispatch("kernel_glm53_store_half", [UInt32(rows * 512)], [s("kvNorm"), kvDestination],
            groups: .init(width: (rows * 512 + 127) / 128, height: 1, depth: 1))
        try project(graph, "indexer.attn_k.weight", s("norm"), s("indexK"), w, d, rows)
        try project(graph, "indexer.pool_gate.weight", s("norm"), s("poolGate"), w, d, rows)
        try graph.glm53PoolUpdate(rawKeys: s("indexK"), gates: s("poolGate"),
            norm: w["indexer.k_norm.weight"]!, bias: w["indexer.k_norm.bias"]!,
            ape: w["indexer.pool_ape.weight"]!, cache: state["pool"]!,
            tailKeys: state["tailK"]!, tailGates: state["tailGate"]!,
            position: position, rows: rows, capacity: contextCapacity)
        try project(graph, "indexer.attn_q_b.weight", s("qNorm"), s("indexQ"), w, d, rows)
        try project(graph, "indexer.proj.weight", s("norm"), s("indexWeights"), w, d, rows)
        try project(graph, "attn_k_b.weight", s("q"), s("qLow"), w, d, rows)
        for token in 0..<rows {
            let visible = position + token + 1
            let count = visible <= 2048 ? visible : 2048 + visible % 4
            if visible > 2048 {
                let pools = visible / 4
                try graph.glm53Dispatch("kernel_glm53_index_scores", [UInt32(pools), 512, 32, Float(1.0/64).bitPattern],
                    [row(s("indexQ"), token, 4096), row(s("indexWeights"), token, 32), state["pool"]!, s("scores")],
                    groups: .init(width: pools, height: 1, depth: 1), threads: 32)
                try graph.indexerTopKIndices(scores: s("scores"), out: s("pools"), nScores: pools, topK: 512)
                // Increasing raw positions give stable accumulation order.
                try graph.sortTopKAsc(indices: s("pools"), sorted: s("sortedPools"), topK: 512)
            }
            let args: [UInt32] = [UInt32(visible), UInt32(count), 64, Float(1.0/16).bitPattern]
            try graph.glm53Dispatch("kernel_glm53_select_raw", args, [s("sortedPools"), s("selected")],
                groups: .init(width: (count + 127) / 128, height: 1, depth: 1))
            try graph.glm53Dispatch("kernel_glm53_attention", args,
                [row(s("qLow"), token, 32768), state["kv"]!, s("selected"), row(s("latent"), token, 32768)],
                groups: .init(width: 64, height: 1, depth: 1), shared: (count + 4) * 4)
        }
        try project(graph, "attn_v_b.weight", s("latent"), s("heads"), w, d, rows)
        try project(graph, "attn_output.weight", s("heads"), s("attnOut"), w, d, rows)
    }

    private func routedFFN(descriptions: [String:GLM53Weights.Weight], rows: Int,
                           cancelled: @Sendable () -> Bool) throws {
        let ids = (s("routeIDs").buffer.contents() + s("routeIDs").byteOffset).assumingMemoryBound(to: UInt32.self)
        let routes = s("routeWeights").floatArray(rows * 8)
        var applications: [Int: [(token: UInt32, slot: UInt32, weight: Float)]] = [:]
        for token in 0..<rows {
            for slot in 0..<8 {
                let i = token * 8 + slot, expert = Int(ids[i]), weight = routes[i]
                guard expert < 288, weight.isFinite, weight >= 0 else {
                    throw SwiftModelDecoderError.gpu("GLM 5.3 router produced an invalid expert selection")
                }
                applications[expert, default: []].append((UInt32(token), UInt32(i), weight))
            }
        }
        let expertIDs = applications.keys.sorted()
        // Each wave retains at most 16 selected experts' small mapped ranges.
        // It never creates one Metal buffer for the 288-expert tensor bank.
        for begin in stride(from: 0, to: expertIDs.count, by: 16) {
            if cancelled() { throw SwiftModelDecoderError.cancelled }
            try autoreleasepool {
            let graph = GraphContext(runtime); try graph.begin()
            for expert in expertIDs[begin..<min(expertIDs.count, begin + 16)] {
                let items = applications[expert]!, count = items.count
                let tokenIDs = try uints(items.map(\.token))
                let slots = try uints(items.map(\.slot))
                let routeWeights = try GPUTensor.floats(runtime, items.map(\.weight))
                try graph.glm53Rows("kernel_glm53_gather_rows", width: 4096, rows: count,
                    buffers: [s("norm"), tokenIDs, s("expertX")])
                for (part, out) in [("gate", "expertGate"), ("up", "expertUp")] {
                    let weight = descriptions["ffn_\(part)_exps.weight"]!
                    let mapped = try weight.map(runtime, model: model, expert: expert)
                    try graph.glm53Projection(mapped, type: weight.type, input: s("expertX"), output: s(out),
                        columns: 4096, width: 2048, rows: count)
                }
                try graph.glm53Rows("kernel_glm53_swiglu_route", width: 2048, rows: count,
                    buffers: [s("expertGate"), s("expertUp"), routeWeights, s("expertMid")])
                let down = descriptions["ffn_down_exps.weight"]!
                let mappedDown = try down.map(runtime, model: model, expert: expert)
                try graph.glm53Projection(mappedDown, type: down.type, input: s("expertMid"), output: s("expertOut"),
                    columns: 2048, width: 4096, rows: count)
                try graph.glm53Rows("kernel_glm53_scatter_contribution", width: 4096, rows: count,
                    buffers: [s("expertOut"), slots, s("expertContributions")])
            }
            try complete(graph)
            }
        }
        let combine = GraphContext(runtime); try combine.begin()
        try combine.glm53Rows("kernel_glm53_combine_experts", width: 4096, rows: rows,
                             buffers: [s("expertContributions"), s("ffnOut")])
        try complete(combine)
    }
}
