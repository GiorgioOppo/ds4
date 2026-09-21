import Foundation
import Metal
import DS4Core

/// V4.1 has its own delayed HC, compressed KV, index-source and Engram graph.
/// Only dense weights are mapped to Metal. The enormous Engram tables and
/// routed expert tensors remain on SSD and are read in bounded selections.
public final class DeepSeek41Model: SwiftModelDecoder {
    public let contextCapacity: Int
    public let vocabularySize = 129280
    public private(set) var position = 0
    public let configuration: DeepSeek41Configuration
    private let model: GGUFModel
    private let runtime: MetalRuntime
    private let weights: DeepSeek41Weights
    private let engram: DeepSeek41Engram
    private let pager: DeepSeek41ExpertPager
    private let buffers: [String:GPUTensor]
    private let windows: [GPUTensor], compressed: [GPUTensor], indexCache: [GPUTensor]
    private let previousKV: [GPUTensor], previousScores: [GPUTensor]
    private let batchCapacity: Int, maskWidth: Int
    private var invalid = false

    public init(model: GGUFModel, contextSize: Int) throws {
        let configuration = try DeepSeek41Configuration(model:model)
        guard contextSize > 0 && contextSize <= configuration.maximumContextLength else {
            throw SwiftModelDecoderError.contextOverflow(requested:contextSize,capacity:configuration.maximumContextLength)
        }
        let engram = try DeepSeek41Engram(model:model)
        let runtime = try MetalRuntime(additionalSources:[DeepSeek41MetalSource.source,DeepSeek41MetalSource.graph])
        guard runtime.device.hasUnifiedMemory else { throw SwiftModelDecoderError.gpu("V4.1 richiede memoria unificata") }
        let weights = try DeepSeek41Weights(model:model,runtime:runtime)
        let cap = min(128,contextSize), maskWidth = (contextSize+7)/8
        let rowSizes: [String:Int] = ["residual":20480,"after":20480,"flat":20480,"mix":24,"attnSplit":24,"ffnSplit":24,"pre":4,
            "x":5120,"norm":5120,"block":5120,"qr":1280,"q":32768,"kv":512,"poolKV":512,"poolScore":512,
            "indexQ":4096,"indexWeights":32,"selectedComp":512,"blockMask":maskWidth,"heads":32768,"low":8192,
            "routeLogits":384,"routeProbs":384,"selected":6,"routeWeights":6,"experts":30720,
            "sharedGate":2304,"sharedUp":2304,"sharedMid":2304,"shared":5120,"routed":5120,
            "engramRows":6144,"engramKV":25600,"packX":5120,"packGate":2304,"packUp":2304,"packMid":2304,"packOut":5120,"packWeights":1,"zeroIDs":1]
        let cacheFloats = (0..<4).reduce(0) { $0 + (contextSize / ($1 < 3 ? 2 : 1) + 1) * 640 }
        let graphFloats = cap * rowSizes.values.reduce(0,+) + 40*128*512 + cacheFloats + contextSize + maskWidth + 129280 + 512*512 + 4*1024
        let scratchBytes = GraphContext.flashScratchBytes(nHead:64,nKeys:640)
        let bytes = UInt64(graphFloats)*4 + UInt64(scratchBytes.kvF16+scratchBytes.mask+scratchBytes.pad+scratchBytes.tmp) + UInt64(6*(2*weights.maximumGateBytes+weights.maximumDownBytes))
        let used = UInt64(runtime.device.currentAllocatedSize), budget = runtime.device.recommendedMaxWorkingSetSize
        guard used < budget, bytes < budget-used else { throw SwiftModelDecoderError.gpu("V4.1: contesto e staging richiedono \(bytes/1_048_576) MiB oltre i pesi dense") }
        var buffers: [String:GPUTensor] = [:]
        for (name,width) in rowSizes { buffers[name] = try .zeros(runtime,floatCount:cap*width) }
        for (name,size) in ["latent":512,"indexK":128,"scores":contextSize,"blockScores":maskWidth,"heap":2048,"blockIDs":2048,"selectedKV":512*512,"logits":129280] {
            buffers[name] = try .zeros(runtime,floatCount:size)
        }
        for (name,size) in ["kvF16":scratchBytes.kvF16,"mask":scratchBytes.mask,"pad":scratchBytes.pad,"tmp":scratchBytes.tmp] {
            buffers[name] = try .zerosBytes(runtime,byteLength:size)
        }
        windows = try (0..<40).map { _ in try .zeros(runtime,floatCount:128*512) }
        compressed = try (0..<4).map { try .lazyZeros(runtime,floatCount:(contextSize / ($0 < 3 ? 2 : 1) + 1)*512) }
        indexCache = try (0..<4).map { try .lazyZeros(runtime,floatCount:(contextSize / ($0 < 3 ? 2 : 1) + 1)*128) }
        previousKV = try (0..<4).map { _ in try .zeros(runtime,floatCount:512) }
        previousScores = try (0..<4).map { _ in try .zeros(runtime,floatCount:512) }
        self.pager = try DeepSeek41ExpertPager(model:model,weights:weights,runtime:runtime)
        self.model = model; self.runtime = runtime; self.weights = weights; self.engram = engram
        self.configuration = configuration; self.contextCapacity = contextSize; self.buffers = buffers
        self.batchCapacity = cap; self.maskWidth = maskWidth
    }

    public func reset() throws {
        // All rings/frontiers and n-gram history belong to this conversation.
        // Future compressed rows are overwritten before they become visible.
        for tensor in windows + previousKV + previousScores {
            memset(tensor.buffer.contents().advanced(by:tensor.byteOffset),0,tensor.byteLength)
        }
        engram.reset(); position = 0; invalid = false
    }
    public func evaluate(tokens:[Int],cancelled:@Sendable () -> Bool) throws -> [Float] {
        try evaluateRows(tokens:tokens,embeddings:nil,cancelled:cancelled)
    }
    /// Vision rows use the same absolute one-dimensional RoPE as the text
    /// trunk. Their n-grams are masked and the VL routing bias is required.
    public func evaluateRows(tokens:[Int],embeddings:[[Float]?]?,cancelled:@Sendable () -> Bool) throws -> [Float] {
        guard !invalid else { throw SwiftModelDecoderError.invalidState }
        guard !tokens.isEmpty, tokens.allSatisfy({ $0 >= 0 && $0 < vocabularySize }),
              embeddings == nil || embeddings!.count == tokens.count else { throw SwiftModelDecoderError.invalidInput("V4.1: token o embedding non validi") }
        if let embeddings {
            for row in embeddings.compactMap({$0}) where row.count != 5120 || !row.allSatisfy(\.isFinite) {
                throw SwiftModelDecoderError.invalidInput("V4.1: embedding vision deve avere 5120 valori finiti")
            }
            if embeddings.contains(where:{$0 != nil}) {
                guard (0..<40).allSatisfy({weights.resident["blk.\($0).exp_probs_b_vl.bias"] != nil}) else { throw DeepSeek41Error.invalidTensor("exp_probs_b_vl.bias for image rows") }
            }
        }
        guard tokens.count <= contextCapacity-position else { throw SwiftModelDecoderError.contextOverflow(requested:position+tokens.count,capacity:contextCapacity) }
        do {
            for start in stride(from:0,to:tokens.count,by:batchCapacity) {
                if cancelled() { throw SwiftModelDecoderError.cancelled }
                let end = min(tokens.count,start+batchCapacity)
                try chunk(tokens:Array(tokens[start..<end]),embeddings:embeddings.map{Array($0[start..<end])},publishLogits:end == tokens.count,cancelled:cancelled)
            }
            let result = b("logits").floatArray(vocabularySize)
            guard result.allSatisfy(\.isFinite) else { throw SwiftModelDecoderError.gpu("V4.1 logits non finiti") }
            return result
        } catch { invalid = true; throw error }
    }
    private func b(_ name:String) -> GPUTensor { buffers[name]! }
    private func w(_ layer:Int,_ name:String) -> DS41Weight { weights.resident["blk.\(layer)."+name]! }
    private func row(_ name:String,_ row:Int,_ width:Int) -> GPUTensor { b(name).rowView(row:row,cols:width) }
    private func check(_ cancelled:@Sendable () -> Bool) throws { if cancelled() { throw SwiftModelDecoderError.cancelled } }
    private func hcMix(_ ops:DeepSeek41Ops,_ layer:Int,_ count:Int,ffn:Bool) throws {
        let kind = ffn ? "ffn" : "attn", source = b(ffn ? "after" : "residual")
        try ops.graph.rmsNorm(source,weight:nil,out:b("flat"),rows:count,n:20480,eps:1e-20)
        try ops.project(w(layer,"hc_\(kind)_fn.weight"),b("flat"),b("mix"),count:count,round:false)
        try ops.graph.hcSplitSinkhorn(mix:b("mix"),scale:w(layer,"hc_\(kind)_scale.weight").gpu,base:w(layer,"hc_\(kind)_base.weight").gpu,out:b(ffn ? "ffnSplit":"attnSplit"),nRows:count,sinkhornIters:20,eps:1e-6)
    }
    private func hcExpand(_ ops:DeepSeek41Ops,_ count:Int,ffn:Bool) throws {
        let split = b(ffn ? "ffnSplit":"attnSplit"), out = b(ffn ? "residual":"after")
        try ops.graph.hcExpand4(blockOut:b("block"),residual:b(ffn ? "after":"residual"),post:split,comb:split,blockAdd:nil,out:out,nEmbd:5120,nTokens:count,postByteOffset:16,combByteOffset:32,splitTokenStride:96)
        try ops.quantize(out,width:20480,rows:count)
    }
    private func chunk(tokens:[Int],embeddings:[[Float]?]?,publishLogits:Bool,cancelled:@Sendable () -> Bool) throws {
        let count = tokens.count
        let mask = embeddings.map { $0.map { $0 == nil } }
        let rows = try engram.read(tokens:tokens,mask:mask)
        try check(cancelled)
        let graph = GraphContext(runtime), ops = DeepSeek41Ops(runtime:runtime,graph:graph)
        try graph.begin()
        for t in 0..<count {
            if let embedding = embeddings?[t] {
                embedding.withUnsafeBytes { row("x",t,5120).buffer.contents().advanced(by:row("x",t,5120).byteOffset).copyMemory(from:$0.baseAddress!,byteCount:$0.count) }
            } else {
                try graph.getRowsF16(table:weights.resident["token_embd.weight"]!.gpu,id:tokens[t],out:row("x",t,5120),nEmbd:5120,nVocab:vocabularySize)
            }
        }
        try graph.repeatHC(src:b("x"),out:b("residual"),nEmbd:5120,nTokens:count,nHC:4)
        let pre = b("pre").buffer.contents().assumingMemoryBound(to:Float.self)
        for t in 0..<count { pre[t*4]=1;pre[t*4+1]=0;pre[t*4+2]=0;pre[t*4+3]=0 }
        for layer in 0..<40 {
            try check(cancelled)
            if layer == 1 || layer == 14 {
                let values = rows[layer == 1 ? 0 : 1]
                values.withUnsafeBytes { b("engramRows").buffer.contents().copyMemory(from:$0.baseAddress!,byteCount:$0.count) }
                try ops.project(w(layer,"engram_kv.weight"),b("engramRows"),b("engramKV"),count:count)
                let maskBuffer = try GPUTensor.bytes(runtime,(mask ?? [Bool](repeating:true,count:count)).map{$0 ? 1 : 0},elementCount:count)
                try ops.dispatch("kernel_dsv41_engram_add",[5120,UInt32(count),Float(1e-20).bitPattern,mask == nil ? 0:1],
                    [b("residual"),b("engramKV"),w(layer,"engram_q_norm.weight").gpu,w(layer,"engram_k_norm.weight").gpu,maskBuffer],x:count,y:4,threads:32)
            }
            try hcMix(ops,layer,count,ffn:false)
            // Delayed pre: the newly computed attention split is not consumed here.
            try graph.hcWeightedSum(x:b("residual"),weights:b("pre"),out:b("x"),nEmbd:5120,nHC:4,nTokens:count)
            try ops.quantize(b("x"),width:5120,rows:count)
            try ops.norm(b("x"),w(layer,"attn_norm.weight").gpu,b("norm"),width:5120,rows:count)
            try attention(ops,layer,count:count)
            try hcExpand(ops,count,ffn:false)
            try hcMix(ops,layer,count,ffn:true)
            try graph.hcWeightedSum(x:b("after"),weights:b("attnSplit"),out:b("x"),nEmbd:5120,nHC:4,nTokens:count,weightsTokenStride:96)
            try ops.quantize(b("x"),width:5120,rows:count)
            try ops.norm(b("x"),w(layer,"ffn_norm.weight").gpu,b("norm"),width:5120,rows:count)
            try sharedAndRoute(ops,layer,count:count,mask:mask)
            try ops.finish() // route IDs become visible; dense work and shared FFN complete
            try check(cancelled)
            try routed(ops,layer,count:count,cancelled:cancelled)
            try graph.begin()
            try graph.moeSum6(experts:b("experts"),out:b("routed"),width:5120,tokens:count)
            try graph.add(b("routed"),b("shared"),out:b("block"),width:5120,rows:count)
            try ops.quantize(b("block"),width:5120,rows:count)
            try hcExpand(ops,count,ffn:true)
            for t in 0..<count { try ops.copy(row("ffnSplit",t,24),row("pre",t,4),floats:4) }
            try ops.finish()
            try check(cancelled)
            try graph.begin()
        }
        if publishLogits {
            try graph.hcWeightedSum(x:row("residual",count-1,20480),weights:row("pre",count-1,4),out:b("x"),nEmbd:5120,nHC:4,nTokens:1)
            try ops.quantize(b("x"),width:5120,rows:1)
            try ops.norm(b("x"),weights.resident["output_norm.weight"]!.gpu,b("norm"),width:5120,rows:1)
            try ops.project(weights.resident["output.weight"]!,b("norm"),b("logits"),count:1,round:false)
        }
        try ops.finish(); try check(cancelled); position += count
    }

    private func attention(_ ops:DeepSeek41Ops,_ layer:Int,count:Int) throws {
        let graph = ops.graph, ratio = DeepSeek41Configuration.compressionRatio(layer:layer)
        let owner = layer < 8 ? 0 : layer < 14 ? 1 : layer < 20 ? 2 : 3
        try ops.project(w(layer,"attn_q_a.weight"),b("norm"),b("qr"),count:count)
        try ops.norm(b("qr"),w(layer,"attn_q_a_norm.weight").gpu,b("qr"),width:1280,rows:count)
        try ops.project(w(layer,"attn_q_b.weight"),b("qr"),b("q"),count:count)
        try ops.project(w(layer,"attn_kv.weight"),b("norm"),b("kv"),count:count)
        try ops.norm(b("kv"),w(layer,"attn_kv_a_norm.weight").gpu,b("kv"),width:512,rows:count)
        try ops.rope(b("q"),width:512,heads:64,rows:count,start:position,compressed:ratio != 0)
        try ops.rope(b("kv"),width:512,heads:1,rows:count,start:position,compressed:ratio != 0)
        try ops.quantize(b("kv"),width:512,rows:count,mode:1)
        let source = DeepSeek41Configuration.kvSourceLayers.contains(layer)
        let indexSource = DeepSeek41Configuration.indexSourceLayers.contains(layer)
        if source {
            try ops.project(w(layer,"attn_compressor_kv.weight"),b("norm"),b("poolKV"),count:count,round:ratio == 1)
            if ratio == 2 { try ops.project(w(layer,"attn_compressor_gate.weight"),b("norm"),b("poolScore"),count:count,round:false) }
        }
        if indexSource {
            try ops.project(w(layer,"indexer.attn_q_b.weight"),b("qr"),b("indexQ"),count:count)
            try ops.rope(b("indexQ"),width:128,heads:32,rows:count,start:position,compressed:true)
            try ops.quantize(b("indexQ"),width:128,rows:count*32,mode:2)
            try ops.project(w(layer,"indexer.proj.weight"),b("norm"),b("indexWeights"),count:count)
        }
        for t in 0..<count {
            let pos = position+t, nComp = ratio == 0 ? 0 : (pos+1)/ratio
            try ops.copy(row("kv",t,512),windows[layer],floats:512,outputOffset:(pos%128)*512)
            if source {
                if ratio == 2 && pos%2 == 0 {
                    try ops.copy(row("poolKV",t,512),previousKV[owner],floats:512)
                    try ops.copy(row("poolScore",t,512),previousScores[owner],floats:512)
                } else {
                    if ratio == 2 {
                        try ops.dispatch("kernel_dsv41_pool2",[512,1,1],[b("latent"),row("poolKV",t,512),row("poolScore",t,512),previousKV[owner],previousScores[owner]],x:2)
                    } else { try ops.copy(row("poolKV",t,512),b("latent"),floats:512) }
                    try ops.norm(b("latent"),w(layer,"attn_compressor_norm.weight").gpu,b("latent"),width:512,rows:1)
                    try ops.project(w(layer,"indexer.attn_k.weight"),b("latent"),b("indexK"),count:1)
                    try ops.norm(b("indexK"),w(layer,"indexer.k_norm.weight").gpu,b("indexK"),width:128,rows:1)
                    try ops.rope(b("indexK"),width:128,heads:1,rows:1,start:pos+1-ratio,compressed:true)
                    try ops.quantize(b("indexK"),width:128,rows:1,mode:2)
                    try ops.copy(b("indexK"),indexCache[owner],floats:128,outputOffset:(nComp-1)*128)
                    try ops.rope(b("latent"),width:512,heads:1,rows:1,start:pos+1-ratio,compressed:true)
                    try ops.quantize(b("latent"),width:512,rows:1,mode:3)
                    try ops.copy(b("latent"),compressed[owner],floats:512,outputOffset:(nComp-1)*512)
                }
            }
            let top = min(nComp,512)
            if nComp > 0 && indexSource {
                try ops.dispatch("kernel_dsv41_swift_index_scores",[UInt32(nComp),1,UInt32(pos),UInt32(ratio)],
                    [row("indexQ",t,4096),row("indexWeights",t,32),indexCache[owner],b("scores")],x:nComp,threads:128,shared:132*4)
                let blocks = (nComp+7)/8, mask = row("blockMask",t,maskWidth)
                if layer == 20 {
                    try ops.dispatch("kernel_dsv41_candidate_blocks",[UInt32(nComp),1,UInt32(pos),UInt32(ratio)],[b("scores"),b("blockScores"),b("scores")],x:(blocks+255)/256)
                    try ops.topK(b("blockScores"),count:blocks,keep:min(blocks,2048),heap:b("heap"),out:b("blockIDs"))
                    try ops.dispatch("kernel_dsv41_swift_mask",[UInt32(blocks),UInt32(min(blocks,2048))],[b("blockIDs"),mask],x:(blocks+255)/256)
                } else if layer > 20 {
                    try ops.dispatch("kernel_dsv41_candidate_filter",[UInt32(nComp),1,UInt32(pos),UInt32(ratio)],[b("scores"),b("scores"),mask],x:(nComp+255)/256)
                }
                try ops.topK(b("scores"),count:nComp,keep:top,heap:b("heap"),out:row("selectedComp",t,512))
            }
            if top > 0 {
                try ops.dispatch("kernel_dsv41_swift_gather",[512,UInt32(top)],[compressed[owner],row("selectedComp",t,512),b("selectedKV")],x:2,y:top)
            }
            let nRaw = min(pos+1,128)
            try graph.flashAttnCore(q:row("q",t,32768),kvF32:windows[layer],kvF16:b("kvF16"),mask:b("mask"),sinks:w(layer,"attn_sinks.weight").gpu,pad:b("pad"),tmp:b("tmp"),heads:row("heads",t,32768),nHead:64,nKeys:nRaw,rawStartRow:(pos+1-nRaw)%128,hasSinks:true,comp:top > 0 ? b("selectedKV") : nil,nComp:top)
        }
        try ops.quantize(b("heads"),width:32768,rows:count)
        try ops.rope(b("heads"),width:512,heads:64,rows:count,start:position,compressed:ratio != 0,inverse:true)
        for t in 0..<count {
            try graph.attnOutLowQ8(outputA:w(layer,"attn_output_a.weight").gpu,heads:row("heads",t,32768),low:row("low",t,8192),nGroups:8,groupDim:4096,rank:1024)
        }
        try ops.quantize(b("low"),width:8192,rows:count)
        try ops.project(w(layer,"attn_output_b.weight"),b("low"),b("block"),count:count)
    }

    private func sharedAndRoute(_ ops:DeepSeek41Ops,_ layer:Int,count:Int,mask:[Bool]?) throws {
        try ops.project(w(layer,"ffn_gate_shexp.weight"),b("norm"),b("sharedGate"),count:count)
        try ops.project(w(layer,"ffn_up_shexp.weight"),b("norm"),b("sharedUp"),count:count)
        try ops.graph.swiglu(gate:b("sharedGate"),up:b("sharedUp"),out:b("sharedMid"),n:count*2304,limit:10)
        try ops.quantize(b("sharedMid"),width:2304,rows:count)
        try ops.project(w(layer,"ffn_down_shexp.weight"),b("sharedMid"),b("shared"),count:count)
        try ops.project(w(layer,"ffn_gate_inp.weight"),b("norm"),b("routeLogits"),count:count,round:false)
        try ops.graph.routerProbabilitiesBatch(logits:b("routeLogits"),probabilities:b("routeProbs"),width:384,rows:count)
        if let mask, mask.contains(false) {
            for t in 0..<count {
                try ops.graph.routerFinalizeTop6(probs:row("routeProbs",t,384),selected:row("selected",t,6),bias:w(layer,mask[t] ? "exp_probs_b.bias":"exp_probs_b_vl.bias").gpu,weights:row("routeWeights",t,6),nExperts:384,expertWeightScale:1.5)
            }
        } else {
            try ops.graph.routerFinalizeTop6Batch(probs:b("routeProbs"),selected:b("selected"),bias:w(layer,"exp_probs_b.bias").gpu,weights:b("routeWeights"),nExperts:384,nTok:count,probsRow:384,selRow:6,weightsRow:6,expertWeightScale:1.5)
        }
    }
    private func routed(_ ops:DeepSeek41Ops,_ layer:Int,count:Int,cancelled:@Sendable () -> Bool) throws {
        let pointer = b("selected").buffer.contents().assumingMemoryBound(to:Int32.self)
        var selections: [Int:[Int32]] = [:]
        for token in 0..<count {
            let unique = Set((0..<6).map { pointer[token*6+$0] })
            guard unique.count == 6 else { throw SwiftModelDecoderError.gpu("V4.1 router ha selezionato esperti duplicati") }
        }
        for slot in 0..<count*6 {
            let id = Int(pointer[slot])
            guard (0..<384).contains(id) else { throw SwiftModelDecoderError.gpu("V4.1 router ID non valido") }
            selections[id,default:[]].append(Int32(slot))
        }
        guard b("routeWeights").floatArray(count*6).allSatisfy(\.isFinite) else { throw SwiftModelDecoderError.gpu("V4.1 routing non finito") }
        let ids = selections.keys.sorted()
        for start in stride(from:0,to:ids.count,by:6) {
            try check(cancelled)
            let wave = Array(ids[start..<min(start+6,ids.count)])
            let records = try wave.enumerated().map { try pager.read(layer:layer,expert:$0.element,slot:$0.offset,cancelled:cancelled) }
            try ops.graph.begin()
            for (index,id) in wave.enumerated() {
                let slots = selections[id]!, n = slots.count
                let slotBuffer = try GPUTensor.bytes(runtime,slots.withUnsafeBytes{Array($0)},elementCount:n)
                try ops.dispatch("kernel_dsv41_swift_route_gather",[5120,UInt32(n)],
                    [b("norm"),slotBuffer,b("routeWeights"),b("packX"),b("packWeights")],x:20,y:n)
                let (gate,up,down) = records[index]
                if n == 1, let quant = MoEQuant.from(ggufType:gate.type), quant != .q2_K {
                    try ops.graph.moePairSwiGLU(quant,gateExp:gate.gpu,upExp:up.gpu,ids:b("zeroIDs"),activation:b("packX"),weights:b("packWeights"),gateScratch:b("packGate"),upScratch:b("packUp"),mid:b("packMid"),k:1,inDim:5120,outDim:2304,clamp:10)
                } else {
                    try ops.project(gate,b("packX"),b("packGate"),count:n,round:false,routed:true,zeroIDs:b("zeroIDs"))
                    try ops.project(up,b("packX"),b("packUp"),count:n,round:false,routed:true,zeroIDs:b("zeroIDs"))
                    try ops.graph.moeSwiGLUWeight(gate:b("packGate"),up:b("packUp"),weights:b("packWeights"),mid:b("packMid"),width:2304,rows:n,clampValue:10)
                }
                try ops.project(down,b("packMid"),b("packOut"),count:n,round:false,routed:true,zeroIDs:b("zeroIDs"))
                try ops.dispatch("kernel_dsv41_swift_route_scatter",[5120,UInt32(n)],[b("packOut"),slotBuffer,b("experts")],x:20,y:n)
            }
            try ops.finish()
        }
    }
}
