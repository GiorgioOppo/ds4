import XCTest
import Metal
@testable import DS4Core
@testable import DS4Metal

extension DeepSeek41MetalTests {
    func checkQueryLowRankProjection(_ runtime: MetalRuntime, _ ops: DeepSeek41Ops) throws {
        // Production QA dimensions with four exact, sparse Q8 coefficients per
        // row. Nine tokens selects GEMM; scalar continuation selects GEMV.
        let columns = 5120, rank = 1280, tokens = 9, rowBytes = 5120/32*34
        var packed = [UInt8](repeating:0,count:rowBytes*rank)
        let scale = Float16(1.0/16).bitPattern
        let coefficients: [Int8] = [-2,-1,1,2]
        for row in 0..<rank {
            for block in 0..<columns/32 {
                let offset = row*rowBytes+block*34
                packed[offset] = UInt8(truncatingIfNeeded:scale)
                packed[offset+1] = UInt8(truncatingIfNeeded:scale >> 8)
            }
            for term in 0..<4 {
                let column = (row*13+term*811)%columns
                packed[row*rowBytes+column/32*34+2+column%32] = UInt8(bitPattern:coefficients[term])
            }
        }
        let weight = try GPUTensor.bytes(runtime,packed,elementCount:columns*rank)
        let spec = GGUFModel.Tensor(name:"synthetic.attn_q_a.weight",dims:[UInt64(columns),UInt64(rank)],
            type:8,elements:UInt64(columns*rank),relOffset:0,absOffset:0,bytes:UInt64(packed.count))
        let matrix = DS41Weight(tensor:spec,gpu:weight,columns:columns,rows:rank)
        let values = (0..<tokens*columns).map { Float(($0*7)%127-63)/64 }
        let normValues = (0..<rank).map { Float(64+$0%7)/64 }
        let input = try GPUTensor.floats(runtime,values), norm = try GPUTensor.floats(runtime,normValues)
        let batched = try GPUTensor.zeros(runtime,floatCount:tokens*rank)
        let scalar = try GPUTensor.zeros(runtime,floatCount:tokens*rank)
        let normalized = try GPUTensor.zeros(runtime,floatCount:tokens*rank)
        try ops.graph.begin()
        try ops.project(matrix,input,batched,count:tokens)
        for t in 0..<tokens {
            try ops.project(matrix,input.rowView(row:t,cols:columns),scalar.rowView(row:t,cols:rank),count:1)
        }
        try ops.norm(batched,norm,normalized,width:rank,rows:tokens)
        try ops.finish()
        let actual = batched.floatArray(), actualNorm = normalized.floatArray()
        XCTAssertEqual(actual.map(\.bitPattern),scalar.floatArray().map(\.bitPattern))
        func rounded(_ x: Float) -> Float {
            let bits = x.bitPattern
            return Float(bitPattern:(bits &+ 0x7fff &+ ((bits >> 16)&1)) & 0xffff0000)
        }
        for t in 0..<tokens {
            var expected = [Float](repeating:0,count:rank)
            for row in 0..<rank {
                var sum: Float = 0
                for term in 0..<4 {
                    let column = (row*13+term*811)%columns
                    sum += values[t*columns+column]*Float(coefficients[term])/16
                }
                expected[row] = rounded(sum)
                XCTAssertEqual(actual[t*rank+row].bitPattern,expected[row].bitPattern)
            }
            let rms = sqrt(expected.reduce(Double(0)) { $0+Double($1)*Double($1) }/Double(rank)+1e-20)
            for row in 0..<rank {
                let value = rounded(Float(Double(expected[row])/rms*Double(normValues[row])))
                XCTAssertEqual(actualNorm[t*rank+row],value,accuracy:max(1e-7,abs(value)/128))
                XCTAssertEqual(actualNorm[t*rank+row].bitPattern & 0xffff,0,"QA normalization must end at a BF16 boundary")
            }
        }
    }

    /// Exercise the same per-query ring updates as the V4.1 layer-major graph.
    /// Poisoned unused cache rows and future compressed rows must stay invisible.
    func checkCausalAttentionRing(_ runtime: MetalRuntime, _ ops: DeepSeek41Ops) throws {
        let width = 512, heads = 2, count = 263
        let raw = (0..<count*width).map { Float(($0 / width * 17 + $0 % width * 3) % 97 - 48) / 128 }
        let compressed = (0..<132*width).map { Float(($0 / width * 29 + $0 % width * 7) % 89 - 44) / 64 }
        let queries = (0..<count*heads*width).map { Float(($0 * 13) % 31 - 15) / 64 }
        let sinkValues: [Float] = [-0.5, 0.75]
        let x = try GPUTensor.floats(runtime,raw), comp = try GPUTensor.floats(runtime,compressed)
        let q = try GPUTensor.floats(runtime,queries), sinks = try GPUTensor.floats(runtime,sinkValues)
        let scratch = GraphContext.flashScratchBytes(nHead:heads,nKeys:260)
        let kv16 = try GPUTensor.uninitializedBytes(runtime,byteLength:scratch.kvF16,elementCount:1)
        let mask = try GPUTensor.uninitializedBytes(runtime,byteLength:scratch.mask,elementCount:1)
        let pad = try GPUTensor.uninitializedBytes(runtime,byteLength:scratch.pad,elementCount:1)
        let tmp = try GPUTensor.uninitializedBytes(runtime,byteLength:scratch.tmp,elementCount:1)
        func run(chunks: [Int], fusedStage: Bool) throws -> [Float] {
            let ring = try GPUTensor.floats(runtime,[Float](repeating:37,count:128*width))
            let out = try GPUTensor.floats(runtime,[Float](repeating:99,count:count*heads*width+4))
            var start = 0
            for chunk in chunks {
                try ops.graph.begin()
                for position in start..<start+chunk {
                    let nRaw = min(position+1,128), nComp = (position+1)/2
                    try ops.copy(x.rowView(row:position,cols:width),ring,floats:width,outputOffset:(position%128)*width)
                    try ops.graph.flashAttnCore(q:q.rowView(row:position,cols:heads*width),kvF32:ring,
                        kvF16:kv16,mask:mask,sinks:sinks,pad:pad,tmp:tmp,
                        heads:out.rowView(row:position,cols:heads*width),nHead:heads,nKeys:nRaw,
                        rawStartRow:(position+1-nRaw)%128,hasSinks:true,comp:comp,nComp:nComp,
                        fusedStage:fusedStage)
                }
                try ops.finish()
                start += chunk
            }
            XCTAssertEqual(start,count)
            XCTAssertEqual(Array(out.floatArray().suffix(4)),[99,99,99,99])
            return out.floatArray(count*heads*width)
        }
        // Crossing both 127→128 and 255→256 inside an append catches a ring
        // overwrite being incorrectly scheduled ahead of earlier queries.
        let chunked = try run(chunks:[1,125,4,1,127,5],fusedStage:true)
        let scalar = try run(chunks:[Int](repeating:1,count:count),fusedStage:false)
        XCTAssertEqual(chunked.map(\.bitPattern),scalar.map(\.bitPattern))
        let scale = 1 / sqrt(Double(width))
        for position in [0,1,125,126,127,128,129,130,254,255,256,257,262] {
            let first = max(0,position+1-128), nComp = (position+1)/2
            let rows = (first...position).map { Array(raw[$0*width..<($0+1)*width]) }
                + (0..<nComp).map { Array(compressed[$0*width..<($0+1)*width]) }
            for head in 0..<heads {
                let qi = (position*heads+head)*width
                var scores: [Double] = []
                for row in rows {
                    var dot = Double(0)
                    for d in 0..<width { dot += Double(queries[qi+d])*Double(row[d]) }
                    scores.append(dot*scale)
                }
                let maximum = max(scores.max()!,Double(sinkValues[head]))
                let probabilities = scores.map { exp($0-maximum) }
                let denominator = probabilities.reduce(exp(Double(sinkValues[head])-maximum),+)
                for d in 0..<width {
                    var sum = Double(0)
                    for row in rows.indices { sum += probabilities[row]*Double(rows[row][d]) }
                    XCTAssertEqual(Double(chunked[qi+d]),sum/denominator,accuracy:0.0005,
                                   "causal ring position \(position), head \(head), channel \(d)")
                }
            }
        }
    }

    func checkIndexerAndRouter(_ runtime: MetalRuntime, _ ops: DeepSeek41Ops) throws {
        let tokens = 3, keys = 67, start = 127, ratio = 2
        let queries = (0..<tokens*32*128).map { Float(($0*13)%31-15)/16 }
        let weights = (0..<tokens*32).map { Float($0%9-4)/8 }
        let cache = (0..<keys*128).map { Float(($0*19)%47-23)/32 }
        let q = try GPUTensor.floats(runtime,queries), w = try GPUTensor.floats(runtime,weights)
        let k = try GPUTensor.floats(runtime,cache), scores = try GPUTensor.zeros(runtime,floatCount:tokens*keys)
        try ops.graph.begin()
        try ops.dispatch("kernel_dsv41_swift_index_scores",[UInt32(keys),UInt32(tokens),UInt32(start),UInt32(ratio)],
                         [q,w,k,scores],x:keys,y:tokens,threads:128,shared:132*4)
        try ops.finish()
        let actual = scores.floatArray()
        for t in 0..<tokens { for key in 0..<keys {
            if key >= (start+t+1)/ratio {
                XCTAssertEqual(actual[t*keys+key],-.infinity)
            } else {
                var expected: Float = 0
                for h in 0..<32 {
                    var dot: Float = 0
                    for d in 0..<128 { dot += queries[(t*32+h)*128+d]*cache[key*128+d] }
                    expected += max(dot/64,0)*weights[t*32+h]
                }
                XCTAssertEqual(actual[t*keys+key],expected,accuracy:1e-6)
            }
        } }
        // Text and image biases select different experts. The selected weights
        // must still normalize un-biased sqrt(softplus(logits)), totaling 1.5.
        let logits = (0..<tokens*384).map { Float($0%384-192)/64 + Float($0/384)/8 }
        var biases = [[Float](repeating:0,count:384),[Float](repeating:0,count:384)]
        for j in 0..<6 { biases[0][j] = 20 + Float(j); biases[1][383-j] = 20 + Float(j) }
        let input = try GPUTensor.floats(runtime,logits)
        let probs = try GPUTensor.zeros(runtime,floatCount:tokens*384)
        let selected = try GPUTensor.zeros(runtime,floatCount:tokens*6)
        let route = try GPUTensor.zeros(runtime,floatCount:tokens*6)
        let biasBuffers = try biases.map { try GPUTensor.floats(runtime,$0) }
        try ops.graph.begin()
        try ops.graph.routerProbabilitiesBatch(logits:input,probabilities:probs,width:384,rows:tokens)
        for t in 0..<tokens {
            try ops.graph.routerFinalizeTop6(probs:probs.rowView(row:t,cols:384),
                selected:selected.rowView(row:t,cols:6),bias:biasBuffers[t%2],weights:route.rowView(row:t,cols:6),
                nExperts:384,expertWeightScale:1.5)
        }
        try ops.finish()
        let ids = selected.buffer.contents().assumingMemoryBound(to:Int32.self), result = route.floatArray()
        for t in 0..<tokens {
            let cpuProbs = (0..<384).map { sqrt(log1p(exp(Double(logits[t*384+$0])))) }
            let expected = (0..<384).sorted {
                let a = cpuProbs[$0]+Double(biases[t%2][$0]), b = cpuProbs[$1]+Double(biases[t%2][$1])
                return a == b ? $0 < $1 : a > b
            }.prefix(6)
            XCTAssertEqual((0..<6).map { Int(ids[t*6+$0]) },Array(expected))
            let sum = expected.reduce(Double(0)) { $0+cpuProbs[$1] }
            for (slot,id) in expected.enumerated() {
                XCTAssertEqual(Double(result[t*6+slot]),cpuProbs[id]/sum*1.5,accuracy:1e-6)
            }
        }
    }

    func checkDelayedHC(_ runtime: MetalRuntime, _ ops: DeepSeek41Ops) throws {
        let width = 512, tokens = 3
        func rounded(_ value: Float) -> Float {
            let bits = value.bitPattern
            return Float(bitPattern:(bits &+ 0x7fff &+ ((bits >> 16)&1)) & 0xffff0000)
        }
        let residual = (0..<tokens*4*width).map { Float($0%131-65)/64 }
        let block = (0..<tokens*width).map { Float($0%71-35)/128 }
        let previous: [Float] = [1,0,0,0, 0.125,0.5,0.25,0.125, 0.5,0.25,0.125,0.125]
        var split = [Float](repeating:0,count:tokens*24)
        for t in 0..<tokens {
            for h in 0..<4 {
                split[t*24+h] = Float(4-h)/16
                split[t*24+4+h] = Float(h+1)/8
                for j in 0..<4 { split[t*24+8+j*4+h] = Float((h+j+t)%4+1)/16 }
            }
        }
        let r = try GPUTensor.floats(runtime,residual), b = try GPUTensor.floats(runtime,block)
        let pre = try GPUTensor.floats(runtime,previous), s = try GPUTensor.floats(runtime,split)
        let collapsed = try GPUTensor.zeros(runtime,floatCount:tokens*width)
        let expanded = try GPUTensor.zeros(runtime,floatCount:tokens*4*width)
        let next = try GPUTensor.zeros(runtime,floatCount:tokens*width)
        try ops.graph.begin()
        try ops.graph.hcWeightedSum(x:r,weights:pre,out:collapsed,nEmbd:width,nHC:4,nTokens:tokens)
        try ops.quantize(collapsed,width:width,rows:tokens)
        try ops.graph.hcExpand4(blockOut:b,residual:r,post:s,comb:s,blockAdd:nil,out:expanded,
            nEmbd:width,nTokens:tokens,postByteOffset:16,combByteOffset:32,splitTokenStride:96)
        try ops.quantize(expanded,width:width*4,rows:tokens)
        try ops.graph.hcWeightedSum(x:expanded,weights:s,out:next,nEmbd:width,nHC:4,nTokens:tokens,weightsTokenStride:96)
        try ops.quantize(next,width:width,rows:tokens)
        try ops.finish()
        let gotCollapsed = collapsed.floatArray(), gotExpanded = expanded.floatArray(), gotNext = next.floatArray()
        var differsFromCurrentPre = false
        for t in 0..<tokens { for d in 0..<width {
            var expected: Float = 0, wrongPre: Float = 0, nextExpected: Float = 0
            for h in 0..<4 {
                expected += residual[(t*4+h)*width+d]*previous[t*4+h]
                wrongPre += residual[(t*4+h)*width+d]*split[t*24+h]
                var expandedExpected = block[t*width+d]*split[t*24+4+h]
                for j in 0..<4 { expandedExpected += residual[(t*4+j)*width+d]*split[t*24+8+j*4+h] }
                let bf = rounded(expandedExpected)
                XCTAssertEqual(gotExpanded[(t*4+h)*width+d].bitPattern,bf.bitPattern)
                nextExpected += bf*split[t*24+h]
            }
            XCTAssertEqual(gotCollapsed[t*width+d].bitPattern,rounded(expected).bitPattern)
            XCTAssertEqual(gotNext[t*width+d].bitPattern,rounded(nextExpected).bitPattern)
            differsFromCurrentPre = differsFromCurrentPre || rounded(expected) != rounded(wrongPre)
        } }
        XCTAssertTrue(differsFromCurrentPre,"Fixture must detect accidentally consuming the current HC pre gate")
    }
}
