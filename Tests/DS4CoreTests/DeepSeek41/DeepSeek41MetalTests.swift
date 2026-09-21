import XCTest
import Metal
@testable import DS4Core
@testable import DS4Metal

final class DeepSeek41MetalTests: XCTestCase {
    private func bf16(_ value: Float) -> Float {
        var bits = value.bitPattern
        if bits & 0x7f800000 != 0x7f800000 { bits &+= 0x7fff + ((bits >> 16) & 1) }
        return Float(bitPattern:bits & 0xffff0000)
    }
    private func nearest(_ value: Float, fp4: Bool) -> Float {
        var levels: [Float] = [0,0.5,1,1.5,2,3,4,6]
        if !fp4 {
            levels = []
            for code in 0..<127 {
                let mantissa = Float(code & 7)
                let level: Float
                if code < 8 { level = ldexpf(mantissa, -9) }
                else { level = ldexpf(1 + mantissa / 8, Int32((code >> 3) - 7)) }
                levels.append(level)
            }
        }
        var best = 0, distance = Float.infinity
        for (index,level) in levels.enumerated() {
            let error = abs(abs(value)-level)
            if error < distance || (error == distance && index%2 == 0 && best%2 != 0) { best=index;distance=error }
        }
        return value.sign == .minus ? -levels[best] : levels[best]
    }
    func testActivationFormatsPoolingSelectionAndRotary() throws {
        guard MTLCreateSystemDefaultDevice()?.hasUnifiedMemory == true else { throw XCTSkip("Requires Apple Silicon Metal") }
        let runtime = try MetalRuntime(additionalSources:[DeepSeek41MetalSource.source,DeepSeek41MetalSource.graph])
        let graph = GraphContext(runtime), ops = DeepSeek41Ops(runtime:runtime,graph:graph)
        for mode: UInt32 in 0...3 {
            let width = 512, rows = 3, block = mode == 3 ? 16 : 32
            let source = (0..<width*rows).map { index -> Float in
                if index < 32 { return index%2 == 0 ? Float(0) : -Float(0) }
                return Float((index*73)%997-498)/127 * Float(1 << ((index/block)%4))
            }
            let tensor = try GPUTensor.floats(runtime,source)
            try graph.begin(); try ops.quantize(tensor,width:width,rows:rows,mode:mode); try ops.finish()
            let result = tensor.floatArray()
            for start in stride(from:0,to:source.count,by:block) {
                let maximum = source[start..<start+block].map{abs(bf16($0))}.max()!
                var scale: Float = 1
                if mode == 1 { scale=pow(2,ceil(log2(max(maximum,1e-4)/448))) }
                if mode == 2 { scale=pow(2,ceil(log2(max(maximum,Float(sign:.plus,exponent:-124,significand:1.5))/6))) }
                if mode == 3 { scale=nearest(max(maximum,6/512)/6,fp4:false) }
                for i in start..<start+block {
                    let original = bf16(source[i]), expected = mode == 0 ? original : bf16(nearest(original/scale,fp4:mode != 1)*scale)
                    XCTAssertEqual(result[i].bitPattern,expected.bitPattern,"mode \(mode), coefficient \(i)")
                }
            }
        }
        try checkBF16Ties(runtime,ops)
        try checkPoolContinuation(runtime,ops)
        try checkCandidateSelection(runtime,ops)
        try checkRotary(runtime,ops)
        try checkQueryLowRankProjection(runtime,ops)
        try checkCausalAttentionRing(runtime,ops)
        try checkIndexerAndRouter(runtime,ops)
        try checkDelayedHC(runtime,ops)
    }
    private func checkBF16Ties(_ runtime: MetalRuntime,_ ops: DeepSeek41Ops) throws {
        let bits: [UInt32] = [0x3f808000,0x3f818000,0xbf808000,0xbf818000,0x00000001,0x80000001,0x7f800000,0xff800000,0x7fc12345]
        for offset in [0,1] {
            let data = [Float](repeating:-99,count:offset)+bits.map(Float.init(bitPattern:))+[123]
            let full = try GPUTensor.floats(runtime,data)
            let slice = full.subview(byteOffset:offset*4,byteLength:bits.count*4,count:bits.count)
            try ops.graph.begin();try ops.quantize(slice,width:bits.count,rows:1);try ops.finish()
            XCTAssertEqual(slice.floatArray().map(\.bitPattern),bits.map{bf16(Float(bitPattern:$0)).bitPattern})
            XCTAssertEqual(full.floatArray().last,123)
            if offset != 0 { XCTAssertEqual(full.floatArray().first,-99) }
        }
    }
    private func checkPoolContinuation(_ runtime: MetalRuntime,_ ops: DeepSeek41Ops) throws {
        let width = 512, count = 6
        let kv = (0..<width*count).map{Float($0%83-41)/16}
        let scores = (0..<width*count).map{Float($0%47-23)/4}
        let x = try GPUTensor.floats(runtime,kv), sc = try GPUTensor.floats(runtime,scores)
        let previous = try GPUTensor.zeros(runtime,floatCount:width), previousScore = try GPUTensor.zeros(runtime,floatCount:width)
        let batched = try GPUTensor.zeros(runtime,floatCount:count/2*width), sequential = try GPUTensor.zeros(runtime,floatCount:count/2*width)
        try ops.graph.begin()
        try ops.dispatch("kernel_dsv41_pool2",[UInt32(width),UInt32(count/2),0],[batched,x,sc,previous,previousScore],x:2,y:count/2)
        for row in 0..<count {
            if row%2 == 0 {
                try ops.copy(x.rowView(row:row,cols:width),previous,floats:width)
                try ops.copy(sc.rowView(row:row,cols:width),previousScore,floats:width)
            } else {
                try ops.dispatch("kernel_dsv41_pool2",[UInt32(width),1,1],
                    [sequential.rowView(row:row/2,cols:width),x.rowView(row:row,cols:width),sc.rowView(row:row,cols:width),previous,previousScore],x:2)
            }
        }
        try ops.finish()
        XCTAssertEqual(batched.floatArray().map(\.bitPattern),sequential.floatArray().map(\.bitPattern))
        let result = batched.floatArray()
        for pair in 0..<count/2 { for c in 0..<width {
            let a=pair*2*width+c,b=a+width
            let gate=1/(1+exp(Double(scores[b])-Double(scores[a])))
            let expected=bf16(Float(Double(kv[a])*gate+Double(kv[b])*(1-gate)))
            XCTAssertEqual(result[pair*width+c],expected,accuracy:max(1e-6,abs(expected)/128))
        } }
    }
    private func checkCandidateSelection(_ runtime: MetalRuntime,_ ops: DeepSeek41Ops) throws {
        // Latest partial block must remain selectable even with the lowest score.
        let scores = try GPUTensor.floats(runtime,[9,9,8,7,6,5,4,3,-100])
        let blocks = try GPUTensor.zeros(runtime,floatCount:2), heap = try GPUTensor.zeros(runtime,floatCount:8)
        let selected = try GPUTensor.zeros(runtime,floatCount:8), mask = try GPUTensor.zeros(runtime,floatCount:2)
        try ops.graph.begin()
        try ops.dispatch("kernel_dsv41_candidate_blocks",[9,1,8,1],[scores,blocks,scores],x:1)
        try ops.topK(blocks,count:2,keep:1,heap:heap,out:selected)
        try ops.dispatch("kernel_dsv41_swift_mask",[2,1],[selected,mask],x:1)
        try ops.dispatch("kernel_dsv41_candidate_filter",[9,1,8,1],[scores,scores,mask],x:1)
        try ops.finish()
        XCTAssertEqual(mask.floatArray(),[-Float.infinity,0])
        XCTAssertEqual(scores.floatArray(),[Float](repeating:-.infinity,count:8)+[-100])
        let tieScores = try GPUTensor.floats(runtime,[2,3,3,1,3,-1])
        try ops.graph.begin();try ops.topK(tieScores,count:6,keep:3,heap:heap,out:selected);try ops.finish()
        let ptr = selected.buffer.contents().assumingMemoryBound(to:Int32.self)
        XCTAssertEqual((0..<3).map{ptr[$0]},[1,2,4])
    }
    private func checkRotary(_ runtime: MetalRuntime,_ ops: DeepSeek41Ops) throws {
        let values = (0..<512*2).map{bf16(Float($0%61-30)/16)}
        let x = try GPUTensor.floats(runtime,values)
        try ops.graph.begin();try ops.rope(x,width:512,heads:1,rows:2,start:70000,compressed:true);try ops.finish()
        let result = x.floatArray()
        for row in 0..<2 {
            XCTAssertEqual(Array(result[row*512..<row*512+448]),Array(values[row*512..<row*512+448]))
            XCTAssertTrue(result[row*512+448..<row*512+512].allSatisfy(\.isFinite))
            XCTAssertNotEqual(Array(result[row*512+448..<row*512+512]),Array(values[row*512+448..<row*512+512]))
        }
    }
}
