import Foundation
import Metal
import DS4Core

struct DS41Weight {
    let tensor: GGUFModel.Tensor
    let gpu: GPUTensor
    let columns: Int, rows: Int
    var type: UInt32 { tensor.type }
}

/// Architecture-specific wrappers preserve BF16 boundaries; shared V4 HC
/// fusion cannot be used because V4.1 consumes the previous split's pre gate.
struct DeepSeek41Ops {
    let runtime: MetalRuntime
    let graph: GraphContext

    func dispatch(_ name: String, _ words: [UInt32], _ buffers: [GPUTensor],
                  x: Int, y: Int = 1, z: Int = 1, threads: Int = 256, shared: Int = 0) throws {
        let pipeline = try runtime.pipeline(name), encoder = graph.encoder
        guard threads <= pipeline.maxTotalThreadsPerThreadgroup else { throw MetalError.unsupported("V4.1 kernel thread count: \(name)") }
        encoder.setComputePipelineState(pipeline)
        words.withUnsafeBytes { encoder.setBytes($0.baseAddress!, length: $0.count, index: 0) }
        for (index, tensor) in buffers.enumerated() { encoder.setBuffer(tensor.buffer, offset: tensor.byteOffset, index: index + 1) }
        if shared > 0 { encoder.setThreadgroupMemoryLength(shared, index: 0) }
        encoder.dispatchThreadgroups(MTLSize(width: x, height: y, depth: z), threadsPerThreadgroup: MTLSize(width: threads, height: 1, depth: 1))
    }
    func quantize(_ x: GPUTensor, width: Int, rows: Int, mode: UInt32 = 0) throws {
        if mode == 0 && x.byteOffset % 16 == 0 {
            let count = UInt64(width * rows)
            try dispatch("kernel_dsv41_bf16_linear",[UInt32(truncatingIfNeeded:count),UInt32(count >> 32)],[x],x:(width*rows+1023)/1024)
        } else {
            let block = mode == 3 ? 16 : 32
            try dispatch("kernel_dsv41_quantize",[UInt32(width),UInt32(rows),mode],[x],x:(width+block-1)/block,y:rows,threads:32)
        }
    }
    func norm(_ input: GPUTensor, _ weight: GPUTensor, _ output: GPUTensor, width: Int, rows: Int) throws {
        try graph.rmsNorm(input,weight:weight,out:output,rows:rows,n:width,eps:1e-20)
        try quantize(output,width:width,rows:rows)
    }
    func project(_ weight: DS41Weight, _ input: GPUTensor, _ output: GPUTensor, count: Int,
                 round: Bool = true, routed: Bool = false, zeroIDs: GPUTensor? = nil) throws {
        let c = weight.columns, r = weight.rows
        if routed, let quant = MoEQuant.from(ggufType:weight.type), let zeroIDs {
            try graph.moeMatvecID(quant,experts:weight.gpu,ids:zeroIDs,activation:input,out:output,k:count,inDim:c,outDim:r,perExpertAct:true)
        } else if weight.type == 8 && count > 8 {
            try graph.encodeMMDenseQ8(weight:weight.gpu,act:input,actBase:0,out:output,inDim:c,outDim:r,nTok:count)
        } else if weight.type == 12 && count > 8 {
            try graph.encodeMMDenseQ4K(weight:weight.gpu,act:input,actBase:0,out:output,inDim:c,outDim:r,nTok:count)
        } else if weight.type == 1 && count > 8 {
            try graph.encodeMMDenseF16(weight:weight.gpu,act:input,actBase:0,out:output,inDim:c,outDim:r,nTok:count)
        } else if weight.type == 2 {
            try dispatch("kernel_dsv41_swift_q4_0",[UInt32(c),UInt32(r),UInt32(count),UInt32(c/32*18)],[weight.gpu,input,output],x:(r+3)/4,y:count,threads:128)
        } else if [10,13,14,16].contains(weight.type) {
            try dispatch("kernel_dsv41_swift_kquant",[UInt32(c),UInt32(r),UInt32(count),weight.type],[weight.gpu,input,output],x:(r+3)/4,y:count,threads:128)
        } else if weight.type == 39 {
            try dispatch("kernel_dsv41_swift_mxfp4",[UInt32(c),UInt32(r),UInt32(count),0],[weight.gpu,input,output],x:(r+3)/4,y:count,threads:128)
        } else {
            for row in 0..<count {
                let x = input.rowView(row:row,cols:c), out = output.rowView(row:row,cols:r)
                switch weight.type {
                case 0: try graph.matmulF32(weight:weight.gpu,x:x,out:out,inDim:c,outDim:r)
                case 1: try graph.matmulF16(weight:weight.gpu,x:x,out:out,inDim:c,outDim:r)
                case 8: try graph.matmulQ8_0(weight:weight.gpu,x:x,out:out,inDim:c,outDim:r)
                case 12: try graph.matmulQ4_K(weight:weight.gpu,x:x,out:out,inDim:c,outDim:r)
                default: throw DeepSeek41Error.invalidTensor(weight.tensor.name)
                }
            }
        }
        if round { try quantize(output,width:r,rows:count) }
    }
    func rope(_ x: GPUTensor, width: Int, heads: Int, rows: Int, start: Int,
              stride: Int = 1, compressed: Bool, inverse: Bool = false) throws {
        let base: Float = compressed ? 160000 : 10000
        let low = Float(floor(64 * log(65536 / (32 * 2 * Double.pi)) / (2 * log(Double(base)))))
        let high = Float(ceil(64 * log(65536 / (2 * Double.pi)) / (2 * log(Double(base)))))
        var args = [UInt32(width),UInt32(heads),UInt32(rows),UInt32(start),inverse ? 1 : 0,UInt32(stride)]
        for i in 0..<32 {
            var frequency: Float = 1 / pow(base,Float(i)/32)
            if compressed {
                let ramp = min(1,max(0,(Float(i)-low)/(high-low))), smooth = 1-ramp
                frequency = (frequency/16)*(1-smooth) + frequency*smooth
            }
            args.append(frequency.bitPattern)
        }
        try dispatch("kernel_dsv41_rope",args,[x],x:heads,y:rows,threads:32)
    }
    func topK(_ scores: GPUTensor, count: Int, keep: Int, heap: GPUTensor, out: GPUTensor) throws {
        try graph.indexerTopKIndices(scores:scores,out:heap,nScores:count,topK:keep)
        try dispatch("kernel_dsv41_swift_sort_selected",[UInt32(keep)],[scores,heap,out],x:(keep+255)/256)
    }
    func copy(_ input: GPUTensor, _ output: GPUTensor, floats: Int, inputOffset: Int = 0, outputOffset: Int = 0) throws {
        try graph.blitCopies([(src:input,srcOff:inputOffset*4,dst:output,dstOff:outputOffset*4,bytes:floats*4)])
    }
    func finish() throws {
        graph.commit()
        if let error = graph.lastError { throw SwiftModelDecoderError.gpu(error.localizedDescription) }
    }
}
