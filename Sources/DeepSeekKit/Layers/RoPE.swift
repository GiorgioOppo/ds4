import Foundation
import MLX
import MLXNN

public final class RoPE {
    public let ropeHeadDim: Int
    public let freqs: Tensor
    
    public init(ropeHeadDim: Int, freqs: Tensor) {
        self.ropeHeadDim = ropeHeadDim
        self.freqs = freqs
    }
    
    public static func makeFreqs(config: ModelConfig, useYarn: Bool) -> Tensor {
        let originalSeqLen = useYarn ? config.originalSeqLen : 0
        let theta = useYarn ? config.compressRopeTheta : config.ropeTheta
        let arr = YaRN.precomputeFreqsCis(dim: config.ropeHeadDim,
                                          seqlen: config.maxSeqLen,
                                          originalSeqLen: originalSeqLen,
                                          base: theta,
                                          factor: config.ropeFactor,
                                          betaFast: config.betaFast,
                                          betaSlow: config.betaSlow)
        
        let mlxArray = arr.withUnsafeBytes { ptr in
            let data = Data(bytes: ptr.baseAddress!, count: ptr.count)
            return MLXArray(data, [config.maxSeqLen, config.ropeHeadDim / 2, 2], type: Float.self)
        }
        return Tensor(array: mlxArray, dtype: .f32)
    }
    
    /// `x`: [tokens, heads, headDim] f32.
    public func apply(_ x: Tensor, startPos: Int, inverse: Bool) -> Tensor {
        let T = x.shape[0], H = x.shape[1], D = x.shape[2]
        
        let xArr = x.array
        let nopeDim = D - ropeHeadDim
        
        let nopePart = xArr[0..., 0..., 0..<nopeDim]
        let ropePart = xArr[0..., 0..., nopeDim..<D]
        
        let freqSlice = freqs.array[startPos..<(startPos + T)]
        
        let xRope = ropePart.reshaped([T, H, ropeHeadDim / 2, 2])
        
        var cosFreq = freqSlice[0..., 0..., 0]
        var sinFreq = freqSlice[0..., 0..., 1]
        
        if inverse {
            sinFreq = -sinFreq
        }
        
        cosFreq = cosFreq.expandedDimensions(axes: [1])
        sinFreq = sinFreq.expandedDimensions(axes: [1])
        
        let x0 = xRope[0..., 0..., 0..., 0]
        let x1 = xRope[0..., 0..., 0..., 1]
        
        let out0 = x0 * cosFreq - x1 * sinFreq
        let out1 = x0 * sinFreq + x1 * cosFreq
        
        let outRope = stacked([out0, out1], axis: -1).reshaped([T, H, ropeHeadDim])
        
        let outFull = concatenated([nopePart, outRope], axis: -1)
        
        return Tensor(array: outFull, dtype: x.dtype)
    }
}
