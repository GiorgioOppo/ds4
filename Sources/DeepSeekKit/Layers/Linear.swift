import Foundation
import MLX
import MLXNN

private let fp4Lut = MLXArray((0..<16).map { Float(dequantE2M1(UInt8($0))) })
private let e8m0Lut = MLXArray((0..<256).map { Float(dequantE8M0(UInt8($0))) })

public final class Linear {
    public let inFeatures: Int
    public let outFeatures: Int
    
    // Eager mode: weight stored directly
    private var _weight: Tensor?
    private var _scale: Tensor?
    
    // Lazy mode: load from disk on demand
    private var _weightName: String?
    private var _scaleName: String?
    private weak var _loader: WeightLoader?
    
    public var weight: Tensor {
        if let w = _weight { return w }
        // Lazy load
        guard let loader = _loader, let name = _weightName else {
            fatalError("Linear: no weight and no loader configured for lazy loading")
        }
        let w = (try? loader.load(name)) ?? Tensor.empty(shape: [outFeatures, inFeatures], dtype: .f32)
        return w
    }
    
    public var scale: Tensor? {
        if _weight != nil { return _scale }
        // Lazy load
        guard let loader = _loader, let name = _scaleName else { return nil }
        return try? loader.load(name)
    }

    public let castOutputToBF16: Bool
    public let useW8A8Activations: Bool
    public var inverseChannelScale: Tensor? = nil
    
    /// Original dtype of the weight (needed for FP4 detection even in lazy mode)
    public let weightDType: DType

    // Eager init (existing API)
    public init(inFeatures: Int, outFeatures: Int, weight: Tensor, scale: Tensor?,
                castOutputToBF16: Bool = false,
                useW8A8Activations: Bool = false,
                inverseChannelScale: Tensor? = nil) {
        self.inFeatures = inFeatures
        self.outFeatures = outFeatures
        self._weight = weight
        self._scale = scale
        self.weightDType = weight.dtype
        self.castOutputToBF16 = castOutputToBF16
        self.useW8A8Activations = useW8A8Activations
        self.inverseChannelScale = inverseChannelScale
    }
    
    // Lazy init (streaming mode)
    public init(inFeatures: Int, outFeatures: Int,
                weightName: String, scaleName: String?,
                weightDType: DType,
                loader: WeightLoader,
                castOutputToBF16: Bool = false) {
        self.inFeatures = inFeatures
        self.outFeatures = outFeatures
        self._weight = nil
        self._scale = nil
        self._weightName = weightName
        self._scaleName = scaleName
        self._loader = loader
        self.weightDType = weightDType
        self.castOutputToBF16 = castOutputToBF16
        self.useW8A8Activations = false
    }
    
    public func getDequantizedWeight() -> MLXArray {
        let w = self.weight
        var wArr = w.array
        let isFP4 = weightDType == .fp4E2M1
        
        if isFP4 {
            let u8Arr = wArr.asType(.uint8)
            let low = bitwiseAnd(u8Arr, 15)
            let high = rightShift(u8Arr, 4)
            let stackedArr = MLX.stacked([low, high], axis: -1)
            let indices = stackedArr.reshaped([outFeatures, inFeatures])
            
            var unpacked = take(fp4Lut, indices)
            
            if let s = self.scale {
                var sArr = s.array
                if s.dtype == .e8m0 {
                    sArr = take(e8m0Lut, sArr.asType(.uint8))
                } else if s.dtype != .f32 {
                    sArr = sArr.asType(.float32)
                }
                
                let sExpanded = sArr.expandedDimensions(axes: [-1])
                let unpackedReshaped = unpacked.reshaped([outFeatures, inFeatures / 32, 32])
                let scaled = unpackedReshaped * sExpanded
                unpacked = scaled.reshaped([outFeatures, inFeatures])
            }
            wArr = unpacked // Retain as Float32 to avoid casting back to UInt8
        } else if let s = self.scale, s.shape.count == 2 {
            // Block scaled FP8 weights (DeepSeek W8A8)
            var sArr = s.array
            
            if s.dtype == .e8m0 {
                sArr = take(e8m0Lut, sArr.asType(.uint8))
            } else if s.dtype != .f32 {
                // If it's native fp8 or f16, MLX can multiply it directly, but 
                // just in case we can upcast to f32. Actually let's just leave it 
                // as is unless it's uint8/e8m0.
                if sArr.dtype == .uint8 {
                    // Fallback if the loader didn't tag it .e8m0 but it is uint8
                    sArr = take(e8m0Lut, sArr)
                }
            }
            
            // Expected scale shape is [outBlocks, inBlocks]
            let outBlocks = s.shape[0]
            let inBlocks = s.shape[1]
            
            let outBlockSize = outFeatures / outBlocks
            let inBlockSize = inFeatures / inBlocks
            
            let wReshaped = wArr.reshaped([outBlocks, outBlockSize, inBlocks, inBlockSize])
            let sReshaped = sArr.reshaped([outBlocks, 1, inBlocks, 1])
            
            let scaled = wReshaped * sReshaped
            wArr = scaled.reshaped([outFeatures, inFeatures])
        }
        return wArr
    }

    public func callAsFunction(_ xIn: Tensor) -> Tensor {
        var xArr = xIn.array
        
        if let invScale = inverseChannelScale {
            xArr = xArr * invScale.array
        }
        
        var wArr = getDequantizedWeight()
        
        // Cast weight to xArr's dtype if they differ
        if wArr.dtype != xArr.dtype {
            wArr = wArr.asType(xArr.dtype)
        }
        
        // In FP4, the last dimension is inFeatures / 2 because of packing
        let isFP4 = weightDType == .fp4E2M1
        let transposeNeeded = isFP4 ? (self.weight.array.shape.last == inFeatures / 2) : (wArr.shape.last == inFeatures)
        
        let yArr: MLXArray
        if transposeNeeded {
            yArr = matmul(xArr, wArr.transposed())
        } else {
            yArr = matmul(xArr, wArr)
        }
        
        var outArr = yArr
        if !isFP4, let s = scale, s.shape.count != 2 {
             outArr = outArr * s.array
        }
        
        if castOutputToBF16 {
            outArr = outArr.asType(.bfloat16)
        }
        
        return Tensor(array: outArr, dtype: castOutputToBF16 ? .bf16 : .f32)
    }
}
