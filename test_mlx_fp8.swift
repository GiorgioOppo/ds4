import Foundation
import MLX

func simulateFP8KV(_ x: MLXArray) -> MLXArray {
    // x shape: [B * S, nopeHeadDim]
    // split into blocks of 64
    let B_S = x.shape[0]
    let nopeHeadDim = x.shape[1]
    let blocks = nopeHeadDim / 64
    let x_blocked = x.reshaped([B_S, blocks, 64])
    
    let amax = maximum(MLX.abs(x_blocked).max(axes: [-1], keepDims: true), MLXArray(1e-4))
    let scale = exp2(ceil(log2(amax / 448.0)))
    let scaled = clip(x_blocked / scale, min: -448.0, max: 448.0)
    
    // E4M3 rounding (3 mantissa bits)
    let abs_scaled = MLX.abs(scaled)
    let p = exp2(floor(log2(maximum(abs_scaled, MLXArray(1e-9)))))
    let m = scaled / p
    let rounded_m = round(m * 8.0) / 8.0
    let y_rounded = rounded_m * p
    
    // Handle subnormals and zero (simplified)
    let result = MLX.where(abs_scaled .< 0.001, MLXArray.zeros(like: scaled), y_rounded)
    
    return (result * scale).reshaped([B_S, nopeHeadDim])
}

let x = MLXArray([1.0, 450.0, 0.0001, -123.4]).reshaped([1, 4])
// We will just test if it compiles and runs
print(x)
