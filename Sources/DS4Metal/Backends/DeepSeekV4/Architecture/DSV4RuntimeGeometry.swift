import Foundation
import DS4Core

/// Instance-owned DeepSeek-V4 geometry used to remove Flash-only constants from
/// runtime objects without breaking callers that still use `DSV4Shape`.
///
/// A geometry built from a validated configuration preserves its per-layer
/// compression and RoPE metadata. Building directly from a known shape uses the
/// architecture defaults and is useful before a GGUF has been opened.
public struct DSV4RuntimeGeometry: Sendable {
    public let shape: DeepSeekV4Shape
    public let dims: DSV4Dims
    public let nLayers: Int
    public let compressRatios: [Int]
    public let expertWeightScale: Float
    public let nHashLayer: Int
    public let sinkhornIterations: Int

    private let ropeFreqBase: Float
    private let ropeScaleFactor: Float
    private let ropeYarnBetaFast: Float
    private let ropeYarnBetaSlow: Float
    private let compressRopeFreqBase: Float
    private let ropeOrigCtx: Int

    /// Geometry for one of the two known architecture profiles. RoPE and
    /// compression use the validated engine defaults associated with the shape.
    public init(shape: DeepSeekV4Shape) {
        let layerCount = Int(shape.nLayer)
        self.init(
            shape: shape,
            compressRatios: (0..<layerCount).map {
                Int(shape.expectedCompressRatio(layer: UInt32($0)))
            },
            swigluClamp: DeepSeekV4Defaults.swigluClampExp,
            ropeFreqBase: DeepSeekV4Defaults.ropeFreqBase,
            ropeScaleFactor: DeepSeekV4Defaults.ropeScaleFactor,
            ropeYarnBetaFast: DeepSeekV4Defaults.ropeYarnBetaFast,
            ropeYarnBetaSlow: DeepSeekV4Defaults.ropeYarnBetaSlow,
            compressRopeFreqBase: DeepSeekV4Defaults.compressRopeFreqBase,
            ropeOrigCtx: Int(DeepSeekV4Defaults.ropeOrigCtx)
        )
    }

    /// Geometry for a validated GGUF configuration. This is the initializer the
    /// inference service should use once model inspection has completed.
    public init(configuration: DeepSeekV4Configuration) {
        self.init(
            shape: configuration.shape,
            compressRatios: configuration.compressRatios.map(Int.init),
            swigluClamp: configuration.swigluClampExp.first
                ?? DeepSeekV4Defaults.swigluClampExp,
            ropeFreqBase: configuration.ropeFreqBase,
            ropeScaleFactor: configuration.ropeScaleFactor,
            ropeYarnBetaFast: configuration.ropeYarnBetaFast,
            ropeYarnBetaSlow: configuration.ropeYarnBetaSlow,
            compressRopeFreqBase: configuration.compressRopeFreqBase,
            ropeOrigCtx: Int(configuration.ropeOrigCtx)
        )
    }

    private init(
        shape: DeepSeekV4Shape,
        compressRatios: [Int],
        swigluClamp: Float,
        ropeFreqBase: Float,
        ropeScaleFactor: Float,
        ropeYarnBetaFast: Float,
        ropeYarnBetaSlow: Float,
        compressRopeFreqBase: Float,
        ropeOrigCtx: Int
    ) {
        let nLayers = Int(shape.nLayer)
        precondition(compressRatios.count == nLayers,
                     "expected \(nLayers) compression ratios, got \(compressRatios.count)")

        let nHead = Int(shape.nHead)
        let headDim = Int(shape.nHeadDim)
        var dims = DSV4Dims(
            nEmbd: Int(shape.nEmbd),
            nHC: Int(shape.nHC),
            headDim: headDim,
            nHead: nHead,
            qRank: Int(shape.nLoraQ),
            qDim: nHead * headDim,
            sharedFfn: Int(shape.nFFExp),
            nExperts: Int(shape.nExpert),
            expertFfn: Int(shape.nFFExp),
            k: Int(shape.nExpertUsed),
            nRot: Int(shape.nRot),
            vocab: Int(shape.nVocab),
            nOutGroup: Int(shape.nOutGroup),
            nLoraO: Int(shape.nLoraO),
            swigluClamp: swigluClamp,
            expertWeightScale: shape.expertWeightScale,
            nHashLayers: Int(shape.nHashLayer),
            sinkhornIterations: Int(shape.nHCSinkhornIter)
        )
        dims.nSWA = Int(shape.nSWA)
        dims.nIndexerHead = Int(shape.nIndexerHead)
        dims.nIndexerHeadDim = Int(shape.nIndexerHeadDim)
        dims.indexerTopK = Int(shape.nIndexerTopK)

        self.shape = shape
        self.dims = dims
        self.nLayers = nLayers
        self.compressRatios = compressRatios
        self.expertWeightScale = shape.expertWeightScale
        self.nHashLayer = Int(shape.nHashLayer)
        self.sinkhornIterations = Int(shape.nHCSinkhornIter)
        self.ropeFreqBase = ropeFreqBase
        self.ropeScaleFactor = ropeScaleFactor
        self.ropeYarnBetaFast = ropeYarnBetaFast
        self.ropeYarnBetaSlow = ropeYarnBetaSlow
        self.compressRopeFreqBase = compressRopeFreqBase
        self.ropeOrigCtx = ropeOrigCtx
    }

    /// Source-friendly aliases matching the old singular/static spelling.
    public var nLayer: Int { nLayers }
    public var nHCSinkhornIter: Int { sinkhornIterations }

    public func compressRatio(layer: Int) -> Int {
        precondition(layer >= 0 && layer < nLayers,
                     "layer \(layer) is outside 0..<\(nLayers)")
        return compressRatios[layer]
    }

    public func ropeParams(layer: Int) -> RopeParams {
        if compressRatio(layer: layer) != 0 {
            let freqScale = 1.0 / ropeScaleFactor
            let attnFactor = 1.0 / (1.0 + 0.1 * Foundation.log(1.0 / freqScale))
            return RopeParams(
                nCtxOrig: ropeOrigCtx,
                freqBase: compressRopeFreqBase,
                freqScale: freqScale,
                extFactor: 1,
                attnFactor: attnFactor,
                betaFast: ropeYarnBetaFast,
                betaSlow: ropeYarnBetaSlow
            )
        }
        return RopeParams(
            nCtxOrig: 0,
            freqBase: ropeFreqBase,
            freqScale: 1,
            extFactor: 0,
            attnFactor: 1,
            betaFast: ropeYarnBetaFast,
            betaSlow: ropeYarnBetaSlow
        )
    }
}
