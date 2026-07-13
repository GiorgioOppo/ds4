import Foundation

public struct RopeParams {
    public var nCtxOrig: Int, freqBase: Float, freqScale: Float, extFactor: Float
    public var attnFactor: Float, betaFast: Float, betaSlow: Float
    public init(nCtxOrig: Int, freqBase: Float, freqScale: Float, extFactor: Float,
                attnFactor: Float, betaFast: Float, betaSlow: Float) {
        self.nCtxOrig = nCtxOrig; self.freqBase = freqBase; self.freqScale = freqScale
        self.extFactor = extFactor; self.attnFactor = attnFactor; self.betaFast = betaFast; self.betaSlow = betaSlow
    }
}

