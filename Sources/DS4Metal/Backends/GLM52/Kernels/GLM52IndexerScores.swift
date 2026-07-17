import DS4Core
import Foundation
import Metal

/// Token-major GLM indexer scores with explicit matrix geometry.
public struct GLM52IndexerScoresOutput: Sendable, Equatable {
    public let scores: [Float]
    public let tokenCount: Int
    public let rowCount: Int

    public init(scores: [Float], tokenCount: Int, rowCount: Int) {
        self.scores = scores
        self.tokenCount = tokenCount
        self.rowCount = rowCount
    }

    public subscript(token: Int, row: Int) -> Float {
        scores[token * rowCount + row]
    }
}

private struct GLM52IndexerScorePlan {
    let tokenCount: Int
    let rowCount: Int
    let scoreCount: Int
}

private func glm52IndexerScorePlan(
    queries: [Float],
    headWeights: [Float],
    keyCacheBits: [UInt16],
    pos0: Int,
    scale: Float
) throws -> GLM52IndexerScorePlan {
    let queryWidth = GLM52IndexerScoresReference.queryWidth
    let headCount = GLM52IndexerScoresReference.headCount
    let headDimension = GLM52IndexerScoresReference.headDimension
    guard !queries.isEmpty, queries.count.isMultiple(of: queryWidth) else {
        throw MetalError.unsupported(
            "GLM 5.2 indexer queries must contain non-empty 32x128 rows"
        )
    }
    let tokenCount = queries.count / queryWidth
    let (expectedWeights, weightsOverflow) = tokenCount.multipliedReportingOverflow(
        by: headCount
    )
    guard !weightsOverflow, headWeights.count == expectedWeights else {
        throw MetalError.unsupported(
            "GLM 5.2 indexer head-weight count \(headWeights.count); " +
            "expected \(weightsOverflow ? -1 : expectedWeights)"
        )
    }
    guard !keyCacheBits.isEmpty,
          keyCacheBits.count.isMultiple(of: headDimension) else {
        throw MetalError.unsupported(
            "GLM 5.2 indexer key cache must contain non-empty 128-wide F16 rows"
        )
    }
    let rowCount = keyCacheBits.count / headDimension
    let u32Max = Int(UInt32.max)
    guard tokenCount <= u32Max, rowCount <= u32Max,
          pos0 >= 0, pos0 < rowCount,
          tokenCount <= rowCount - pos0 else {
        throw MetalError.unsupported(
            "GLM 5.2 indexer causal range pos0=\(pos0), tokens=\(tokenCount), " +
            "rows=\(rowCount) is invalid"
        )
    }
    guard scale.isFinite, scale > 0 else {
        throw MetalError.unsupported(
            "GLM 5.2 indexer scale must be finite and positive"
        )
    }
    let (scoreCount, scoreOverflow) = tokenCount.multipliedReportingOverflow(by: rowCount)
    guard !scoreOverflow else {
        throw MetalError.unsupported("GLM 5.2 indexer score size overflow")
    }
    return GLM52IndexerScorePlan(
        tokenCount: tokenCount,
        rowCount: rowCount,
        scoreCount: scoreCount
    )
}

/// Scalar oracle for GLM 5.2's DSA indexer relevance score.
///
/// For each visible cache row it computes
/// `sum_h relu(dot(q[token,h], key[row]) * scale) * headWeight[token,h]`.
/// Future rows are represented by negative infinity, matching the Metal
/// primitive and making a later top-K selection causally safe.
public enum GLM52IndexerScoresReference {
    public static let headCount = 32
    public static let headDimension = 128
    public static let queryWidth = headCount * headDimension
    public static let defaultScale: Float = 1.0 / 64.0

    public static func score(
        queries: [Float],
        headWeights: [Float],
        keyCacheBits: [UInt16],
        pos0: Int,
        scale: Float = defaultScale
    ) throws -> GLM52IndexerScoresOutput {
        let plan = try glm52IndexerScorePlan(
            queries: queries,
            headWeights: headWeights,
            keyCacheBits: keyCacheBits,
            pos0: pos0,
            scale: scale
        )
        var output = [Float](repeating: -Float.infinity, count: plan.scoreCount)
        for token in 0..<plan.tokenCount {
            let lastVisibleRow = pos0 + token
            for row in 0...lastVisibleRow {
                var score: Float = 0
                let keyOffset = row * headDimension
                for head in 0..<headCount {
                    let queryOffset = (token * headCount + head) * headDimension
                    var dotProduct: Float = 0
                    for column in 0..<headDimension {
                        dotProduct += queries[queryOffset + column] *
                            Half.float(keyCacheBits[keyOffset + column])
                    }
                    let activated = max(dotProduct * scale, 0)
                    score += activated * headWeights[token * headCount + head]
                }
                output[token * plan.rowCount + row] = score
            }
        }
        return GLM52IndexerScoresOutput(
            scores: output,
            tokenCount: plan.tokenCount,
            rowCount: plan.rowCount
        )
    }
}

extension MetalRuntime {
    /// Dispatch the fixed 32x128 GLM indexer scorer over an F16 key cache.
    /// This validation wrapper does not select top-K rows or enable GLM decode.
    public func glm52IndexerScores(
        queries: [Float],
        headWeights: [Float],
        keyCacheBits: [UInt16],
        pos0: Int,
        scale: Float = GLM52IndexerScoresReference.defaultScale
    ) throws -> GLM52IndexerScoresOutput {
        let plan = try glm52IndexerScorePlan(
            queries: queries,
            headWeights: headWeights,
            keyCacheBits: keyCacheBits,
            pos0: pos0,
            scale: scale
        )
        guard let queryBuffer = device.makeBuffer(
                  bytes: queries,
                  length: queries.count * MemoryLayout<Float>.stride,
                  options: .storageModeShared
              ),
              let weightsBuffer = device.makeBuffer(
                  bytes: headWeights,
                  length: headWeights.count * MemoryLayout<Float>.stride,
                  options: .storageModeShared
              ),
              let cacheBuffer = device.makeBuffer(
                  bytes: keyCacheBits,
                  length: keyCacheBits.count * MemoryLayout<UInt16>.stride,
                  options: .storageModeShared
              ),
              let scoresBuffer = device.makeBuffer(
                  length: plan.scoreCount * MemoryLayout<Float>.stride,
                  options: .storageModeShared
              ) else {
            throw MetalError.bufferAlloc
        }

        var arguments = [UInt32](repeating: 0, count: 4)
        arguments[0] = UInt32(plan.rowCount)
        arguments[1] = UInt32(plan.tokenCount)
        arguments[2] = UInt32(pos0)
        arguments[3] = scale.bitPattern

        let pipeline = try pipeline("kernel_glm52_indexer_scores_f16")
        guard pipeline.threadExecutionWidth == 32,
              pipeline.maxTotalThreadsPerThreadgroup >= 128 else {
            throw MetalError.unsupported(
                "GLM 5.2 indexer scorer requires four 32-lane SIMD groups"
            )
        }
        guard let commandBuffer = queue.makeCommandBuffer(),
              let encoder = commandBuffer.makeComputeCommandEncoder() else {
            throw MetalError.bufferAlloc
        }
        encoder.setComputePipelineState(pipeline)
        arguments.withUnsafeBytes {
            encoder.setBytes($0.baseAddress!, length: $0.count, index: 0)
        }
        encoder.setBuffer(queryBuffer, offset: 0, index: 1)
        encoder.setBuffer(weightsBuffer, offset: 0, index: 2)
        encoder.setBuffer(cacheBuffer, offset: 0, index: 3)
        encoder.setBuffer(scoresBuffer, offset: 0, index: 4)
        encoder.setThreadgroupMemoryLength(
            (128 + 4) * MemoryLayout<Float>.stride,
            index: 0
        )
        encoder.dispatchThreadgroups(
            MTLSize(width: plan.rowCount, height: plan.tokenCount, depth: 1),
            threadsPerThreadgroup: MTLSize(width: 32, height: 4, depth: 1)
        )
        encoder.endEncoding()
        commandBuffer.commit()
        commandBuffer.waitUntilCompleted()
        if let error = commandBuffer.error { throw error }

        let pointer = scoresBuffer.contents().bindMemory(
            to: Float.self,
            capacity: plan.scoreCount
        )
        return GLM52IndexerScoresOutput(
            scores: Array(UnsafeBufferPointer(
                start: pointer,
                count: plan.scoreCount
            )),
            tokenCount: plan.tokenCount,
            rowCount: plan.rowCount
        )
    }
}
