import DS4Core
import Foundation
import Metal

// Multi-block descending top-k for the GLM 5.2 indexer, the faithful port of
// upstream's ds4_gpu_indexer_topk_tensor: bitonic-argsort each block of a
// score row, then iteratively merge the sorted runs (binary-search partition)
// until one run remains, writing the first top-k indices per token. It reuses
// the vendored kernel_argsort_f32_i32_desc / kernel_argsort_merge_f32_i32_desc
// kernels the DeepSeek router already dispatches — no GLM-specific kernel.
//
// Causality is encoded upstream of this call: future rows carry -INFINITY
// scores (see kernel_glm52_indexer_scores_f16) and sink to the end of the
// descending order. The caller must therefore keep topK at or below the
// number of finite rows; the architecture cap is the indexer's top-2048.
// Ties are resolved by the bitonic network, not by the CPU oracle's
// lowest-index rule — validation fixtures use distinct scores.

extension MetalRuntime {
    /// Top-`topK` row indices of each token-major score row, descending by
    /// score. `scores` is `[tokenCount][rowCount]`; the result is
    /// `[tokenCount][topK]`.
    public func glm52IndexerTopK(scores: [Float],
                                 rowCount: Int,
                                 tokenCount: Int,
                                 topK: Int) throws -> [UInt32] {
        guard rowCount > 0, tokenCount > 0,
              scores.count == rowCount * tokenCount else {
            throw MetalError.unsupported(
                "GLM 5.2 top-k scores must be [\(tokenCount)][\(rowCount)]")
        }
        guard topK > 0, topK <= rowCount,
              topK <= Int(GLM52Shape.v5_2.nIndexerTopK) else {
            throw MetalError.unsupported(
                "GLM 5.2 top-k \(topK) must be 1...min(\(rowCount), "
                + "\(GLM52Shape.v5_2.nIndexerTopK))")
        }
        guard rowCount <= Int(Int32.max), tokenCount <= Int(Int32.max) else {
            throw MetalError.unsupported("GLM 5.2 top-k geometry overflows Int32")
        }

        let sortPipeline = try pipeline("kernel_argsort_f32_i32_desc")
        let mergePipeline = try pipeline("kernel_argsort_merge_f32_i32_desc")

        // Block width: largest power of two the sort threadgroup can hold.
        var maxThreads = sortPipeline.maxTotalThreadsPerThreadgroup
        if maxThreads == 0 { maxThreads = 256 }
        var nth = 1
        while nth < rowCount && 2 * nth <= maxThreads { nth *= 2 }
        let blockCount = (rowCount + nth - 1) / nth
        let blockTopK = min(topK, nth)
        var workWidth = topK
        if blockCount > 1 {
            let lastBlock = rowCount - (blockCount - 1) * nth
            workWidth = (blockCount - 1) * blockTopK + min(lastBlock, blockTopK)
        }
        let onePass = blockCount <= 1

        let outputCount = tokenCount * topK
        let scratchRowBytes = workWidth * MemoryLayout<Int32>.stride
        guard let scoreBuffer = device.makeBuffer(
                  bytes: scores,
                  length: scores.count * MemoryLayout<Float>.stride,
                  options: .storageModeShared),
              let selectedBuffer = device.makeBuffer(
                  length: outputCount * MemoryLayout<Int32>.stride,
                  options: .storageModeShared) else {
            throw MetalError.bufferAlloc
        }
        var scratchBuffer: MTLBuffer?
        if !onePass {
            scratchBuffer = device.makeBuffer(
                length: 2 * scratchRowBytes * tokenCount,
                options: .storageModeShared)
            guard scratchBuffer != nil else { throw MetalError.bufferAlloc }
        }

        guard let commandBuffer = queue.makeCommandBuffer() else {
            throw MetalError.bufferAlloc
        }

        // Stage 1: sort every nth-wide block of every token row.
        let sortArguments = Self.argsortArgs(
            n: rowCount, rows: tokenCount, ne0: workWidth, topK: blockTopK)
        guard let sortEncoder = commandBuffer.makeComputeCommandEncoder() else {
            throw MetalError.bufferAlloc
        }
        sortEncoder.setComputePipelineState(sortPipeline)
        sortArguments.withUnsafeBytes {
            sortEncoder.setBytes($0.baseAddress!, length: $0.count, index: 0)
        }
        sortEncoder.setBuffer(scoreBuffer, offset: 0, index: 1)
        sortEncoder.setBuffer(onePass ? selectedBuffer : scratchBuffer!,
                              offset: 0, index: 2)
        sortEncoder.setThreadgroupMemoryLength(
            ((nth * MemoryLayout<Int32>.stride) + 15) & ~15, index: 0)
        sortEncoder.dispatchThreadgroups(
            MTLSize(width: blockCount * tokenCount, height: 1, depth: 1),
            threadsPerThreadgroup: MTLSize(width: nth, height: 1, depth: 1))
        sortEncoder.endEncoding()

        // Stage 2: merge sorted runs, ping-ponging the scratch halves; the
        // final merge lands directly in the packed [token][topK] output.
        var currentOffset = 0
        var nextOffset = scratchRowBytes * tokenCount
        var runLength = blockTopK
        while runLength < workWidth {
            let mergeCount = (workWidth + 2 * runLength - 1) / (2 * runLength)
            let finalMerge = mergeCount == 1
            var mergeThreads = mergePipeline.maxTotalThreadsPerThreadgroup
            if mergeThreads == 0 || mergeThreads > 512 { mergeThreads = 512 }
            mergeThreads = max(1, min(mergeThreads, runLength))

            let mergeArguments = Self.glm52ArgsortMergeArgs(
                n: rowCount,
                rows: tokenCount,
                ne0: workWidth,
                topK: finalMerge ? topK : workWidth,
                runLength: runLength)
            guard let encoder = commandBuffer.makeComputeCommandEncoder() else {
                throw MetalError.bufferAlloc
            }
            encoder.setComputePipelineState(mergePipeline)
            mergeArguments.withUnsafeBytes {
                encoder.setBytes($0.baseAddress!, length: $0.count, index: 0)
            }
            encoder.setBuffer(scoreBuffer, offset: 0, index: 1)
            encoder.setBuffer(scratchBuffer!, offset: currentOffset, index: 2)
            encoder.setBuffer(finalMerge ? selectedBuffer : scratchBuffer!,
                              offset: finalMerge ? 0 : nextOffset, index: 3)
            encoder.dispatchThreadgroups(
                MTLSize(width: mergeCount * tokenCount, height: 1, depth: 1),
                threadsPerThreadgroup: MTLSize(width: mergeThreads,
                                               height: 1, depth: 1))
            encoder.endEncoding()

            swap(&currentOffset, &nextOffset)
            runLength <<= 1
        }

        commandBuffer.commit()
        commandBuffer.waitUntilCompleted()
        if let error = commandBuffer.error { throw error }

        let pointer = selectedBuffer.contents().bindMemory(
            to: Int32.self, capacity: outputCount)
        return UnsafeBufferPointer(start: pointer, count: outputCount).map {
            UInt32(bitPattern: $0)
        }
    }

    /// 88-byte ds4_metal_args_argsort_merge, mirroring the upstream layout:
    /// int64 ne00..ne03, uint64 nb00..nb03, int32 ne0..ne3, top_k, len.
    static func glm52ArgsortMergeArgs(n: Int,
                                      rows: Int,
                                      ne0: Int,
                                      topK: Int,
                                      runLength: Int) -> [UInt8] {
        var bytes = [UInt8](repeating: 0, count: 88)
        func i64(_ offset: Int, _ value: Int64) {
            withUnsafeBytes(of: value.littleEndian) {
                for k in 0..<8 { bytes[offset + k] = $0[k] }
            }
        }
        func i32(_ offset: Int, _ value: Int32) {
            withUnsafeBytes(of: value.littleEndian) {
                for k in 0..<4 { bytes[offset + k] = $0[k] }
            }
        }
        let rowBytes = Int64(n) * 4
        i64(0, Int64(n)); i64(8, Int64(rows)); i64(16, 1); i64(24, 1)
        i64(32, 4); i64(40, rowBytes)
        i64(48, Int64(rows) * rowBytes); i64(56, Int64(rows) * rowBytes)
        i32(64, Int32(ne0)); i32(68, Int32(rows)); i32(72, 1); i32(76, 1)
        i32(80, Int32(topK)); i32(84, Int32(runLength))
        return bytes
    }
}
