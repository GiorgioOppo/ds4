import DS4Core
import Foundation
import Metal

// Multi-block descending top-k for the GLM 5.2 indexer. The generic path is the
// faithful port of upstream's ds4_gpu_indexer_topk_tensor: bitonic-argsort each
// score-row block, then iteratively merge the sorted runs. GLM's production
// geometry has a much more useful invariant, though: topK is always 2048.
// Above 8192 rows the optimized path truncates and compacts every intermediate
// merge run to 2048 entries, avoiding needless comparisons and scratch traffic.
// DS4_INDEXER_TOPK_FAST=0 forces the generic path for reproducible A/B tests.
//
// Causality is encoded upstream of this call: future rows carry -INFINITY
// scores (see kernel_glm52_indexer_scores_f16) and sink to the end of the
// descending order. The caller must therefore keep topK at or below the
// number of finite rows; the architecture cap is the indexer's top-2048.
// The initial bitonic runs do not promise the CPU oracle's lowest-index tie
// order; the compact merge inherits that limitation, so cross-path fixtures
// use distinct scores.

private struct GLM52IndexerTopKLayout {
    let sortThreads: Int
    let blockCount: Int
    let blockTopK: Int
    let workWidth: Int

    var onePass: Bool { blockCount <= 1 }
    var scratchRowBytes: Int {
        workWidth * MemoryLayout<Int32>.stride
    }
}

extension MetalRuntime {
    private static let glm52TopK2048 = 2_048
    private static let glm52TopKCompactMinimum = 8_193

    private func glm52IndexerTopKLayout(
        rowCount: Int,
        topK: Int
    ) throws -> GLM52IndexerTopKLayout {
        let sortPipeline = try pipeline("kernel_argsort_f32_i32_desc")
        var maxThreads = sortPipeline.maxTotalThreadsPerThreadgroup
        if maxThreads == 0 { maxThreads = 256 }
        var nth = 1
        while nth < rowCount && 2 * nth <= maxThreads { nth *= 2 }
        let blockCount = (rowCount + nth - 1) / nth
        let blockTopK = min(topK, nth)
        var workWidth = topK
        if blockCount > 1 {
            let lastBlock = rowCount - (blockCount - 1) * nth
            workWidth = (blockCount - 1) * blockTopK
                + min(lastBlock, blockTopK)
        }
        return GLM52IndexerTopKLayout(
            sortThreads: nth,
            blockCount: blockCount,
            blockTopK: blockTopK,
            workWidth: workWidth)
    }

    /// True only when the exact GLM top-2048 geometry has enough merge levels
    /// and scratch for compaction to help. Kept internal so parity/benchmark
    /// tests can distinguish a real fast-path run from the portable fallback.
    func glm52SupportsFastIndexerTopK(
        rowCount: Int,
        tokenCount: Int = 1,
        outputBytes: Int = .max,
        scratchBytes: Int = .max
    ) throws -> Bool {
        guard rowCount >= Self.glm52TopKCompactMinimum,
              tokenCount > 0,
              outputBytes >= tokenCount * Self.glm52TopK2048
                * MemoryLayout<UInt32>.stride else {
            return false
        }
        let layout = try glm52IndexerTopKLayout(
            rowCount: rowCount, topK: Self.glm52TopK2048)
        guard layout.blockCount > 1,
              scratchBytes >= 2 * layout.scratchRowBytes * tokenCount else {
            return false
        }
        return try pipeline(
            "kernel_glm52_indexer_topk_merge_compact"
        ).maxTotalThreadsPerThreadgroup > 0
    }

    /// Variante ENCODE del top-k: gli STESSI stadi del wrapper standalone
    /// qui sotto, ma dentro il command buffer del chiamante — gli score
    /// restano sul device (niente readback + re-upload né secondo command
    /// buffer) e il chiamante rilegge SOLO i topK indici dopo il commit.
    /// `sortScratch` deve tenere 2×workWidth int32 per riga (lo scratch del
    /// decode è dimensionato a 2×scoreCapacity). `preferFastPath` è esposto
    /// internamente per i test A/B: nil usa il knob latched, false forza il
    /// fallback, true lo preferisce ma rispetta comunque i limiti hardware.
    func glm52EncodeIndexerTopK(into commandBuffer: MTLCommandBuffer,
                                scores: MTLBuffer, rowCount: Int, topK: Int,
                                output: MTLBuffer,
                                sortScratch: MTLBuffer,
                                tokenCount: Int = 1,
                                preferFastPath: Bool? = nil) throws {
        guard rowCount > 0, tokenCount > 0, topK > 0,
              topK <= rowCount,
              scores.length >= rowCount * tokenCount
                * MemoryLayout<Float>.stride,
              output.length >= tokenCount * topK
                * MemoryLayout<UInt32>.stride else {
            throw MetalError.unsupported(
                "GLM 5.2 encoded top-k buffer geometry is invalid")
        }

        let layout = try glm52IndexerTopKLayout(
            rowCount: rowCount, topK: topK)
        let wantsFast = preferFastPath ?? GLM52IndexerTopKDispatch.enabled
        let useCompactMerge: Bool
        if wantsFast && topK == Self.glm52TopK2048 {
            useCompactMerge = try glm52SupportsFastIndexerTopK(
                rowCount: rowCount,
                tokenCount: tokenCount,
                outputBytes: output.length,
                scratchBytes: sortScratch.length)
        } else {
            useCompactMerge = false
        }
        let onePass = layout.onePass
        let requiredScratch = onePass
            ? 0
            : 2 * layout.scratchRowBytes * tokenCount
        guard sortScratch.length >= requiredScratch else {
            throw MetalError.unsupported(
                "GLM 5.2 top-k scratch \(sortScratch.length) B, "
                + "required \(requiredScratch) B")
        }
        func words(_ bytes: [UInt8]) -> [UInt32] {
            bytes.withUnsafeBytes { raw in
                (0..<bytes.count / 4).map {
                    raw.loadUnaligned(fromByteOffset: $0 * 4,
                                      as: UInt32.self)
                }
            }
        }
        try glm52GraphEncode(
            into: commandBuffer,
            pipelineName: "kernel_argsort_f32_i32_desc",
            arguments: words(Self.argsortArgs(
                n: rowCount,
                rows: tokenCount,
                ne0: layout.workWidth,
                topK: layout.blockTopK)),
            buffers: [scores, onePass ? output : sortScratch],
            threadgroups: MTLSize(
                width: layout.blockCount * tokenCount,
                height: 1, depth: 1),
            threadsPerThreadgroup: MTLSize(
                width: layout.sortThreads, height: 1, depth: 1),
            threadgroupMemoryLength:
                ((layout.sortThreads * MemoryLayout<Int32>.stride) + 15)
                    & ~15)

        if useCompactMerge {
            var currentOffset = 0
            var nextOffset = layout.scratchRowBytes * tokenCount
            var currentSetCount = layout.blockCount
            var currentStride = layout.blockTopK
            var currentTotal = layout.workWidth
            var currentRowStride = layout.workWidth
            let mergePipeline = try pipeline(
                "kernel_glm52_indexer_topk_merge_compact")
            var mergeThreads =
                mergePipeline.maxTotalThreadsPerThreadgroup
            if mergeThreads == 0 || mergeThreads > 512 {
                mergeThreads = 512
            }
            mergeThreads = max(1, min(mergeThreads, topK))

            while currentSetCount > 1 {
                let nextSetCount = (currentSetCount + 1) / 2
                var nextTotal = 0
                for outputSet in 0..<nextSetCount {
                    let first = 2 * outputSet * currentStride
                    let left = first < currentTotal
                        ? min(currentStride, currentTotal - first) : 0
                    let second = first + currentStride
                    let right = second < currentTotal
                        ? min(currentStride, currentTotal - second) : 0
                    nextTotal += min(topK, left + right)
                }
                let finalMerge = nextSetCount == 1
                let outputRowStride = finalMerge ? topK : nextTotal
                try glm52GraphEncode(
                    into: commandBuffer,
                    pipelineName:
                        "kernel_glm52_indexer_topk_merge_compact",
                    arguments: [
                        UInt32(rowCount),
                        UInt32(tokenCount),
                        UInt32(currentTotal),
                        UInt32(currentSetCount),
                        UInt32(currentStride),
                        UInt32(currentRowStride),
                        UInt32(topK),
                        UInt32(outputRowStride),
                    ],
                    buffers: [
                        scores,
                        sortScratch,
                        finalMerge ? output : sortScratch,
                    ],
                    offsets: [
                        0,
                        currentOffset,
                        finalMerge ? 0 : nextOffset,
                    ],
                    threadgroups: MTLSize(
                        width: nextSetCount * tokenCount,
                        height: 1, depth: 1),
                    threadsPerThreadgroup: MTLSize(
                        width: mergeThreads, height: 1, depth: 1))
                if !finalMerge {
                    swap(&currentOffset, &nextOffset)
                }
                currentSetCount = nextSetCount
                currentStride = topK
                currentTotal = nextTotal
                currentRowStride = outputRowStride
            }
            return
        }

        var currentOffset = 0
        var nextOffset = layout.scratchRowBytes * tokenCount
        var runLength = layout.blockTopK
        let mergePipeline = try pipeline("kernel_argsort_merge_f32_i32_desc")
        while runLength < layout.workWidth {
            let mergeCount = (layout.workWidth + 2 * runLength - 1)
                / (2 * runLength)
            let finalMerge = mergeCount == 1
            var mergeThreads = mergePipeline.maxTotalThreadsPerThreadgroup
            if mergeThreads == 0 || mergeThreads > 512 { mergeThreads = 512 }
            mergeThreads = max(1, min(mergeThreads, runLength))
            try glm52GraphEncode(
                into: commandBuffer,
                pipelineName: "kernel_argsort_merge_f32_i32_desc",
                arguments: words(Self.glm52ArgsortMergeArgs(
                    n: rowCount, rows: tokenCount,
                    ne0: layout.workWidth,
                    topK: finalMerge ? topK : layout.workWidth,
                    runLength: runLength)),
                buffers: [scores, sortScratch,
                          finalMerge ? output : sortScratch],
                offsets: [0, currentOffset, finalMerge ? 0 : nextOffset],
                threadgroups: MTLSize(
                    width: mergeCount * tokenCount,
                    height: 1, depth: 1),
                threadsPerThreadgroup: MTLSize(width: mergeThreads,
                                               height: 1, depth: 1))
            swap(&currentOffset, &nextOffset)
            runLength <<= 1
        }
    }

    /// Top-`topK` row indices of each token-major score row, descending by
    /// score. `scores` is `[tokenCount][rowCount]`; the result is
    /// `[tokenCount][topK]`.
    public func glm52IndexerTopK(scores: [Float],
                                 rowCount: Int,
                                 tokenCount: Int,
                                 topK: Int,
                                 preferFastPath: Bool? = nil) throws
        -> [UInt32] {
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

        let layout = try glm52IndexerTopKLayout(
            rowCount: rowCount, topK: topK)
        let outputCount = tokenCount * topK
        guard let scoreBuffer = device.makeBuffer(
                  bytes: scores,
                  length: scores.count * MemoryLayout<Float>.stride,
                  options: .storageModeShared),
              let selectedBuffer = device.makeBuffer(
                  length: outputCount * MemoryLayout<Int32>.stride,
                  options: .storageModeShared) else {
            throw MetalError.bufferAlloc
        }
        // Both merge policies ping-pong inside two workWidth-sized Int32
        // planes; compaction only reduces the live prefix at later levels.
        let scratchBytes = max(
            16, 2 * layout.scratchRowBytes * tokenCount)
        guard let scratchBuffer = device.makeBuffer(
            length: scratchBytes,
            options: .storageModeShared) else {
            throw MetalError.bufferAlloc
        }

        guard let commandBuffer = queue.makeCommandBuffer() else {
            throw MetalError.bufferAlloc
        }
        try glm52EncodeIndexerTopK(
            into: commandBuffer,
            scores: scoreBuffer,
            rowCount: rowCount,
            topK: topK,
            output: selectedBuffer,
            sortScratch: scratchBuffer,
            tokenCount: tokenCount,
            preferFastPath: preferFastPath)
        try glm52GraphCommit(commandBuffer)

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
