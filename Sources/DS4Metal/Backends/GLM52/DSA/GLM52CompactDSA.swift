import Foundation
import DS4Core

/// Storage precision for GLM's compact KV-LoRA, RoPE-tail and indexer-key
/// caches.  The reference Metal graph uses F16; F32 remains useful for CPU
/// correctness oracles.
public enum GLM52CompactCachePrecision: Int, Sendable, Equatable {
    case float16 = 2
    case float32 = 4

    public var bytesPerElement: UInt64 { UInt64(rawValue) }
}

/// Byte breakdown of a compact DSA cache at one concrete row capacity.
public struct GLM52CompactDSAAllocation: Sendable, Equatable {
    public let rows: Int
    public let kvLoRABytes: UInt64
    public let ropeTailBytes: UInt64
    public let indexerKeyBytes: UInt64
    public let totalBytes: UInt64
}

public enum GLM52CompactDSAError: Error, Sendable, Equatable, CustomStringConvertible {
    case invalidCapacity(current: Int, required: Int, logicalContext: Int)
    case logicalContextExceedsModel(requested: Int, maximum: Int)
    case byteCountOverflow
    case budgetExceeded(requiredBytes: UInt64, budgetBytes: UInt64)

    public var description: String {
        switch self {
        case .invalidCapacity(let current, let required, let logical):
            return "invalid compact DSA capacity current=\(current) required=\(required) context=\(logical)"
        case .logicalContextExceedsModel(let requested, let maximum):
            return "GLM compact DSA context \(requested) exceeds model limit \(maximum)"
        case .byteCountOverflow:
            return "GLM compact DSA byte count overflow"
        case .budgetExceeded(let required, let budget):
            return "GLM compact DSA requires \(required) bytes, above the \(budget)-byte budget"
        }
    }
}

/// Logical layout of the compact cache used by GLM DSA.
///
/// Every normal transformer layer stores 512 KV-LoRA elements and the 64-value
/// RoPE tail per token.  Only the 21 full-indexer layers additionally store one
/// 128-value indexer key.  Expanded 64-head K/V rows are deliberately absent.
public struct GLM52CompactDSALayout: Sendable, Equatable {
    public let shape: GLM52Shape
    public let precision: GLM52CompactCachePrecision

    public init(shape: GLM52Shape = .v5_2,
                precision: GLM52CompactCachePrecision = .float16) {
        self.shape = shape
        self.precision = precision
    }

    public var normalLayerCount: Int { Int(shape.inferenceLayerCount) }

    public var fullIndexerLayers: [Int] {
        GLM52IndexSharePolicy.fullIndexerLayers(shape: shape)
    }

    public var kvLoRABytesPerToken: UInt64 {
        UInt64(normalLayerCount) * UInt64(shape.nKVLoRA) * precision.bytesPerElement
    }

    public var ropeTailBytesPerToken: UInt64 {
        UInt64(normalLayerCount) * UInt64(shape.nRot) * precision.bytesPerElement
    }

    public var indexerKeyBytesPerToken: UInt64 {
        UInt64(fullIndexerLayers.count) * UInt64(shape.nIndexerHeadDim)
            * precision.bytesPerElement
    }

    public var bytesPerToken: UInt64 {
        kvLoRABytesPerToken + ropeTailBytesPerToken + indexerKeyBytesPerToken
    }

    public func allocation(rows: Int) throws -> GLM52CompactDSAAllocation {
        guard rows >= 0 else {
            throw GLM52CompactDSAError.invalidCapacity(
                current: rows, required: rows, logicalContext: rows)
        }
        let rowCount = UInt64(rows)
        let kv = try Self.multiply(kvLoRABytesPerToken, rowCount)
        let rope = try Self.multiply(ropeTailBytesPerToken, rowCount)
        let indexer = try Self.multiply(indexerKeyBytesPerToken, rowCount)
        let (first, overflow1) = kv.addingReportingOverflow(rope)
        let (total, overflow2) = first.addingReportingOverflow(indexer)
        guard !overflow1, !overflow2 else { throw GLM52CompactDSAError.byteCountOverflow }
        return GLM52CompactDSAAllocation(
            rows: rows,
            kvLoRABytes: kv,
            ropeTailBytes: rope,
            indexerKeyBytes: indexer,
            totalBytes: total
        )
    }

    private static func multiply(_ lhs: UInt64, _ rhs: UInt64) throws -> UInt64 {
        let (value, overflow) = lhs.multipliedReportingOverflow(by: rhs)
        guard !overflow else { throw GLM52CompactDSAError.byteCountOverflow }
        return value
    }
}

/// Append-only slab growth plan.  A backend can allocate each returned range as
/// a new packed slab, avoiding both full-context preallocation and a temporary
/// old+new cache copy during growth.
public struct GLM52CompactDSACapacityPlan: Sendable, Equatable {
    public let currentCapacity: Int
    public let targetCapacity: Int
    public let newSlabRanges: [Range<Int>]
    public let additionalBytes: UInt64
    public let residentBytesAfterGrowth: UInt64

    public var grows: Bool { targetCapacity > currentCapacity }
}

/// Lazy capacity policy for large logical contexts.
///
/// A 100k-token session therefore starts with no compact allocation and grows
/// only when rows become live.  Fixed append-only slabs bound allocation
/// latency and avoid reallocating/copying the already populated cache.
public struct GLM52CompactDSACapacityPolicy: Sendable, Equatable {
    public let slabRows: Int

    public init(slabRows: Int = 1_024) {
        precondition(slabRows > 0, "GLM compact DSA slabRows must be positive")
        self.slabRows = slabRows
    }

    public func plan(currentCapacity: Int,
                     requiredRows: Int,
                     logicalContext: Int,
                     layout: GLM52CompactDSALayout = .init(),
                     maxResidentBytes: UInt64? = nil) throws
        -> GLM52CompactDSACapacityPlan {
        let modelLimit = Int(layout.shape.originalContextLength)
        guard logicalContext <= modelLimit else {
            throw GLM52CompactDSAError.logicalContextExceedsModel(
                requested: logicalContext, maximum: modelLimit)
        }
        guard logicalContext > 0,
              currentCapacity >= 0,
              requiredRows >= 0,
              currentCapacity <= logicalContext,
              requiredRows <= logicalContext else {
            throw GLM52CompactDSAError.invalidCapacity(
                current: currentCapacity,
                required: requiredRows,
                logicalContext: logicalContext
            )
        }

        let target: Int
        if requiredRows <= currentCapacity {
            target = currentCapacity
        } else {
            let quotient = requiredRows / slabRows
            let hasRemainder = requiredRows % slabRows != 0
            let rounded = (quotient + (hasRemainder ? 1 : 0)) * slabRows
            target = min(rounded, logicalContext)
        }

        let before = try layout.allocation(rows: currentCapacity)
        let after = try layout.allocation(rows: target)
        if let maxResidentBytes, after.totalBytes > maxResidentBytes {
            throw GLM52CompactDSAError.budgetExceeded(
                requiredBytes: after.totalBytes,
                budgetBytes: maxResidentBytes
            )
        }

        var slabs: [Range<Int>] = []
        var start = currentCapacity
        while start < target {
            // Bound the increment before adding so a public policy configured
            // with `Int.max` cannot overflow while planning a small context.
            let end = start + min(slabRows, target - start)
            slabs.append(start..<end)
            start = end
        }

        return GLM52CompactDSACapacityPlan(
            currentCapacity: currentCapacity,
            targetCapacity: target,
            newSlabRanges: slabs,
            additionalBytes: after.totalBytes - before.totalBytes,
            residentBytesAfterGrowth: after.totalBytes
        )
    }
}
