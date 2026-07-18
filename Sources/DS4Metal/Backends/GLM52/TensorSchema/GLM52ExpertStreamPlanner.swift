import Foundation
import DS4Core

public enum GLM52ExpertProjection: String, Sendable, CaseIterable {
    case gate
    case up
    case down
}

public struct GLM52ExpertTensorLayout: Sendable, Equatable {
    public let descriptor: GLM52WeightDescriptor
    public let blockElements: UInt64
    public let blockBytes: UInt64
    public let rowBytes: UInt64
    public let expertBytes: UInt64
    public let expertCount: UInt64
    public let endOffset: UInt64
}

/// One contiguous file read.  Ranges are kept projection- and expert-scoped;
/// the planner never joins adjacent ranges belonging to different experts.
public struct GLM52ExpertByteRange: Sendable, Equatable {
    public let projection: GLM52ExpertProjection
    public let tensorName: String
    public let tensorType: UInt32
    public let expertID: UInt32
    public let absoluteOffset: UInt64
    public let byteCount: UInt64
}

public struct GLM52ExpertReadPlan: Sendable, Equatable {
    public let expertID: UInt32
    public let gate: GLM52ExpertByteRange
    public let up: GLM52ExpertByteRange
    public let down: GLM52ExpertByteRange
    public let totalBytes: UInt64

    public var ranges: [GLM52ExpertByteRange] { [gate, up, down] }
}

public struct GLM52ExpertStreamPlan: Sendable, Equatable {
    public let layer: Int
    /// Router rank order is preserved exactly.
    public let experts: [GLM52ExpertReadPlan]
    public let totalBytes: UInt64

    public var expertIDs: [UInt32] { experts.map(\.expertID) }
    public var ranges: [GLM52ExpertByteRange] { experts.flatMap(\.ranges) }
}

public enum GLM52ExpertStreamPlannerError: Error, Sendable, Equatable,
    CustomStringConvertible {
    case invalidLayer(Int)
    case invalidSelectionWidth(Int, expertCount: UInt64)
    case unsupportedQuantization(name: String, type: UInt32)
    case invalidDimensions(name: String, dims: [UInt64])
    case unalignedInnerDimension(name: String, dimension: UInt64, blockElements: UInt64)
    case arithmeticOverflow(name: String)
    case byteCountMismatch(name: String, expected: UInt64, got: UInt64)
    case mismatchedGateUpTypes(gate: UInt32, up: UInt32)
    case mismatchedProjectionGeometry(gate: [UInt64], up: [UInt64], down: [UInt64])
    case wrongSelectionCount(expected: Int, got: Int)
    case duplicateExpert(UInt32)
    case expertOutOfRange(UInt32, expertCount: UInt64)
    case plannedRangeOutsideTensor(name: String, expert: UInt32)
    case totalByteCountOverflow

    public var description: String {
        switch self {
        case .invalidLayer(let layer):
            return "invalid GLM 5.2 streaming layer: \(layer)"
        case .invalidSelectionWidth(let width, let count):
            return "invalid GLM 5.2 expert selection width \(width) for \(count) experts"
        case .unsupportedQuantization(let name, let type):
            return "\(name): unsupported routed-expert type \(GGUF.typeName(type))"
        case .invalidDimensions(let name, let dims):
            return "\(name): expected three non-zero expert dimensions, got \(dims)"
        case .unalignedInnerDimension(let name, let dimension, let block):
            return "\(name): inner dimension \(dimension) is not divisible by quantization block \(block)"
        case .arithmeticOverflow(let name):
            return "\(name): expert byte layout overflows UInt64"
        case .byteCountMismatch(let name, let expected, let got):
            return "\(name): directory size \(got) does not match quantized layout \(expected)"
        case .mismatchedGateUpTypes(let gate, let up):
            return "routed gate/up types differ (\(GGUF.typeName(gate)) vs \(GGUF.typeName(up)))"
        case .mismatchedProjectionGeometry(let gate, let up, let down):
            return "routed expert geometry does not transpose correctly: gate \(gate), up \(up), down \(down)"
        case .wrongSelectionCount(let expected, let got):
            return "GLM 5.2 routed streaming expects top-\(expected), got \(got) experts"
        case .duplicateExpert(let expert):
            return "GLM 5.2 routed streaming selection repeats expert \(expert)"
        case .expertOutOfRange(let expert, let count):
            return "GLM 5.2 expert \(expert) is outside 0..<\(count)"
        case .plannedRangeOutsideTensor(let name, let expert):
            return "\(name): planned range for expert \(expert) escapes the tensor payload"
        case .totalByteCountOverflow:
            return "GLM 5.2 expert stream-plan byte count overflows UInt64"
        }
    }
}

/// Pure byte-range planner for GLM 5.2 routed experts.
///
/// GGUF stores expert matrices as `[input, output, expert]`.  Each expert is
/// therefore one contiguous third-dimension slice.  Quantized row sizes are
/// derived from GGUF block geometry instead of from a bits-per-weight estimate.
public struct GLM52ExpertStreamPlanner: Sendable {
    public let layer: Int
    public let selectionWidth: Int
    public let gateLayout: GLM52ExpertTensorLayout
    public let upLayout: GLM52ExpertTensorLayout
    public let downLayout: GLM52ExpertTensorLayout

    public var expertCount: UInt64 { gateLayout.expertCount }

    public init(layer: Int,
                weights: GLM52RoutedExpertWeights,
                selectionWidth: Int = Int(GLM52Shape.v5_2.nExpertUsed),
                shape: GLM52Shape = .v5_2) throws {
        let firstMoELayer = Int(shape.nLeadingDense)
        let inferenceLayerCount = Int(shape.inferenceLayerCount)
        guard layer >= firstMoELayer, layer < inferenceLayerCount else {
            throw GLM52ExpertStreamPlannerError.invalidLayer(layer)
        }

        let gate = try Self.layout(weights.gate, projection: .gate)
        let up = try Self.layout(weights.up, projection: .up)
        let down = try Self.layout(weights.down, projection: .down)

        guard selectionWidth > 0, UInt64(selectionWidth) <= gate.expertCount else {
            throw GLM52ExpertStreamPlannerError.invalidSelectionWidth(
                selectionWidth, expertCount: gate.expertCount)
        }
        guard gate.descriptor.type == up.descriptor.type else {
            throw GLM52ExpertStreamPlannerError.mismatchedGateUpTypes(
                gate: gate.descriptor.type, up: up.descriptor.type)
        }

        let gd = gate.descriptor.dims
        let ud = up.descriptor.dims
        let dd = down.descriptor.dims
        guard gd == ud,
              gd[0] == dd[1],
              gd[1] == dd[0],
              gd[2] == dd[2] else {
            throw GLM52ExpertStreamPlannerError.mismatchedProjectionGeometry(
                gate: gd, up: ud, down: dd)
        }

        self.layer = layer
        self.selectionWidth = selectionWidth
        self.gateLayout = gate
        self.upLayout = up
        self.downLayout = down
    }

    public init(weightMap: GLM52WeightMap,
                layer: Int,
                selectionWidth: Int? = nil) throws {
        try self.init(
            layer: layer,
            weights: weightMap.routedExperts(layer: layer),
            selectionWidth: selectionWidth ?? Int(weightMap.configuration.shape.nExpertUsed),
            shape: weightMap.configuration.shape
        )
    }

    /// Build exactly one independent gate/up/down triplet per selected expert.
    /// Input order is router rank order and is not sorted or coalesced.
    /// A full router batch never exceeds the selection width; SHORTER plans
    /// are legitimate — the per-expert streaming provider fetches records
    /// one at a time into its slot cache.
    public func plan(selectedExperts: [UInt32]) throws -> GLM52ExpertStreamPlan {
        guard !selectedExperts.isEmpty,
              selectedExperts.count <= selectionWidth else {
            throw GLM52ExpertStreamPlannerError.wrongSelectionCount(
                expected: selectionWidth, got: selectedExperts.count)
        }

        var seen = Set<UInt32>()
        seen.reserveCapacity(selectedExperts.count)
        var reads: [GLM52ExpertReadPlan] = []
        reads.reserveCapacity(selectedExperts.count)
        var totalBytes: UInt64 = 0

        for expert in selectedExperts {
            guard UInt64(expert) < expertCount else {
                throw GLM52ExpertStreamPlannerError.expertOutOfRange(
                    expert, expertCount: expertCount)
            }
            guard seen.insert(expert).inserted else {
                throw GLM52ExpertStreamPlannerError.duplicateExpert(expert)
            }

            let gate = try Self.range(.gate, layout: gateLayout, expert: expert)
            let up = try Self.range(.up, layout: upLayout, expert: expert)
            let down = try Self.range(.down, layout: downLayout, expert: expert)
            let (gateAndUpBytes, gateUpOverflow) = gate.byteCount
                .addingReportingOverflow(up.byteCount)
            let (readBytes, readOverflow) = gateAndUpBytes
                .addingReportingOverflow(down.byteCount)
            guard !gateUpOverflow, !readOverflow else {
                throw GLM52ExpertStreamPlannerError.totalByteCountOverflow
            }
            let read = GLM52ExpertReadPlan(
                expertID: expert,
                gate: gate,
                up: up,
                down: down,
                totalBytes: readBytes
            )
            let (nextTotal, overflow) = totalBytes.addingReportingOverflow(readBytes)
            guard !overflow else {
                throw GLM52ExpertStreamPlannerError.totalByteCountOverflow
            }
            totalBytes = nextTotal
            reads.append(read)
        }

        return GLM52ExpertStreamPlan(layer: layer, experts: reads, totalBytes: totalBytes)
    }

    private static func layout(_ descriptor: GLM52WeightDescriptor,
                               projection: GLM52ExpertProjection) throws
        -> GLM52ExpertTensorLayout {
        let accepted = projection == .down
            ? GLM52TensorSchema.routedDownTypes
            : GLM52TensorSchema.routedGateUpTypes
        guard accepted.contains(descriptor.type),
              let info = GGUF.typeInfo(descriptor.type),
              info.blockElems > 0 else {
            throw GLM52ExpertStreamPlannerError.unsupportedQuantization(
                name: descriptor.name, type: descriptor.type)
        }
        guard descriptor.dims.count == 3,
              descriptor.dims.allSatisfy({ $0 > 0 }) else {
            throw GLM52ExpertStreamPlannerError.invalidDimensions(
                name: descriptor.name, dims: descriptor.dims)
        }

        let blockElements = UInt64(info.blockElems)
        let blockBytes = UInt64(info.blockBytes)
        let input = descriptor.dims[0]
        guard input.isMultiple(of: blockElements) else {
            throw GLM52ExpertStreamPlannerError.unalignedInnerDimension(
                name: descriptor.name,
                dimension: input,
                blockElements: blockElements
            )
        }

        let rowBlocks = input / blockElements
        let (rowBytes, rowOverflow) = rowBlocks.multipliedReportingOverflow(by: blockBytes)
        let (expertBytes, expertOverflow) = rowBytes.multipliedReportingOverflow(
            by: descriptor.dims[1])
        let (tensorBytes, tensorOverflow) = expertBytes.multipliedReportingOverflow(
            by: descriptor.dims[2])
        guard !rowOverflow, !expertOverflow, !tensorOverflow else {
            throw GLM52ExpertStreamPlannerError.arithmeticOverflow(name: descriptor.name)
        }
        guard tensorBytes == descriptor.bytes else {
            throw GLM52ExpertStreamPlannerError.byteCountMismatch(
                name: descriptor.name, expected: tensorBytes, got: descriptor.bytes)
        }
        let (endOffset, offsetOverflow) = descriptor.absOffset.addingReportingOverflow(
            descriptor.bytes)
        guard !offsetOverflow else {
            throw GLM52ExpertStreamPlannerError.arithmeticOverflow(name: descriptor.name)
        }

        return GLM52ExpertTensorLayout(
            descriptor: descriptor,
            blockElements: blockElements,
            blockBytes: blockBytes,
            rowBytes: rowBytes,
            expertBytes: expertBytes,
            expertCount: descriptor.dims[2],
            endOffset: endOffset
        )
    }

    private static func range(_ projection: GLM52ExpertProjection,
                              layout: GLM52ExpertTensorLayout,
                              expert: UInt32) throws -> GLM52ExpertByteRange {
        let (relativeOffset, relativeOverflow) = UInt64(expert)
            .multipliedReportingOverflow(by: layout.expertBytes)
        let (absoluteOffset, absoluteOverflow) = layout.descriptor.absOffset
            .addingReportingOverflow(relativeOffset)
        let (rangeEnd, rangeOverflow) = absoluteOffset
            .addingReportingOverflow(layout.expertBytes)
        guard !relativeOverflow,
              !absoluteOverflow,
              !rangeOverflow,
              absoluteOffset >= layout.descriptor.absOffset,
              rangeEnd <= layout.endOffset else {
            throw GLM52ExpertStreamPlannerError.plannedRangeOutsideTensor(
                name: layout.descriptor.name, expert: expert)
        }

        return GLM52ExpertByteRange(
            projection: projection,
            tensorName: layout.descriptor.name,
            tensorType: layout.descriptor.type,
            expertID: expert,
            absoluteOffset: absoluteOffset,
            byteCount: layout.expertBytes
        )
    }
}
