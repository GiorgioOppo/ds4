import Foundation

// Offline GGUF -> GGUF requantization: the read -> dequant -> requant -> write
// pass the upstream C project performs with `gguf-tools/deepseek4-quantize.c`
// (selective `--tensor-type`). Here it reuses the byte-exact `QuantEncode`
// encoders (previously test-only) and the `Quantize` dequantizers, wired to the
// `GGUFWriter`. Everything is pure Swift (no Metal), so it runs anywhere the
// toolchain does and is validated by round-trip tests.
//
// Scope: dense/2-D tensors whose inner dimension is a block multiple. Tensors we
// cannot dequantize or requantize are copied through byte-for-byte, so a run
// never corrupts a model — worst case it leaves a tensor at its original type and
// logs the reason.

public enum GGUFRequantizer {

    /// Dequantize `count` elements of a GGUF `type` at `src` into a fresh F32
    /// array, or nil for types this port cannot dequantize (e.g. iq2_*).
    public static func dequantizeToF32(type: UInt32, src: UnsafeRawPointer, count: Int) -> [Float]? {
        var out = [Float](repeating: 0, count: count)
        let ok: Bool = out.withUnsafeMutableBufferPointer { dst -> Bool in
            guard let d = dst.baseAddress else { return false }
            switch type {
            case 0: // f32
                memcpy(d, src, count * 4)
            case 1: // f16
                for i in 0..<count {
                    d[i] = Half.float(src.loadUnaligned(fromByteOffset: i * 2, as: UInt16.self))
                }
            case 8:  guard count % 32 == 0 else { return false }; Quantize.dequantQ8_0(src, count: count, into: d)
            case 10: guard count % 256 == 0 else { return false }; Quantize.dequantQ2_K(src, count: count, into: d)
            case 12: guard count % 256 == 0 else { return false }; Quantize.dequantQ4_K(src, count: count, into: d)
            case 13: guard count % 256 == 0 else { return false }; Quantize.dequantQ5_K(src, count: count, into: d)
            case 14: guard count % 256 == 0 else { return false }; Quantize.dequantQ6_K(src, count: count, into: d)
            default: return false
            }
            return true
        }
        return ok ? out : nil
    }

    public struct Options {
        /// Target GGUF type for a tensor. Return nil (or the tensor's current
        /// type) to copy it through unchanged.
        public var targetType: (GGUFModel.Tensor) -> UInt32?
        /// Per-column importance matrix for a tensor (length == inner dim), or
        /// nil. Required for iq2_xxs (type 16); optional quality aid for K-quants.
        public var imatrix: (GGUFModel.Tensor) -> [Float]?
        public var log: ((String) -> Void)?

        public init(targetType: @escaping (GGUFModel.Tensor) -> UInt32?,
                    imatrix: @escaping (GGUFModel.Tensor) -> [Float]? = { _ in nil },
                    log: ((String) -> Void)? = nil) {
            self.targetType = targetType
            self.imatrix = imatrix
            self.log = log
        }

        /// Simple policy: remap any tensor whose current type is a key of `map`
        /// to the mapped target, subject to `include(name)`. Others pass through.
        public static func remap(_ map: [UInt32: UInt32],
                                 include: @escaping (String) -> Bool = { _ in true },
                                 log: ((String) -> Void)? = nil) -> Options {
            Options(targetType: { t in include(t.name) ? map[t.type] : nil }, log: log)
        }
    }

    public struct Report: Sendable {
        public var requantized = 0
        public var passthrough = 0
        public var skipped = 0
        /// (tensor, reason) for tensors that could not be requantized as asked.
        public var skips: [(name: String, reason: String)] = []
    }

    /// True if this port can dequantize `count` elements of `type` to F32.
    public static func canDequantize(type: UInt32, count: Int) -> Bool {
        switch type {
        case 0, 1: return true                          // f32, f16
        case 8: return count % 32 == 0                  // q8_0
        case 10, 12, 13, 14: return count % 256 == 0    // q2_k, q4_k, q5_k, q6_k
        default: return false
        }
    }

    /// Requantize `source` into a new GGUF at `path`. Metadata is preserved in
    /// order; only tensor payloads change. Tensor bytes are produced lazily, so
    /// only one tensor is materialized at a time while writing (large models do
    /// not have to fit in memory).
    @discardableResult
    public static func requantize(source: GGUFModel, options: Options, to path: String) throws -> Report {
        var report = Report()
        var inputs: [GGUFWriter.TensorInput] = []
        inputs.reserveCapacity(source.tensors.count)

        for t in source.tensors {
            guard let want = options.targetType(t), want != t.type else {
                inputs.append(passthrough(t, source)); report.passthrough += 1; continue
            }
            switch plan(t, to: want, options: options) {
            case .requant(let byteCount, let imatrix, let columns, let rows):
                options.log?("requant \(t.name): \(t.typeName) -> \(GGUF.typeName(want)) (\(rows)x\(columns))")
                inputs.append(GGUFWriter.TensorInput(
                    name: t.name, dims: t.dims, type: want, byteCount: byteCount,
                    provider: { try requantBytes(t, to: want, source: source, imatrix: imatrix,
                                                 columns: columns, rows: rows, byteCount: byteCount) }))
                report.requantized += 1
            case .skip(let reason):
                options.log?("skip \(t.name): \(reason)")
                report.skips.append((t.name, reason))
                inputs.append(passthrough(t, source)); report.skipped += 1
            }
        }

        let writer = try GGUFWriter(metadata: try source.allMetadata(),
                                    tensors: inputs, alignment: source.alignment)
        try writer.write(to: path)
        return report
    }

    // MARK: - Internals

    private enum Plan {
        case requant(byteCount: Int, imatrix: [Float]?, columns: Int, rows: Int)
        case skip(String)
    }

    /// Lazy passthrough: copies the source tensor's bytes when written.
    private static func passthrough(_ t: GGUFModel.Tensor, _ source: GGUFModel) -> GGUFWriter.TensorInput {
        GGUFWriter.TensorInput(name: t.name, dims: t.dims, type: t.type,
                               byteCount: Int(t.bytes), provider: { source.tensorData(t) })
    }

    /// Decide the output type/size for a tensor WITHOUT dequantizing, so the
    /// directory can be written before any payload is produced.
    private static func plan(_ t: GGUFModel.Tensor, to want: UInt32, options: Options) -> Plan {
        guard QuantEncode.canQuantize(want) else { return .skip("target \(GGUF.typeName(want)) not encodable") }
        guard let info = GGUF.typeInfo(want) else { return .skip("unknown target type \(want)") }
        guard let firstDimension = t.dims.first,
              firstDimension <= UInt64(Int.max),
              t.elements <= UInt64(Int.max) else {
            return .skip("shape \(t.dims) is too large")
        }
        let columns = Int(firstDimension)
        let elements = Int(t.elements)
        guard columns > 0, columns % Int(info.blockElems) == 0, elements % columns == 0 else {
            return .skip("shape \(t.dims) not block-aligned for \(info.name)")
        }
        guard canDequantize(type: t.type, count: elements) else {
            return .skip("cannot dequantize source type \(t.typeName)")
        }
        let im = options.imatrix(t)
        if QuantEncode.requiresImatrix(want) && im == nil {
            return .skip("target \(info.name) requires an imatrix")
        }
        if let im, im.count != columns {
            return .skip("imatrix length \(im.count) != inner dim \(columns)")
        }
        let rows = elements / columns
        let rowBytes = QuantEncode.rowSize(type: want, columns: columns)
        let (byteCount, overflow) = rowBytes.multipliedReportingOverflow(
            by: rows)
        guard rowBytes > 0, !overflow else {
            return .skip("output allocation for shape \(t.dims) is too large")
        }
        return .requant(byteCount: byteCount, imatrix: im, columns: columns, rows: rows)
    }

    /// Produce the requantized bytes for one tensor (dequant -> requant).
    private static func requantBytes(_ t: GGUFModel.Tensor, to want: UInt32, source: GGUFModel,
                                     imatrix im: [Float]?, columns: Int, rows: Int,
                                     byteCount: Int) throws -> Data {
        let (elementCount, overflow) = rows.multipliedReportingOverflow(
            by: columns)
        guard !overflow,
              let f32 = dequantizeToF32(type: t.type,
                                        src: source.mapBase.advanced(by: Int(t.absOffset)),
                                        count: elementCount) else {
            throw GGUFError.message("requant: dequantization of \(t.name) failed")
        }
        var dst = Data(count: byteCount)
        let written: Int = dst.withUnsafeMutableBytes { dstRaw -> Int in
            guard let dstBase = dstRaw.baseAddress else { return 0 }
            return f32.withUnsafeBufferPointer { srcBuf -> Int in
                guard let srcBase = srcBuf.baseAddress else { return 0 }
                if let im {
                    return im.withUnsafeBufferPointer { imBuf in
                        QuantEncode.quantizeChunk(type: want, src: srcBase, dst: dstBase,
                                                  start: 0, rows: rows, columns: columns,
                                                  imatrix: imBuf.baseAddress)
                    }
                }
                return QuantEncode.quantizeChunk(type: want, src: srcBase, dst: dstBase,
                                                 start: 0, rows: rows, columns: columns,
                                                 imatrix: nil)
            }
        }
        guard written == byteCount else {
            throw GGUFError.message("requant: encoder wrote \(written)/\(byteCount) bytes for \(t.name)")
        }
        return dst
    }
}
