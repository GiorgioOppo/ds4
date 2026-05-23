import Foundation
import Metal

/// Inspect a tensor slice by name. Loads the tensor through the
/// `WeightLoader` (driving `ensureLayer` for streaming-pool layers)
/// and dequantizes the requested `[row, cols)` window to `[Float]`
/// using the same scale convention the GEMM kernels use:
///
///   • i4 / i2 / i8 weights expect a `<base>.scale` companion in F16,
///     stored as `[N, K/128]` (per row × per-128-block, symmetric).
///   • i4 packs two nibbles per byte: low = col 2k, high = col 2k+1.
///   • i2 packs four 2-bit codes per byte, LSB-first within each byte.
///   • f32 / f16 / bf16 weights are read directly with no scale.
///
/// Intended for diagnostics: compare against a Python reference dump
/// of the same row when the model produces uniform-looking noise to
/// localize whether the bug is in quantization metadata
/// (block size, nibble order, transpose, dtype mismatch) or
/// elsewhere in the forward pass.
public enum TensorDump {
    public struct Result {
        public let name: String
        public let shape: [Int]
        public let dtype: DType
        public let row: Int
        public let cols: Range<Int>
        public let values: [Float]
        /// Name of the scale companion that was loaded, if applicable.
        public let scaleName: String?
    }

    public enum DumpError: Swift.Error, CustomStringConvertible {
        case notFound(String, candidates: [String])
        case missingScale(weight: String, scale: String)
        case rankUnsupported(name: String, rank: Int)
        case rowOutOfRange(Int, count: Int)
        case colsOutOfRange(Range<Int>, count: Int)
        case dtypeUnsupported(DType)
        case blockSizeMismatch(rowLen: Int, blockSize: Int)

        public var description: String {
            switch self {
            case .notFound(let n, let cands):
                if cands.isEmpty {
                    return "tensor not found in checkpoint: \(n)"
                }
                let list = cands.prefix(15).joined(separator: "\n  ")
                return """
                tensor not found in checkpoint: \(n)
                did you mean (closest \(min(cands.count, 15)) of \(cands.count)):
                  \(list)
                """
            case .missingScale(let w, let s):
                return "\(w) is quantized (needs scale companion \(s)) but \(s) is missing"
            case .rankUnsupported(let n, let r):
                return "tensor \(n) has rank \(r); TensorDump supports rank 1 or 2"
            case .rowOutOfRange(let r, let c):
                return "row \(r) out of range 0..<\(c)"
            case .colsOutOfRange(let r, let c):
                return "cols \(r.lowerBound)..\(r.upperBound) out of range 0..<\(c)"
            case .dtypeUnsupported(let d):
                return "TensorDump does not yet support dtype \(d)"
            case .blockSizeMismatch(let r, let b):
                return "row length \(r) is not a multiple of quant block size \(b)"
            }
        }
    }

    /// Rank candidate names by a coarse similarity to `target`: prefer
    /// names that share the longest common prefix, break ties by the
    /// Levenshtein-lite difference. Cheap; we don't expect O(N²)
    /// behavior to matter for the ~70k-tensor checkpoints we ship.
    public static func candidates(for target: String,
                                    among names: [String],
                                    limit: Int = 15) -> [String] {
        let scored = names.map { name -> (String, Int) in
            var prefix = 0
            for (a, b) in zip(name, target) {
                if a == b { prefix += 1 } else { break }
            }
            // Higher prefix = better. Lower length-diff = better.
            // Pack so sort key is descending by prefix, ascending by diff.
            let score = -prefix * 1000 + abs(name.count - target.count)
            return (name, score)
        }
        return scored.sorted { $0.1 < $1.1 }.prefix(limit).map(\.0)
    }

    /// Block size baked into the quant kernels. Mirrors `kInt4GroupK`
    /// / `kInt2GroupK` / INT8's group constant. All three are 128 by
    /// construction so we hard-code it here.
    private static let quantBlock = 128

    /// Dequantize a single row slice.
    ///
    /// - Parameter cols: if `nil`, defaults to `0..<min(32, rowLen)`.
    public static func dumpRow(_ name: String,
                                row: Int,
                                cols: Range<Int>?,
                                loader: WeightLoader) throws -> Result {
        if let layer = layerIndex(in: name) {
            loader.ensureLayer(layer)
        }
        guard let w = try loader.load(name) else {
            let suggestions = candidates(for: name, among: loader.allKnownNames)
            throw DumpError.notFound(name, candidates: suggestions)
        }

        let rows: Int, rowLen: Int
        switch w.shape.count {
        case 2: rows = w.shape[0]; rowLen = w.shape[1]
        case 1: rows = 1;          rowLen = w.shape[0]
        default: throw DumpError.rankUnsupported(name: name, rank: w.shape.count)
        }
        guard row >= 0, row < rows else {
            throw DumpError.rowOutOfRange(row, count: rows)
        }
        let range = cols ?? 0..<min(32, rowLen)
        guard range.lowerBound >= 0, range.upperBound <= rowLen,
              range.lowerBound <= range.upperBound else {
            throw DumpError.colsOutOfRange(range, count: rowLen)
        }

        // Resolve scale companion when the weight is quantized.
        var scale: Tensor? = nil
        var scaleName: String? = nil
        let needsScale: Bool
        switch w.dtype {
        case .i8, .i4, .i2, .fp8E4M3, .fp4E2M1: needsScale = true
        default: needsScale = false
        }
        if needsScale {
            let base: String = name.hasSuffix(".weight")
                ? String(name.dropLast(".weight".count))
                : name
            let sName = "\(base).scale"
            guard let s = try loader.load(sName) else {
                throw DumpError.missingScale(weight: name, scale: sName)
            }
            scale = s
            scaleName = sName
        }

        let values = try dequantRow(w: w, row: row, cols: range,
                                     rowLen: rowLen, scale: scale,
                                     name: name)
        return Result(name: name, shape: w.shape, dtype: w.dtype,
                       row: row, cols: range, values: values,
                       scaleName: scaleName)
    }

    /// Extract `K` from `"layers.K.<rest>"`. Returns nil for top-level
    /// tensors (embed/head/norm/etc.) which live in the shared pool
    /// slot and don't need `ensureLayer`.
    private static func layerIndex(in name: String) -> Int? {
        guard name.hasPrefix("layers.") else { return nil }
        let rest = name.dropFirst("layers.".count)
        guard let dot = rest.firstIndex(of: ".") else { return nil }
        return Int(rest[rest.startIndex..<dot])
    }

    private static func dequantRow(w: Tensor, row: Int, cols: Range<Int>,
                                    rowLen: Int, scale: Tensor?,
                                    name: String) throws -> [Float] {
        // Disabled for MLX migration
        return []
    }

    @inline(__always)
    private static func f16ToFloat(_ h: UInt16) -> Float {
        let sign = UInt32(h >> 15) & 0x1
        let exp  = UInt32(h >> 10) & 0x1F
        let mant = UInt32(h) & 0x3FF
        let f: UInt32
        if exp == 0 {
            if mant == 0 {
                f = sign << 31
            } else {
                var e: UInt32 = 1
                var m = mant
                while m & 0x400 == 0 { m <<= 1; e &+= 1 }
                m &= 0x3FF
                f = (sign << 31) | (((127 &- 15 &- e &+ 1) & 0xFF) << 23) | (m << 13)
            }
        } else if exp == 0x1F {
            f = (sign << 31) | (0xFF << 23) | (mant << 13)
        } else {
            f = (sign << 31) | ((exp &+ 127 &- 15) << 23) | (mant << 13)
        }
        return Float(bitPattern: f)
    }
}

/// Parse `NAME[:row=R][:cols=A..B]` into the three components.
/// Returns nil if `spec` doesn't start with a tensor name; throws on
/// malformed `row=` / `cols=` fragments.
public enum TensorDumpSpec {
    public struct Parsed {
        public let name: String
        public let row: Int
        public let cols: Range<Int>?
    }

    public enum ParseError: Swift.Error, CustomStringConvertible {
        case empty
        case badRow(String)
        case badCols(String)
        case unknownFragment(String)

        public var description: String {
            switch self {
            case .empty:              return "tensor spec is empty"
            case .badRow(let s):      return "could not parse `row=` fragment: \(s)"
            case .badCols(let s):     return "could not parse `cols=A..B` fragment: \(s)"
            case .unknownFragment(let s): return "unknown fragment in tensor spec: \(s)"
            }
        }
    }

    public static func parse(_ spec: String) throws -> Parsed {
        let parts = spec.split(separator: ":", omittingEmptySubsequences: false)
            .map(String.init)
        guard let first = parts.first, !first.isEmpty else {
            throw ParseError.empty
        }
        var row = 0
        var cols: Range<Int>? = nil
        for frag in parts.dropFirst() {
            if frag.hasPrefix("row=") {
                let rest = String(frag.dropFirst("row=".count))
                guard let r = Int(rest) else { throw ParseError.badRow(frag) }
                row = r
            } else if frag.hasPrefix("cols=") {
                let rest = String(frag.dropFirst("cols=".count))
                let bounds = rest.components(separatedBy: "..")
                guard bounds.count == 2,
                      let lo = Int(bounds[0]),
                      let hi = Int(bounds[1]),
                      lo <= hi else {
                    throw ParseError.badCols(frag)
                }
                cols = lo..<hi
            } else if !frag.isEmpty {
                throw ParseError.unknownFragment(frag)
            }
        }
        return Parsed(name: first, row: row, cols: cols)
    }
}
