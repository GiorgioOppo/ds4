import Foundation

/// GGUF file format primitives. This module parses the header and
/// tensor table of a GGUF v3 file as produced by `llama.cpp` and its
/// converter scripts. It does NOT decode the tensor data itself —
/// callers receive offsets into the underlying buffer and a
/// `GGUFType`, then route to a per-dtype dequant kernel (Q4_0 / Q4_K /
/// Q8_0 — deferred to a follow-up) or pass-through for F32/F16/BF16.
///
/// Reference: https://github.com/ggml-org/ggml/blob/master/docs/gguf.md
///
/// Memory model: the parser reads strictly from the leading metadata
/// region; it does not touch the tensor data segment at parse time.
/// `GGUFLoader` mmap's the whole file and constructs `Tensor`s that
/// share that mapping (zero-copy on Apple Silicon's unified memory),
/// the same trick `SafeTensorsFile` uses.

// MARK: - Constants and types

/// ggml_type values used in the GGUF tensor table.
/// Values match `ggml.h:GGML_TYPE_*` and must not be renumbered.
public enum GGUFType: Int32, Sendable {
    case f32   = 0
    case f16   = 1
    case q4_0  = 2
    case q4_1  = 3
    case q5_0  = 6
    case q5_1  = 7
    case q8_0  = 8
    case q8_1  = 9
    case q2_K  = 10
    case q3_K  = 11
    case q4_K  = 12
    case q5_K  = 13
    case q6_K  = 14
    case q8_K  = 15
    case iq2_xxs = 16
    case iq2_xs  = 17
    case iq3_xxs = 18
    case iq1_s   = 19
    case iq4_nl  = 20
    case iq3_s   = 21
    case iq2_s   = 22
    case iq4_xs  = 23
    case i8      = 24
    case i16     = 25
    case i32     = 26
    case i64     = 27
    case f64     = 28
    case iq1_m   = 29
    case bf16    = 30

    /// Bytes per logical element (or per block element for quantized
    /// types). The block size and bytes-per-block are returned by
    /// `blockSize` / `bytesPerBlock` separately so callers can size
    /// output buffers without per-type switches.
    public var bytesPerBlock: Int {
        switch self {
        case .f32:                          return 4
        case .f16, .bf16:                   return 2
        case .f64, .i64:                    return 8
        case .i32:                          return 4
        case .i16:                          return 2
        case .i8:                           return 1
        case .q4_0:                         return 18   // 1×F16 scale + 32×4bit = 16+2
        case .q4_1:                         return 20   // F16 d + F16 m + 16
        case .q5_0:                         return 22   // F16 + 4 hi bits + 16
        case .q5_1:                         return 24
        case .q8_0:                         return 34   // F16 scale + 32×8bit
        case .q8_1:                         return 36
        case .q2_K:                         return 84
        case .q3_K:                         return 110
        case .q4_K:                         return 144
        case .q5_K:                         return 176
        case .q6_K:                         return 210
        case .q8_K:                         return 292
        // Newer IQ types — sized from llama.cpp src/ggml-quants.h.
        case .iq2_xxs:                      return 66
        case .iq2_xs:                       return 74
        case .iq3_xxs:                      return 98
        case .iq1_s, .iq1_m:                return 50
        case .iq4_nl:                       return 18
        case .iq3_s:                        return 110
        case .iq2_s:                        return 82
        case .iq4_xs:                       return 136
        }
    }

    /// Number of logical (dequantized) elements per stored block.
    public var blockSize: Int {
        switch self {
        case .f32, .f16, .bf16, .f64, .i64, .i32, .i16, .i8:
                                            return 1
        case .q4_0, .q4_1, .q5_0, .q5_1, .q8_0, .q8_1, .iq4_nl:
                                            return 32
        case .q2_K, .q3_K, .q4_K, .q5_K, .q6_K, .q8_K,
             .iq2_xxs, .iq2_xs, .iq3_xxs, .iq1_s, .iq3_s, .iq2_s,
             .iq4_xs, .iq1_m:
                                            return 256
        }
    }

    /// True iff this type can be returned to a caller as a `Tensor` of
    /// the equivalent native `DType` with no per-block dequantisation.
    public var isPassThroughDType: Bool {
        switch self {
        case .f32, .f16, .bf16, .i32, .i8:  return true
        default:                            return false
        }
    }
}

/// ggml metadata value types.
public enum GGUFValueType: UInt32, Sendable {
    case uint8   = 0
    case int8    = 1
    case uint16  = 2
    case int16   = 3
    case uint32  = 4
    case int32   = 5
    case float32 = 6
    case bool    = 7
    case string  = 8
    case array   = 9
    case uint64  = 10
    case int64   = 11
    case float64 = 12
}

/// One metadata key/value pair from the GGUF header.
public enum GGUFValue: Sendable {
    case uint64(UInt64)
    case int64(Int64)
    case float64(Double)
    case bool(Bool)
    case string(String)
    case array([GGUFValue])
}

/// One row of the tensor info table.
public struct GGUFTensorInfo: Sendable {
    public let name: String
    public let shape: [Int]        // row-major
    public let type: GGUFType
    /// Absolute byte offset in the file where the tensor data begins
    /// (i.e. `tensor_data_offset + relative_offset`).
    public let absoluteOffset: Int
    /// Total byte count for this tensor's data, computed from shape
    /// and type. Independent of the next-tensor padding.
    public let byteCount: Int
}

/// Errors emitted by the parser.
public enum GGUFError: Error, CustomStringConvertible {
    case badMagic(String)
    case unsupportedVersion(UInt32)
    case malformed(String)
    case unsupportedType(Int32)
    case io(String)

    public var description: String {
        switch self {
        case .badMagic(let s):        return "GGUF: bad magic '\(s)' (expected 'GGUF')"
        case .unsupportedVersion(let v): return "GGUF: unsupported version \(v) (need 2 or 3)"
        case .malformed(let m):       return "GGUF: malformed — \(m)"
        case .unsupportedType(let t): return "GGUF: unsupported ggml_type \(t)"
        case .io(let m):              return "GGUF: i/o error — \(m)"
        }
    }
}

// MARK: - Parser

/// Parses the GGUF metadata header from a byte buffer. Pure value
/// type, holds no resources. Caller is responsible for mapping the
/// file (or reading it into memory) before invoking `parse`.
public struct GGUFHeader: Sendable {
    public let version: UInt32
    public let metadata: [String: GGUFValue]
    public let tensors: [GGUFTensorInfo]
    /// Byte offset where the tensor data segment begins. Tensor
    /// `absoluteOffset` values are computed against this.
    public let tensorDataOffset: Int
    /// Alignment of the tensor data segment, from
    /// `general.alignment` metadata (default 32).
    public let alignment: Int

    public static func parse(buffer: UnsafeRawBufferPointer) throws -> GGUFHeader {
        var reader = ByteReader(buffer: buffer)
        guard let magicBytes = reader.read(count: 4) else {
            throw GGUFError.malformed("file too short for magic")
        }
        let magic = String(decoding: magicBytes, as: UTF8.self)
        guard magic == "GGUF" else { throw GGUFError.badMagic(magic) }

        let version: UInt32 = try reader.readLE()
        guard version == 2 || version == 3 else {
            throw GGUFError.unsupportedVersion(version)
        }
        let tensorCount: UInt64 = try reader.readLE()
        let kvCount: UInt64 = try reader.readLE()

        // Metadata KV pairs.
        var metadata: [String: GGUFValue] = [:]
        metadata.reserveCapacity(Int(kvCount))
        for _ in 0..<Int(kvCount) {
            let key = try reader.readGGUFString()
            let typeRaw: UInt32 = try reader.readLE()
            guard let type = GGUFValueType(rawValue: typeRaw) else {
                throw GGUFError.malformed("unknown KV value type \(typeRaw) for key '\(key)'")
            }
            let value = try reader.readGGUFValue(type: type)
            metadata[key] = value
        }
        let alignment: Int = {
            if case .uint64(let v)? = metadata["general.alignment"] { return Int(v) }
            if case .int64(let v)?  = metadata["general.alignment"] { return Int(v) }
            return 32
        }()

        // Tensor info table (offsets here are RELATIVE to
        // `tensorDataOffset`, which sits right after this section
        // padded to `alignment`).
        var rawInfos: [(name: String, shape: [Int], type: GGUFType, relOffset: Int)] = []
        rawInfos.reserveCapacity(Int(tensorCount))
        for _ in 0..<Int(tensorCount) {
            let name = try reader.readGGUFString()
            let nDims: UInt32 = try reader.readLE()
            var shape: [Int] = []
            shape.reserveCapacity(Int(nDims))
            for _ in 0..<Int(nDims) {
                let d: UInt64 = try reader.readLE()
                shape.append(Int(d))
            }
            let typeRaw: Int32 = try reader.readLE()
            guard let type = GGUFType(rawValue: typeRaw) else {
                throw GGUFError.unsupportedType(typeRaw)
            }
            let off: UInt64 = try reader.readLE()
            rawInfos.append((name, shape, type, Int(off)))
        }

        // Pad to alignment.
        let metadataEnd = reader.position
        let pad = (alignment - (metadataEnd % alignment)) % alignment
        let tensorDataOffset = metadataEnd + pad

        // Compute absolute offsets and byte counts.
        var tensors: [GGUFTensorInfo] = []
        tensors.reserveCapacity(rawInfos.count)
        for info in rawInfos {
            let elementCount = info.shape.reduce(1, *)
            let blocks = (elementCount + info.type.blockSize - 1) / info.type.blockSize
            let bytes = blocks * info.type.bytesPerBlock
            tensors.append(GGUFTensorInfo(
                name: info.name,
                shape: info.shape,
                type: info.type,
                absoluteOffset: tensorDataOffset + info.relOffset,
                byteCount: bytes))
        }

        return GGUFHeader(version: version,
                          metadata: metadata,
                          tensors: tensors,
                          tensorDataOffset: tensorDataOffset,
                          alignment: alignment)
    }
}

// MARK: - Byte reader

private struct ByteReader {
    let buffer: UnsafeRawBufferPointer
    var position: Int = 0

    init(buffer: UnsafeRawBufferPointer) { self.buffer = buffer }

    mutating func read(count: Int) -> UnsafeRawBufferPointer? {
        guard position + count <= buffer.count else { return nil }
        let slice = UnsafeRawBufferPointer(rebasing: buffer[position..<(position + count)])
        position += count
        return slice
    }

    mutating func readLE<T: FixedWidthInteger>() throws -> T {
        let size = MemoryLayout<T>.size
        guard let slice = read(count: size) else {
            throw GGUFError.malformed("read past EOF for \(T.self)")
        }
        var value: T = 0
        for i in 0..<size {
            value |= T(slice[i]) << T(i * 8)
        }
        return value
    }

    mutating func readDouble() throws -> Double {
        let bits: UInt64 = try readLE()
        return Double(bitPattern: bits)
    }

    mutating func readFloat() throws -> Float {
        let bits: UInt32 = try readLE()
        return Float(bitPattern: bits)
    }

    mutating func readGGUFString() throws -> String {
        let len: UInt64 = try readLE()
        guard let slice = read(count: Int(len)) else {
            throw GGUFError.malformed("string truncated (len \(len))")
        }
        return String(decoding: slice, as: UTF8.self)
    }

    mutating func readGGUFValue(type: GGUFValueType) throws -> GGUFValue {
        switch type {
        case .uint8:    let v: UInt8  = try readLE(); return .uint64(UInt64(v))
        case .int8:     let v: Int8   = try readLE(); return .int64(Int64(v))
        case .uint16:   let v: UInt16 = try readLE(); return .uint64(UInt64(v))
        case .int16:    let v: Int16  = try readLE(); return .int64(Int64(v))
        case .uint32:   let v: UInt32 = try readLE(); return .uint64(UInt64(v))
        case .int32:    let v: Int32  = try readLE(); return .int64(Int64(v))
        case .uint64:   let v: UInt64 = try readLE(); return .uint64(v)
        case .int64:    let v: Int64  = try readLE(); return .int64(v)
        case .float32:  return .float64(Double(try readFloat()))
        case .float64:  return .float64(try readDouble())
        case .bool:
            let v: UInt8 = try readLE()
            return .bool(v != 0)
        case .string:   return .string(try readGGUFString())
        case .array:
            let elemTypeRaw: UInt32 = try readLE()
            guard let elemType = GGUFValueType(rawValue: elemTypeRaw) else {
                throw GGUFError.malformed("unknown array element type \(elemTypeRaw)")
            }
            let count: UInt64 = try readLE()
            var values: [GGUFValue] = []
            values.reserveCapacity(Int(count))
            for _ in 0..<Int(count) {
                values.append(try readGGUFValue(type: elemType))
            }
            return .array(values)
        }
    }
}
