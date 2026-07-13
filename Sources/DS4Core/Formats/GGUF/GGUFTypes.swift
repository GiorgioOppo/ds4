import Foundation
#if canImport(Darwin)
import Darwin
#endif

// Faithful Swift port of the GGUF loader in ds4.c (model_open, parse_metadata,
// parse_tensors, the cursor reader, the tensor-type table, and the metadata
// accessors). The model file is mmap'd once and tensor bytes stay in place;
// callers reach weights by absolute offset into the mapping.
//
// Differences from C, none behavioral: where ds4.c calls ds4_die() (aborting the
// process) this throws GGUFError, and KV keys / tensor names are decoded to Swift
// String at parse time (lookups are identical). Phase 2 of the C->Swift port.

public enum GGUFValueType: UInt32, Sendable {
    case uint8 = 0, int8 = 1, uint16 = 2, int16 = 3, uint32 = 4, int32 = 5
    case float32 = 6, bool = 7, string = 8, array = 9
    case uint64 = 10, int64 = 11, float64 = 12
}

public enum GGUFError: Error, CustomStringConvertible {
    case cannotOpen(String)
    case mmapFailed(String)
    case tooSmall
    case notGGUF
    case unsupportedVersion(UInt32)
    case truncated(UInt64)
    case message(String)

    public var description: String {
        switch self {
        case .cannotOpen(let p): return "cannot open model: \(p)"
        case .mmapFailed(let p): return "cannot mmap model: \(p)"
        case .tooSmall: return "model file is too small to be GGUF"
        case .notGGUF: return "model is not a GGUF file"
        case .unsupportedVersion(let v): return "only GGUF v3 is supported (got \(v))"
        case .truncated(let at): return "truncated GGUF file at byte \(at)"
        case .message(let m): return m
        }
    }
}

/// GGUF tensor-type descriptor (matches gguf_type_info / gguf_types[] in ds4.c).
public struct GGUFTypeInfo: Sendable {
    public let name: String
    public let blockElems: UInt32
    public let blockBytes: UInt32
}

public enum GGUF {
    public static let magic: UInt32 = 0x4655_4747 // "GGUF" little-endian
    public static let maxDims = 8

    /// The sparse type table from ds4.c (indices without an entry are unknown).
    public static let typeTable: [UInt32: GGUFTypeInfo] = [
        0:  .init(name: "f32",     blockElems: 1,   blockBytes: 4),
        1:  .init(name: "f16",     blockElems: 1,   blockBytes: 2),
        2:  .init(name: "q4_0",    blockElems: 32,  blockBytes: 18),
        3:  .init(name: "q4_1",    blockElems: 32,  blockBytes: 20),
        6:  .init(name: "q5_0",    blockElems: 32,  blockBytes: 22),
        7:  .init(name: "q5_1",    blockElems: 32,  blockBytes: 24),
        8:  .init(name: "q8_0",    blockElems: 32,  blockBytes: 34),
        9:  .init(name: "q8_1",    blockElems: 32,  blockBytes: 40),
        10: .init(name: "q2_k",    blockElems: 256, blockBytes: 84),
        11: .init(name: "q3_k",    blockElems: 256, blockBytes: 110),
        12: .init(name: "q4_k",    blockElems: 256, blockBytes: 144),
        13: .init(name: "q5_k",    blockElems: 256, blockBytes: 176),
        14: .init(name: "q6_k",    blockElems: 256, blockBytes: 210),
        15: .init(name: "q8_k",    blockElems: 256, blockBytes: 292),
        16: .init(name: "iq2_xxs", blockElems: 256, blockBytes: 66),
        17: .init(name: "iq2_xs",  blockElems: 256, blockBytes: 74),
        18: .init(name: "iq3_xxs", blockElems: 256, blockBytes: 98),
        19: .init(name: "iq1_s",   blockElems: 256, blockBytes: 110),
        20: .init(name: "iq4_nl",  blockElems: 256, blockBytes: 50),
        21: .init(name: "iq3_s",   blockElems: 256, blockBytes: 110),
        22: .init(name: "iq2_s",   blockElems: 256, blockBytes: 82),
        23: .init(name: "iq4_xs",  blockElems: 256, blockBytes: 136),
        24: .init(name: "i8",      blockElems: 1,   blockBytes: 1),
        25: .init(name: "i16",     blockElems: 1,   blockBytes: 2),
        26: .init(name: "i32",     blockElems: 1,   blockBytes: 4),
        27: .init(name: "i64",     blockElems: 1,   blockBytes: 8),
        28: .init(name: "f64",     blockElems: 1,   blockBytes: 8),
        29: .init(name: "iq1_m",   blockElems: 256, blockBytes: 56),
        30: .init(name: "bf16",    blockElems: 1,   blockBytes: 2),
    ]

    public static func typeInfo(_ type: UInt32) -> GGUFTypeInfo? { typeTable[type] }
    public static func typeName(_ type: UInt32) -> String { typeTable[type]?.name ?? "unknown" }

    /// Port of tensor_nbytes: block-rounded byte size, or nil for unknown types.
    public static func tensorNBytes(type: UInt32, elements: UInt64) -> UInt64? {
        guard let info = typeTable[type], info.blockElems != 0 else { return nil }
        let be = UInt64(info.blockElems)
        let bb = UInt64(info.blockBytes)
        let blocks = (elements + be - 1) / be
        guard blocks <= UInt64.max / bb else { return nil }
        return blocks * bb
    }

    static func alignUp(_ value: UInt64, _ alignment: UInt64) -> UInt64 {
        let rem = value % alignment
        return rem == 0 ? value : value + alignment - rem
    }
}

