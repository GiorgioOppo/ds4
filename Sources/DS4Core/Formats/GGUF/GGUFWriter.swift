import Foundation

// GGUF serialization (writer), the inverse of GGUFModel/GGUFCursor. The upstream
// C project generates GGUFs offline with `gguf-tools/deepseek4-quantize.c`; the
// Swift port could previously only READ GGUFs. This adds a faithful writer so the
// same v3 layout the reader parses can be produced in-process: header, typed
// metadata KVs, the tensor directory, and the aligned tensor-data section.
//
// Byte layout (little-endian, matching GGUFCursor):
//   header : magic u32 ("GGUF"), version u32 (=3), n_tensors u64, n_kv u64
//   kv[]   : key (u64 len + bytes), value-type u32, value payload
//   tinfo[]: name (u64 len + bytes), n_dims u32, dims u64[n_dims], type u32,
//            rel-offset u64  (offset into the data section)
//   <pad to `alignment`>
//   data[] : each tensor padded up to `alignment`, raw block bytes

/// A typed GGUF metadata value. Strings are kept as raw bytes so tokenizer
/// tables round-trip byte-for-byte (the reader also keys them by exact bytes).
public indirect enum GGUFMetadataValue: Sendable {
    case uint8(UInt8)
    case int8(Int8)
    case uint16(UInt16)
    case int16(Int16)
    case uint32(UInt32)
    case int32(Int32)
    case uint64(UInt64)
    case int64(Int64)
    case float32(Float)
    case float64(Double)
    case bool(Bool)
    case string([UInt8])
    /// A homogeneous array. `elementType` is the GGUF value type of every element;
    /// each element's own case must match it.
    case array(elementType: GGUFValueType, elements: [GGUFMetadataValue])

    /// Convenience: a UTF-8 string value (named to avoid clashing with the
    /// `string([UInt8])` case).
    public static func text(_ s: String) -> GGUFMetadataValue { .string(Array(s.utf8)) }

    /// The GGUF value-type code written before the payload.
    public var valueType: GGUFValueType {
        switch self {
        case .uint8:   return .uint8
        case .int8:    return .int8
        case .uint16:  return .uint16
        case .int16:   return .int16
        case .uint32:  return .uint32
        case .int32:   return .int32
        case .uint64:  return .uint64
        case .int64:   return .int64
        case .float32: return .float32
        case .float64: return .float64
        case .bool:    return .bool
        case .string:  return .string
        case .array:   return .array
        }
    }
}

public enum GGUFWriterError: Error, CustomStringConvertible {
    case invalidAlignment(UInt64)
    case emptyDims(String)
    case tooManyDims(String, Int)
    case unknownTensorType(String, UInt32)
    case tensorDataSizeMismatch(name: String, expected: UInt64, got: UInt64)
    case arrayElementTypeMismatch(expected: GGUFValueType, got: GGUFValueType)
    case cannotCreate(String)
    case writeFailed(String)

    public var description: String {
        switch self {
        case .invalidAlignment(let a): return "invalid GGUF alignment \(a)"
        case .emptyDims(let n): return "tensor \(n) has no dimensions"
        case .tooManyDims(let n, let d): return "tensor \(n) has \(d) dimensions (max \(GGUF.maxDims))"
        case .unknownTensorType(let n, let t): return "tensor \(n) has unknown GGUF type \(t)"
        case .tensorDataSizeMismatch(let n, let e, let g):
            return "tensor \(n) data is \(g) bytes, expected \(e) for its type/shape"
        case .arrayElementTypeMismatch(let e, let g):
            return "GGUF array element type \(g) does not match declared \(e)"
        case .cannotCreate(let p): return "cannot create GGUF file: \(p)"
        case .writeFailed(let p): return "failed writing GGUF file: \(p)"
        }
    }
}

/// Builds a GGUF v3 file from ordered metadata and tensors. Metadata and tensor
/// order are preserved exactly (GGUF is order-sensitive for reproducibility).
public struct GGUFWriter {

    /// One tensor to serialize. The payload is the raw, already-quantized block
    /// bytes whose length must equal `GGUF.tensorNBytes(type:, elements:)`.
    ///
    /// Bytes are produced lazily via `provider` so `write(to:)` can materialize
    /// one tensor at a time (a multi-GB model never has to fit in memory at
    /// once). Use the `data:` initializer for eager bytes.
    public struct TensorInput {
        public let name: String
        public let dims: [UInt64]
        public let type: UInt32
        /// Declared payload size; validated against the type/shape and against
        /// the provider's actual output.
        public let byteCount: Int
        let provider: () throws -> Data

        /// Eager: the bytes are already in hand.
        public init(name: String, dims: [UInt64], type: UInt32, data: Data) {
            self.name = name; self.dims = dims; self.type = type
            self.byteCount = data.count
            self.provider = { data }
        }

        /// Lazy: `provider` yields the bytes when the tensor is written. Its
        /// output length must equal `byteCount`.
        public init(name: String, dims: [UInt64], type: UInt32,
                    byteCount: Int, provider: @escaping () throws -> Data) {
            self.name = name; self.dims = dims; self.type = type
            self.byteCount = byteCount; self.provider = provider
        }

        var elements: UInt64 { dims.reduce(1, *) }
    }

    public private(set) var metadata: [(key: String, value: GGUFMetadataValue)]
    public private(set) var tensors: [TensorInput]
    public let alignment: UInt64

    /// `alignment` defaults to 32 (the GGUF default). If the metadata carries a
    /// `general.alignment` uint32 it takes precedence, so the written file and
    /// its declared alignment never disagree.
    public init(metadata: [(key: String, value: GGUFMetadataValue)] = [],
                tensors: [TensorInput] = [],
                alignment: UInt64 = 32) throws {
        var effective = alignment
        for (k, v) in metadata where k == "general.alignment" {
            if case .uint32(let a) = v, a != 0 { effective = UInt64(a) }
        }
        guard effective != 0 else { throw GGUFWriterError.invalidAlignment(effective) }
        self.metadata = metadata
        self.tensors = tensors
        self.alignment = effective
    }

    public mutating func put(_ key: String, _ value: GGUFMetadataValue) {
        metadata.append((key, value))
    }

    public mutating func add(_ tensor: TensorInput) {
        tensors.append(tensor)
    }

    // MARK: - Serialization

    /// Serialize everything into a single in-memory `Data`. Convenient for tests
    /// and small files; for multi-GB models prefer `write(to:)`, which streams the
    /// tensor payloads one at a time.
    public func build() throws -> Data {
        var out = Data()
        try appendHeaderAndDirectory(into: &out)
        let dataPos = GGUF.alignUp(UInt64(out.count), alignment)
        pad(&out, to: dataPos)
        let (relOffsets, _) = try layoutTensorData()
        for (i, t) in tensors.enumerated() {
            pad(&out, to: dataPos + relOffsets[i])
            out.append(try producedData(t))
        }
        return out
    }

    /// Stream the file to `path`: the small header/directory prefix is built in
    /// memory, then each tensor's bytes are written in turn (kept out of a single
    /// giant buffer).
    public func write(to path: String) throws {
        var prefix = Data()
        try appendHeaderAndDirectory(into: &prefix)
        let dataPos = GGUF.alignUp(UInt64(prefix.count), alignment)
        pad(&prefix, to: dataPos)
        let (relOffsets, _) = try layoutTensorData()

        FileManager.default.createFile(atPath: path, contents: nil)
        guard let fh = FileHandle(forWritingAtPath: path) else {
            throw GGUFWriterError.cannotCreate(path)
        }
        defer { try? fh.close() }
        do {
            try fh.write(contentsOf: prefix)
            var written = UInt64(prefix.count)
            for (i, t) in tensors.enumerated() {
                let target = dataPos + relOffsets[i]
                if target > written {
                    try fh.write(contentsOf: Data(count: Int(target - written)))
                    written = target
                }
                let bytes = try producedData(t)
                try fh.write(contentsOf: bytes)
                written += UInt64(bytes.count)
            }
        } catch {
            throw GGUFWriterError.writeFailed("\(path) — \(error)")
        }
    }

    // MARK: - Internals

    private func appendHeaderAndDirectory(into out: inout Data) throws {
        // Header
        appendU32(&out, GGUF.magic)
        appendU32(&out, 3)                                   // version
        appendU64(&out, UInt64(tensors.count))
        appendU64(&out, UInt64(metadata.count))

        // Metadata KVs
        for (key, value) in metadata {
            appendString(&out, Array(key.utf8))
            appendU32(&out, value.valueType.rawValue)
            try appendPayload(&out, value)
        }

        // Tensor directory
        let (relOffsets, _) = try layoutTensorData()
        for (i, t) in tensors.enumerated() {
            guard !t.dims.isEmpty else { throw GGUFWriterError.emptyDims(t.name) }
            guard t.dims.count <= GGUF.maxDims else {
                throw GGUFWriterError.tooManyDims(t.name, t.dims.count)
            }
            guard let expected = GGUF.tensorNBytes(type: t.type, elements: t.elements) else {
                throw GGUFWriterError.unknownTensorType(t.name, t.type)
            }
            guard UInt64(t.byteCount) == expected else {
                throw GGUFWriterError.tensorDataSizeMismatch(
                    name: t.name, expected: expected, got: UInt64(t.byteCount))
            }
            appendString(&out, Array(t.name.utf8))
            appendU32(&out, UInt32(t.dims.count))
            for d in t.dims { appendU64(&out, d) }
            appendU32(&out, t.type)
            appendU64(&out, relOffsets[i])
        }
    }

    /// Per-tensor relative offsets (each aligned to `alignment`) and the total
    /// data-section byte size. Also validates each tensor's byte count.
    private func layoutTensorData() throws -> (relOffsets: [UInt64], total: UInt64) {
        var relOffsets: [UInt64] = []
        relOffsets.reserveCapacity(tensors.count)
        var off: UInt64 = 0
        for t in tensors {
            guard let expected = GGUF.tensorNBytes(type: t.type, elements: t.elements) else {
                throw GGUFWriterError.unknownTensorType(t.name, t.type)
            }
            guard UInt64(t.byteCount) == expected else {
                throw GGUFWriterError.tensorDataSizeMismatch(
                    name: t.name, expected: expected, got: UInt64(t.byteCount))
            }
            off = GGUF.alignUp(off, alignment)
            relOffsets.append(off)
            off += expected
        }
        return (relOffsets, off)
    }

    /// Fetch a tensor's bytes from its provider and verify the length matches
    /// the declared `byteCount` (the directory was already written with it).
    private func producedData(_ t: TensorInput) throws -> Data {
        let d = try t.provider()
        guard d.count == t.byteCount else {
            throw GGUFWriterError.tensorDataSizeMismatch(
                name: t.name, expected: UInt64(t.byteCount), got: UInt64(d.count))
        }
        return d
    }

    private func appendPayload(_ out: inout Data, _ value: GGUFMetadataValue) throws {
        switch value {
        case .uint8(let v):   out.append(v)
        case .int8(let v):    appendLE(&out, UInt8(bitPattern: v))
        case .uint16(let v):  appendLE(&out, v)
        case .int16(let v):   appendLE(&out, UInt16(bitPattern: v))
        case .uint32(let v):  appendU32(&out, v)
        case .int32(let v):   appendU32(&out, UInt32(bitPattern: v))
        case .uint64(let v):  appendU64(&out, v)
        case .int64(let v):   appendU64(&out, UInt64(bitPattern: v))
        case .float32(let v): appendU32(&out, v.bitPattern)
        case .float64(let v): appendU64(&out, v.bitPattern)
        case .bool(let v):    out.append(v ? 1 : 0)
        case .string(let bytes): appendString(&out, bytes)
        case .array(let elementType, let elements):
            appendU32(&out, elementType.rawValue)
            appendU64(&out, UInt64(elements.count))
            for e in elements {
                guard e.valueType == elementType else {
                    throw GGUFWriterError.arrayElementTypeMismatch(
                        expected: elementType, got: e.valueType)
                }
                try appendPayload(&out, e)   // element payload only (type is homogeneous)
            }
        }
    }

    // MARK: - Little-endian primitives

    private func appendLE<T: FixedWidthInteger & UnsignedInteger>(_ out: inout Data, _ v: T) {
        var le = v.littleEndian
        withUnsafeBytes(of: &le) { out.append(contentsOf: $0) }
    }
    private func appendU32(_ out: inout Data, _ v: UInt32) { appendLE(&out, v) }
    private func appendU64(_ out: inout Data, _ v: UInt64) { appendLE(&out, v) }

    /// A GGUF string: u64 length then raw bytes (not NUL terminated).
    private func appendString(_ out: inout Data, _ bytes: [UInt8]) {
        appendU64(&out, UInt64(bytes.count))
        out.append(contentsOf: bytes)
    }

    private func pad(_ out: inout Data, to target: UInt64) {
        let cur = UInt64(out.count)
        if target > cur { out.append(Data(count: Int(target - cur))) }
    }
}
