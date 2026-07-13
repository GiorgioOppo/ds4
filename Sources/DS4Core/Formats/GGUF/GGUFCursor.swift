import Foundation

/// Bounds-checked little-endian reader over the mmap (port of ds4_cursor).
struct GGUFCursor {
    let base: UnsafePointer<UInt8>
    let size: UInt64
    var pos: UInt64

    init(base: UnsafePointer<UInt8>, size: UInt64, pos: UInt64 = 0) {
        self.base = base; self.size = size; self.pos = pos
    }

    private func has(_ n: UInt64) throws {
        if n > size || pos > size - n { throw GGUFError.truncated(pos) }
    }

    mutating func skip(_ n: UInt64) throws { try has(n); pos += n }

    mutating func u32() throws -> UInt32 {
        try has(4)
        let v = UnsafeRawPointer(base + Int(pos)).loadUnaligned(as: UInt32.self)
        pos += 4
        return v
    }

    mutating func u64() throws -> UInt64 {
        try has(8)
        let v = UnsafeRawPointer(base + Int(pos)).loadUnaligned(as: UInt64.self)
        pos += 8
        return v
    }

    mutating func read<T>(as type: T.Type) throws -> T {
        let n = UInt64(MemoryLayout<T>.size)
        try has(n)
        let v = UnsafeRawPointer(base + Int(pos)).loadUnaligned(as: T.self)
        pos += n
        return v
    }

    /// A GGUF string: u64 length then raw UTF-8 bytes (not NUL terminated).
    mutating func string() throws -> String {
        let len = try u64()
        try has(len)
        let bytes = UnsafeBufferPointer(start: base + Int(pos), count: Int(len))
        pos += len
        return String(decoding: bytes, as: UTF8.self)
    }

    /// Raw bytes of a GGUF string (no UTF-8 decode). Used by the tokenizer,
    /// which keys its tables by exact bytes like the C implementation.
    mutating func stringBytes() throws -> [UInt8] {
        let len = try u64()
        try has(len)
        let buf = UnsafeBufferPointer(start: base + Int(pos), count: Int(len))
        pos += len
        return Array(buf)
    }

    static func scalarSize(_ type: UInt32) -> UInt64 {
        switch type {
        case GGUFValueType.uint8.rawValue, GGUFValueType.int8.rawValue, GGUFValueType.bool.rawValue:
            return 1
        case GGUFValueType.uint16.rawValue, GGUFValueType.int16.rawValue:
            return 2
        case GGUFValueType.uint32.rawValue, GGUFValueType.int32.rawValue, GGUFValueType.float32.rawValue:
            return 4
        case GGUFValueType.uint64.rawValue, GGUFValueType.int64.rawValue, GGUFValueType.float64.rawValue:
            return 8
        default:
            return 0
        }
    }

    /// Port of skip_value: advance past a metadata value of `type`.
    mutating func skipValue(_ type: UInt32, depth: Int = 0) throws {
        if depth > 8 { throw GGUFError.message("metadata array nesting is too deep at byte \(pos)") }

        let scalar = GGUFCursor.scalarSize(type)
        if scalar != 0 { try skip(scalar); return }

        if type == GGUFValueType.string.rawValue {
            _ = try string()
            return
        }
        if type == GGUFValueType.array.rawValue {
            let itemType = try u32()
            let len = try u64()
            let itemSize = GGUFCursor.scalarSize(itemType)
            if itemSize != 0 {
                if len > UInt64.max / itemSize { throw GGUFError.message("metadata array is too large at byte \(pos)") }
                try skip(len * itemSize)
                return
            }
            var i: UInt64 = 0
            while i < len { try skipValue(itemType, depth: depth + 1); i += 1 }
            return
        }
        throw GGUFError.message("unknown GGUF metadata type at byte \(pos)")
    }
}

