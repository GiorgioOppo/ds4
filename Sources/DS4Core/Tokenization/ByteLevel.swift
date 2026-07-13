import Foundation

// MARK: - GPT-2 byte <-> codepoint mapping

enum ByteLevel {
    /// Port of gpt2_byte_to_codepoint.
    static func byteToCodepoint(_ b: UInt8) -> UInt32 {
        if (b >= 33 && b <= 126) || (b >= 161 && b <= 172) || (b >= 174) {
            return UInt32(b)
        }
        var n: UInt32 = 0
        for x in 0..<256 {
            if (x >= 33 && x <= 126) || (x >= 161 && x <= 172) || (x >= 174) { continue }
            if x == Int(b) { return 256 + n }
            n += 1
        }
        return UInt32(b)
    }

    /// Port of gpt2_codepoint_to_byte. Returns nil (-1 in C) when unmapped.
    static func codepointToByte(_ cp: UInt32) -> UInt8? {
        if (cp >= 33 && cp <= 126) || (cp >= 161 && cp <= 172) || (cp >= 174 && cp <= 255) {
            return UInt8(cp)
        }
        var n: UInt32 = 0
        for b in 0..<256 {
            if (b >= 33 && b <= 126) || (b >= 161 && b <= 172) || (b >= 174) { continue }
            if cp == 256 + n { return UInt8(b) }
            n += 1
        }
        return nil
    }

    static func utf8Put(_ cp: UInt32, into out: inout [UInt8]) {
        if cp <= 0x7f {
            out.append(UInt8(cp))
        } else if cp <= 0x7ff {
            out.append(UInt8(0xc0 | (cp >> 6)))
            out.append(UInt8(0x80 | (cp & 0x3f)))
        } else if cp <= 0xffff {
            out.append(UInt8(0xe0 | (cp >> 12)))
            out.append(UInt8(0x80 | ((cp >> 6) & 0x3f)))
            out.append(UInt8(0x80 | (cp & 0x3f)))
        } else {
            out.append(UInt8(0xf0 | (cp >> 18)))
            out.append(UInt8(0x80 | ((cp >> 12) & 0x3f)))
            out.append(UInt8(0x80 | ((cp >> 6) & 0x3f)))
            out.append(UInt8(0x80 | (cp & 0x3f)))
        }
    }

    static func utf8Len(_ c: UInt8) -> Int {
        if c < 0x80 { return 1 }
        if (c & 0xe0) == 0xc0 { return 2 }
        if (c & 0xf0) == 0xe0 { return 3 }
        if (c & 0xf8) == 0xf0 { return 4 }
        return 1
    }

    /// byte_encode: map raw bytes to printable codepoints encoded as UTF-8.
    static func byteEncode(_ input: ArraySlice<UInt8>) -> [UInt8] {
        var out: [UInt8] = []
        out.reserveCapacity(input.count * 2)
        for b in input { utf8Put(byteToCodepoint(b), into: &out) }
        return out
    }

    /// Decode one codepoint at `pos`, returning (codepoint, nextPos). Mirrors
    /// utf8_peek_one with the same truncation handling.
    static func decodeOne(_ s: [UInt8], _ len: Int, _ pos: Int) -> (cp: UInt32, next: Int) {
        let c0 = s[pos]
        var n = utf8Len(c0)
        if pos + n > len { n = 1 }
        switch n {
        case 1: return (UInt32(c0), pos + 1)
        case 2: return ((UInt32(c0 & 0x1f) << 6) | UInt32(s[pos+1] & 0x3f), pos + 2)
        case 3: return ((UInt32(c0 & 0x0f) << 12) | (UInt32(s[pos+1] & 0x3f) << 6) | UInt32(s[pos+2] & 0x3f), pos + 3)
        default: return ((UInt32(c0 & 0x07) << 18) | (UInt32(s[pos+1] & 0x3f) << 12) | (UInt32(s[pos+2] & 0x3f) << 6) | UInt32(s[pos+3] & 0x3f), pos + 4)
        }
    }
}

