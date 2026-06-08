import Foundation

// Faithful Swift port of the ds4.c tokenizer: GPT-2 byte-level BPE with the
// JoyAI/DeepSeek pre-tokenizer, special-token-aware rendered-chat tokenization,
// DeepSeek chat prompt rendering, and detokenization. Tables are keyed by exact
// bytes, matching the C str_i32_table, so behavior is identical including the
// single-byte fallback. Phase 3 of the C->Swift port.
//
// Validation: `./ds4 --dump-tokens -p "..."` prints the C token stream; the
// tests compare the Swift output against it for many prompts.

public enum ThinkMode: Sendable {
    case none, high, max
    public var enabled: Bool { self != .none }
}

/// Reasoning-effort prefix injected for Think Max (verbatim from ds4.c).
public let DS4ReasoningEffortMaxPrefix =
    "Reasoning Effort: Absolute maximum with no shortcuts permitted.\n" +
    "You MUST be very thorough in your thinking and comprehensively decompose the problem to resolve the root cause, rigorously stress-testing your logic against all potential paths, edge cases, and adversarial scenarios.\n" +
    "Explicitly write out your entire deliberation process, documenting every intermediate step, considered alternative, and rejected hypothesis to ensure absolutely no assumption is left unchecked.\n\n"

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

// MARK: - Vocabulary

public final class Tokenizer {
    public let tokens: [[UInt8]]          // id -> raw byte string
    public let nVocab: Int
    public let bosId, eosId, userId, assistantId, thinkStartId, thinkEndId, dsmlId: Int32

    private let tokenToId: [[UInt8]: Int32]
    private let mergeRank: [[UInt8]: Int32]
    /// Literal special tokens recognized when tokenizing already-rendered chat,
    /// sorted longest-first so the longest token wins at any position. Built from
    /// every CONTROL/USER_DEFINED vocab entry (so tool-call markup tokenizes as
    /// single ids), falling back to the seven named specials.
    private let specials: [(bytes: [UInt8], id: Int32)]

    public enum TokError: Error { case missingTable(String), missingSpecial(String) }

    public init(model: GGUFModel) throws {
        guard let tokenBytes = model.stringArrayBytes("tokenizer.ggml.tokens") else {
            throw TokError.missingTable("tokenizer.ggml.tokens")
        }
        guard let mergeBytes = model.stringArrayBytes("tokenizer.ggml.merges") else {
            throw TokError.missingTable("tokenizer.ggml.merges")
        }

        self.tokens = tokenBytes
        self.nVocab = tokenBytes.count

        var t2i: [[UInt8]: Int32] = Dictionary(minimumCapacity: tokenBytes.count)
        for (i, b) in tokenBytes.enumerated() { t2i[b] = Int32(i) }   // last wins, like table_put
        self.tokenToId = t2i

        var mr: [[UInt8]: Int32] = Dictionary(minimumCapacity: mergeBytes.count)
        for (i, b) in mergeBytes.enumerated() { mr[b] = Int32(i) }
        self.mergeRank = mr

        func lookup(_ s: String) throws -> Int32 {
            guard let id = t2i[Array(s.utf8)] else { throw TokError.missingSpecial(s) }
            return id
        }
        self.bosId = try lookup("<｜begin▁of▁sentence｜>")
        self.eosId = try lookup("<｜end▁of▁sentence｜>")
        self.userId = try lookup("<｜User｜>")
        self.assistantId = try lookup("<｜Assistant｜>")
        self.thinkStartId = try lookup("<think>")
        self.thinkEndId = try lookup("</think>")
        self.dsmlId = try lookup("｜DSML｜")

        // Prefer the full set of CONTROL (type 3) tokens from
        // tokenizer.ggml.token_type, so chat/tool markup (role markers, tool-call
        // begin/end/sep, tool outputs) is recognized as atomic tokens — not just
        // the seven named specials. Skip empty tokens.
        var sp: [(bytes: [UInt8], id: Int32)] = []
        if let types = model.intArray("tokenizer.ggml.token_type"), types.count == tokenBytes.count {
            for (i, t) in types.enumerated() where t == 3 {
                let b = tokenBytes[i]
                if !b.isEmpty { sp.append((b, Int32(i))) }
            }
        }
        if sp.isEmpty {
            sp = [
                (Array("<｜begin▁of▁sentence｜>".utf8), bosId),
                (Array("<｜end▁of▁sentence｜>".utf8), eosId),
                (Array("<｜User｜>".utf8), userId),
                (Array("<｜Assistant｜>".utf8), assistantId),
                (Array("<think>".utf8), thinkStartId),
                (Array("</think>".utf8), thinkEndId),
                (Array("｜DSML｜".utf8), dsmlId),
            ]
        }
        // Longest-first: at each position the longest matching special wins.
        sp.sort { $0.bytes.count > $1.bytes.count }
        self.specials = sp
    }

    /// Look up the id of an arbitrary special/normal token by its exact bytes
    /// (used by chat/tool rendering to find model-specific markup tokens).
    public func tokenId(_ s: String) -> Int32? { tokenToId[Array(s.utf8)] }

    // MARK: BPE

    private func bpeRank(_ a: [UInt8], _ b: [UInt8]) -> Int32 {
        var key = a
        key.append(0x20)
        key.append(contentsOf: b)
        return mergeRank[key] ?? -1
    }

    /// Port of bpe_emit_piece.
    private func emitPiece(_ piece: ArraySlice<UInt8>, into out: inout [Int32]) {
        let encoded = ByteLevel.byteEncode(piece)
        let encLen = encoded.count

        var sym: [[UInt8]] = []
        var off = 0
        while off < encLen {
            var n = ByteLevel.utf8Len(encoded[off])
            if off + n > encLen { n = 1 }
            sym.append(Array(encoded[off..<off+n]))
            off += n
        }

        while true {
            var bestI = -1
            var bestRank = Int32.max
            var i = 0
            while i + 1 < sym.count {
                let r = bpeRank(sym[i], sym[i + 1])
                if r >= 0 && r < bestRank { bestRank = r; bestI = i }
                i += 1
            }
            if bestI < 0 { break }
            sym[bestI].append(contentsOf: sym[bestI + 1])
            sym.remove(at: bestI + 1)
        }

        for s in sym {
            if let token = tokenToId[s] {
                out.append(token)
            } else {
                for b in s {
                    if let token = tokenToId[[b]] { out.append(token) }
                }
            }
        }
    }

    // MARK: JoyAI/DeepSeek pre-tokenizer (port of bpe_tokenize_text)

    private func asciiAlpha(_ c: UInt8) -> Bool { (c >= 65 && c <= 90) || (c >= 97 && c <= 122) }
    private func asciiDigit(_ c: UInt8) -> Bool { c >= 48 && c <= 57 }
    private func asciiSpace(_ c: UInt8) -> Bool { c == 0x20 || c == 0x09 || c == 0x0a || c == 0x0d || c == 0x0b || c == 0x0c }
    private func asciiNewline(_ c: UInt8) -> Bool { c == 0x0a || c == 0x0d }
    private func punctSym(_ c: UInt8) -> Bool {
        (c >= 0x21 && c <= 0x2f) || (c >= 0x3a && c <= 0x40) ||
        (c >= 0x5b && c <= 0x60) || (c >= 0x7b && c <= 0x7e)
    }
    private func isCJK(_ cp: UInt32) -> Bool {
        (cp >= 0x4e00 && cp <= 0x9fa5) || (cp >= 0x3040 && cp <= 0x309f) || (cp >= 0x30a0 && cp <= 0x30ff)
    }
    private func nextChar(_ s: [UInt8], _ len: Int, _ pos: Int) -> Int {
        var n = ByteLevel.utf8Len(s[pos]); if pos + n > len { n = 1 }; return pos + n
    }
    private func letterLike(_ s: [UInt8], _ pos: Int) -> Bool {
        let c = s[pos]
        if c < 128 { return asciiAlpha(c) }
        return true
    }
    private func consumeLetters(_ s: [UInt8], _ len: Int, _ start: Int) -> Int {
        var pos = start
        while pos < len && letterLike(s, pos) { pos = nextChar(s, len, pos) }
        return pos
    }
    private func cjkAt(_ s: [UInt8], _ len: Int, _ pos: Int) -> Bool {
        if s[pos] < 128 { return false }
        let (cp, _) = ByteLevel.decodeOne(s, len, pos)
        return isCJK(cp)
    }

    private func preTokenize(_ text: [UInt8], into out: inout [Int32]) {
        let len = text.count
        var pos = 0
        while pos < len {
            let start = pos
            let c = text[pos]

            if asciiDigit(c) {
                var nd = 0
                while pos < len && asciiDigit(text[pos]) && nd < 3 { pos += 1; nd += 1 }
            } else if cjkAt(text, len, pos) {
                repeat { pos = nextChar(text, len, pos) } while pos < len && cjkAt(text, len, pos)
            } else if punctSym(c) && pos + 1 < len && asciiAlpha(text[pos + 1]) {
                pos += 1
                while pos < len && asciiAlpha(text[pos]) { pos += 1 }
            } else if letterLike(text, pos) {
                pos = consumeLetters(text, len, pos)
            } else if !asciiNewline(c) && !punctSym(c) && pos + 1 < len && letterLike(text, pos + 1) {
                pos += 1
                pos = consumeLetters(text, len, pos)
            } else if c == 0x20 && pos + 1 < len && punctSym(text[pos + 1]) {
                pos += 1
                while pos < len && punctSym(text[pos]) { pos += 1 }
                while pos < len && asciiNewline(text[pos]) { pos += 1 }
            } else if punctSym(c) {
                while pos < len && punctSym(text[pos]) { pos += 1 }
                while pos < len && asciiNewline(text[pos]) { pos += 1 }
            } else if asciiSpace(c) {
                var p = pos
                var lastNewlineEnd = 0
                while p < len && asciiSpace(text[p]) {
                    let sc = text[p]; p += 1
                    if asciiNewline(sc) { lastNewlineEnd = p }
                }
                if lastNewlineEnd != 0 {
                    pos = lastNewlineEnd
                } else if p < len && p > pos + 1 && (letterLike(text, p) || punctSym(text[p])) {
                    pos = p - 1
                } else {
                    pos = p
                }
            } else {
                pos = nextChar(text, len, pos)
            }

            if pos == start { pos = nextChar(text, len, pos) }
            emitPiece(text[start..<pos], into: &out)
        }
    }

    // MARK: Public tokenization

    /// Tokenize plain text (no special-token recognition). Port of ds4_tokenize_text.
    public func tokenize(_ text: String) -> [Int32] {
        var out: [Int32] = []
        preTokenize(Array(text.utf8), into: &out)
        return out
    }

    private func tokenizeSpan(_ bytes: ArraySlice<UInt8>, into out: inout [Int32]) {
        if bytes.isEmpty { return }
        preTokenize(Array(bytes), into: &out)
    }

    /// Tokenize already-rendered chat text, recognizing literal special tokens.
    /// Port of tokenize_rendered_chat_vocab.
    public func tokenizeRenderedChat(_ text: String) -> [Int32] {
        let t = Array(text.utf8)
        var out: [Int32] = []
        var span = 0
        var p = 0
        let len = t.count
        outer: while p < len {
            for sp in specials {
                let n = sp.bytes.count
                if p + n <= len && Array(t[p..<p+n]) == sp.bytes {
                    tokenizeSpan(t[span..<p], into: &out)
                    out.append(sp.id)
                    p += n
                    span = p
                    continue outer
                }
            }
            p += 1
        }
        tokenizeSpan(t[span..<len], into: &out)
        return out
    }

    /// Port of encode_chat_prompt: BOS, optional max-effort prefix, system, user
    /// prompt, assistant marker, and <think>/</think> per mode.
    public func encodeChatPrompt(system: String?, prompt: String, think: ThinkMode) -> [Int32] {
        var out: [Int32] = [bosId]
        if think == .max { preTokenize(Array(DS4ReasoningEffortMaxPrefix.utf8), into: &out) }
        if let system, !system.isEmpty { preTokenize(Array(system.utf8), into: &out) }
        out.append(userId)
        preTokenize(Array(prompt.utf8), into: &out)
        out.append(assistantId)
        out.append(think.enabled ? thinkStartId : thinkEndId)
        return out
    }

    // MARK: Detokenization (port of ds4_token_text)

    private static let fullwidthBar: [UInt8] = [0xef, 0xbd, 0x9c] // U+FF5C

    private func isLiteralSpecial(_ s: [UInt8]) -> Bool {
        let bar = Tokenizer.fullwidthBar
        if s.count < bar.count { return false }
        var i = 0
        while i + bar.count <= s.count {
            if Array(s[i..<i+bar.count]) == bar { return true }
            i += 1
        }
        return false
    }

    /// Decode a token id to its raw output bytes.
    public func tokenText(_ id: Int32) -> [UInt8] {
        if id < 0 || Int(id) >= nVocab { return [] }
        let s = tokens[Int(id)]
        if isLiteralSpecial(s) { return s }
        var out: [UInt8] = []
        var pos = 0
        let len = s.count
        while pos < len {
            let (cp, next) = ByteLevel.decodeOne(s, len, pos)
            pos = next
            if let b = ByteLevel.codepointToByte(cp) { out.append(b) }
        }
        return out
    }
}
