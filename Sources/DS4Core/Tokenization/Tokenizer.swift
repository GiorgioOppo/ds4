import Foundation

// Faithful Swift port of the ds4.c tokenizer: GPT-2 byte-level BPE with the
// JoyAI/DeepSeek pre-tokenizer, special-token-aware rendered-chat tokenization,
// DeepSeek chat prompt rendering, and detokenization. Tables are keyed by exact
// bytes, matching the C str_i32_table, so behavior is identical including the
// single-byte fallback. Phase 3 of the C->Swift port.
//
// Validation: `./ds4 --dump-tokens -p "..."` prints the C token stream; the
// tests compare the Swift output against it for many prompts.

// MARK: - Vocabulary

public final class Tokenizer {
    public let tokens: [[UInt8]]          // id -> raw byte string
    public let nVocab: Int
    public let bosId, eosId, userId, assistantId, thinkStartId, thinkEndId, dsmlId: Int32

    private let tokenToId: [[UInt8]: Int32]
    private let mergeRank: [[UInt8]: Int32]
    /// Literal special tokens recognized when tokenizing already-rendered chat.
    /// Built from every CONTROL/USER_DEFINED vocab entry (so tool-call markup
    /// tokenizes as single ids), plus the seven named specials, indexed by FIRST
    /// byte (256 bucket, longest-first dentro il bucket): la scansione del
    /// rendered chat salta in O(1) i byte che non possono aprire nessun token
    /// speciale. Senza questo indice ogni byte del transcript confrontava TUTTI
    /// gli speciali (con un'allocazione per confronto): su un transcript Xcode
    /// da decine di KB erano secondi di CPU spesi in silenzio tra l'arrivo
    /// della richiesta e l'inizio del prefill.
    private let specialsByFirstByte: [[(bytes: [UInt8], id: Int32)]]

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

        // Collect the full set of CONTROL (type 3) tokens from
        // tokenizer.ggml.token_type, so chat/tool markup (role markers, tool-call
        // begin/end/sep, tool outputs) is recognized as atomic tokens. Skip empty.
        var sp: [(bytes: [UInt8], id: Int32)] = []
        if let types = model.intArray("tokenizer.ggml.token_type"), types.count == tokenBytes.count {
            for (i, t) in types.enumerated() where t == 3 {
                let b = tokenBytes[i]
                if !b.isEmpty { sp.append((b, Int32(i))) }
            }
        }
        // ALWAYS add the seven named control tokens (union, dedup by id) — not just
        // as a fallback. A GGUF may tag these USER_DEFINED (type 4) instead of
        // CONTROL (type 3); if ｜DSML｜ were missing here, the rendered tool example
        // "<｜DSML｜tool_calls>" in the prompt would BPE-split into "DS"+"ML" and the
        // model would learn to emit those two text pieces instead of the atomic id.
        let named: [(bytes: [UInt8], id: Int32)] = [
            (Array("<｜begin▁of▁sentence｜>".utf8), bosId),
            (Array("<｜end▁of▁sentence｜>".utf8), eosId),
            (Array("<｜User｜>".utf8), userId),
            (Array("<｜Assistant｜>".utf8), assistantId),
            (Array("<think>".utf8), thinkStartId),
            (Array("</think>".utf8), thinkEndId),
            (Array("｜DSML｜".utf8), dsmlId),
        ]
        var seen = Set(sp.map { $0.id })
        for n in named where !seen.contains(n.id) { sp.append(n); seen.insert(n.id) }
        // Longest-first: at each position the longest matching special wins.
        sp.sort { $0.bytes.count > $1.bytes.count }
        var byFirst = [[(bytes: [UInt8], id: Int32)]](repeating: [], count: 256)
        for s in sp { byFirst[Int(s.bytes[0])].append(s) }   // sp è già longest-first
        self.specialsByFirstByte = byFirst
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
            // Solo i byte che aprono almeno uno speciale pagano il confronto;
            // il confronto stesso è sulla slice, senza allocare copie.
            let candidates = specialsByFirstByte[Int(t[p])]
            if !candidates.isEmpty {
                for sp in candidates {
                    let n = sp.bytes.count
                    if p + n <= len && t[p..<p+n].elementsEqual(sp.bytes) {
                        tokenizeSpan(t[span..<p], into: &out)
                        out.append(sp.id)
                        p += n
                        span = p
                        continue outer
                    }
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

