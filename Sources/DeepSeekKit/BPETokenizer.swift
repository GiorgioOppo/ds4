import Foundation

/// Byte-level BPE tokenizer compatible with HuggingFace `tokenizer.json` files
/// produced by the `tokenizers` library. This is the format DeepSeek-V4 ships.
///
/// The implementation handles the subset actually used by DeepSeek:
///   - `model.type == "BPE"`, with `vocab` (str → id) and `merges` (array of
///     "a b" strings or [a, b] pairs)
///   - `pre_tokenizer` of type `ByteLevel` or a `Sequence` containing one,
///     with the GPT-2 / cl100k regex
///   - `added_tokens` for the DeepSeek special tokens
///   - byte-level encoding via the GPT-2 byte-to-unicode map
///
/// Limitations vs the full HF spec:
///   - no normalization step (DeepSeek's tokenizer.json has `normalizer: null`)
///   - no truncation/padding (we encode and let the caller manage length)
///   - regex split uses NSRegularExpression with the most-common ByteLevel
///     pattern; if a tokenizer specifies a different `pretokenizer.pattern.Regex`
///     we fall back to that pattern verbatim
public final class BPETokenizer: Tokenizer {

    public let vocab: [String: Int]              // token (in byte-encoded unicode) -> id
    public let invVocab: [Int: String]
    public let mergeRank: [String: Int]          // "a b" -> rank (lower = earlier merge)
    public let addedTokens: [String: Int]        // text -> id (always emitted as-is)
    public let invAddedTokens: [Int: String]
    public let bosId: Int?
    public let eosId: Int?

    private let pretokRegex: NSRegularExpression
    private let specialPattern: NSRegularExpression?
    private let byteToUnicode: [UInt8: String]
    private let unicodeToByte: [String: UInt8]

    public init(jsonData: Data) throws {
        guard let root = try JSONSerialization.jsonObject(with: jsonData) as? [String: Any] else {
            throw err("tokenizer.json is not a JSON object")
        }

        // ---- model.vocab + merges --------------------------------------
        guard let model = root["model"] as? [String: Any] else { throw err("missing model") }
        guard let modelType = model["type"] as? String, modelType == "BPE" else {
            throw err("only BPE tokenizers supported (got \(model["type"] ?? "?"))")
        }
        guard let rawVocab = model["vocab"] as? [String: Int] else { throw err("missing model.vocab") }
        var vocab = rawVocab
        var inv: [Int: String] = [:]
        for (k, v) in vocab { inv[v] = k }

        var rank: [String: Int] = [:]
        if let mergesArr = model["merges"] as? [Any] {
            for (i, m) in mergesArr.enumerated() {
                if let s = m as? String {
                    rank[s] = i
                } else if let pair = m as? [String], pair.count == 2 {
                    rank["\(pair[0]) \(pair[1])"] = i
                }
            }
        }

        // ---- added_tokens (special tokens) -----------------------------
        var added: [String: Int] = [:]
        var invAdded: [Int: String] = [:]
        if let addedArr = root["added_tokens"] as? [[String: Any]] {
            for entry in addedArr {
                if let id = entry["id"] as? Int, let content = entry["content"] as? String {
                    added[content] = id
                    invAdded[id] = content
                    if vocab[content] == nil { vocab[content] = id; inv[id] = content }
                }
            }
        }
        self.vocab = vocab
        self.invVocab = inv
        self.mergeRank = rank
        self.addedTokens = added
        self.invAddedTokens = invAdded

        // ---- pre_tokenizer regex ---------------------------------------
        let defaultRegex = "'(?i:[sdmt]|ll|ve|re)| ?\\p{L}+| ?\\p{N}+| ?[^\\s\\p{L}\\p{N}]+|\\s+(?!\\S)|\\s+"
        var pattern = defaultRegex
        if let pre = root["pre_tokenizer"] as? [String: Any] {
            pattern = Self.findRegex(in: pre) ?? defaultRegex
        }
        self.pretokRegex = try NSRegularExpression(pattern: pattern)

        // ---- special-token splitter (matches any added_token) ----------
        if !added.isEmpty {
            let alts = added.keys
                .sorted { $0.count > $1.count }
                .map { NSRegularExpression.escapedPattern(for: $0) }
                .joined(separator: "|")
            self.specialPattern = try NSRegularExpression(pattern: alts)
        } else {
            self.specialPattern = nil
        }

        // ---- byte-level char map ---------------------------------------
        let (b2u, u2b) = Self.makeByteToUnicode()
        self.byteToUnicode = b2u
        self.unicodeToByte = u2b

        // ---- bos/eos ---------------------------------------------------
        // post_processor.special_tokens.{bos_token, eos_token} or top-level model.bos_token
        var bos: Int? = nil
        var eos: Int? = nil
        if let post = root["post_processor"] as? [String: Any],
           let specials = post["special_tokens"] as? [String: Any] {
            if let b = specials["bos_token"] as? [String: Any], let ids = b["ids"] as? [Int] {
                bos = ids.first
            }
            if let e = specials["eos_token"] as? [String: Any], let ids = e["ids"] as? [Int] {
                eos = ids.first
            }
        }
        // Fallback to common DeepSeek special-token names.
        if bos == nil { bos = added["<｜begin▁of▁sentence｜>"] }
        if eos == nil { eos = added["<｜end▁of▁sentence｜>"] }
        self.bosId = bos
        self.eosId = eos
    }

    // MARK: - Encode

    public func encode(_ text: String) -> [Int] {
        var out: [Int] = []
        // Split off any added tokens first; emit them as-is.
        let nsText = text as NSString
        let full = NSRange(location: 0, length: nsText.length)
        var cursor = 0
        if let sp = specialPattern {
            sp.enumerateMatches(in: text, range: full) { match, _, _ in
                guard let m = match else { return }
                if m.range.location > cursor {
                    let chunk = nsText.substring(with: NSRange(location: cursor, length: m.range.location - cursor))
                    out.append(contentsOf: encodeChunk(chunk))
                }
                let token = nsText.substring(with: m.range)
                if let id = addedTokens[token] { out.append(id) }
                cursor = m.range.location + m.range.length
            }
        }
        if cursor < nsText.length {
            let chunk = nsText.substring(with: NSRange(location: cursor, length: nsText.length - cursor))
            out.append(contentsOf: encodeChunk(chunk))
        }
        return out
    }

    private func encodeChunk(_ text: String) -> [Int] {
        var out: [Int] = []
        var coveredAny = false
        let ns = text as NSString
        pretokRegex.enumerateMatches(in: text, range: NSRange(location: 0, length: ns.length)) { match, _, _ in
            guard let m = match else { return }
            coveredAny = true
            let piece = ns.substring(with: m.range)
            appendEncodedPiece(piece, into: &out)
        }
        // Fallback: if the pre-tokenizer regex didn't match anything in a
        // non-empty chunk (e.g. tokenizer.json's pre_tokenizer regex is a
        // special-token splitter rather than a word splitter), encode the
        // whole chunk as a single byte-level piece instead of silently
        // dropping it.
        if !coveredAny && ns.length > 0 {
            appendEncodedPiece(text, into: &out)
        }
        return out
    }

    private func appendEncodedPiece(_ piece: String, into out: inout [Int]) {
        var encoded = ""
        for byte in piece.utf8 {
            encoded += byteToUnicode[byte]!
        }
        for tok in bpe(encoded) {
            if let id = vocab[tok] {
                out.append(id)
            } else {
                for b in tok.utf8 {
                    if let id = vocab[String(UnicodeScalar(UInt32(b))!)] {
                        out.append(id)
                    }
                }
            }
        }
    }

    private func bpe(_ word: String) -> [String] {
        var pieces = word.map { String($0) }
        if pieces.count <= 1 { return pieces }

        while true {
            var bestRank = Int.max
            var bestIdx = -1
            for i in 0..<(pieces.count - 1) {
                let pair = pieces[i] + " " + pieces[i + 1]
                if let r = mergeRank[pair], r < bestRank {
                    bestRank = r; bestIdx = i
                }
            }
            if bestIdx < 0 { break }
            pieces[bestIdx] = pieces[bestIdx] + pieces[bestIdx + 1]
            pieces.remove(at: bestIdx + 1)
        }
        return pieces
    }

    // MARK: - Decode

    public func decode(_ ids: [Int]) -> String {
        var bytes: [UInt8] = []
        for id in ids {
            if let added = invAddedTokens[id] {
                // Special tokens emit their literal UTF-8 bytes.
                bytes.append(contentsOf: Array(added.utf8))
                continue
            }
            guard let token = invVocab[id] else { continue }
            for ch in token {
                let s = String(ch)
                if let b = unicodeToByte[s] {
                    bytes.append(b)
                } else {
                    // Should not happen with a well-formed vocab; fall back to UTF-8.
                    bytes.append(contentsOf: Array(s.utf8))
                }
            }
        }
        return String(decoding: bytes, as: UTF8.self)
    }

    // MARK: - Helpers

    private static func makeByteToUnicode() -> ([UInt8: String], [String: UInt8]) {
        // Visible-printable bytes that map to themselves.
        var bs: [UInt8] = []
        bs.append(contentsOf: 33...126)
        bs.append(contentsOf: 161...172)
        bs.append(contentsOf: 174...255)
        var cs: [UInt32] = bs.map { UInt32($0) }
        var n: UInt32 = 0
        for b in 0...255 {
            if !bs.contains(UInt8(b)) {
                bs.append(UInt8(b))
                cs.append(256 + n)
                n += 1
            }
        }
        var b2u: [UInt8: String] = [:]
        var u2b: [String: UInt8] = [:]
        for (b, c) in zip(bs, cs) {
            let s = String(UnicodeScalar(c)!)
            b2u[b] = s
            u2b[s] = b
        }
        return (b2u, u2b)
    }

    private static func findRegex(in any: [String: Any]) -> String? {
        // ByteLevel pretok exposes its regex directly when type == "ByteLevel" with use_regex,
        // but tokenizer.json doesn't include the literal pattern — the rust runtime hardcodes it.
        // We dig for an explicit "pattern.Regex" first (Sequence([Split(Regex(...)), ByteLevel])
        // is a common composition), and fall back to the ByteLevel default outside.
        if let typ = any["type"] as? String, typ == "Sequence",
           let list = any["pretokenizers"] as? [[String: Any]] {
            for p in list {
                if let s = findRegex(in: p) { return s }
            }
        }
        if let pat = any["pattern"] as? [String: Any], let r = pat["Regex"] as? String {
            return r
        }
        return nil
    }
}

private func err(_ msg: String) -> NSError {
    NSError(domain: "BPETokenizer", code: -1, userInfo: [NSLocalizedDescriptionKey: msg])
}
