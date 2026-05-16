import Foundation

/// Minimal WordPiece tokenizer, BERT-style. Reads the `vocab` field of
/// a HuggingFace `tokenizer.json` whose `model.type == "WordPiece"`.
///
/// Algorithm: whitespace + punctuation pre-tokenisation, then greedy
/// longest-match on each word — first match starts at the head of the
/// word, subsequent matches use the `##` continuation prefix.
public final class WordPieceTokenizer: Tokenizer {
    private let vocab: [String: Int]
    private let invVocab: [Int: String]
    private let unkToken: String
    private let unkId: Int
    private let clsId: Int?
    private let sepId: Int?
    private let padId: Int?
    private let maxInputCharsPerWord: Int

    public let bosId: Int?
    public let eosId: Int?
    public let stopTokenIds: Set<Int>

    public init(jsonData: Data,
                unkToken: String = "[UNK]",
                maxInputCharsPerWord: Int = 200) throws {
        guard let json = try JSONSerialization.jsonObject(with: jsonData) as? [String: Any],
              let model = json["model"] as? [String: Any],
              let vocab = model["vocab"] as? [String: Int]
        else {
            throw NSError(domain: "WordPieceTokenizer", code: 1,
                          userInfo: [NSLocalizedDescriptionKey: "tokenizer.json: missing model.vocab"])
        }
        self.vocab = vocab
        var inv: [Int: String] = [:]
        inv.reserveCapacity(vocab.count)
        for (tok, id) in vocab { inv[id] = tok }
        self.invVocab = inv
        self.unkToken = unkToken
        self.unkId = vocab[unkToken] ?? -1
        self.maxInputCharsPerWord = maxInputCharsPerWord
        self.clsId = vocab["[CLS]"]
        self.sepId = vocab["[SEP]"]
        self.padId = vocab["[PAD]"]
        self.bosId = self.clsId
        self.eosId = self.sepId
        var stops = Set<Int>()
        if let s = self.sepId { stops.insert(s) }
        self.stopTokenIds = stops
    }

    public func encode(_ text: String) -> [Int] {
        var out: [Int] = []
        for word in basicTokenize(text) {
            let chars = Array(word)
            if chars.count > maxInputCharsPerWord {
                if unkId >= 0 { out.append(unkId) }
                continue
            }
            var matched = true
            var pieces: [Int] = []
            var start = 0
            while start < chars.count {
                var end = chars.count
                var cur: Int? = nil
                while start < end {
                    var substr = String(chars[start..<end])
                    if start > 0 { substr = "##" + substr }
                    if let id = vocab[substr] {
                        cur = id
                        break
                    }
                    end -= 1
                }
                if let id = cur {
                    pieces.append(id)
                    start = end
                } else {
                    matched = false
                    break
                }
            }
            if matched { out.append(contentsOf: pieces) }
            else if unkId >= 0 { out.append(unkId) }
        }
        return out
    }

    public func decode(_ ids: [Int]) -> String {
        var out = ""
        for id in ids {
            guard let tok = invVocab[id] else { continue }
            if tok.hasPrefix("##") {
                out += String(tok.dropFirst(2))
            } else {
                if !out.isEmpty { out += " " }
                out += tok
            }
        }
        return out
    }

    /// Whitespace + punctuation split. Mirrors HuggingFace's
    /// `BasicTokenizer` default behaviour without lowercasing
    /// (callers can lowercase ahead of time if their vocab is uncased).
    private func basicTokenize(_ text: String) -> [String] {
        var words: [String] = []
        var current = ""
        for ch in text {
            if ch.isWhitespace {
                if !current.isEmpty { words.append(current); current = "" }
            } else if ch.isPunctuation {
                if !current.isEmpty { words.append(current); current = "" }
                words.append(String(ch))
            } else {
                current.append(ch)
            }
        }
        if !current.isEmpty { words.append(current) }
        return words
    }
}
