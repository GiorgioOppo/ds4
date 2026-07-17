import Foundation

/// Special-token ids needed by GLM 5.2 chat, tools and generation stopping.
public struct GLM52SpecialTokenIDs: Sendable, Equatable {
    public let mask: Int32
    public let startOfPrompt: Int32
    public let endOfText: Int32
    public let system: Int32
    public let user: Int32
    public let assistant: Int32
    public let observation: Int32
    public let thinkOpen: Int32
    public let thinkClose: Int32
    public let toolCallOpen: Int32
    public let toolCallClose: Int32
    public let toolResponseOpen: Int32
    public let toolResponseClose: Int32
    public let argumentKeyOpen: Int32
    public let argumentKeyClose: Int32
    public let argumentValueOpen: Int32
    public let argumentValueClose: Int32

    public var generationStops: Set<Int32> {
        [endOfText, system, user, assistant, observation]
    }

    public var all: Set<Int32> {
        generationStops.union([
            mask, startOfPrompt, thinkOpen, thinkClose,
            toolCallOpen, toolCallClose,
            toolResponseOpen, toolResponseClose,
            argumentKeyOpen, argumentKeyClose,
            argumentValueOpen, argumentValueClose,
        ])
    }
}

/// ChatGLM4/GLM pre-tokenizer used by GGUFs declaring
/// `tokenizer.ggml.pre = "glm4"`.
///
/// Kept as a standalone pure splitter so its regex-equivalent boundaries can
/// be regression-tested without a 700+ GB model or its vocabulary.
enum GLM4Pretokenizer {
    private struct Info {
        let codepoint: UInt32
        let next: Int
        let isLetter: Bool
        let isNumber: Bool
        let isWhitespace: Bool
    }

    static func split(_ text: String) -> [[UInt8]] {
        split(Array(text.utf8))
    }

    static func split(_ bytes: [UInt8]) -> [[UInt8]] {
        var pieces: [[UInt8]] = []
        var position = 0
        while position < bytes.count {
            let start = position
            guard let current = info(bytes, at: position) else { break }

            if current.codepoint == 0x27, let next = info(bytes, at: current.next) {
                let first = asciiLower(next.codepoint)
                if first == 0x73 || first == 0x74 || first == 0x6d || first == 0x64 {
                    position = next.next
                    pieces.append(Array(bytes[start..<position]))
                    continue
                }
                if let second = info(bytes, at: next.next) {
                    let secondLower = asciiLower(second.codepoint)
                    if (first == 0x72 && secondLower == 0x65)
                        || (first == 0x76 && secondLower == 0x65)
                        || (first == 0x6c && secondLower == 0x6c) {
                        position = second.next
                        pieces.append(Array(bytes[start..<position]))
                        continue
                    }
                }
            }

            if current.codepoint != 0x0d,
               current.codepoint != 0x0a,
               !current.isNumber {
                let next = info(bytes, at: current.next)
                if current.isLetter || next?.isLetter == true {
                    position = current.next
                    while let scan = info(bytes, at: position), scan.isLetter {
                        position = scan.next
                    }
                    pieces.append(Array(bytes[start..<position]))
                    continue
                }
            }

            if current.isNumber {
                var count = 0
                while count < 3, let scan = info(bytes, at: position), scan.isNumber {
                    position = scan.next
                    count += 1
                }
                pieces.append(Array(bytes[start..<position]))
                continue
            }

            var punctuation = current
            var punctuationPosition = position
            if current.codepoint == 0x20, let afterSpace = info(bytes, at: current.next) {
                punctuationPosition = current.next
                punctuation = afterSpace
            }
            if !punctuation.isWhitespace,
               !punctuation.isLetter,
               !punctuation.isNumber {
                position = punctuationPosition
                while let scan = info(bytes, at: position),
                      !scan.isWhitespace,
                      !scan.isLetter,
                      !scan.isNumber {
                    position = scan.next
                }
                while let scan = info(bytes, at: position),
                      scan.codepoint == 0x0d || scan.codepoint == 0x0a {
                    position = scan.next
                }
                pieces.append(Array(bytes[start..<position]))
                continue
            }

            if current.isWhitespace {
                var scanPosition = position
                var lastNewlineEnd: Int?
                var lastWhitespaceStart = position
                var whitespaceCount = 0
                while let scan = info(bytes, at: scanPosition), scan.isWhitespace {
                    lastWhitespaceStart = scanPosition
                    if scan.codepoint == 0x0d || scan.codepoint == 0x0a {
                        lastNewlineEnd = scan.next
                    }
                    scanPosition = scan.next
                    whitespaceCount += 1
                }
                if let lastNewlineEnd {
                    position = lastNewlineEnd
                } else if whitespaceCount > 1, scanPosition < bytes.count {
                    position = lastWhitespaceStart
                } else {
                    position = scanPosition
                }
                pieces.append(Array(bytes[start..<position]))
                continue
            }

            position = current.next
            if position == start { position += 1 }
            pieces.append(Array(bytes[start..<position]))
        }
        return pieces
    }

    private static func info(_ bytes: [UInt8], at position: Int) -> Info? {
        guard position < bytes.count else { return nil }
        let decoded = ByteLevel.decodeOne(bytes, bytes.count, position)
        let whitespace = unicodeWhitespace(decoded.cp)
        let number = unicodeNumber(decoded.cp)
        let letter: Bool
        if decoded.cp < 128 {
            letter = (decoded.cp >= 65 && decoded.cp <= 90)
                || (decoded.cp >= 97 && decoded.cp <= 122)
        } else {
            letter = !whitespace && !number && !unicodePunctuationOrSymbol(decoded.cp)
        }
        return Info(codepoint: decoded.cp, next: decoded.next,
                    isLetter: letter, isNumber: number, isWhitespace: whitespace)
    }

    private static func asciiLower(_ codepoint: UInt32) -> UInt32 {
        codepoint >= 65 && codepoint <= 90 ? codepoint + 32 : codepoint
    }

    private static func unicodeWhitespace(_ cp: UInt32) -> Bool {
        if cp < 128 {
            return cp == 0x20 || cp == 0x09 || cp == 0x0a || cp == 0x0d
                || cp == 0x0b || cp == 0x0c
        }
        return cp == 0x0085 || cp == 0x00a0 || cp == 0x1680
            || (cp >= 0x2000 && cp <= 0x200a)
            || cp == 0x2028 || cp == 0x2029 || cp == 0x202f
            || cp == 0x205f || cp == 0x3000
    }

    private static func unicodeNumber(_ cp: UInt32) -> Bool {
        if cp >= 0x30 && cp <= 0x39 { return true }
        return numberRanges.contains { $0.contains(cp) }
    }

    private static let numberRanges: [ClosedRange<UInt32>] = [
            0x0660...0x0669, 0x06f0...0x06f9, 0x07c0...0x07c9,
            0x0966...0x096f, 0x09e6...0x09ef, 0x0a66...0x0a6f,
            0x0ae6...0x0aef, 0x0b66...0x0b6f, 0x0be6...0x0bef,
            0x0c66...0x0c6f, 0x0ce6...0x0cef, 0x0d66...0x0d6f,
            0x0de6...0x0def, 0x0e50...0x0e59, 0x0ed0...0x0ed9,
            0x0f20...0x0f29, 0x1040...0x1049, 0x1090...0x1099,
            0x17e0...0x17e9, 0x1810...0x1819, 0xff10...0xff19,
    ]

    private static func unicodePunctuationOrSymbol(_ cp: UInt32) -> Bool {
        if cp < 128 {
            return (cp >= 0x21 && cp <= 0x2f)
                || (cp >= 0x3a && cp <= 0x40)
                || (cp >= 0x5b && cp <= 0x60)
                || (cp >= 0x7b && cp <= 0x7e)
        }
        if punctuationRanges.contains(where: { $0.contains(cp) }) { return true }
        return cp == 0x00b4 || (cp >= 0x00b6 && cp <= 0x00b8)
            || cp == 0x00bb || cp == 0x00bf || cp == 0x00d7 || cp == 0x00f7
            || cp == 0x0387 || cp == 0x05c3 || cp == 0x061b
            || cp == 0x066a || cp == 0x066d || cp == 0x06d4
    }

    private static let punctuationRanges: [ClosedRange<UInt32>] = [
            0x00a1...0x00a9, 0x00ab...0x00ac, 0x00ae...0x00b1,
            0x02c2...0x02df, 0x02e5...0x02eb, 0x02ed...0x02ff,
            0x0375...0x037e, 0x0384...0x0385, 0x055a...0x055f,
            0x0589...0x058a, 0x05be...0x05c0, 0x05c6...0x05c7,
            0x0609...0x060a, 0x060c...0x060d, 0x061e...0x061f,
            0x2000...0x206f, 0x20a0...0x20cf, 0x2100...0x214f,
            0x2190...0x23ff, 0x2460...0x24ff, 0x2500...0x2775,
            0x2794...0x2bff, 0x2e00...0x2e7f, 0x3000...0x303f,
            0xfd3e...0xfd3f, 0xfe10...0xfe6f, 0xff01...0xff0f,
            0xff1a...0xff20, 0xff3b...0xff40, 0xff5b...0xff65,
            0x1f000...0x1faff,
    ]
}

/// GPT-2 byte-level BPE with GLM4 pre-tokenization and native rendered-chat
/// special-token scanning.
public final class GLM52Tokenizer: TokenizerProtocol {
    public enum TokenizerError: Error, Sendable, Equatable, CustomStringConvertible {
        case missingTable(String)
        case unsupportedTokenizerModel(String)
        case unsupportedPretokenizer(String)
        case missingSpecial(String)
        case inconsistentSpecialID(String, expected: Int32, actual: Int32)
        case vocabularyTooLarge

        public var description: String {
            switch self {
            case .missingTable(let name): return "missing GLM tokenizer table \(name)"
            case .unsupportedTokenizerModel(let name): return "unsupported GLM tokenizer model \(name)"
            case .unsupportedPretokenizer(let name): return "unsupported GLM pre-tokenizer \(name)"
            case .missingSpecial(let name): return "missing GLM special token \(name)"
            case .inconsistentSpecialID(let name, let expected, let actual):
                return "GLM metadata id for \(name) is \(actual), expected \(expected)"
            case .vocabularyTooLarge: return "GLM vocabulary exceeds Int32 token ids"
            }
        }
    }

    public let architecture: ModelArchitectureID = .glmDSA
    public let tokens: [[UInt8]]
    public let nVocab: Int
    public let special: GLM52SpecialTokenIDs

    private let tokenToID: [[UInt8]: Int32]
    private let mergeRank: [[UInt8]: Int32]
    private let specialsByFirstByte: [[(bytes: [UInt8], id: Int32)]]

    public convenience init(model: GGUFModel) throws {
        let detected = try ModelArchitectureDetector.detect(in: model)
        guard detected.id == .glmDSA else {
            throw ModelArchitectureError.unsupportedArchitecture(detected.id)
        }
        if let tokenizerModel = model.string("tokenizer.ggml.model"),
           tokenizerModel.trimmingCharacters(in: .whitespacesAndNewlines).lowercased() != "gpt2" {
            throw TokenizerError.unsupportedTokenizerModel(tokenizerModel)
        }
        if let pretokenizer = model.string("tokenizer.ggml.pre"),
           pretokenizer.trimmingCharacters(in: .whitespacesAndNewlines).lowercased() != "glm4" {
            throw TokenizerError.unsupportedPretokenizer(pretokenizer)
        }
        guard let tokens = model.stringArrayBytes("tokenizer.ggml.tokens") else {
            throw TokenizerError.missingTable("tokenizer.ggml.tokens")
        }
        guard let merges = model.stringArrayBytes("tokenizer.ggml.merges") else {
            throw TokenizerError.missingTable("tokenizer.ggml.merges")
        }
        let bos = Self.metadataTokenID(model.u64Compat("tokenizer.ggml.bos_token_id"))
        let eos = Self.metadataTokenID(model.u64Compat("tokenizer.ggml.eos_token_id"))
        try self.init(tokens: tokens, merges: merges, bosTokenID: bos, eosTokenID: eos)
    }

    /// Model-free initializer used by deterministic tokenizer fixtures.
    init(tokens: [[UInt8]], merges: [[UInt8]],
         bosTokenID: Int32? = nil, eosTokenID: Int32? = nil) throws {
        guard tokens.count <= Int(Int32.max) else { throw TokenizerError.vocabularyTooLarge }
        self.tokens = tokens
        self.nVocab = tokens.count

        var tokenToID: [[UInt8]: Int32] = Dictionary(minimumCapacity: tokens.count)
        for (index, token) in tokens.enumerated() { tokenToID[token] = Int32(index) }
        self.tokenToID = tokenToID

        var mergeRank: [[UInt8]: Int32] = Dictionary(minimumCapacity: merges.count)
        for (index, merge) in merges.enumerated() { mergeRank[merge] = Int32(index) }
        self.mergeRank = mergeRank

        func required(_ literal: String) throws -> Int32 {
            guard let id = tokenToID[Array(literal.utf8)] else {
                throw TokenizerError.missingSpecial(literal)
            }
            return id
        }

        let p = GLM52ConversationProtocol.self
        let mask = try required(p.mask)
        let endOfText = try required(p.endOfText)
        if let bosTokenID, bosTokenID != mask {
            throw TokenizerError.inconsistentSpecialID(
                p.mask, expected: mask, actual: bosTokenID
            )
        }
        if let eosTokenID, eosTokenID != endOfText {
            throw TokenizerError.inconsistentSpecialID(
                p.endOfText, expected: endOfText, actual: eosTokenID
            )
        }

        self.special = try GLM52SpecialTokenIDs(
            mask: mask,
            startOfPrompt: required(p.startOfPrompt),
            endOfText: endOfText,
            system: required(p.system),
            user: required(p.user),
            assistant: required(p.assistant),
            observation: required(p.observation),
            thinkOpen: required(p.thinkOpen),
            thinkClose: required(p.thinkClose),
            toolCallOpen: required(p.toolCallOpen),
            toolCallClose: required(p.toolCallClose),
            toolResponseOpen: required(p.toolResponseOpen),
            toolResponseClose: required(p.toolResponseClose),
            argumentKeyOpen: required(p.argumentKeyOpen),
            argumentKeyClose: required(p.argumentKeyClose),
            argumentValueOpen: required(p.argumentValueOpen),
            argumentValueClose: required(p.argumentValueClose)
        )

        let literalIDs: [(String, Int32)] = [
            (p.mask, special.mask), (p.startOfPrompt, special.startOfPrompt),
            (p.endOfText, special.endOfText),
            (p.system, special.system), (p.user, special.user),
            (p.assistant, special.assistant), (p.observation, special.observation),
            (p.thinkOpen, special.thinkOpen), (p.thinkClose, special.thinkClose),
            (p.toolCallOpen, special.toolCallOpen), (p.toolCallClose, special.toolCallClose),
            (p.toolResponseOpen, special.toolResponseOpen),
            (p.toolResponseClose, special.toolResponseClose),
            (p.argumentKeyOpen, special.argumentKeyOpen),
            (p.argumentKeyClose, special.argumentKeyClose),
            (p.argumentValueOpen, special.argumentValueOpen),
            (p.argumentValueClose, special.argumentValueClose),
        ]
        var buckets = [[(bytes: [UInt8], id: Int32)]](repeating: [], count: 256)
        for (literal, id) in literalIDs {
            let bytes = Array(literal.utf8)
            buckets[Int(bytes[0])].append((bytes, id))
        }
        for index in buckets.indices {
            buckets[index].sort { $0.bytes.count > $1.bytes.count }
        }
        self.specialsByFirstByte = buckets
    }

    public func tokenID(_ literal: String) -> Int32? {
        tokenToID[Array(literal.utf8)]
    }

    public func tokenize(_ text: String) -> [Int32] {
        var output: [Int32] = []
        for piece in GLM4Pretokenizer.split(text) {
            emitPiece(piece[...], into: &output)
        }
        return output
    }

    public func tokenizeRenderedChat(_ text: String) -> [Int32] {
        let bytes = Array(text.utf8)
        var output: [Int32] = []
        var spanStart = 0
        var position = 0
        while position < bytes.count {
            var match: (bytes: [UInt8], id: Int32)?
            for candidate in specialsByFirstByte[Int(bytes[position])] {
                let end = position + candidate.bytes.count
                if end <= bytes.count,
                   bytes[position..<end].elementsEqual(candidate.bytes) {
                    match = candidate
                    break
                }
            }
            guard let match else {
                position += 1
                continue
            }
            tokenizeSpan(bytes[spanStart..<position], into: &output)
            output.append(match.id)
            position += match.bytes.count
            spanStart = position
        }
        tokenizeSpan(bytes[spanStart..<bytes.count], into: &output)
        return output
    }

    public func tokenText(_ id: Int32) -> [UInt8] {
        guard id >= 0, Int(id) < tokens.count else { return [] }
        let token = tokens[Int(id)]
        if special.all.contains(id) { return token }
        var output: [UInt8] = []
        var position = 0
        while position < token.count {
            let decoded = ByteLevel.decodeOne(token, token.count, position)
            position = decoded.next
            if let byte = ByteLevel.codepointToByte(decoded.cp) { output.append(byte) }
        }
        return output
    }

    public func isStopToken(_ id: Int32, reasoning: ThinkMode) -> Bool {
        if special.generationStops.contains(id) { return true }
        return reasoning == .none && (id == special.thinkOpen || id == special.thinkClose)
    }

    public func neutralizeSpecialTokenLiterals(in text: String) -> String {
        GLM52ConversationProtocol.neutralizeControlTokens(in: text)
    }

    public func encodeChatPrompt(system: String? = nil, prompt: String,
                                 reasoning: ThinkMode = .none) throws -> [Int32] {
        let rendered = try GLM52ChatRenderer.renderPrompt(
            system: system,
            prompt: prompt,
            reasoning: reasoning
        )
        return tokenizeRenderedChat(rendered)
    }

    private func tokenizeSpan(_ bytes: ArraySlice<UInt8>, into output: inout [Int32]) {
        guard !bytes.isEmpty else { return }
        for piece in GLM4Pretokenizer.split(Array(bytes)) {
            emitPiece(piece[...], into: &output)
        }
    }

    private func emitPiece(_ piece: ArraySlice<UInt8>, into output: inout [Int32]) {
        let encoded = ByteLevel.byteEncode(piece)
        var symbols: [[UInt8]] = []
        var position = 0
        while position < encoded.count {
            var length = ByteLevel.utf8Len(encoded[position])
            if position + length > encoded.count { length = 1 }
            symbols.append(Array(encoded[position..<position + length]))
            position += length
        }

        while true {
            var bestIndex = -1
            var bestRank = Int32.max
            guard symbols.count > 1 else { break }
            for index in 0..<(symbols.count - 1) {
                var key = symbols[index]
                key.append(0x20)
                key.append(contentsOf: symbols[index + 1])
                if let rank = mergeRank[key], rank < bestRank {
                    bestIndex = index
                    bestRank = rank
                }
            }
            guard bestIndex >= 0 else { break }
            symbols[bestIndex].append(contentsOf: symbols[bestIndex + 1])
            symbols.remove(at: bestIndex + 1)
        }

        for symbol in symbols {
            if let id = tokenToID[symbol] {
                output.append(id)
            } else {
                for byte in symbol {
                    if let id = tokenToID[[byte]] { output.append(id) }
                }
            }
        }
    }

    private static func metadataTokenID(_ value: UInt64?) -> Int32? {
        guard let value, value <= UInt64(Int32.max) else { return nil }
        return Int32(value)
    }
}
