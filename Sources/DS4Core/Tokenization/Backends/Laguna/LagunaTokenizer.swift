import Foundation

/// Special-token ids needed by Laguna S 2.1 chat, tools and generation
/// stopping.
///
/// Laguna keeps a deliberately small control set: `<assistant>`,
/// `</assistant>`, `<think>`, `</think>`, `<tool_call>` and `</tool_call>` are
/// vocabulary tokens, while `<system>`, `<user>`, `<tool_response>` and the
/// argument tags remain ordinary BPE text.
public struct LagunaSpecialTokenIDs: Sendable, Equatable {
    public let beginOfSequence: Int32
    public let endOfSequence: Int32
    /// `</assistant>` — the end-of-turn id (metadata `eot_token_id` when
    /// present, otherwise the vocabulary lookup).
    public let endOfTurn: Int32
    public let assistant: Int32
    public let thinkOpen: Int32
    public let thinkClose: Int32
    public let toolCallOpen: Int32
    public let toolCallClose: Int32

    /// `vocab_token_is_generation_stop` in upstream `ds4.c`: the Laguna family
    /// stops on EOS and on the `</assistant>` end-of-turn token only.
    public var generationStops: Set<Int32> {
        [endOfSequence, endOfTurn]
    }

    public var all: Set<Int32> {
        generationStops.union([
            beginOfSequence, assistant,
            thinkOpen, thinkClose,
            toolCallOpen, toolCallClose,
        ])
    }
}

/// Laguna S 2.1 pre-tokenizer.
///
/// Laguna's first pre-tokenizer regex separates non-newline spans from runs of
/// LF bytes before applying the GPT-2-style expression (the GLM4 split shape
/// with single-digit number groups).  This is observable for CRLF: CR remains
/// in the preceding span and LF starts a new one, so they must not merge into
/// one BPE piece (`bpe_tokenize_text_laguna` in upstream `ds4.c`).
enum LagunaPretokenizer {
    static func split(_ text: String) -> [[UInt8]] {
        split(Array(text.utf8))
    }

    static func split(_ bytes: [UInt8]) -> [[UInt8]] {
        var pieces: [[UInt8]] = []
        var position = 0
        while position < bytes.count {
            let start = position
            if bytes[position] == 0x0a {
                while position < bytes.count, bytes[position] == 0x0a { position += 1 }
            } else {
                while position < bytes.count, bytes[position] != 0x0a { position += 1 }
            }
            pieces.append(contentsOf: GLM4Pretokenizer.split(
                Array(bytes[start..<position]),
                maxDigits: 1
            ))
        }
        return pieces
    }
}

/// GPT-2 byte-level BPE with Laguna pre-tokenization and native rendered-chat
/// special-token scanning.
///
/// Upstream selects the pre-tokenizer from the model family, not from
/// `tokenizer.ggml.pre`, so this initializer intentionally does not reject a
/// particular `pre` string; the architecture and geometry validation already
/// pin the exact supported GGUFs.
public final class LagunaTokenizer: TokenizerProtocol {
    public enum TokenizerError: Error, Sendable, Equatable, CustomStringConvertible {
        case missingTable(String)
        case missingMetadata(String)
        case missingSpecial(String)
        case invalidTokenID(String, Int64)
        case vocabularyTooLarge

        public var description: String {
            switch self {
            case .missingTable(let name): return "missing Laguna tokenizer table \(name)"
            case .missingMetadata(let key): return "missing Laguna tokenizer metadata \(key)"
            case .missingSpecial(let name): return "missing Laguna special token \(name)"
            case .invalidTokenID(let key, let value):
                return "Laguna tokenizer metadata \(key) has out-of-range id \(value)"
            case .vocabularyTooLarge: return "Laguna vocabulary exceeds Int32 token ids"
            }
        }
    }

    public let architecture: ModelArchitectureID = .laguna
    public let tokens: [[UInt8]]
    public let nVocab: Int
    public let special: LagunaSpecialTokenIDs

    private let tokenToID: [[UInt8]: Int32]
    private let mergeRank: [[UInt8]: Int32]
    private let specialsByFirstByte: [[(bytes: [UInt8], id: Int32)]]

    public convenience init(model: GGUFModel) throws {
        let detected = try ModelArchitectureDetector.detect(in: model)
        guard detected.id == .laguna else {
            throw ModelArchitectureError.unsupportedArchitecture(detected.id)
        }
        guard let tokens = model.stringArrayBytes("tokenizer.ggml.tokens") else {
            throw TokenizerError.missingTable("tokenizer.ggml.tokens")
        }
        guard let merges = model.stringArrayBytes("tokenizer.ggml.merges") else {
            throw TokenizerError.missingTable("tokenizer.ggml.merges")
        }
        // BOS/EOS come from metadata and are required, matching the reference
        // loader; the end-of-turn id may fall back to the `</assistant>`
        // vocabulary entry.
        let bos = try Self.requiredTokenID(model, "tokenizer.ggml.bos_token_id",
                                           vocabularySize: tokens.count)
        let eos = try Self.requiredTokenID(model, "tokenizer.ggml.eos_token_id",
                                           vocabularySize: tokens.count)
        let eot = try Self.optionalTokenID(model, "tokenizer.ggml.eot_token_id",
                                           vocabularySize: tokens.count)
        try self.init(tokens: tokens, merges: merges,
                      bosTokenID: bos, eosTokenID: eos, eotTokenID: eot)
    }

    /// Model-free initializer used by deterministic tokenizer fixtures.
    init(tokens: [[UInt8]], merges: [[UInt8]],
         bosTokenID: Int32, eosTokenID: Int32, eotTokenID: Int32? = nil) throws {
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

        let p = LagunaConversationProtocol.self
        // `??` would wrap the fallback in an autoclosure, and a closure formed
        // here may not capture anything through `self` before every stored
        // property is initialized (Swift 6 definite initialization).
        let endOfTurn: Int32
        if let eotTokenID {
            endOfTurn = eotTokenID
        } else {
            endOfTurn = try required(p.assistantClose)
        }
        let special = try LagunaSpecialTokenIDs(
            beginOfSequence: bosTokenID,
            endOfSequence: eosTokenID,
            endOfTurn: endOfTurn,
            assistant: required(p.assistantOpen),
            thinkOpen: required(p.thinkOpen),
            thinkClose: required(p.thinkClose),
            toolCallOpen: required(p.toolCallOpen),
            toolCallClose: required(p.toolCallClose)
        )
        self.special = special

        // The active Laguna rows of the upstream rendered-chat scanner: the
        // sequence marker maps onto the EOS id (Poolside reuses one token for
        // BOS and EOS), and the cross-family rows whose ids stay set for
        // Laguna (`special_token_at` skips a row only when its id is
        // negative) remain atomic exactly like in the reference.
        let crossFamily: [(String, Int32)] =
            p.crossFamilyScannerLiterals.map { row in
                switch row.special {
                case .beginOfSequence: return (row.literal, special.beginOfSequence)
                case .endOfSequence: return (row.literal, special.endOfSequence)
                case .assistant: return (row.literal, special.assistant)
                }
            }
        let literalIDs: [(String, Int32)] = [
            (p.bosMarker, special.endOfSequence),
            (p.assistantOpen, special.assistant),
            (p.assistantClose, special.endOfTurn),
            (p.thinkOpen, special.thinkOpen),
            (p.thinkClose, special.thinkClose),
            (p.toolCallOpen, special.toolCallOpen),
            (p.toolCallClose, special.toolCallClose),
        ] + crossFamily
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
        for piece in LagunaPretokenizer.split(text) {
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

    /// Laguna stops generation on EOS or `</assistant>` only; reasoning and
    /// tool markers are content, never turn boundaries.
    public func isStopToken(_ id: Int32) -> Bool {
        special.generationStops.contains(id)
    }

    /// `ds4_token_is_stop_for_think_mode`: with thinking disabled the prompt
    /// already supplied the protocol close tag, so a generated `<think>` or
    /// `</think>` is a control marker to stop on, not assistant content.
    public func isStopToken(_ id: Int32, reasoning: ThinkMode) -> Bool {
        if isStopToken(id) { return true }
        if !reasoning.enabled {
            return id == special.thinkOpen || id == special.thinkClose
        }
        return false
    }

    public func neutralizeSpecialTokenLiterals(in text: String) -> String {
        LagunaConversationProtocol.neutralizeControlTokens(in: text)
    }

    /// The reference CLI prompt encoder (`encode_chat_prompt` in `ds4.c`):
    /// the metadata BOS id, an optional `<system>` block (no default system
    /// prompt on this path), the `<user>` block, then the assistant opener
    /// with the thinking marker as dedicated tokens.  Each wrapped block is
    /// scanned as one contiguous rendered string so BPE merges can cross
    /// tag/content boundaries (`laguna_chat_append_wrapped`).  Server-style
    /// transcripts (default Poolside system prompt, tools section) go through
    /// `LagunaChatRenderer.render` + `tokenizeRenderedChat` instead.
    public func encodeChatPrompt(system: String? = nil, prompt: String,
                                 reasoning: ThinkMode = .none) throws -> [Int32] {
        let p = LagunaConversationProtocol.self
        var output: [Int32] = [special.beginOfSequence]
        if let system, !system.isEmpty {
            output += tokenizeRenderedChat(
                p.systemOpen + system + p.systemClose + "\n"
            )
        }
        output += tokenizeRenderedChat(
            p.userOpen + prompt + p.userClose + "\n"
        )
        output.append(special.assistant)
        output.append(reasoning.enabled ? special.thinkOpen : special.thinkClose)
        return output
    }

    private func tokenizeSpan(_ bytes: ArraySlice<UInt8>, into output: inout [Int32]) {
        guard !bytes.isEmpty else { return }
        for piece in LagunaPretokenizer.split(Array(bytes)) {
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

    private static func requiredTokenID(_ model: GGUFModel, _ key: String,
                                        vocabularySize: Int) throws -> Int32 {
        guard let raw = model.u64Compat(key) else {
            throw TokenizerError.missingMetadata(key)
        }
        return try validatedTokenID(raw, key: key, vocabularySize: vocabularySize)
    }

    private static func optionalTokenID(_ model: GGUFModel, _ key: String,
                                        vocabularySize: Int) throws -> Int32? {
        guard let raw = model.u64Compat(key) else { return nil }
        return try validatedTokenID(raw, key: key, vocabularySize: vocabularySize)
    }

    private static func validatedTokenID(_ raw: UInt64, key: String,
                                         vocabularySize: Int) throws -> Int32 {
        guard raw <= UInt64(Int32.max), Int(raw) < vocabularySize else {
            throw TokenizerError.invalidTokenID(key, Int64(bitPattern: UInt64(raw)))
        }
        return Int32(raw)
    }
}
