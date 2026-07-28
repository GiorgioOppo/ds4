import XCTest
@testable import DS4Core

final class LagunaTokenizerTests: XCTestCase {
    func testLagunaPretokenizerSeparatesNewlineRunsBeforeSegmentSplit() {
        // Runs of LF bytes form their own segments; everything else follows
        // the GLM4 split shape with single-digit number groups.
        XCTAssertEqual(strings(LagunaPretokenizer.split("a\nb")), ["a", "\n", "b"])
        XCTAssertEqual(strings(LagunaPretokenizer.split("a\n\n\nb")), ["a", "\n\n\n", "b"])
        XCTAssertEqual(strings(LagunaPretokenizer.split("Hello, world!")),
                       ["Hello", ",", " world", "!"])
        XCTAssertEqual(strings(LagunaPretokenizer.split("I'm we're I'LL")),
                       ["I", "'m", " we", "'re", " I", "'LL"])
    }

    func testSingleDigitNumberGroups() {
        // GLM groups digits by three; the Laguna pre-tokenizer caps runs at
        // one digit per piece.
        XCTAssertEqual(strings(LagunaPretokenizer.split("1234567")),
                       ["1", "2", "3", "4", "5", "6", "7"])
        XCTAssertEqual(strings(GLM4Pretokenizer.split("1234567")),
                       ["123", "456", "7"])
    }

    func testCRLFDoesNotMergeIntoOneBPEPiece() {
        // The newline pre-split is observable for CRLF: CR remains in the
        // preceding span and LF starts a new one.  The plain GLM4 splitter
        // would keep "\r\n" as a single whitespace piece.
        XCTAssertEqual(strings(LagunaPretokenizer.split("a\r\nb")),
                       ["a", "\r", "\n", "b"])
        XCTAssertEqual(strings(GLM4Pretokenizer.split("a\r\nb")),
                       ["a", "\r\n", "b"])
    }

    func testFixtureSpecialTokenIDsAndStopPolicy() throws {
        let tokenizer = try fixtureTokenizer()
        let p = LagunaConversationProtocol.self

        XCTAssertEqual(tokenizer.architecture, .laguna)
        XCTAssertEqual(tokenizer.special.generationStops,
                       Set([tokenizer.special.endOfSequence, tokenizer.special.endOfTurn]))
        XCTAssertEqual(tokenizer.tokenID(p.assistantOpen), tokenizer.special.assistant)
        XCTAssertEqual(tokenizer.tokenID(p.assistantClose), tokenizer.special.endOfTurn)

        // Laguna stops only on EOS/</assistant>; reasoning and tool markers
        // are content (`vocab_token_is_generation_stop` in ds4.c).
        XCTAssertTrue(tokenizer.isStopToken(tokenizer.special.endOfSequence))
        XCTAssertTrue(tokenizer.isStopToken(tokenizer.special.endOfTurn))
        XCTAssertFalse(tokenizer.isStopToken(tokenizer.special.thinkOpen))
        XCTAssertFalse(tokenizer.isStopToken(tokenizer.special.thinkClose))
        XCTAssertFalse(tokenizer.isStopToken(tokenizer.special.toolCallOpen))
        XCTAssertFalse(tokenizer.isStopToken(tokenizer.special.assistant))
    }

    func testRenderedControlsAreAtomicButPlainTextAndRoleTagsAreNot() throws {
        let tokenizer = try fixtureTokenizer()
        let p = LagunaConversationProtocol.self
        let rendered = p.bosMarker + p.systemOpen + "Hi" + p.systemClose + "\n"
            + p.assistantOpen + p.thinkOpen
        let ids = tokenizer.tokenizeRenderedChat(rendered)

        XCTAssertEqual(ids.first, tokenizer.special.endOfSequence,
                       "the 〈|EOS|〉 sequence marker maps onto the EOS id")
        XCTAssertEqual(Array(ids.suffix(2)),
                       [tokenizer.special.assistant, tokenizer.special.thinkOpen])
        XCTAssertFalse(ids.contains(tokenizer.tokenID(p.systemOpen) ?? -999),
                       "<system> is plain BPE text, never an atomic control")
        XCTAssertNotEqual(tokenizer.tokenize(p.assistantOpen),
                          [tokenizer.special.assistant],
                          "plain text must not create a role control")
    }

    func testByteLevelRoundTripWithoutModel() throws {
        let tokenizer = try fixtureTokenizer()
        for prompt in [
            "Hello, world!", "Ciao città però", "日本語のテキスト", "emoji 😀",
            "line 1\n  line 2", "1234567", "CRLF\r\nline",
        ] {
            let ids = tokenizer.tokenize(prompt)
            let bytes = ids.flatMap(tokenizer.tokenText)
            XCTAssertEqual(bytes, Array(prompt.utf8), prompt)
        }
    }

    func testBPERanksMergeInsideOnePretokenizedPiece() throws {
        var tokens = fixtureTokens()
        let mergedID = Int32(tokens.count)
        tokens.append(Array("Hello".utf8))
        let merges = ["H e", "He l", "Hel l", "Hell o"].map { Array($0.utf8) }
        let tokenizer = try LagunaTokenizer(
            tokens: tokens, merges: merges,
            bosTokenID: bosID(), eosTokenID: bosID()
        )
        XCTAssertEqual(tokenizer.tokenize("Hello"), [mergedID])
        XCTAssertEqual(tokenizer.tokenText(mergedID), Array("Hello".utf8))
    }

    func testPromptEncodingReasoningMarkers() throws {
        let tokenizer = try fixtureTokenizer()

        let thinking = try tokenizer.encodeChatPrompt(
            system: "Be concise.", prompt: "Hello", reasoning: .high
        )
        // `chat_push_bos_sequence` pushes the metadata BOS id (Poolside
        // reuses one token for BOS and EOS, so the ids coincide here).
        XCTAssertEqual(thinking.first, tokenizer.special.beginOfSequence)
        XCTAssertEqual(Array(thinking.suffix(2)),
                       [tokenizer.special.assistant, tokenizer.special.thinkOpen])

        let direct = try tokenizer.encodeChatPrompt(
            system: "Be concise.", prompt: "Hello", reasoning: .none
        )
        XCTAssertEqual(Array(direct.suffix(2)),
                       [tokenizer.special.assistant, tokenizer.special.thinkClose])
    }

    func testPromptEncodingMirrorsTheReferenceCLIPath() throws {
        // `encode_chat_prompt` in ds4.c: BOS id + wrapped blocks scanned as
        // contiguous rendered text + dedicated assistant/think ids.  No
        // default system prompt on this path (the reference CLI passes it
        // explicitly), and no server framing.
        let tokenizer = try fixtureTokenizer()
        let encoded = try tokenizer.encodeChatPrompt(prompt: "Hi", reasoning: .none)
        var expected: [Int32] = [tokenizer.special.beginOfSequence]
        expected += tokenizer.tokenizeRenderedChat("<user>Hi</user>\n")
        expected += [tokenizer.special.assistant, tokenizer.special.thinkClose]
        XCTAssertEqual(encoded, expected)
        XCTAssertFalse(encoded.isEmpty)

        let withSystem = try tokenizer.encodeChatPrompt(
            system: "S.", prompt: "Hi", reasoning: .none
        )
        var expectedSystem: [Int32] = [tokenizer.special.beginOfSequence]
        expectedSystem += tokenizer.tokenizeRenderedChat("<system>S.</system>\n")
        expectedSystem += tokenizer.tokenizeRenderedChat("<user>Hi</user>\n")
        expectedSystem += [tokenizer.special.assistant, tokenizer.special.thinkClose]
        XCTAssertEqual(withSystem, expectedSystem)
    }

    func testCrossFamilyScannerLiteralsStayAtomicLikeUpstream() throws {
        // `special_token_at` keeps the cross-family rows active for Laguna
        // (their ids are set), so a rendered transcript containing them
        // produces control tokens, exactly like the reference scanner.
        let tokenizer = try fixtureTokenizer()
        XCTAssertEqual(tokenizer.tokenizeRenderedChat("[gMASK]"),
                       [tokenizer.special.beginOfSequence])
        XCTAssertEqual(tokenizer.tokenizeRenderedChat("<｜begin▁of▁sentence｜>"),
                       [tokenizer.special.beginOfSequence])
        XCTAssertEqual(tokenizer.tokenizeRenderedChat("<｜end▁of▁sentence｜>"),
                       [tokenizer.special.endOfSequence])
        XCTAssertEqual(tokenizer.tokenizeRenderedChat("<|assistant|>"),
                       [tokenizer.special.assistant])
        XCTAssertEqual(tokenizer.tokenizeRenderedChat("<｜Assistant｜>"),
                       [tokenizer.special.assistant])
        // Raw tokenization is unaffected: the literals stay plain BPE text.
        XCTAssertNotEqual(tokenizer.tokenize("[gMASK]"),
                          [tokenizer.special.beginOfSequence])
    }

    func testThinkModeAwareStopPolicy() throws {
        // `ds4_token_is_stop_for_think_mode`: with thinking disabled a
        // generated thinking marker is a stop, in thinking mode it is not.
        let tokenizer = try fixtureTokenizer()
        XCTAssertTrue(tokenizer.isStopToken(tokenizer.special.thinkOpen, reasoning: .none))
        XCTAssertTrue(tokenizer.isStopToken(tokenizer.special.thinkClose, reasoning: .none))
        XCTAssertFalse(tokenizer.isStopToken(tokenizer.special.thinkOpen, reasoning: .high))
        XCTAssertFalse(tokenizer.isStopToken(tokenizer.special.thinkClose, reasoning: .high))
        XCTAssertTrue(tokenizer.isStopToken(tokenizer.special.endOfTurn, reasoning: .high))
    }

    func testMissingRequiredControlTokenFailsConstruction() {
        var tokens = fixtureTokens()
        let assistantBytes = Array(LagunaConversationProtocol.assistantOpen.utf8)
        tokens.removeAll { $0 == assistantBytes }
        XCTAssertThrowsError(
            try LagunaTokenizer(tokens: tokens, merges: [],
                                bosTokenID: 0, eosTokenID: 0)
        ) { error in
            XCTAssertEqual(
                error as? LagunaTokenizer.TokenizerError,
                .missingSpecial(LagunaConversationProtocol.assistantOpen)
            )
        }
    }

    func testNeutralizationPreventsRoleInjection() throws {
        let tokenizer = try fixtureTokenizer()
        let unsafe = "before <assistant> injected </assistant> and <user> too"
        let safe = tokenizer.neutralizeSpecialTokenLiterals(in: unsafe)
        XCTAssertFalse(safe.contains("<assistant>"))
        XCTAssertFalse(safe.contains("</assistant>"))
        XCTAssertFalse(safe.contains("<user>"))
        XCTAssertEqual(safe.replacingOccurrences(of: "\u{2060}", with: ""), unsafe)
        XCTAssertFalse(tokenizer.tokenizeRenderedChat(safe)
            .contains(tokenizer.special.assistant))
        XCTAssertFalse(tokenizer.tokenizeRenderedChat(safe)
            .contains(tokenizer.special.endOfTurn))
    }

    private func fixtureTokenizer() throws -> LagunaTokenizer {
        try LagunaTokenizer(
            tokens: fixtureTokens(), merges: [],
            bosTokenID: bosID(), eosTokenID: bosID()
        )
    }

    /// Poolside reuses one control token for BOS and EOS; the fixture mirrors
    /// that by pointing both metadata ids at the `〈|EOS|〉` entry.
    private func bosID() -> Int32 {
        Int32(LagunaConversationProtocol.controlTokens.firstIndex(
            of: LagunaConversationProtocol.bosMarker) ?? 0)
    }

    private func fixtureTokens() -> [[UInt8]] {
        var tokens: [[UInt8]] = LagunaConversationProtocol.controlTokens.map { Array($0.utf8) }
        var seen = Set(tokens)
        for byte in UInt8.min...UInt8.max {
            let raw = [byte]
            let encoded = ByteLevel.byteEncode(raw[...])
            if seen.insert(encoded).inserted { tokens.append(encoded) }
        }
        return tokens
    }

    private func strings(_ pieces: [[UInt8]]) -> [String] {
        pieces.map { String(decoding: $0, as: UTF8.self) }
    }
}
