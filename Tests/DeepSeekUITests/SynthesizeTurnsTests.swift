import XCTest
@testable import DeepSeekUI

/// `ChatStore.synthesizeTurns` is the legacy ↔ v2 bridge. PR 3a
/// exposes it as the read surface for `turns(of:)`; PR 3b will read
/// summary files directly for v2 chats, but the synth path must
/// keep working for legacy chats that never get a v2 folder. These
/// tests pin the grouping semantics so any future tweak that breaks
/// the existing "consecutive assistants belong to the same turn"
/// rule fails loudly.
final class SynthesizeTurnsTests: XCTestCase {

    // MARK: grouping

    func testSingleUserNoReply_returnsOneTurnNoRounds() {
        let user = StoredMessage(role: .user, content: "hi")
        let turns = ChatStore.synthesizeTurns(from: [user])
        XCTAssertEqual(turns.count, 1)
        XCTAssertEqual(turns[0].userText, "hi")
        XCTAssertTrue(turns[0].roundIDs.isEmpty)
        XCTAssertFalse(turns[0].flags.contains(.isSystem))
    }

    func testUserPlusAssistant_returnsOneTurnOneRound() {
        let user = StoredMessage(role: .user, content: "q")
        let asst = StoredMessage(role: .assistant, content: "a")
        let turns = ChatStore.synthesizeTurns(from: [user, asst])
        XCTAssertEqual(turns.count, 1)
        XCTAssertEqual(turns[0].roundIDs, [asst.id])
        XCTAssertEqual(turns[0].finalContentPreview, "a")
    }

    func testToolRoundtrips_collapseIntoOneTurn() {
        let user = StoredMessage(role: .user, content: "do it")
        let r1 = StoredMessage(role: .assistant, content: "",
                                toolCalls: [
                                    StoredToolCall(name: "ls",
                                                    args: "{}")
                                ])
        let r2 = StoredMessage(role: .assistant, content: "",
                                toolCalls: [
                                    StoredToolCall(name: "grep",
                                                    args: "{}")
                                ])
        let r3 = StoredMessage(role: .assistant, content: "done")
        let turns = ChatStore.synthesizeTurns(from: [user, r1, r2, r3])
        XCTAssertEqual(turns.count, 1)
        XCTAssertEqual(turns[0].roundIDs, [r1.id, r2.id, r3.id])
        XCTAssertEqual(turns[0].finalContentPreview, "done")
        XCTAssertEqual(turns[0].toolCallCount, 2)
        XCTAssertTrue(turns[0].flags.contains(.hasToolCalls))
    }

    func testTwoUserTurns_returnsTwoTurns() {
        let u1 = StoredMessage(role: .user, content: "first")
        let a1 = StoredMessage(role: .assistant, content: "one")
        let u2 = StoredMessage(role: .user, content: "second")
        let a2 = StoredMessage(role: .assistant, content: "two")
        let turns = ChatStore.synthesizeTurns(from: [u1, a1, u2, a2])
        XCTAssertEqual(turns.count, 2)
        XCTAssertEqual(turns[0].userText, "first")
        XCTAssertEqual(turns[0].finalContentPreview, "one")
        XCTAssertEqual(turns[1].userText, "second")
        XCTAssertEqual(turns[1].finalContentPreview, "two")
    }

    func testSystemMessage_standsAloneAsSystemTurn() {
        let sys = StoredMessage(role: .system, content: "you are a helper")
        let u = StoredMessage(role: .user, content: "hi")
        let a = StoredMessage(role: .assistant, content: "hello")
        let turns = ChatStore.synthesizeTurns(from: [sys, u, a])
        XCTAssertEqual(turns.count, 2)
        XCTAssertTrue(turns[0].flags.contains(.isSystem))
        XCTAssertTrue(turns[0].roundIDs.isEmpty)
        XCTAssertEqual(turns[0].userText, "you are a helper")
        XCTAssertEqual(turns[1].userText, "hi")
        XCTAssertEqual(turns[1].finalContentPreview, "hello")
    }

    // MARK: flag aggregation

    func testHasReasoningFlag_setWhenAnyRoundCarriesReasoning() {
        let u = StoredMessage(role: .user, content: "q")
        let a = StoredMessage(
            role: .assistant, content: "a",
            reasoningContent: "thinking out loud")
        let turns = ChatStore.synthesizeTurns(from: [u, a])
        XCTAssertTrue(turns[0].flags.contains(.hasReasoning))
    }

    func testHasPrefillTraceFlag_setWhenAnyRoundCarriesTrace() {
        let u = StoredMessage(role: .user, content: "q")
        let a = StoredMessage(
            role: .assistant, content: "a",
            prefillTrace: "decoded prompt…")
        let turns = ChatStore.synthesizeTurns(from: [u, a])
        XCTAssertTrue(turns[0].flags.contains(.hasPrefillTrace))
    }

    // MARK: preview truncation

    func testShortContent_isNotTruncated() {
        let u = StoredMessage(role: .user, content: "q")
        let short = String(repeating: "x", count: 100)
        let a = StoredMessage(role: .assistant, content: short)
        let turns = ChatStore.synthesizeTurns(from: [u, a])
        XCTAssertEqual(turns[0].finalContentPreview, short)
        XCTAssertFalse(turns[0].finalContentIsTruncated)
    }

    func testLongContent_isTruncatedToPreviewSize() {
        let u = StoredMessage(role: .user, content: "q")
        let long = String(repeating: "x", count: 3000)
        let a = StoredMessage(role: .assistant, content: long)
        let turns = ChatStore.synthesizeTurns(from: [u, a])
        XCTAssertEqual(turns[0].finalContentPreview.count, 2048)
        XCTAssertTrue(turns[0].finalContentIsTruncated)
    }

    // MARK: synthesizeTurnsAndRounds (bulk path used by syncV2State)

    func testBulkSynth_singlePassMatchesPerTurnAndPerRoundPaths() {
        let u1 = StoredMessage(role: .user, content: "a")
        let a1 = StoredMessage(role: .assistant, content: "ans1",
                                reasoningContent: "r1")
        let a2 = StoredMessage(role: .assistant, content: "ans2",
                                toolCalls: [
                                    StoredToolCall(name: "t",
                                                    args: "{}")
                                ])
        let u2 = StoredMessage(role: .user, content: "b")
        let a3 = StoredMessage(role: .assistant, content: "ans3")
        let messages = [u1, a1, a2, u2, a3]

        let pairs = ChatStore.synthesizeTurnsAndRounds(from: messages)
        XCTAssertEqual(pairs.count, 2)

        // Turn 1: lead u1, two rounds (a1, a2)
        XCTAssertEqual(pairs[0].summary.userText, "a")
        XCTAssertEqual(pairs[0].summary.roundIDs, [a1.id, a2.id])
        XCTAssertEqual(pairs[0].rounds.count, 2)
        XCTAssertEqual(pairs[0].rounds[0].content, "ans1")
        XCTAssertEqual(pairs[0].rounds[0].reasoningContent, "r1")
        XCTAssertEqual(pairs[0].rounds[0].roundIndex, 0)
        XCTAssertEqual(pairs[0].rounds[1].content, "ans2")
        XCTAssertEqual(pairs[0].rounds[1].toolCalls.first?.name, "t")
        XCTAssertEqual(pairs[0].rounds[1].roundIndex, 1)

        // Turn 2: lead u2, one round (a3)
        XCTAssertEqual(pairs[1].summary.userText, "b")
        XCTAssertEqual(pairs[1].rounds.count, 1)
        XCTAssertEqual(pairs[1].rounds[0].content, "ans3")

        // Equivalence: synthesizeTurns is just .map { $0.summary }
        let turns = ChatStore.synthesizeTurns(from: messages)
        XCTAssertEqual(turns, pairs.map { $0.summary })
    }

    // MARK: synthesizeRound

    func testSynthesizeRound_returnsAssistantPayloadByID() {
        let u = StoredMessage(role: .user, content: "q")
        let a1 = StoredMessage(role: .assistant, content: "first",
                                reasoningContent: "r1")
        let a2 = StoredMessage(role: .assistant, content: "second",
                                toolCalls: [
                                    StoredToolCall(name: "x",
                                                    args: "{}")
                                ])
        let round = ChatStore.synthesizeRound(
            from: [u, a1, a2], turnID: u.id, roundID: a2.id)
        XCTAssertNotNil(round)
        XCTAssertEqual(round?.content, "second")
        XCTAssertEqual(round?.roundIndex, 1)
        XCTAssertEqual(round?.toolCalls.first?.name, "x")
    }

    func testSynthesizeRound_returnsNilOnUnknownTurnID() {
        let u = StoredMessage(role: .user, content: "q")
        let a = StoredMessage(role: .assistant, content: "a")
        let round = ChatStore.synthesizeRound(
            from: [u, a], turnID: UUID(), roundID: a.id)
        XCTAssertNil(round)
    }

    func testSynthesizeRound_returnsNilOnUnknownRoundID() {
        let u = StoredMessage(role: .user, content: "q")
        let a = StoredMessage(role: .assistant, content: "a")
        let round = ChatStore.synthesizeRound(
            from: [u, a], turnID: u.id, roundID: UUID())
        XCTAssertNil(round)
    }
}

/// Convenience init that lets test cases build a `StoredToolCall`
/// without going through `ToolCall` from DeepSeekKit. The runtime
/// type has only stored properties + a single `init(_ tc: ToolCall)`
/// — the @testable boundary lets us reach the internal stored
/// fields directly from an extension here.
private extension StoredToolCall {
    init(name: String, args: String, id: String? = nil) {
        self.name = name
        self.args = args
        self.id = id
    }
}
