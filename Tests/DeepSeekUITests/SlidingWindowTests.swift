import XCTest
@testable import DeepSeekUI

/// `ChatStore.applySlidingWindow` is the cost-cap for remote
/// chats: it bounds how much history `runRemoteLoop` re-sends to
/// the provider on each turn. Tests pin the boundary behaviour so a
/// future refactor that quietly drops system messages or skews the
/// "last N user turns" definition fails loudly.
final class SlidingWindowTests: XCTestCase {

    private func user(_ s: String) -> StoredMessage {
        StoredMessage(role: .user, content: s)
    }
    private func assistant(_ s: String) -> StoredMessage {
        StoredMessage(role: .assistant, content: s)
    }
    private func system(_ s: String) -> StoredMessage {
        StoredMessage(role: .system, content: s)
    }

    func testZeroCapReturnsEverything() {
        let msgs = [user("a"), assistant("A"), user("b"), assistant("B")]
        let out = ChatStore.applySlidingWindow(msgs, keepLastNUserTurns: 0)
        XCTAssertEqual(out.map(\.content), ["a", "A", "b", "B"])
    }

    func testCapAboveCountIsNoOp() {
        let msgs = [user("a"), assistant("A")]
        let out = ChatStore.applySlidingWindow(msgs, keepLastNUserTurns: 5)
        XCTAssertEqual(out.map(\.content), ["a", "A"])
    }

    /// Keeping 1 user turn drops the first user+assistant pair from
    /// a 2-turn chat. Older context is invisible to the model on
    /// the next call but still visible to the user in the UI.
    func testKeepsLastUserTurnAndItsAssistant() {
        let msgs = [user("a"), assistant("A"), user("b"), assistant("B")]
        let out = ChatStore.applySlidingWindow(msgs, keepLastNUserTurns: 1)
        XCTAssertEqual(out.map(\.content), ["b", "B"])
    }

    /// System messages are never dropped — they carry the agent
    /// prompt + project inventory, which are constant across turns
    /// and tiny compared with the chat itself.
    func testSystemMessagesAreRetainedEvenWhenOlderThanCutoff() {
        let msgs = [system("S0"),
                    user("a"), assistant("A"),
                    system("S1"),
                    user("b"), assistant("B")]
        let out = ChatStore.applySlidingWindow(msgs, keepLastNUserTurns: 1)
        // S0 + S1 stay despite being older than the cutoff (`a`).
        // Only the user/assistant pair from before is dropped.
        XCTAssertEqual(out.map(\.role), [.system, .system, .user, .assistant])
        XCTAssertEqual(out.map(\.content), ["S0", "S1", "b", "B"])
    }

    /// Tool-using turns: when an assistant emits tool_calls and the
    /// store records the tool outputs back on the same StoredMessage,
    /// the windowed slice still keeps those outputs paired with the
    /// kicking user message. We test the simpler case where the tool
    /// outputs come AFTER the assistant entry in the array.
    func testKeepsToolOutputsAttachedToTheirAssistant() {
        let toolMsg = StoredMessage(role: .assistant,
                                     content: "calling tool",
                                     toolCalls: [],
                                     toolOutputs: ["result"])
        let msgs: [StoredMessage] = [
            user("first"),
            assistant("A"),
            user("now run the tool"),
            toolMsg,
        ]
        let out = ChatStore.applySlidingWindow(msgs, keepLastNUserTurns: 1)
        XCTAssertEqual(out.map(\.role), [.user, .assistant])
        XCTAssertEqual(out.last?.toolOutputs, ["result"])
    }
}
