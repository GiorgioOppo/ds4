import XCTest
@testable import DeepSeekUI

/// Round-trip Codable coverage for the v2 chat storage types. The
/// goal is to catch regressions where adding/removing/renaming a
/// field silently breaks on-disk format compatibility — failed
/// round-trips here mean previously persisted chats won't load after
/// the next release.
final class ChatManifestCodableTests: XCTestCase {

    // MARK: ChatManifest

    func testChatManifest_roundTrip_defaults() throws {
        let original = ChatManifest(
            id: UUID(),
            title: "Hello",
            createdAt: Date(timeIntervalSince1970: 1_700_000_000),
            modelDirPath: "/Users/me/models/deepseek-v4",
            endpoint: nil,
            projectID: nil,
            agentID: nil,
            cumulativeCostUSD: nil,
            lastEncodedMode: nil,
            turnIDs: [],
            schemaVersion: 2)
        let decoded = try roundTrip(original)
        XCTAssertEqual(decoded, original)
    }

    func testChatManifest_roundTrip_fullyPopulated() throws {
        let original = ChatManifest(
            id: UUID(),
            title: "Long title with emoji",
            createdAt: Date(timeIntervalSince1970: 1_700_000_001),
            modelDirPath: "",
            endpoint: .openRouter(modelID: "anthropic/claude-sonnet-4-6"),
            projectID: UUID(),
            agentID: UUID(),
            cumulativeCostUSD: 12.345_678,
            lastEncodedMode: "high",
            turnIDs: [UUID(), UUID(), UUID()],
            schemaVersion: 2)
        let decoded = try roundTrip(original)
        XCTAssertEqual(decoded, original)
    }

    func testChatManifest_effectiveEndpoint_prefersTyped() {
        let id = UUID()
        let withTyped = ChatManifest(
            id: id, title: "x", modelDirPath: "/legacy",
            endpoint: .anthropic(modelID: "claude-opus-4-7"),
            turnIDs: [])
        if case .anthropic(let m) = withTyped.effectiveEndpoint {
            XCTAssertEqual(m, "claude-opus-4-7")
        } else {
            XCTFail("Expected .anthropic effective endpoint")
        }
    }

    func testChatManifest_effectiveEndpoint_fallsBackToModelDirPath() {
        let m = ChatManifest(
            id: UUID(), title: "x",
            modelDirPath: "/Users/me/models/x",
            endpoint: nil,
            turnIDs: [])
        if case .localDirectory(let p) = m.effectiveEndpoint {
            XCTAssertEqual(p, "/Users/me/models/x")
        } else {
            XCTFail("Expected .localDirectory effective endpoint")
        }
    }

    // MARK: TurnSummary + TurnFlags

    func testTurnSummary_roundTrip() throws {
        let original = TurnSummary(
            id: UUID(),
            createdAt: Date(timeIntervalSince1970: 1_700_000_100),
            userMessageID: UUID(),
            userText: "Tell me about Italy",
            userTokenCount: 12,
            finalContentPreview: "Italy is a country in southern Europe…",
            finalContentIsTruncated: true,
            roundIDs: [UUID(), UUID()],
            flags: [.hasReasoning, .hasToolCalls],
            toolCallCount: 3,
            totalGeneratedTokens: 540,
            turnCostUSD: 0.0042)
        let decoded = try roundTrip(original)
        XCTAssertEqual(decoded, original)
    }

    func testTurnFlags_optionSetSemantics() throws {
        var f: TurnFlags = []
        XCTAssertFalse(f.contains(.hasReasoning))
        f.insert(.hasReasoning)
        f.insert(.hasToolCalls)
        XCTAssertTrue(f.contains(.hasReasoning))
        XCTAssertTrue(f.contains(.hasToolCalls))
        XCTAssertFalse(f.contains(.hasPrefillTrace))
        // Round-trip preserves bits exactly.
        let decoded = try roundTrip(f)
        XCTAssertEqual(decoded, f)
    }

    // MARK: StoredRound

    func testStoredRound_roundTrip_minimal() throws {
        let original = StoredRound(
            id: UUID(),
            roundIndex: 0,
            content: "Hello world")
        let decoded = try roundTrip(original)
        XCTAssertEqual(decoded, original)
    }

    func testStoredRound_roundTrip_fullyPopulated() throws {
        let original = StoredRound(
            id: UUID(),
            roundIndex: 3,
            content: "Here is the answer",
            reasoningContent: "Let me think step by step…",
            toolCalls: [
                StoredToolCall(name: "read",
                                args: #"{"path":"/tmp/x"}"#),
                StoredToolCall(name: "grep",
                                args: #"{"pattern":"foo"}"#),
            ],
            toolOutputs: ["file contents A", "matches B"],
            prefillTrace: "<bos>… prompt tokens decoded …",
            tokenCount: 128)
        let decoded = try roundTrip(original)
        XCTAssertEqual(decoded, original)
    }

    // MARK: PendingSnapshot tagged union

    func testPendingSnapshot_local() throws {
        let pt = PendingTurn(
            assistantMessageID: UUID(),
            promptTokens: [1, 2, 3],
            generatedTokens: [4, 5],
            mode: "chat")
        let snap = PendingSnapshot.local(pt)
        let decoded = try roundTrip(snap)
        XCTAssertEqual(decoded.kind, .local)
        XCTAssertEqual(decoded.local, pt)
        XCTAssertNil(decoded.remote)
    }

    func testPendingSnapshot_remote() throws {
        let rt = RemotePendingTurn(
            assistantMessageID: UUID(),
            userMessageID: UUID(),
            userText: "what time is it?",
            mode: "high",
            issuedAt: Date(timeIntervalSince1970: 1_700_000_200))
        let snap = PendingSnapshot.remote(rt)
        let decoded = try roundTrip(snap)
        XCTAssertEqual(decoded.kind, .remote)
        XCTAssertEqual(decoded.remote, rt)
        XCTAssertNil(decoded.local)
    }

    // MARK: helpers

    private func roundTrip<T: Codable & Equatable>(_ value: T) throws -> T {
        let encoder = JSONEncoder()
        encoder.dateEncodingStrategy = .iso8601
        encoder.outputFormatting = [.prettyPrinted, .sortedKeys]
        let data = try encoder.encode(value)

        let decoder = JSONDecoder()
        decoder.dateDecodingStrategy = .iso8601
        return try decoder.decode(T.self, from: data)
    }
}

/// Constructor convenience: most call sites only need the two
/// required fields, so this matches the `StoredToolCall(_ tc:
/// ToolCall)` shorthand without forcing the test to import
/// DeepSeekKit just to build a `ToolCall`.
private extension StoredToolCall {
    init(name: String, args: String, id: String? = nil) {
        self.name = name
        self.args = args
        self.id = id
    }
}
