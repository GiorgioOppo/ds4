import Foundation
import XCTest
import DS4Core
import DS4Engine
@testable import DwarfStar

final class ChatPersistenceTests: XCTestCase {
    /// Old sessions have neither modelText nor images. They must still open and
    /// rebuild a usable text-only history after upgrading the application.
    @MainActor
    func testSessionWithoutVisionFieldsRemainsReadable() async throws {
        let legacy = Data(#"""
        {
          "id": "legacy-chat", "title": "Existing chat", "agentId": "general",
          "systemNote": "", "modelName": "Flash", "createdAt": 0, "updatedAt": 0,
          "messages": [{
            "role": "user", "reasoning": "", "text": "Explain Swift actors",
            "attachments": ["notes.txt"], "toolCalls": []
          }]
        }
        """#.utf8)
        let session = try JSONDecoder().decode(ChatSession.self, from: legacy)
        let stored = try XCTUnwrap(session.messages.first)
        XCTAssertNil(stored.modelText)
        XCTAssertNil(stored.images)

        let message = UIMessage(stored: stored)
        XCTAssertEqual(message.attachments, ["notes.txt"])
        XCTAssertTrue(message.images.isEmpty)
        XCTAssertEqual(ChatStore.chatTurns(from: [message]), [.user("Explain Swift actors")])
    }

    /// Text-file contents used to disappear when reopening a chat. Preserve the
    /// full model input alongside image bytes while keeping the visible text short.
    @MainActor
    func testSessionRoundTripPreservesImageAndImportedTextForReplay() async throws {
        let image = ChatImage(name: "diagram.png", data: Data([0x89, 0x50, 0x4E, 0x47, 0, 1, 2, 3]))
        let modelText = "--- Attached file: notes.txt ---\nThe measured value is 42.\n--- end: notes.txt ---\n\nExplain this diagram."
        let user = UIMessage(role: .user, text: "Explain this diagram.",
                             attachments: ["notes.txt"], modelText: modelText, images: [image])
        var original = ChatSession(id: "image-chat", agentId: "general")
        original.messages = [StoredMessage(from: user)]
        let encoded = try JSONEncoder().encode(original)
        let decoded = try JSONDecoder().decode(ChatSession.self, from: encoded)
        let restored = UIMessage(stored: try XCTUnwrap(decoded.messages.first))

        XCTAssertEqual(restored.text, "Explain this diagram.")
        XCTAssertEqual(restored.modelText, modelText)
        XCTAssertEqual(restored.images, [image])
        XCTAssertEqual(ChatStore.chatTurns(from: [restored]), [.user(modelText)])
        let replay = ChatStore.visionChatTurns(from: [restored])
        XCTAssertEqual(replay.count, 1)
        XCTAssertEqual(replay.first?.turn, .user(modelText))
        XCTAssertEqual(replay.first?.images, [image])
    }

    @MainActor
    func testVisionReplayKeepsImagesOnTheirOriginalUserTurn() async throws {
        let firstImage = ChatImage(name: "first.png", data: Data([1, 2]))
        let secondImage = ChatImage(name: "second.png", data: Data([3, 4]))
        let calls = [ToolCall(id: "read-1", name: "project_read", argumentsJSON: "{}")]
        let messages = [
            UIMessage(role: .system, text: "Do not duplicate the system prompt."),
            UIMessage(role: .user, text: "Compare", images: [firstImage]),
            UIMessage(role: .assistant, text: "I will inspect the file.", toolCalls: calls),
            UIMessage(role: .tool, text: "file contents"),
            UIMessage(role: .user, text: "Now this", images: [secondImage]),
            UIMessage(role: .assistant, text: "")
        ]
        let replay = ChatStore.visionChatTurns(from: messages)
        XCTAssertEqual(replay.count, 4)
        XCTAssertEqual(replay[0].images, [firstImage])
        XCTAssertEqual(replay[1].turn, .assistant(text: "I will inspect the file.", toolCalls: calls))
        XCTAssertTrue(replay[1].images.isEmpty)
        XCTAssertTrue(replay[2].images.isEmpty)
        XCTAssertEqual(replay[3].images, [secondImage])
        XCTAssertEqual(replay[3].turn, .user("Now this"))
    }

    @MainActor
    func testImageOnlyConversationGetsAStableSearchableTitle() async throws {
        let first = UIMessage(role: .user, text: "", images: [
            ChatImage(name: "schema-impianto.png", data: Data([1, 2]))
        ])
        XCTAssertEqual(ChatStore.deriveTitle(from: [first]), "schema-impianto.png")
        XCTAssertEqual(ChatStore.deriveTitle(from: [first, UIMessage(role: .user, text: "E questo dettaglio?")]),
                       "schema-impianto.png")
        XCTAssertEqual(ChatStore.deriveTitle(from: [UIMessage(role: .user, text: "", attachments: ["report.txt"])]),
                       "report.txt")
        XCTAssertEqual(ChatStore.deriveTitle(from: [UIMessage(role: .user, text: "Prima riga\nDettagli", attachments: ["report.txt"])]),
                       "Prima riga")
        XCTAssertEqual(ChatStore.deriveTitle(from: []), ChatSession.untitled)
    }
}
