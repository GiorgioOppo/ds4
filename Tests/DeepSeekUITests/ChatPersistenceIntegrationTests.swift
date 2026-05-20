import XCTest
@testable import DeepSeekUI

/// Integration coverage for `ChatPersistence` against the real
/// Application Support layout. Each test uses a fresh UUID and tears
/// the per-chat folder down in `tearDown`, so the user's actual chat
/// store is never touched — no `{UUID}/` collisions, no leftover
/// files.
@MainActor
final class ChatPersistenceIntegrationTests: XCTestCase {

    private var persistence: ChatPersistence!
    private var chatID: UUID!

    override func setUp() async throws {
        try await super.setUp()
        persistence = ChatPersistence()
        chatID = UUID()
    }

    override func tearDown() async throws {
        // Wipe everything the test wrote, regardless of which path
        // it touched. Best-effort; failures in tearDown shouldn't
        // mask a test failure.
        if let chatID {
            try? persistence?.deleteChat(id: chatID)
        }
        persistence = nil
        chatID = nil
        try await super.tearDown()
    }

    // MARK: writeManifestImmediate

    func testWriteManifestImmediate_createsFolderAndFile() throws {
        let manifest = ChatManifest(
            id: chatID,
            title: "PR2 integration",
            createdAt: Date(timeIntervalSince1970: 1_700_000_500),
            modelDirPath: "/Users/me/models/x",
            endpoint: nil,
            projectID: nil,
            agentID: nil,
            cumulativeCostUSD: nil,
            lastEncodedMode: nil,
            turnIDs: [],
            schemaVersion: 2)
        try persistence.writeManifestImmediate(manifest)
        XCTAssertTrue(PersistencePaths.isV2Chat(id: chatID))
        let readBack = try persistence.readManifest(id: chatID)
        XCTAssertEqual(readBack, manifest)
    }

    func testWriteManifestImmediate_overwritesAndDropsPendingDebounce() throws {
        let firstManifest = ChatManifest(
            id: chatID, title: "first", modelDirPath: "", turnIDs: [])
        let secondManifest = ChatManifest(
            id: chatID, title: "second", modelDirPath: "", turnIDs: [])
        // Queue a debounced write that hasn't flushed yet, then
        // immediately overwrite it. The immediate write must drop
        // the pending task so the older value doesn't land
        // afterwards and clobber the freshly-written one.
        persistence.scheduleManifestSave(firstManifest)
        try persistence.writeManifestImmediate(secondManifest)
        // No flushAll — the test asserts that the pending debounce
        // is gone, so reading right now must return `second`.
        let readBack = try persistence.readManifest(id: chatID)
        XCTAssertEqual(readBack.title, "second")
    }

    // MARK: deleteChat

    func testDeleteChat_wipesFolder() throws {
        let manifest = ChatManifest(
            id: chatID, title: "to-delete",
            modelDirPath: "", turnIDs: [])
        try persistence.writeManifestImmediate(manifest)
        XCTAssertTrue(PersistencePaths.isV2Chat(id: chatID))
        try persistence.deleteChat(id: chatID)
        XCTAssertFalse(PersistencePaths.isV2Chat(id: chatID))
        // readManifest must now throw — the file is gone.
        XCTAssertThrowsError(try persistence.readManifest(id: chatID))
    }
}
