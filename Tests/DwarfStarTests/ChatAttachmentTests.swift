import Foundation
import XCTest
@testable import DwarfStar

final class ChatAttachmentTests: XCTestCase {
    @MainActor
    func testTextDecoderRejectsBinaryControlBytes() async throws {
        let url = FileManager.default.temporaryDirectory
            .appendingPathComponent("dwarfstar-binary-\(UUID().uuidString)")
        defer { try? FileManager.default.removeItem(at: url) }
        for control: UInt8 in [0, 1, 8, 14, 31] {
            try Data([0x41, control, 0x42]).write(to: url)
            XCTAssertNil(ChatStore.readText(url), "Control byte \(control) must not enter the prompt via Latin-1 fallback.")
        }
    }

    @MainActor
    func testTextDecoderAcceptsUTF8WhitespaceAndLegacyLatin1() async throws {
        let url = FileManager.default.temporaryDirectory
            .appendingPathComponent("dwarfstar-text-\(UUID().uuidString)")
        defer { try? FileManager.default.removeItem(at: url) }
        let unicode = "Città 🪐\t42\r\nSeconda riga\n"
        try Data(unicode.utf8).write(to: url)
        XCTAssertEqual(ChatStore.readText(url), unicode)
        try Data([0x63, 0x61, 0x66, 0xE9]).write(to: url)
        XCTAssertEqual(ChatStore.readText(url), "café")
    }

    @MainActor
    func testImageBytesAreNeverFoldedIntoTheTextPrompt() async throws {
        let text = ChatAttachment(name: "notes.txt", content: "Document contents")
        let image = ChatAttachment(name: "private-image.png", content: "Must not be serialized as text",
                                   imageData: Data([0x89, 0x50, 0, 0x47]))
        let composed = ChatStore.composeUserText(typed: "Describe the diagram", attachments: [image, text])
        XCTAssertTrue(composed.contains("Document contents"))
        XCTAssertTrue(composed.hasSuffix("Describe the diagram"))
        XCTAssertFalse(composed.contains("private-image.png"))
        XCTAssertFalse(composed.contains("Must not be serialized as text"))
        XCTAssertEqual(ChatStore.composeUserText(typed: "", attachments: [image]), "")
        XCTAssertEqual(image.bytes, 4)
    }
}
