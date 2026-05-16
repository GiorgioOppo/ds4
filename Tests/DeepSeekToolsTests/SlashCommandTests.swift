import XCTest
@testable import DeepSeekTools

final class SlashCommandTests: XCTestCase {
    func testParsesSimpleCommand() {
        let parsed = ParsedSlashCommand.parse("/help")
        XCTAssertEqual(parsed?.name, "help")
        XCTAssertEqual(parsed?.rest, "")
    }

    func testParsesCommandWithArguments() {
        let parsed = ParsedSlashCommand.parse("/mode plan")
        XCTAssertEqual(parsed?.name, "mode")
        XCTAssertEqual(parsed?.rest, "plan")
    }

    func testIgnoresLeadingSlashWithSpace() {
        // "/ something" is prose, not a command.
        XCTAssertNil(ParsedSlashCommand.parse("/ something"))
    }

    func testIgnoresPlainProse() {
        XCTAssertNil(ParsedSlashCommand.parse("ciao"))
    }

    func testLowercasesCommandName() {
        XCTAssertEqual(ParsedSlashCommand.parse("/HELP")?.name, "help")
    }

    func testBuiltInCommandSetIsStable() {
        // Sanity-check that we ship the documented set; the chat
        // command palette + docs reference these by name.
        let names = Set(BuiltInSlashCommands.all.map(\.name))
        for required in ["help", "clear", "mode", "tools", "permissions"] {
            XCTAssertTrue(names.contains(required), "missing /\(required)")
        }
    }
}
