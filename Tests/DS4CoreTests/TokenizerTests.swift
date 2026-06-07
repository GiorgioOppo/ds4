import XCTest
import Foundation
@testable import DS4Core

/// Validates the Swift tokenizer against the C tokenizer via `ds4 --dump-tokens`,
/// which prints the exact token-id stream the engine uses. Gated on both the
/// real GGUF and the built ./ds4 binary being present.
final class TokenizerTests: XCTestCase {

    static let modelPath = "/Users/oppog/Downloads/ds4-main/gguf/DeepSeek-V4-Flash-Q4KExperts-F16HC-F16Compressor-F16Indexer-Q8Attn-Q8Shared-Q8Out-chat-v2-imatrix.gguf"
    static let ds4Binary = "/Users/oppog/Downloads/ds4-main/ds4"
    static let projectRoot = "/Users/oppog/Downloads/ds4-main"

    /// Run `ds4 --dump-tokens --prompt-file <file>` and parse the first line.
    /// The prompt is passed via a file (not a process argument) so both the C
    /// tokenizer and the Swift one receive byte-identical input — passing
    /// non-ASCII as a process argument lets macOS re-normalize the UTF-8.
    private func cReference(_ prompt: String) -> [Int32]? {
        let file = FileManager.default.temporaryDirectory
            .appendingPathComponent("tok-\(UUID().uuidString).txt")
        guard (try? Data(prompt.utf8).write(to: file)) != nil else { return nil }
        defer { try? FileManager.default.removeItem(at: file) }

        let proc = Process()
        proc.executableURL = URL(fileURLWithPath: Self.ds4Binary)
        proc.arguments = ["--dump-tokens", "--prompt-file", file.path, "-m", Self.modelPath]
        proc.currentDirectoryURL = URL(fileURLWithPath: Self.projectRoot)
        let pipe = Pipe()
        proc.standardOutput = pipe
        proc.standardError = FileHandle.nullDevice
        do { try proc.run() } catch { return nil }
        let data = pipe.fileHandleForReading.readDataToEndOfFile()
        proc.waitUntilExit()
        guard let text = String(data: data, encoding: .utf8),
              let firstLine = text.split(separator: "\n").first else { return nil }
        let trimmed = firstLine.trimmingCharacters(in: CharacterSet(charactersIn: "[] \t"))
        if trimmed.isEmpty { return [] }
        return trimmed.split(separator: ",").compactMap { Int32($0.trimmingCharacters(in: .whitespaces)) }
    }

    func testTokenizerMatchesCDumpTokens() throws {
        try XCTSkipUnless(FileManager.default.fileExists(atPath: Self.modelPath), "real GGUF not present")
        try XCTSkipUnless(FileManager.default.fileExists(atPath: Self.ds4Binary), "./ds4 not built")

        let model = try GGUFModel(path: Self.modelPath, metalMapping: false)
        let tok = try Tokenizer(model: model)

        let prompts: [String] = [
            "Ciao, come stai?",
            "Hello, world!",
            "The quick brown fox; jumps>over\n",
            "func main() {\n    int x = 42;\n    return x;\n}",
            "    four-space indent\n\ttab indent",
            "123 4567 89 1000000",
            "a.b.c-d_e/f\\g",
            "line1\nline2\n\n\nline3",
            "Перевод на русский язык",
            "日本語のテキストです",
            "emoji 😀 mixed テスト end",
            "Trailing spaces    \nand >;\n code",
            "<｜User｜>hi there<｜Assistant｜>ok",
            "<think>reasoning</think>answer",
            "Mixed｜DSML｜marker test",
            "UPPER lower MiXeD CaSe",
            "punctuation!!! ??? ...,,,;;;",
            "tab\tseparated\tvalues",
        ]

        for prompt in prompts {
            guard let expected = cReference(prompt) else {
                XCTFail("could not get C reference for \(prompt.debugDescription)")
                continue
            }
            let got = tok.tokenizeRenderedChat(prompt)
            XCTAssertEqual(got, expected,
                           "tokenization mismatch for \(prompt.debugDescription)\n  C:     \(expected)\n  Swift: \(got)")
        }
    }

    /// Round-trip: decoding the C token ids should reproduce the prompt bytes
    /// (for prompts with no special tokens, which don't round-trip to text).
    func testDetokenizeRoundTrip() throws {
        try XCTSkipUnless(FileManager.default.fileExists(atPath: Self.modelPath), "real GGUF not present")
        let model = try GGUFModel(path: Self.modelPath, metalMapping: false)
        let tok = try Tokenizer(model: model)

        let prompts = ["Hello, world!", "func f() { return 42; }", "Ciao città però", "日本語"]
        for prompt in prompts {
            let ids = tok.tokenize(prompt)
            var bytes: [UInt8] = []
            for id in ids { bytes.append(contentsOf: tok.tokenText(id)) }
            XCTAssertEqual(bytes, Array(prompt.utf8), "round-trip failed for \(prompt.debugDescription)")
        }
    }
}
