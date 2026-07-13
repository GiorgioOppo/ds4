import XCTest
import Foundation
import DS4Engine

/// Validates the native tokenizer diagnostic (DS4Engine.Diagnostics) against the
/// real model's tokenizer (the underlying Tokenizer is already cross-checked vs
/// ./ds4 in TokenizerTests; this confirms the dump wrapper).
final class DiagnosticsTests: XCTestCase {
    static let gguf = "/Users/oppog/Downloads/ds4-main/gguf/DeepSeek-V4-Flash-Q4KExperts-F16HC-F16Compressor-F16Indexer-Q8Attn-Q8Shared-Q8Out-chat-v2-imatrix.gguf"

    func testDumpTokens() throws {
        try XCTSkipUnless(FileManager.default.fileExists(atPath: Self.gguf), "GGUF not present")
        let dump = try Diagnostics.dumpTokens(modelPath: Self.gguf, text: "Ciao, come stai?")
        // First line is the token count; body has id<TAB>text lines.
        let lines = dump.split(separator: "\n", omittingEmptySubsequences: true)
        XCTAssertTrue(lines.first?.contains("token") ?? false, "missing count header: \(dump.prefix(40))")
        XCTAssertTrue(dump.contains("\t"), "no id<TAB>text lines")
        let n = Int(lines.first?.split(separator: " ").first ?? "0") ?? 0
        XCTAssertGreaterThan(n, 0)
        print("  dumpTokens: \(n) tokens for 'Ciao, come stai?'")
    }
}
