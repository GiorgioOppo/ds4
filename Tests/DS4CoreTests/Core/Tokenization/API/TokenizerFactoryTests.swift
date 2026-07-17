import Foundation
import XCTest
@testable import DS4Core

final class TokenizerFactoryTests: XCTestCase {
    func testPureSelectionRoutesDeepSeekAndGLMWithoutRuntimeFallback() throws {
        let deepSeek = try ModelArchitectureDetector.detect(generalArchitecture: "deepseek4")
        XCTAssertEqual(try TokenizerFactory.backend(for: deepSeek), .deepSeekV4)
        XCTAssertEqual(try ConversationBackendPolicy.backend(for: deepSeek), .deepSeekDSML)

        let glm = try ModelArchitectureDetector.detect(generalArchitecture: "glm-dsa")
        XCTAssertEqual(glm.backendAvailability, .recognizedButNotImplemented)
        XCTAssertEqual(try TokenizerFactory.backend(for: glm), .glm52,
                       "frontend availability must not claim or require a GLM runtime")
        XCTAssertEqual(try ConversationBackendPolicy.backend(for: glm), .glm52Native)
    }

    func testFactoryConstructsBothImplementedTokenizers() throws {
        let deepURL = try tokenizerGGUF(
            architecture: "deepseek4",
            tokenizerModel: "gpt2",
            pretokenizer: "joyai-llm",
            tokens: [
                "<｜begin▁of▁sentence｜>", "<｜end▁of▁sentence｜>",
                "<｜User｜>", "<｜Assistant｜>", "<think>", "</think>", "｜DSML｜",
            ]
        )
        defer { try? FileManager.default.removeItem(at: deepURL) }
        let deepModel = try GGUFModel(path: deepURL.path, metalMapping: false)
        let deepTokenizer = try TokenizerFactory.make(for: deepModel)
        XCTAssertTrue(deepTokenizer is DeepSeekV4Tokenizer)
        XCTAssertEqual(deepTokenizer.architecture, .deepSeekV4)

        let glmTokens = GLM52ConversationProtocol.controlTokens
        let glmURL = try tokenizerGGUF(
            architecture: "glm-dsa",
            tokenizerModel: "gpt2",
            pretokenizer: "glm4",
            tokens: glmTokens,
            bos: UInt32(glmTokens.firstIndex(of: "[gMASK]")!),
            eos: UInt32(glmTokens.firstIndex(of: "<|endoftext|>")!)
        )
        defer { try? FileManager.default.removeItem(at: glmURL) }
        let glmModel = try GGUFModel(path: glmURL.path, metalMapping: false)
        let glmTokenizer = try TokenizerFactory.make(for: glmModel)
        XCTAssertTrue(glmTokenizer is GLM52Tokenizer)
        XCTAssertEqual(glmTokenizer.architecture, .glmDSA)
    }

    func testQwenFailsExplicitlyInsteadOfFallingBackToDeepSeek() throws {
        let detected = try ModelArchitectureDetector.detect(
            generalArchitecture: "qwen3",
            hasDeepSeekV4Metadata: true
        )
        XCTAssertThrowsError(try TokenizerFactory.backend(for: detected)) { error in
            XCTAssertEqual(
                error as? TokenizerFactoryError,
                .tokenizerNotImplemented(ModelArchitectureID("qwen3"), family: .qwen)
            )
        }
        XCTAssertThrowsError(try ConversationBackendPolicy.backend(for: detected)) { error in
            XCTAssertEqual(
                error as? ConversationBackendSelectionError,
                .conversationBackendNotImplemented(ModelArchitectureID("qwen3"), family: .qwen)
            )
        }
    }

    func testUnknownExplicitArchitectureIsAuthoritative() {
        XCTAssertThrowsError(try TokenizerFactory.backend(
            generalArchitecture: "future-llm",
            hasDeepSeekV4Metadata: true
        )) { error in
            XCTAssertEqual(error as? TokenizerFactoryError,
                           .unsupportedArchitecture(ModelArchitectureID("future-llm")))
        }
        XCTAssertThrowsError(try ConversationBackendPolicy.backend(
            generalArchitecture: "future-llm",
            hasDeepSeekV4Metadata: true
        )) { error in
            XCTAssertEqual(error as? ConversationBackendSelectionError,
                           .unsupportedArchitecture(ModelArchitectureID("future-llm")))
        }
    }

    func testMissingArchitectureAndLegacyFallbackAreDistinct() throws {
        XCTAssertThrowsError(try TokenizerFactory.backend(generalArchitecture: nil)) { error in
            XCTAssertEqual(error as? TokenizerFactoryError, .missingArchitecture)
        }
        XCTAssertThrowsError(try ConversationBackendPolicy.backend(generalArchitecture: nil)) { error in
            XCTAssertEqual(error as? ConversationBackendSelectionError, .missingArchitecture)
        }
        XCTAssertEqual(
            try TokenizerFactory.backend(
                generalArchitecture: nil, hasDeepSeekV4Metadata: true
            ),
            .deepSeekV4
        )
        XCTAssertEqual(
            try ConversationBackendPolicy.backend(
                generalArchitecture: "  ", hasDeepSeekV4Metadata: true
            ),
            .deepSeekDSML
        )
    }

    func testConstructionWrapsTokenizerMetadataErrorsWithSelectedBackend() throws {
        let url = try tokenizerGGUF(
            architecture: "glm-dsa",
            tokenizerModel: "gpt2",
            pretokenizer: "glm4",
            tokens: []
        )
        defer { try? FileManager.default.removeItem(at: url) }
        let model = try GGUFModel(path: url.path, metalMapping: false)
        XCTAssertThrowsError(try TokenizerFactory.make(for: model)) { error in
            guard case TokenizerFactoryError.invalidTokenizerMetadata(let backend, let reason) = error else {
                return XCTFail("unexpected error: \(error)")
            }
            XCTAssertEqual(backend, .glm52)
            XCTAssertTrue(reason.contains("[gMASK]"))
        }
    }

    private enum MetadataValue {
        case string(String)
        case u32(UInt32)
        case strings([String])
    }

    private func tokenizerGGUF(
        architecture: String,
        tokenizerModel: String,
        pretokenizer: String,
        tokens: [String],
        bos: UInt32? = nil,
        eos: UInt32? = nil
    ) throws -> URL {
        var metadata: [(String, MetadataValue)] = [
            ("general.architecture", .string(architecture)),
            ("tokenizer.ggml.model", .string(tokenizerModel)),
            ("tokenizer.ggml.pre", .string(pretokenizer)),
            ("tokenizer.ggml.tokens", .strings(tokens)),
            ("tokenizer.ggml.merges", .strings([])),
        ]
        if let bos { metadata.append(("tokenizer.ggml.bos_token_id", .u32(bos))) }
        if let eos { metadata.append(("tokenizer.ggml.eos_token_id", .u32(eos))) }

        var data = Data()
        func appendU32(_ input: UInt32) {
            var value = input.littleEndian
            withUnsafeBytes(of: &value) { data.append(contentsOf: $0) }
        }
        func appendU64(_ input: UInt64) {
            var value = input.littleEndian
            withUnsafeBytes(of: &value) { data.append(contentsOf: $0) }
        }
        func appendString(_ value: String) {
            let bytes = Data(value.utf8)
            appendU64(UInt64(bytes.count))
            data.append(bytes)
        }

        appendU32(GGUF.magic)
        appendU32(3)
        appendU64(0)
        appendU64(UInt64(metadata.count))
        for (key, value) in metadata {
            appendString(key)
            switch value {
            case .string(let string):
                appendU32(GGUFValueType.string.rawValue)
                appendString(string)
            case .u32(let integer):
                appendU32(GGUFValueType.uint32.rawValue)
                appendU32(integer)
            case .strings(let strings):
                appendU32(GGUFValueType.array.rawValue)
                appendU32(GGUFValueType.string.rawValue)
                appendU64(UInt64(strings.count))
                strings.forEach(appendString)
            }
        }
        while data.count < 32 || data.count % 32 != 0 { data.append(0) }
        let url = FileManager.default.temporaryDirectory
            .appendingPathComponent("tokenizer-factory-\(UUID().uuidString).gguf")
        try data.write(to: url)
        return url
    }
}
