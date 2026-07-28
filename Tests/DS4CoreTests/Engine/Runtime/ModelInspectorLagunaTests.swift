import XCTest
import Foundation
import DS4Core
@testable import DS4Engine

final class ModelInspectorLagunaTests: XCTestCase {
    private enum Value {
        case string(String)
        case u32(UInt32)
    }

    private func makeGGUF(_ metadata: [(String, Value)]) throws -> URL {
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
        appendU64(0) // tensor count
        appendU64(UInt64(metadata.count))
        for (key, value) in metadata {
            appendString(key)
            switch value {
            case .string(let text):
                appendU32(GGUFValueType.string.rawValue)
                appendString(text)
            case .u32(let number):
                appendU32(GGUFValueType.uint32.rawValue)
                appendU32(number)
            }
        }
        while data.count < 32 || !data.count.isMultiple(of: 32) { data.append(0) }

        let url = FileManager.default.temporaryDirectory
            .appendingPathComponent("laguna-inspector-\(UUID().uuidString).gguf")
        try data.write(to: url)
        return url
    }

    func testInspectorDescribesLagunaWithoutClaimingARuntime() throws {
        let url = try makeGGUF([
            ("general.architecture", .string("laguna")),
            ("general.name", .string("Laguna S 2.1 fixture")),
            ("laguna.block_count", .u32(48)),
            ("laguna.embedding_length", .u32(3_072)),
            ("laguna.vocab_size", .u32(100_352)),
        ])
        defer { try? FileManager.default.removeItem(at: url) }
        let model = try GGUFModel(path: url.path, metalMapping: false)
        let descriptor = try ModelInspector.inspect(model)

        XCTAssertEqual(descriptor.architecture, .laguna)
        XCTAssertEqual(descriptor.model.architecture.family, .laguna)
        XCTAssertEqual(descriptor.layerCount, 48)
        XCTAssertEqual(descriptor.model.embeddingLength, 3_072)
        XCTAssertEqual(descriptor.model.vocabularySize, 100_352)
        XCTAssertEqual(descriptor.model.name, "Laguna S 2.1 fixture")

        // The gate is off: recognized, complete frontend capabilities, but no
        // runtime claim and no generation capability.
        XCTAssertEqual(descriptor.backendAvailability, .recognizedButNotImplemented)
        XCTAssertTrue(descriptor.model.capabilities.contains(.chat))
        XCTAssertTrue(descriptor.model.capabilities.contains(.tools))
        XCTAssertTrue(descriptor.model.capabilities.contains(.reasoning))
        XCTAssertTrue(descriptor.model.capabilities.contains(.mixtureOfExperts))
        XCTAssertFalse(descriptor.model.capabilities.contains(.compressedAttention))
        XCTAssertEqual(descriptor.capabilities, [])
    }

    func testSelectorRefusesLagunaWithADistinctError() throws {
        let descriptor = try ModelInspector.inspect(
            generalArchitecture: "laguna",
            layerCount: 48
        )
        XCTAssertThrowsError(try BackendSelector.select(descriptor)) { error in
            XCTAssertEqual(
                error as? BackendSelectionError,
                .backendNotImplemented(.laguna)
            )
        }
    }

    func testFallbackNameAndFrontendPolicies() throws {
        let descriptor = try ModelInspector.inspect(generalArchitecture: "laguna")
        XCTAssertEqual(descriptor.model.name, "Laguna S 2.1")

        XCTAssertEqual(
            try TokenizerFactory.backend(generalArchitecture: "laguna"), .laguna
        )
        XCTAssertEqual(
            try ConversationBackendPolicy.backend(generalArchitecture: "laguna"),
            .lagunaNative
        )
    }

    func testLagunaBackendDefinitionStaysGatedOff() {
        XCTAssertEqual(LagunaBackendDefinition.supportedArchitecture, .laguna)
        XCTAssertEqual(LagunaBackendDefinition.expectedBlockCount, 48)
        XCTAssertFalse(LagunaBackendDefinition.runtimeEnabled,
                       "flip LagunaRuntimeGate only after the decoder port passes logits parity")
        XCTAssertEqual(LagunaBackendDefinition.runtimeCapabilities, [])
    }
}
