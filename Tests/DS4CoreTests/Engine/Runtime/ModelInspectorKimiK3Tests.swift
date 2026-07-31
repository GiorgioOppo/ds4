import XCTest
import Foundation
import DS4Core
import DS4Metal
@testable import DS4Engine

final class ModelInspectorKimiK3Tests: XCTestCase {
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
        appendU64(0)
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
        while data.count < 32 || !data.count.isMultiple(of: 32) {
            data.append(0)
        }

        let url = FileManager.default.temporaryDirectory
            .appendingPathComponent(
                "kimi-k3-inspector-\(UUID().uuidString).gguf")
        try data.write(to: url)
        return url
    }

    func testInspectorReadsThePublishedKimiK3MetadataNamespace() throws {
        let url = try makeGGUF([
            ("general.architecture", .string("kimi-k3")),
            ("general.name", .string("Kimi K3 DwarfStar Q2")),
            ("kimi-k3.block_count", .u32(93)),
            ("kimi-k3.embedding_length", .u32(7_168)),
            ("kimi-k3.vocabulary_size", .u32(163_840)),
        ])
        defer { try? FileManager.default.removeItem(at: url) }

        let model = try GGUFModel(path: url.path, metalMapping: false)
        let descriptor = try ModelInspector.inspect(model)

        XCTAssertEqual(descriptor.architecture, .kimiK3)
        XCTAssertEqual(descriptor.family, .kimi)
        XCTAssertEqual(descriptor.layerCount, KimiK3Shape.blockCount)
        XCTAssertEqual(
            descriptor.model.embeddingLength,
            KimiK3Shape.embeddingLength)
        XCTAssertEqual(
            descriptor.model.vocabularySize,
            KimiK3Shape.vocabularySize)
        XCTAssertEqual(descriptor.model.name, "Kimi K3 DwarfStar Q2")
        XCTAssertEqual(
            descriptor.backendAvailability,
            .recognizedButNotImplemented)
        XCTAssertEqual(descriptor.capabilities, [])
        XCTAssertTrue(descriptor.model.capabilities.contains(.chat))
        XCTAssertTrue(
            descriptor.model.capabilities.contains(.mixtureOfExperts))
        XCTAssertTrue(
            descriptor.model.capabilities.contains(.compressedAttention))
    }

    func testKimiFrontendAndRuntimeBoundariesStayClosed() throws {
        let descriptor = try ModelInspector.inspect(
            generalArchitecture: "kimi-k3",
            layerCount: KimiK3Shape.blockCount)
        XCTAssertEqual(descriptor.model.name, "Kimi K3")
        XCTAssertFalse(KimiK3BackendDefinition.runtimeEnabled)
        XCTAssertEqual(KimiK3BackendDefinition.runtimeCapabilities, [])

        XCTAssertThrowsError(try BackendSelector.select(descriptor)) {
            XCTAssertEqual(
                $0 as? BackendSelectionError,
                .backendNotImplemented(.kimiK3))
        }
        XCTAssertThrowsError(
            try TokenizerFactory.backend(generalArchitecture: "kimi-k3")
        )
        XCTAssertThrowsError(
            try ConversationBackendPolicy.backend(
                generalArchitecture: "kimi-k3")
        )
    }
}
