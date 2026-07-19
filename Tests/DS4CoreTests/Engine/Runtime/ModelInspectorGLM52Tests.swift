import XCTest
import Foundation
import DS4Core
@testable import DS4Engine

final class ModelInspectorGLM52Tests: XCTestCase {
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
            .appendingPathComponent("glm-inspector-\(UUID().uuidString).gguf")
        try data.write(to: url)
        return url
    }

    func testInspectorUsesHyphenatedGLMMetadataNamespace() throws {
        let url = try makeGGUF([
            ("general.architecture", .string("glm-dsa")),
            ("general.name", .string("GLM 5.2 fixture")),
            // Deliberately conflicting normalized keys prove that the inspector
            // reads the wire-format namespace, not ModelArchitectureID.rawValue.
            ("glmdsa.block_count", .u32(12)),
            ("glm-dsa.block_count", .u32(79)),
            ("glm-dsa.embedding_length", .u32(6_144)),
            ("glm-dsa.vocab_size", .u32(154_880)),
        ])
        defer { try? FileManager.default.removeItem(at: url) }

        let model = try GGUFModel(path: url.path, metalMapping: false,
                                  prefetchCPU: false)
        let descriptor = try ModelInspector.inspect(model)

        XCTAssertEqual(descriptor.architecture, .glmDSA)
        XCTAssertEqual(descriptor.displayName, "GLM 5.2 fixture")
        XCTAssertEqual(descriptor.layerCount, 79)
        XCTAssertEqual(descriptor.embeddingLength, 6_144)
        XCTAssertEqual(descriptor.vocabularySize, 154_880)
        // La disponibilità segue il gate runtime GLM (overlay dell'inspector
        // sulla detection statica di DS4Core); le capability runtime sono
        // quelle dichiarate dal backend in entrambi gli stati del gate.
        XCTAssertEqual(descriptor.backendAvailability,
                       GLM52BackendDefinition.runtimeEnabled
                           ? .implemented : .recognizedButNotImplemented)
        XCTAssertEqual(descriptor.capabilities,
                       GLM52BackendDefinition.runtimeCapabilities)
    }
}
